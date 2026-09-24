"""Offline behavioral checks: no real SSH, Docker, Git, or GPU operations."""

import json
import fcntl
import os
from pathlib import Path
import subprocess
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_remote.sh"


def remote(tmp_path, mode, *args, state="running", gpu="GPU-test", dirty=False,
           script=SCRIPT, extra_env=None):
    binary = tmp_path / "bin"
    binary.mkdir(exist_ok=True)
    log = tmp_path / "calls.jsonl"
    ssh = binary / "ssh"
    ssh.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, shlex, subprocess, sys\n"
        "command = sys.argv[-1]\n"
        "with open(os.environ['CALL_LOG'], 'a') as f: f.write(json.dumps(command)+'\\n')\n"
        "args = shlex.split(command)\n"
        "if args[:3] == ['docker', 'ps', '-q']:\n"
        " print('container-id' if os.environ['STATE']=='running' else '')\n"
        "elif args[:3] == ['docker', 'ps', '-aq']:\n"
        " print('container-id' if os.environ['STATE']!='missing' else '')\n"
        "elif args[:2] == ['docker', 'exec']:\n"
        " script=args[-1].replace('/tmp/fmad-', os.environ['TEST_ROOT']+'/fmad-')\n"
        " sys.exit(subprocess.run(['bash','-c',script]).returncode)\n"
        "elif args[:2] == ['mkdir', '-p'] and os.environ.get('LOCAL_SYNC_TEST')=='1':\n"
        " sys.exit(subprocess.run(args).returncode)\n",
        encoding="utf-8",
    )
    ssh.chmod(0o755)
    git = binary / "git"
    git.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['CALL_LOG'], 'a') as f: f.write(json.dumps(sys.argv)+'\\n')\n"
        "if sys.argv[1]=='status': print(' M changed.py' if os.environ['DIRTY']=='1' else '')\n"
        "elif sys.argv[1]=='rev-parse': print('abc123')\n",
        encoding="utf-8",
    )
    git.chmod(0o755)
    rsync = binary / "rsync"
    rsync.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, shutil, sys\n"
        "with open(os.environ['CALL_LOG'], 'a') as f: f.write(json.dumps(sys.argv)+'\\n')\n",
        encoding="utf-8",
    )
    if extra_env and extra_env.get("LOCAL_SYNC_TEST") == "1":
        with rsync.open("a") as stream:
            stream.write(
                "dest=sys.argv[-1].split(']:',1)[1]\n"
                "shutil.copyfile(sys.argv[-2],dest)\n"
                "if os.environ.get('CORRUPT_ARCHIVE')=='1': pathlib.Path(dest).write_text('invalid')\n"
            )
    rsync.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}",
               CALL_LOG=str(log), TEST_ROOT=str(tmp_path), STATE=state,
               DIRTY=str(int(dirty)), REMOTE_HOST="fake.invalid", SSH_PASSWORD="",
               REMOTE_GPU_DEVICES=gpu, CONTAINER_NAME="retained",
               REMOTE_WORKDIR="/workspace", REMOTE_REPO_DIR="/remote/repo",
               REMOTE_CODE_MODE="git", REMOTE_PYTHON_HOME="", REMOTE_TORCH_HOME="",
               REMOTE_RESULTS_DIR="/remote/results", LOCAL_RESULTS_DIR=str(tmp_path / "results"))
    env.update(extra_env or {})
    if log.exists():
        log.unlink()
    result = subprocess.run(["bash", str(script), mode, *args], env=env,
                            capture_output=True, text=True, check=False)
    calls = [json.loads(row) for row in log.read_text().splitlines()] if log.exists() else []
    return result, calls


def local_snapshot_fixture(tmp_path):
    source = tmp_path / "source"
    for name in ("src", "configs/remote", "scripts", "tests", "docker"):
        (source / name).mkdir(parents=True)
    for name in ("run.py", "pyproject.toml", "src/method.py", "configs/example.yaml"):
        (source / name).write_text("fixture")
    (source / "configs/remote/private.env").write_text("DO_NOT_TRANSFER")
    (source / "src/private.key").write_text("DO_NOT_TRANSFER")
    script = source / "scripts/run_remote.sh"
    shutil.copyfile(SCRIPT, script)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "legacy.py").write_text("preserve legacy")
    local_root = checkout / ".local-sync"
    env = {"REMOTE_CODE_MODE": "local", "REMOTE_REPO_DIR": str(checkout),
           "REMOTE_LOCAL_ROOT": str(local_root),
           "REMOTE_WORKDIR": str(local_root / "current"), "LOCAL_SYNC_TEST": "1"}
    return source, script, local_root, env


def test_local_snapshot_is_exact_secret_free_and_idempotent(tmp_path):
    source, script, local_root, env = local_snapshot_fixture(tmp_path)
    result, calls = remote(tmp_path, "update-code", script=script, extra_env=env)
    assert result.returncode == 0, result.stderr
    first = (local_root / "current").resolve()
    assert (first / "src/method.py").read_text() == "fixture"
    assert not (first / "configs/remote").exists()
    assert not (first / "src/private.key").exists()
    assert not (first / "legacy.py").exists()
    assert (local_root.parent / "legacy.py").read_text() == "preserve legacy"
    assert not any(isinstance(c, list) and c[1:2] == ["pull"] for c in calls)
    result, calls = remote(tmp_path, "update-code", script=script, extra_env=env)
    assert result.returncode == 0
    assert "no transfer" in result.stdout
    assert not any(isinstance(c, list) for c in calls)
    (source / "src/method.py").unlink()
    (source / "src/new.py").write_text("new local version")
    result, _ = remote(tmp_path, "update-code", script=script, extra_env=env)
    assert result.returncode == 0, result.stderr
    second = (local_root / "current").resolve()
    assert first != second
    assert not (second / "src/method.py").exists()
    assert (second / "src/new.py").read_text() == "new local version"
    assert (first / "src/method.py").read_text() == "fixture"


@pytest.mark.parametrize("failure", ["locked", "corrupt"])
def test_local_snapshot_does_not_activate_failed_update(tmp_path, failure):
    _, script, local_root, env = local_snapshot_fixture(tmp_path)
    with (tmp_path / "fmad-remote-execution.lock").open("w") as lock:
        if failure == "locked":
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            env["CORRUPT_ARCHIVE"] = "1"
        result, _ = remote(tmp_path, "update-code", script=script, extra_env=env)
    assert result.returncode != 0
    assert not (local_root / "current").exists()


def test_local_snapshot_refuses_wrong_workdir_without_remote_calls(tmp_path):
    _, script, _, env = local_snapshot_fixture(tmp_path)
    env["REMOTE_WORKDIR"] = "/workspace"
    result, calls = remote(tmp_path, "update-code", script=script, extra_env=env)
    assert result.returncode != 0
    assert not calls


@pytest.mark.parametrize("state", ["running", "stopped", "missing"])
def test_start_never_creates_or_removes(tmp_path, state):
    result, calls = remote(tmp_path, "start", state=state)
    assert (result.returncode == 0) == (state != "missing")
    assert any("docker start" in c for c in calls) == (state == "stopped")
    assert not any("docker rm" in c or "docker run" in c for c in calls)


def test_stop_preserves_container(tmp_path):
    result, calls = remote(tmp_path, "stop")
    assert result.returncode == 0
    assert any("docker stop" in c for c in calls)
    assert not any("docker rm" in c for c in calls)


def test_setup_fails_without_contacting_remote(tmp_path):
    result, calls = remote(tmp_path, "setup")
    assert result.returncode != 0
    assert not calls


@pytest.mark.parametrize("gpu", ["", "bad'; touch sentinel"])
def test_run_requires_explicit_safe_gpu(tmp_path, gpu):
    result, calls = remote(tmp_path, "run", "--", "true", gpu=gpu)
    assert result.returncode != 0
    assert not calls


def test_run_preserves_argv_and_records_failure_without_removing_container(tmp_path):
    values = ["space inside", "single'quote", 'double"quote', "$(touch sentinel)"]
    code = "import json,sys; print(json.dumps(sys.argv[1:])); sys.exit(7)"
    result, calls = remote(tmp_path, "run", "--", "python3", "-c", code, *values)
    assert result.returncode == 7
    assert json.dumps(values) in result.stdout
    assert not (ROOT / "sentinel").exists()
    execution = next((tmp_path / "fmad-execution-logs").iterdir())
    assert (execution / "status").read_text().strip() == "failed"
    assert (execution / "exit_code").read_text().strip() == "7"
    assert json.dumps(values) in (execution / "stdout.log").read_text()
    assert any("CUDA_VISIBLE_DEVICES='GPU-test'" in c for c in calls if isinstance(c, str))
    assert not any("docker rm" in c for c in calls if isinstance(c, str))


@pytest.mark.parametrize("dirty", [False, True])
def test_update_code_only_fast_forwards_clean_checkout(tmp_path, dirty):
    result, calls = remote(tmp_path, "update-code", dirty=dirty)
    assert (result.returncode == 0) == (not dirty)
    pulls = [c for c in calls if isinstance(c, list) and c[1:2] == ["pull"]]
    assert len(pulls) == int(not dirty)
    if pulls:
        assert pulls[0][1:] == ["pull", "--ff-only"]


def test_pull_allowlists_compact_results(tmp_path):
    result, calls = remote(tmp_path, "pull")
    assert result.returncode == 0
    argv = next(c for c in calls if isinstance(c, list))
    assert "--exclude=*" in argv
    for name in ("metrics.json", "summary.tsv", "config.yaml", "run_manifest.json"):
        assert f"--include={name}" in argv
    assert "--delete" not in argv


def test_pull_metrics_only_omits_provenance(tmp_path):
    result, calls = remote(tmp_path, "pull", "--metrics-only")
    assert result.returncode == 0
    argv = next(c for c in calls if isinstance(c, list))
    assert {arg for arg in argv if arg.startswith("--include=")} == {
        "--include=*/", "--include=metrics.json", "--include=summary.tsv",
    }
    assert "--exclude=*" in argv
    assert "--delete" not in argv
    assert not any(isinstance(c, str) for c in calls)


@pytest.mark.parametrize("args", [("--unknown",), ("--metrics-only", "extra")])
def test_pull_rejects_unknown_arguments_without_remote_operations(tmp_path, args):
    result, calls = remote(tmp_path, "pull", *args)
    assert result.returncode != 0
    assert "Usage:" in result.stderr
    assert not calls
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("mode", ["run", "update-code"])
def test_checkout_lock_blocks_overlapping_work(tmp_path, mode):
    with (tmp_path / "fmad-remote-execution.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, calls = remote(tmp_path, mode, *(["--", "echo", "MUST_NOT_RUN"] if mode == "run" else []))
    assert result.returncode != 0
    assert "MUST_NOT_RUN\n" not in result.stdout
    assert not any(isinstance(c, list) and c[1:2] == ["pull"] for c in calls)


def test_successful_run_starts_stopped_container_and_retains_logs(tmp_path):
    result, calls = remote(tmp_path, "run", "--", "printf", "%s", "completed-output", state="stopped")
    assert result.returncode == 0
    assert any("docker start" in c for c in calls if isinstance(c, str))
    execution = next((tmp_path / "fmad-execution-logs").iterdir())
    assert (execution / "status").read_text().strip() == "completed"
    assert (execution / "stdout.log").read_text() == "completed-output"


@pytest.mark.parametrize("metrics_only", [False, True])
def test_real_local_rsync_filter_omits_large_outputs(tmp_path, metrics_only):
    rsync = shutil.which("rsync")
    if rsync is None:
        pytest.skip("rsync unavailable")
    result, calls = remote(tmp_path, "pull", *(["--metrics-only"] if metrics_only else []))
    assert result.returncode == 0
    argv = next(c for c in calls if isinstance(c, list))[1:-2]
    pos = argv.index("-e")
    del argv[pos:pos + 2]
    source = tmp_path / "source"
    dest = tmp_path / "filtered"
    source.mkdir()
    dest.mkdir()
    retained = {"seed0/run1/metrics.json", "seed0/run1/config.yaml",
                "seed0/run1/run_manifest.json", "summary.tsv"}
    excluded = {"seed0/run1/training.jsonl", "seed0/run1/anomaly_maps/map.npz",
                "seed0/run1/weights/state.pt", "seed0/run1/stdout.log"}
    if metrics_only:
        provenance = {"seed0/run1/config.yaml", "seed0/run1/run_manifest.json"}
        retained -= provenance
        excluded |= provenance
    for name in retained | excluded:
        item = source / name
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_text("fixture")
    copied = subprocess.run([rsync, *argv, f"{source}/", f"{dest}/"], capture_output=True, text=True)
    assert copied.returncode == 0, copied.stderr
    assert {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()} == retained
