import os
from pathlib import Path
import shutil
import subprocess
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["dry", "launch", "existing"])
@pytest.mark.parametrize("suite,prefix", [("query_reference", "QR"), ("reference_consensus", "RC"),
                                          ("mode_reference", "MC"), ("validated_reference", "VR"),
                                          ("correspondence_guard", "CG")])
def test_query_reference_recipe_delegates_with_fresh_output_guard(tmp_path, mode, suite, prefix):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    recipe = scripts / "run_query_reference.sh"
    shutil.copy2(ROOT / "scripts/run_query_reference.sh", recipe)
    preset = tmp_path / "test.env"
    preset.write_text("REMOTE_CODE_MODE=local\nCONTAINER_NAME=existing-container\n")
    calls = tmp_path / "calls"
    _write_executable(scripts / "run_remote.sh", f'''#!/usr/bin/env bash
echo "remote $*" >> {calls}
''')
    _write_executable(scripts / "run_experiments.sh", f'''#!/usr/bin/env bash
echo "launch $*" >> {calls}
''')
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _write_executable(bindir / "docker", f'''#!/usr/bin/env bash
echo "docker $*" >> {calls}
if [[ "$1" == inspect && "${{2:-}}" == -f ]]; then echo true; fi
if [[ "$1" == exec ]]; then exit {9 if mode == 'existing' else 0}; fi
exit 0
''')
    labels = " ".join(f"{prefix}{stage}" for stage in range(4))
    env = dict(os.environ, REMOTE_PRESET=str(preset), RUN_ID=f"{suite}_test", SUITE=suite,
               PATH=f"{bindir}:{os.environ['PATH']}")
    result = subprocess.run(["bash", str(recipe)] + (["--dry-run"] if mode == "dry" else []),
                            env=env, cwd=tmp_path, text=True, capture_output=True)
    if mode == "dry":
        assert result.returncode == 0 and not calls.exists()
        assert "--backend both" in result.stdout and labels in result.stdout
    elif mode == "existing":
        assert result.returncode == 9
        assert "remote " not in calls.read_text() and "launch " not in calls.read_text()
    else:
        assert result.returncode == 0, result.stderr
        actual = calls.read_text()
        assert actual.index("remote exec -- test ! -e") < actual.index("remote update-code")
        assert actual.index("remote update-code") < actual.index("launch --backend both")
        workers = 2 if suite == "correspondence_guard" else 1
        sharding = "stages" if suite == "correspondence_guard" else "objects"
        assert f"--labels {labels}" in actual and f"--workers {workers}" in actual
        assert f"--remote-shard-by {sharding}" in actual
        assert "--local-objects can fabric" in actual
        assert "--remote-objects fruit_jelly rice sheet_metal vial wallplugs walnuts" in actual


def _copy_runner(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(ROOT / "scripts/run_experiments.sh", scripts / "run_experiments.sh")
    return scripts / "run_experiments.sh", scripts


def _write_executable(path, body):
    path.write_text(body)
    path.chmod(0o755)


def test_local_mock_docker_does_not_precreate_lane_and_waits(tmp_path):
    runner, scripts = _copy_runner(tmp_path)
    (tmp_path / "config.yaml").write_text("method: example\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "docker.calls"
    _write_executable(bindir / "docker", f'''#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> {calls}
if [[ "$1" == inspect ]]; then exit 0; fi
if [[ "$1" == exec ]]; then
  [[ "$*" == *"mkdir -p {tmp_path / 'runtime'}"* ]]
  [[ "$*" != *"mkdir -p {tmp_path / 'runtime' / 'lane_0'}"* ]]
  exit 0
fi
exit 3
''')
    output = tmp_path / "runtime"
    logdir = tmp_path / "host-logs"
    command = [str(runner), "--backend", "local-docker", "--configs", str(tmp_path / "config.yaml"),
               "--labels", "N0", "--objects", "can", "--devices", "GPU-test",
               "--output-root", str(output), "--log-dir", str(logdir)]
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
    result = subprocess.run(command, env=env, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert not (output / "lane_0").exists()
    assert (logdir / "lane_0.launch.log").is_file()


def test_remote_mock_script_has_newlines_and_propagates_failure(tmp_path):
    runner, scripts = _copy_runner(tmp_path)
    (tmp_path / "config.yaml").write_text("method: example\n")
    captured = tmp_path / "remote-script"
    custom_bin = tmp_path / "custom-bin"
    custom_bin.mkdir()
    _write_executable(custom_bin / "python3", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(scripts / "run_remote.sh", f'''#!/usr/bin/env bash
set -euo pipefail
printf '%s' "${{!#}}" > {captured}
bash -n <(printf '%s' "${{!#}}")
PATH={custom_bin}:$PATH bash -c "${{!#}}"
exit 7
''')
    output = tmp_path / "runtime"
    command = [str(runner), "--backend", "remote", "--configs", str(tmp_path / "config.yaml"),
               "--labels", "N0", "--objects", "can", "fabric", "--devices", "GPU-a", "GPU-b",
               "--output-root", str(output), "--log-dir", str(tmp_path / "host-logs")]
    env = dict(os.environ, REMOTE_GPU_DEVICES="GPU-a,GPU-b")
    result = subprocess.run(command, env=env, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 7
    generated = captured.read_text()
    assert "\n" in generated
    assert "pids=()" in generated
    assert "wait \"$pid\"" in generated
    assert "bash -c" in generated


@pytest.mark.parametrize("failure", [0, 9])
def test_combined_backend_launches_all_lanes_and_waits(tmp_path, failure):
    runner, scripts = _copy_runner(tmp_path)
    (tmp_path / "config.yaml").write_text("method: example\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls"
    _write_executable(bindir / "docker", f'''#!/usr/bin/env bash
if [[ "$1" == inspect ]]; then exit 0; fi
echo "$*" >> {calls}
exit {failure}
''')
    _write_executable(scripts / "run_remote.sh", f'''#!/usr/bin/env bash
printf '%s' "${{!#}}" >> {calls}
exit 0
''')
    command = ["bash", str(runner), "--backend", "both", "--configs", str(tmp_path / "config.yaml"),
               "--labels", "N0", "--local-device", "GPU-local", "--local-objects", "can",
               "--remote-devices", "GPU-a", "GPU-b", "--remote-objects", "fabric", "rice",
               "--remote-data-root", "/remote/data", "--output-root", "/container-only/does-not-exist",
               "--log-dir", str(tmp_path / "logs"), "--resume"]
    env = dict(os.environ, REMOTE_GPU_DEVICES="GPU-a,GPU-b", PATH=f"{bindir}:{os.environ['PATH']}")
    result = subprocess.run(command, env=env, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == (1 if failure else 0), result.stderr
    actual = calls.read_text()
    assert "CUDA_VISIBLE_DEVICES=GPU-local" in actual
    assert "GPU-a" in actual and "GPU-b" in actual
    assert "/remote/data" in actual and "--resume" in actual
    assert not Path("/container-only/does-not-exist").exists()


@pytest.mark.parametrize("remote_objects,remote_devices", [(["can"], ["GPU-a"]),
    (["fabric", "rice"], ["GPU-a", "GPU-a"])])
def test_combined_duplicates_rejected_before_launch(tmp_path, remote_objects, remote_devices):
    runner, scripts = _copy_runner(tmp_path)
    (tmp_path / "config.yaml").write_text("method: example\n")
    result = subprocess.run(["bash", str(runner), "--backend", "both", "--configs",
        str(tmp_path / "config.yaml"), "--labels", "N0", "--local-device", "GPU-local",
        "--local-objects", "can", "--remote-devices", *remote_devices,
        "--remote-objects", *remote_objects, "--output-root", "/container-only/does-not-exist",
        "--log-dir", str(tmp_path / "logs")], text=True, capture_output=True)
    assert result.returncode == 2 and "duplicate" in result.stderr
    assert not (tmp_path / "logs").exists()


def test_remote_stage_sharding_runs_each_identity_once_with_two_workers(tmp_path):
    runner, scripts = _copy_runner(tmp_path)
    configs = []
    for i in range(4):
        p = tmp_path/f"config{i}.yaml"; p.write_text("method: example\n"); configs.append(str(p))
    captured = tmp_path/"remote-script"
    _write_executable(scripts/"run_remote.sh", f'''#!/usr/bin/env bash
printf '%s' "${{!#}}" > {captured}
bash -n {captured}
''')
    devices = [f"GPU-{i}" for i in range(4)]
    command = ["bash", str(runner), "--backend", "remote", "--configs", *configs,
               "--labels", "CG0", "CG1", "CG2", "CG3", "--devices", *devices,
               "--objects", "fruit_jelly", "rice", "--shard-by", "stages", "--workers", "2",
               "--output-root", "/results/isolated", "--log-dir", str(tmp_path/"logs")]
    result = subprocess.run(command, env=dict(os.environ, REMOTE_GPU_DEVICES=','.join(devices)),
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    import shlex
    invocations = [line for line in captured.read_text().splitlines() if line.startswith("CUDA_VISIBLE_DEVICES=")]
    assert len(invocations) == 4
    identities = []
    for i, line in enumerate(invocations):
        outer = shlex.split(line); inner = shlex.split(outer[outer.index('-c')+1])
        assert inner[inner.index('--labels')+1] == f"CG{i}"
        assert inner[inner.index('--workers')+1] == '2'
        assert inner[inner.index('--output-dir')+1] == f"/results/isolated/lane_{i}"
        assert inner[inner.index('--objects')+1:inner.index('--data-root')] == ['fruit_jelly','rice']
        identities.extend((f"CG{i}",o) for o in ['fruit_jelly','rice'])
    assert len(set(identities)) == 8


def test_both_mode_forwards_remote_stage_sharding_and_workers(tmp_path):
    runner, scripts = _copy_runner(tmp_path)
    for i in range(2):
        (tmp_path/f"config{i}.yaml").write_text("method: example\n")
    bindir = tmp_path/'bin'; bindir.mkdir(); calls=tmp_path/'calls'
    _write_executable(bindir/'docker',f'#!/usr/bin/env bash\nif [[ "$1" == inspect ]]; then exit 0; fi\necho "$*" >> {calls}\n')
    _write_executable(scripts/'run_remote.sh',f'#!/usr/bin/env bash\nprintf "%s" "${{!#}}" >> {calls}\n')
    result=subprocess.run(['bash',str(runner),'--backend','both','--configs',str(tmp_path/'config0.yaml'),str(tmp_path/'config1.yaml'),
        '--labels','CG0','CG1','--local-device','GPU-local','--local-objects','can',
        '--remote-devices','GPU-a','GPU-b','--remote-objects','rice',
        '--remote-shard-by','stages','--workers','2','--output-root','/results/test',
        '--log-dir',str(tmp_path/'logs')],env=dict(os.environ,REMOTE_GPU_DEVICES='GPU-a,GPU-b',PATH=f"{bindir}:{os.environ['PATH']}"),text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    actual=calls.read_text()
    assert 'GPU-a' in actual and 'GPU-b' in actual and '--workers' in actual
