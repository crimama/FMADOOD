from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_remote.sh"


def _fake_tool(path: Path, log: Path, *, env_log: Path | None = None) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        f"pathlib.Path({str(log)!r}).write_bytes(b'\\0'.join(a.encode() for a in sys.argv[1:]))\n"
        + (
            f"pathlib.Path({str(env_log)!r}).write_text(\n"
            "    'PASSWORD_PRESENT=' + str(int(bool(os.environ.get('SSH_PASSWORD')))) + '\\n'\n"
            "    'ASKPASS_CONFIGURED=' + str(int(bool(os.environ.get('SSH_ASKPASS')))) + '\\n'\n"
            "    'ASKPASS_REQUIRE_FORCE=' + str(\n"
            "        int(os.environ.get('SSH_ASKPASS_REQUIRE') == 'force')\n"
            "    ) + '\\n'\n"
            ")\n"
            if env_log
            else ""
        )
        + "sys.exit(0)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _run_sync(
    tmp_path: Path,
    *,
    password: str = "",
    sync_delete: str = "0",
    remote_repo_dir: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str], dict[str, str]]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ssh_log = tmp_path / "ssh.argv"
    rsync_log = tmp_path / "rsync.argv"
    rsync_env = tmp_path / "rsync.env"
    _fake_tool(fake_bin / "ssh", ssh_log)
    _fake_tool(fake_bin / "rsync", rsync_log, env_log=rsync_env)
    if remote_repo_dir is None:
        remote_repo_dir = str(tmp_path / "repo")
    env = os.environ.copy()
    env.update(
        {
            "PATH": str(fake_bin) + os.pathsep + env["PATH"],
            "REMOTE_HOST": "user@example.invalid",
            "REMOTE_PORT": "2207",
            "JUMP_HOST": "jump.example.invalid",
            "JUMP_PORT": "2208",
            "LOCAL_RESULTS_DIR": str(tmp_path / "results"),
            "SSH_PASSWORD": password,
            "SYNC_DELETE": sync_delete,
            "REMOTE_REPO_DIR": remote_repo_dir,
        },
    )
    result = subprocess.run(  # noqa: S603
        ["/bin/bash", str(SCRIPT), "sync"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    argv = rsync_log.read_bytes().split(b"\0") if rsync_log.exists() else []
    argv_text = [item.decode() for item in argv if item]
    rsync_env_text = rsync_env.read_text(encoding="utf-8") if rsync_env.exists() else ""
    captured = {
        "rsync.argv": "\0".join(argv_text),
        "rsync.env": rsync_env_text,
        "ssh.argv": ssh_log.read_bytes().decode() if ssh_log.exists() else "",
    }
    return result, argv_text, captured


def test_password_never_enters_rsync_argv_and_is_inherited_by_rsync(tmp_path: Path) -> None:
    sentinel = "sentinel-password-never-argv"
    result, argv, captured = _run_sync(tmp_path, password=sentinel)

    assert result.returncode == 0, result.stderr
    assert sentinel not in "\0".join(argv)
    assert sentinel not in result.stdout + result.stderr
    assert captured["rsync.env"] == (
        "PASSWORD_PRESENT=1\nASKPASS_CONFIGURED=1\nASKPASS_REQUIRE_FORCE=1\n"
    )
    assert all(sentinel not in value for value in captured.values())
    assert "-e" in argv
    e_value = argv[argv.index("-e") + 1]
    assert sentinel not in e_value
    assert e_value.startswith("ssh ")
    assert "-J jump.example.invalid:2208" in e_value
    assert "-p 2207" in e_value


def test_sync_delete_remains_opt_in(tmp_path: Path) -> None:
    result, argv, _ = _run_sync(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "--delete" not in argv


def test_relative_repo_target_remains_supported_without_delete(tmp_path: Path) -> None:
    result, argv, _ = _run_sync(
        tmp_path, sync_delete="0", remote_repo_dir="relative/repo",
    )

    assert result.returncode == 0, result.stderr
    assert "--delete" not in argv


def test_explicit_empty_repo_target_uses_original_default_without_delete(tmp_path: Path) -> None:
    result, argv, _ = _run_sync(tmp_path, sync_delete="0", remote_repo_dir="")

    assert result.returncode == 0, result.stderr
    assert "--delete" not in argv
    assert "[user@example.invalid]:/tmp/fmad-ood-remote/" in argv


def test_explicit_empty_repo_target_rejected_with_delete(tmp_path: Path) -> None:
    result, argv, captured = _run_sync(tmp_path, sync_delete="1", remote_repo_dir="")

    assert result.returncode != 0
    assert not argv
    assert all(not value for value in captured.values())


@pytest.mark.parametrize(
    "hostile",
    [
        "repo'quote",
        'repo"quote',
        "repo;touch sentinel",
        "repo$(touch sentinel)",
        "repo`touch sentinel`",
        "repo with whitespace",
        "repo\twith-tab",
        "repo\nwith-newline",
        "repo|pipe",
        "repo&background",
        "repo*glob",
        "repo>redirect",
        "repo<redirect",
        "repo\\control",
    ],
)
def test_shell_hostile_repo_targets_rejected_before_fake_tools(
    tmp_path: Path, hostile: str,
) -> None:
    result, argv, captured = _run_sync(tmp_path, remote_repo_dir=str(tmp_path) + "/" + hostile)

    assert result.returncode != 0
    assert not argv
    assert all(not value for value in captured.values())


@pytest.mark.parametrize(
    "target",
    ["", "relative/path", "/", "traversal", "double-slash", "dot", "trailing-slash"],
)
def test_sync_delete_rejects_unsafe_repo_targets(tmp_path: Path, target: str) -> None:
    absolute_variants = {
        "traversal": str(tmp_path) + "/../unsafe",
        "double-slash": str(tmp_path) + "//unsafe",
        "dot": str(tmp_path) + "/./unsafe",
        "trailing-slash": str(tmp_path / "unsafe") + "/",
    }
    target = absolute_variants.get(target, target)
    result, argv, captured = _run_sync(tmp_path, sync_delete="1", remote_repo_dir=target)

    assert result.returncode != 0
    assert not argv
    assert all(not value for value in captured.values())


def test_sync_delete_allows_normal_absolute_repo_target(tmp_path: Path) -> None:
    result, argv, _ = _run_sync(
        tmp_path, sync_delete="1", remote_repo_dir=str(tmp_path / "repo"),
    )

    assert result.returncode == 0, result.stderr
    assert "--delete" in argv


def test_rsync_password_assignment_is_not_external_env_argv() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    run_rsync = source.split("run_rsync() {", 1)[1].split("\ndetect_local_data_root", 1)[0]

    assert ("env " + "SSH_PASSWORD=") not in run_rsync
    assert 'SSH_PASSWORD="${SSH_PASSWORD}" DISPLAY=' in run_rsync


def test_secret_files_are_excluded_but_env_example_is_transferred(tmp_path: Path) -> None:
    rsync = shutil.which("rsync")
    if rsync is None:
        pytest.skip("real local rsync unavailable; transfer-set check is skipped")
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "configs" / "remote").mkdir(parents=True)
    (source / "nested").mkdir()
    for relative in (
        "safe.txt",
        "configs/remote/settings.env.example",
        "nested/service.env.example",
        ".env",
        "nested/secret.env",
        "configs/remote/147.47.134.98.env",
        "nested/id_rsa",
        "nested/id_ed25519.pub",
        "nested/server.pem",
        "nested/server.key",
    ):
        file_path = source / relative
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("sentinel-secret\n", encoding="utf-8")
    excludes = [
        "--include=**/*.env.example",
        "--exclude=configs/remote/*.env",
        "--exclude=**/.env",
        "--exclude=**/*.env",
        "--exclude=**/id_rsa",
        "--exclude=**/id_rsa.*",
        "--exclude=**/id_ed25519",
        "--exclude=**/id_ed25519.*",
        "--exclude=**/*.pem",
        "--exclude=**/*.key",
    ]
    result = subprocess.run(  # noqa: S603
        [rsync, "-ani", "--out-format=%n", *excludes, f"{source}/", f"{destination}/"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    transfer_set = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    assert "safe.txt" in transfer_set
    assert "configs/remote/settings.env.example" in transfer_set
    assert "nested/service.env.example" in transfer_set
    assert ".env" not in transfer_set
    assert "nested/secret.env" not in transfer_set
    assert "configs/remote/147.47.134.98.env" not in transfer_set
    assert not any(
        name.startswith("nested/id_") or name.endswith((".pem", ".key"))
        for name in transfer_set
    )
