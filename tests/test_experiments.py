import csv
import json
import shutil
from types import SimpleNamespace
import subprocess
import threading
import time
from pathlib import Path

import pytest
import yaml

import scripts.experiments as experiments
from scripts.experiments import LEDGER_CSV, comparison_index, replace_section, run, update


@pytest.mark.parametrize("active", [None,
    dict(label="D1", object="can"),
    dict(label="D1", workers=[dict(object="can"), dict(object="fabric")])])
def test_ledger_preserves_history_and_is_idempotent(tmp_path, monkeypatch, active):
    fields = ("row_type,method,scope,class_name,n_classes,backbone,preprocess,split,"
              "support_shots,support_policy,support_identity,seed,training_or_adapter,"
              "resolution,evaluator_metric_contract,seg_AUPRO_0.05,seg_AUROC_0.05,"
              "delta_AUPRO_vs_SuperAD,delta_AUROC_vs_SuperAD,comparison_status,source").split(",")
    ledger = tmp_path / LEDGER_CSV
    ledger.parent.mkdir(parents=True)
    with ledger.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(method="preserved", source="old", row_type="class"))
    markdown = tmp_path / "results/comparisons.md"
    markdown.write_text("# Comparison\n\nHistorical evidence stays here.\n")
    output = tmp_path / "results/local_runs/test_tail"
    run = output / "D0/run-one"
    run.mkdir(parents=True)
    status = dict(state="running", completed=[dict(label="D0", object="can")], active=active,
                  plan=dict(labels=["D0", "D1"], objects=["can", "fabric"]))
    (output / "status.json").write_text(json.dumps(status))
    metrics = {"seg_AUPRO_0.05": .3, "seg_AUROC_0.05": .7, "seg_F1_oracle": .2,
               "metric_contract": "mvtec_ad2_float16_streaming_v1",
               "coverage": dict(non_finite_prediction_count=0, shape_match_count=2, prediction_map_count=2)}
    (run / "metrics.json").write_text(json.dumps(dict(can=metrics)))
    manifest = dict(config_hash="config", run_id="one", code_sha256="code", objects=dict(can=dict(
        protocol=dict(support_content_sha256="support", query_content_sha256="query", ground_truth_sha256="gt"),
        trainable_parameters=12)))
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    (run / "config.yaml").write_text(yaml.safe_dump(dict(method="any_method", seed=3,
        backbone="test_backbone", dataset_config=dict(preprocess="test"),
        protocol=dict(shots=4), method_config=dict(resolutions=[8, 12]))))
    update(tmp_path, output)
    assert "local_runs/test_tail/comparison.md" in markdown.read_text()
    report = output / "comparison.md"
    report.write_text(report.read_text() + "\nManual interpretation stays here.\n")
    markdown.write_text(replace_section(markdown.read_text(), "comparison-latest", "Newer suite summary"))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (ledger, markdown, report)}
    with monkeypatch.context() as patch:
        def unexpected_replace(*args):
            pytest.fail("unchanged collection must not replace any file")
        patch.setattr(Path, "replace", unexpected_replace)
        update(tmp_path, output)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    rows = list(csv.DictReader(ledger.open()))
    assert len(rows) == 2 and rows[0]["method"] == "preserved"
    assert rows[1]["scope"].endswith("_partial")
    assert "Historical evidence stays here." in markdown.read_text()
    assert "Newer suite summary" in markdown.read_text()
    report = output / "comparison.md"
    assert "### Completed class metrics" in report.read_text()
    assert "### Completed class metrics" not in markdown.read_text()
    assert "Previous complete-scope reference" not in report.read_text()
    assert rows[1]["method"] == "any_method_D0"
    assert rows[1]["seed"] == "3" and rows[1]["support_shots"] == "4"
    # The declared plan, not a hard-coded 8-class assumption, controls macro rows.
    status["plan"]["objects"] = ["can"]
    (output / "status.json").write_text(json.dumps(status))
    update(tmp_path, output)
    assert "Manual interpretation stays here." in report.read_text()
    rows = list(csv.DictReader(ledger.open()))
    assert len(rows) == 3 and rows[-1]["row_type"] == "macro"
    assert rows[-1]["scope"].endswith("_full1")
    # Equal lane names on different hosts retain separate reports/CSV runs;
    # only the latest summary is replaced in the compact Markdown index.
    previous_report = report.read_bytes()
    second_output = tmp_path / "results/remote_runs/test_tail"
    shutil.copytree(output, second_output)
    update(tmp_path, second_output)
    update(tmp_path, second_output)
    assert report.read_bytes() == previous_report
    assert (second_output / "comparison.md").is_file()
    assert len(list(csv.DictReader(ledger.open()))) == 5
    assert markdown.read_text().count("## Latest: test_tail") == 1
    assert markdown.read_text().count("<!-- comparison-index:start -->") == 1
    assert "remote_runs/test_tail/comparison.md" in markdown.read_text()
    assert "Historical evidence stays here." in markdown.read_text()
    assert markdown.read_text().count("| S01 | any_method_D0 |") == 2
    # An older run must not move its rows to the end or replace the latest summary.
    before = {p: p.read_bytes() for p in (ledger, markdown, report)}
    update(tmp_path, output)
    assert before == {p: p.read_bytes() for p in before}
    # A metric correction changes only this run's class/macro rows.
    relative = str(output.relative_to(tmp_path)).encode() + b"/"
    other_lines = [line for line in ledger.read_bytes().splitlines(keepends=True) if relative not in line]
    metrics["seg_AUPRO_0.05"] = .31
    (run / "metrics.json").write_text(json.dumps(dict(can=metrics)))
    update(tmp_path, output)
    assert other_lines == [line for line in ledger.read_bytes().splitlines(keepends=True) if relative not in line]
    own_rows = [r for r in csv.DictReader(ledger.open()) if r["source"].startswith(relative.decode())]
    assert len(own_rows) == 2 and all(r["seg_AUPRO_0.05"] == "0.31" for r in own_rows)
    assert "Manual interpretation stays here." in report.read_text()
    # Unmarked historical/manual reports require a targeted edit, never replacement.
    report.write_text("# Existing hand-written comparison\nKeep this interpretation.\n")
    before = {p: p.read_bytes() for p in (ledger, markdown, report)}
    with pytest.raises(ValueError, match="unmanaged comparison report"):
        update(tmp_path, output)
    assert before == {p: p.read_bytes() for p in before}
    before = ledger.read_text()
    metrics["seg_AUPRO_0.05"] = float("nan")
    (run / "metrics.json").write_text(json.dumps(dict(can=metrics)))
    with pytest.raises(ValueError):
        update(tmp_path, output)
    assert ledger.read_text() == before


def test_comparison_index_preserves_macro_scopes_seeds_missing_and_zero():
    base = dict(row_type="macro", method="same_method", scope="full8", n_classes="8",
                seed=0, support_shots="16", split="seen_development", preprocess="fixed",
                **{"seg_AUPRO_0.05": "0", "seg_AUROC_0.05": "N/A"})
    rows = [base, dict(base, seed="1"), dict(base, scope="partial", n_classes="2"),
            dict(base, row_type="class", method="class_row_must_not_be_repeated")]
    index = comparison_index(rows)
    full8, partial = index.split("### 기타 범위", 1)
    assert full8.count("same_method") == 2 and partial.count("same_method") == 1
    assert "| 1 | 16 | 0.00000 | — |" in full8
    assert "| 0 | 16 | 0.00000 | — |" in full8
    assert "class_row_must_not_be_repeated" not in index
    assert "seen_development / fixed" in index


def test_comparison_index_keeps_existing_group_numbers():
    base = dict(row_type="macro", method="old", scope="old_scope", n_classes="8",
                seed="0", support_shots="16", split="seen", preprocess="fixed",
                **{"seg_AUPRO_0.05": "0.3", "seg_AUROC_0.05": "0.7"})
    original = comparison_index([base])
    updated = comparison_index([base, dict(base, method="new", scope="new_scope")], original)
    assert "| S01 | old |" in updated and "| S02 | new |" in updated
    for line in original.splitlines():
        if line.startswith("| S01 |"):
            assert line in updated.splitlines()


def test_csv_update_preserves_raw_history_order_and_multiline_fields():
    original = ('row_type,method,source,note\r\n'
                'class,"old",history,"quoted\r\nannotation"\r\n'
                'class,target,results/run/metrics.json,"keep"\r\n'
                'class,later,other/run,"also keep"\r\n')
    row = dict(row_type="class", method="target", source="results/run/metrics.json", note="keep")
    unchanged, _ = experiments.merge_csv(original, "results/run", [row])
    assert unchanged == original
    updated, rows = experiments.merge_csv(original, "results/run", [dict(row, note="changed")])
    assert updated == original.replace('"keep"\r\n', 'changed\r\n')
    assert [r["method"] for r in rows] == ["old", "target", "later"]
    appended, rows = experiments.merge_csv(updated, "results/new", [dict(row, source="results/new/metrics.json")])
    assert appended.startswith(updated) and len(rows) == 4


def test_atomic_write_does_not_touch_equal_content(tmp_path):
    path = tmp_path / "result.csv"
    path.write_bytes(b"header\r\nvalue\r\n")
    before = path.stat()
    experiments.atomic_write(path, "header\r\nvalue\r\n")
    assert path.read_bytes() == b"header\r\nvalue\r\n"
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert path.stat().st_ino == before.st_ino


@pytest.mark.parametrize("content", ["<!-- key:start -->", "<!-- key:end -->",
    "<!-- key:end --><!-- key:start -->",
    "<!-- key:start --><!-- key:start --><!-- key:end -->"])
def test_replace_section_rejects_malformed_blocks(content):
    with pytest.raises(ValueError, match="malformed comparison block"):
        replace_section(content, "key", "new summary")


@pytest.mark.parametrize("fail", [False, True])
def test_runner_uses_cli_in_order_and_stops_on_failure(tmp_path, monkeypatch, fail):
    configs = [tmp_path / f"config{i}.yaml" for i in range(2)]
    for path in configs:
        path.write_text("method: example\n")
    args = SimpleNamespace(configs=configs, labels=["first", "second"], objects=["a", "b"],
                           output_dir=str(tmp_path / "runs"), data_root="/data/test", device="cpu",
                           timeout=10, split="test_development", resume=False)
    calls = []
    def invoke(command, **kwargs):
        calls.append(command)
        assert command[1] == str(tmp_path / "run.py")
        assert kwargs["check"] is True and kwargs["timeout"] == 10
        if fail:
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr("scripts.experiments.subprocess.run", invoke)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            run(args, tmp_path)
    else:
        run(args, tmp_path)
    status = json.loads((tmp_path / "runs/status.json").read_text())
    assert status["state"] == ("invalid_run" if fail else "complete")
    assert len(calls) == (1 if fail else 4)
    if not fail:
        assert [(v["label"], v["object"]) for v in status["completed"]] == [
            ("first", "a"), ("first", "b"), ("second", "a"), ("second", "b")]
    with pytest.raises(FileExistsError):
        run(args, tmp_path)


def test_runner_resumes_only_missing_items_and_preserves_failure(tmp_path, monkeypatch):
    configs = [tmp_path / f"config{i}.yaml" for i in range(2)]
    for path in configs:
        path.write_text("method: example\n")
    output = tmp_path / "runs"
    output.mkdir()
    plan = dict(labels=["first", "second"], objects=["a", "b"],
                configs=[str(path.resolve()) for path in configs])
    failed = dict(stage=0, label="first", object="b", command=["python"])
    status = dict(started=1, ended=2, state="invalid_run", error="SIGSEGV",
                  active=failed, completed=[dict(stage=0, label="first", object="a")],
                  plan=plan, split="test_development")
    (output / "status.json").write_text(json.dumps(status))
    (output / "first_b.log").write_text("")
    calls = []
    monkeypatch.setattr("scripts.experiments.subprocess.run",
                        lambda command, **kwargs: calls.append((command, kwargs)))
    args = SimpleNamespace(configs=configs, labels=["first", "second"], objects=["a", "b"],
                           output_dir=str(output), data_root="/data/test", device="cpu",
                           timeout=10, split="test_development", resume=True)
    run(args, tmp_path)
    observed = json.loads((output / "status.json").read_text())
    assert observed["state"] == "complete" and len(observed["completed"]) == 4
    assert len(calls) == 3
    assert [call[0][-3] for call in calls] == ["b", "a", "b"]
    assert [call[0][call[0].index("--output-dir") + 1] for call in calls] == [
        str(output / "first"), str(output / "second"), str(output / "second")]
    assert observed["failures"] == [dict(ended=2, active=failed, error="SIGSEGV")]
    assert (output / "first_b.attempt2.log").exists()


@pytest.mark.parametrize("change", ["identical", "changed", "missing", "reordered"])
def test_resume_relocated_configs_require_identical_bytes(tmp_path, monkeypatch, change):
    old_root, new_root = tmp_path / "old_release", tmp_path / "new_release"
    old_root.mkdir()
    new_root.mkdir()
    old_configs, new_configs = [], []
    for index in range(2):
        old, new = old_root / f"{index}.yaml", new_root / f"{index}.yaml"
        old.write_text(f"method: stage{index}\n")
        new.write_bytes(old.read_bytes())
        old_configs.append(str(old))
        new_configs.append(str(new))
    output = tmp_path / "runs"
    output.mkdir()
    old_plan = dict(labels=["first", "second"], objects=["a"], configs=old_configs)
    status = dict(state="invalid_run", split="test_development", plan=old_plan,
                  completed=[dict(stage=0, label="first", object="a")],
                  active=dict(stage=1, label="second", object="a"), error="device mismatch")
    if change == "changed":
        Path(new_configs[1]).write_text("method: different\n")
    elif change == "missing":
        # Keep the old file but make the recorded location unavailable.
        status["plan"]["configs"] = [old_configs[0], str(old_root / "missing.yaml")]
    elif change == "reordered":
        new_configs.reverse()
    status_path = output / "status.json"
    status_path.write_text(json.dumps(status))
    before = status_path.read_bytes()
    calls = []
    monkeypatch.setattr("scripts.experiments.subprocess.run",
                        lambda command, **kwargs: calls.append(command))
    args = SimpleNamespace(configs=new_configs, labels=["first", "second"], objects=["a"],
                           output_dir=str(output), data_root="/data/test", device="cpu",
                           timeout=10, split="test_development", resume=True)
    if change != "identical":
        with pytest.raises(ValueError, match="byte-identical"):
            run(args, tmp_path)
        assert not calls and status_path.read_bytes() == before
        return
    run(args, tmp_path)
    observed = json.loads(status_path.read_text())
    assert observed["state"] == "complete" and len(observed["completed"]) == 2
    assert len(calls) == 1 and calls[0][calls[0].index("--config") + 1] == new_configs[1]
    assert observed["plan"]["configs"] == new_configs
    relocation = observed["config_relocations"][0]
    assert relocation["previous_plan"] == old_plan
    assert relocation["plan"] == observed["plan"] and len(relocation["config_sha256"]) == 2
    assert observed["failures"][0]["error"] == "device mismatch"


def test_parallel_runner_merges_shards_in_object_order(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("method: example\n")
    output = tmp_path / "runs"
    args = SimpleNamespace(configs=[config], labels=["stage"], objects=["b", "a"],
                           output_dir=str(output), data_root="/data/test", device="cpu",
                           timeout=10, split="test_development", resume=False, workers=2)
    active = 0
    max_active = 0
    lock = threading.Lock()

    def invoke(command, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(.02)
            target = Path(command[command.index("--output-dir") + 1])
            target.mkdir(parents=True, exist_ok=True)
            (target / "summary.tsv").write_text(
                "header\n" + command[command.index("--objects") + 1] + "\n")
            (target / "example").mkdir()
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr("scripts.experiments.subprocess.run", invoke)
    run(args, tmp_path)
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "complete" and status["active"] is None
    assert [(item["label"], item["object"]) for item in status["completed"]] == [
        ("stage", "b"), ("stage", "a")]
    assert max_active == 2
    assert (output / "stage" / "example").is_dir()
    assert (output / "stage" / "summary.tsv").read_text() == "header\nb\na\n"


def test_parallel_runner_records_partial_failure_and_resume(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("method: example\n")
    output = tmp_path / "runs"
    base = dict(configs=[config], labels=["stage"], objects=["a", "b"],
                output_dir=str(output), data_root="/data/test", device="cpu",
                timeout=10, split="test_development", workers=2)
    failed_once = {"b"}
    calls = []

    def invoke(command, **kwargs):
        obj = command[command.index("--objects") + 1]
        calls.append(obj)
        if obj in failed_once:
            failed_once.remove(obj)
            raise subprocess.CalledProcessError(3, command)
        target = Path(command[command.index("--output-dir") + 1])
        target.mkdir(parents=True, exist_ok=True)
        (target / "summary.tsv").write_text("header\n" + obj + "\n")
        (target / "example").mkdir(exist_ok=True)

    monkeypatch.setattr("scripts.experiments.subprocess.run", invoke)
    with pytest.raises(subprocess.CalledProcessError):
        run(SimpleNamespace(**base, resume=False), tmp_path)
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "invalid_run"
    assert len(status["completed"]) == 1
    run(SimpleNamespace(**base, resume=True), tmp_path)
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "complete" and len(status["completed"]) == 2
    assert len(status["failures"]) == 1
    assert calls.count("b") == 2


@pytest.mark.parametrize("mutation", ["state", "plan", "split"])
def test_runner_rejects_incompatible_resume(tmp_path, mutation):
    config = tmp_path / "config.yaml"
    config.write_text("method: example\n")
    output = tmp_path / "runs"
    output.mkdir()
    recorded = dict(state="invalid_run", completed=[], split="test_development",
                    plan=dict(labels=["first"], objects=["a"], configs=[str(config.resolve())]))
    if mutation == "state":
        recorded["state"] = "running"
    elif mutation == "plan":
        recorded["plan"]["objects"] = ["b"]
    else:
        recorded["split"] = "other"
    (output / "status.json").write_text(json.dumps(recorded))
    args = SimpleNamespace(configs=[config], labels=["first"], objects=["a"],
                           output_dir=str(output), data_root="/data/test", device="cpu",
                           timeout=10, split="test_development", resume=True)
    with pytest.raises(ValueError):
        run(args, tmp_path)
