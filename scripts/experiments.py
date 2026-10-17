#!/usr/bin/env python3
"""Sequence config-driven CLI runs or collect their compact comparison results.

run always invokes the existing run.py -> fmad.cli -> common pipeline.
collect never starts/stops experiments. In a read-only container mount, run
inside the container and collect on the host using this same script.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

import yaml


METRICS = ("seg_AUPRO_0.05", "seg_AUROC_0.05", "seg_F1_oracle")
LEDGER_CSV = "results/comparisons/psma_mr_vs_history_20260909/psma_mr_vs_history_20260909.csv"
CONTENT_KEYS = ("support_content_sha256", "query_content_sha256", "ground_truth_sha256")


def atomic_write(path, text):
    if path.exists() and path.read_bytes() == text.encode("utf-8"):
        return
    temporary = path.with_name(path.name + ".experiments.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def replace_section(markdown, identity, body):
    """Replace one owned block without rewriting surrounding user notes."""
    start, end = f"<!-- {identity}:start -->", f"<!-- {identity}:end -->"
    section = start + "\n" + body.rstrip() + "\n" + end
    if start in markdown or end in markdown:
        if markdown.count(start) != 1 or markdown.count(end) != 1:
            raise ValueError(f"malformed comparison block: {identity}")
        before, remainder = markdown.split(start, 1)
        if end not in remainder:
            raise ValueError(f"malformed comparison block: {identity}")
        _, after = remainder.split(end, 1)
        return before + section + after
    title, separator, remainder = markdown.partition("\n")
    return title + "\n\n" + section + "\n" + separator + remainder


def merge_csv(original, relative, updates):
    """Replace this run's records in place; retain unrelated CSV bytes/order."""
    lines = original.splitlines(keepends=True)
    reader = csv.DictReader(io.StringIO(original, newline=""))
    fields = reader.fieldnames
    position = reader.line_num
    chunks, rows = ["".join(lines[:position])], []
    newline = "\r\n" if chunks[0].endswith("\r\n") else "\n"

    def key(row):
        return tuple(row.get(k, "") for k in ("row_type", "method", "class_name", "seed", "source"))

    def encode(row):
        stream = io.StringIO(newline="")
        csv.DictWriter(stream, fieldnames=fields, lineterminator=newline).writerow(row)
        return stream.getvalue()

    pending = {}
    for row in updates:
        row = {k: str(v) if v is not None else "" for k, v in row.items()}
        if key(row) in pending or not row["source"].startswith(relative + "/"):
            raise ValueError("duplicate or out-of-scope CSV update")
        pending[key(row)] = row
    for old in reader:
        raw = "".join(lines[position:reader.line_num])
        position = reader.line_num
        if not old["source"].startswith(relative + "/"):
            row = old
        else:
            row = pending.pop(key(old), None)
            if row is None:
                continue  # Superseded identity within this run only.
            if row != old:
                raw = encode(row)
        chunks.append(raw)
        rows.append(row)
    if pending and chunks and not chunks[-1].endswith(("\n", "\r")):
        chunks.append(newline)
    for row in pending.values():
        chunks.append(encode(row))
        rows.append(row)
    return "".join(chunks), rows


def comparison_index(rows, previous=""):
    """Render each recorded macro once; keep class/provenance detail in CSV."""
    macros = [row for row in rows if row["row_type"] == "macro"]
    scopes = list(dict.fromkeys(row["scope"] for row in macros))[::-1]

    def cell(value):
        return str("—" if value is None or value == "" else value).replace("|", "&#124;").replace("\n", " ")

    # Existing scope keys are stable: adding a suite must not renumber history.
    groups = {}
    for line in previous.splitlines():
        cells = [part.strip() for part in line.strip("|").split("|")]
        if len(cells) == 3 and cells[0].startswith("S") and cells[0][1:].isdigit():
            groups[cells[1]] = cells[0]
    next_group = max((int(group[1:]) for group in groups.values()), default=0) + 1
    for scope in scopes:
        if cell(scope) not in groups:
            groups[cell(scope)] = f"S{next_group:02d}"
            next_group += 1

    def value(row, key):
        return "—" if row[key] in ("", "N/A") else f"{float(row[key])*100:.5f}"

    lines = ["## 전체 방법 비교 — CSV macro 행", "",
             "지표 ×100. 각 실행·scope·seed를 별도 보존하며 최신 결과로 과거 행을 덮어쓰지 않습니다.",
             "묶음은 조회용이지 통제된 비교의 보증이 아닙니다. support identity·backbone·해상도·평가 계약·comparison_status는 CSV와 상세 기록을 확인하세요."]
    ordered = [row for scope in scopes for row in macros if row["scope"] == scope]
    def full8(row):
        return str(row["n_classes"]) == "8" and "partial" not in row["scope"]

    for title, selected in (("Full8", [r for r in ordered if full8(r)]),
                            ("기타 범위 — full8과 분리", [r for r in ordered if not full8(r)])):
        lines += ["", f"### {title}", "",
                  "| 묶음 | 방법 | 클래스 | Seed | Support | AUPRO@.05 | pAUROC@.05 |",
                  "|---|---|---:|---|---|---:|---:|"]
        for row in selected:
            cells = [groups[cell(row["scope"])], row["method"], row["n_classes"], row["seed"], row["support_shots"]]
            lines.append("| " + " | ".join(map(cell, cells)) + " | " + " | ".join(value(row, k) for k in METRICS[:2]) + " |")
    lines += ["", "### 비교 범위 키", "",
              "| 묶음 | CSV scope | Split / 전처리 |", "|---|---|---|"]
    for scope in scopes:
        conditions = dict.fromkeys(f"{r['split']} / {r['preprocess']}" for r in macros if r["scope"] == scope)
        lines.append(f"| {groups[cell(scope)]} | {cell(scope)} | {cell('; '.join(conditions))} |")
    return "\n".join(lines)


def wait_for_statuses(paths, timeout):
    """One-shot sequencing gate; never start after a failed predecessor."""
    deadline = time.monotonic() + timeout
    while True:
        states = [json.loads(Path(p).read_text()).get("state") for p in paths]
        if all(state == "complete" for state in states):
            return
        if any(state not in {"running", "waiting", "complete"} for state in states):
            raise RuntimeError(f"predecessor did not complete successfully: {states}")
        if time.monotonic() >= deadline:
            raise TimeoutError("predecessor wait deadline; no experiment started")
        time.sleep(min(15., max(0., deadline - time.monotonic())))


def run(args, root):
    configs = [Path(p).resolve() for p in args.configs]
    labels = args.labels or [p.stem for p in configs]
    if len(labels) != len(configs) or len(set(labels)) != len(labels):
        raise ValueError("provide one unique label per config")
    if any(not label or Path(label).name != label or label in (".", "..") for label in labels):
        raise ValueError("labels must be simple directory names")
    if len(set(args.objects)) != len(args.objects) or any(
            not obj or Path(obj).name != obj or obj in (".", "..") for obj in args.objects):
        raise ValueError("objects must be unique directory names")
    for config in configs:
        if not config.is_file():
            raise FileNotFoundError(config)
    output = Path(args.output_dir).resolve()
    environment = dict(os.environ, FMAD_DINOV2_OFFLINE="1", OMP_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    workers = getattr(args, "workers", 1)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    plan = dict(labels=labels, objects=args.objects, configs=[str(p) for p in configs])
    if args.resume:
        status_path = output / "status.json"
        if not status_path.is_file():
            raise FileNotFoundError(f"resume status missing: {status_path}")
        status = json.loads(status_path.read_text())
        if status.get("state") != "invalid_run":
            raise ValueError("resume requires an invalid_run status")
        previous_plan = status.get("plan", {})
        if (status.get("split") != args.split or
                {k: v for k, v in previous_plan.items() if k != "configs"} !=
                {k: v for k, v in plan.items() if k != "configs"}):
            raise ValueError("resume arguments do not match the recorded plan")
        if previous_plan != plan:
            # Immutable remote releases change absolute config paths after a
            # code fix. Permit relocation only with byte-identical configs;
            # preserve the original paths and content audit, never the metrics.
            old_configs = previous_plan.get("configs", [])
            if len(old_configs) != len(configs):
                raise ValueError("resume config count differs from the recorded plan")
            digests = []
            for old, new in zip(old_configs, configs):
                if not Path(old).is_file() or Path(old).read_bytes() != new.read_bytes():
                    raise ValueError("resume relocated configs must be byte-identical to recorded files")
                digests.append(hashlib.sha256(new.read_bytes()).hexdigest())
            status.setdefault("config_relocations", []).append(dict(
                timestamp=time.time(), previous_plan=previous_plan,
                plan=plan, config_sha256=digests))
            status["plan"] = plan
        completed = {(item["label"], item["object"]) for item in status.get("completed", [])}
        if len(completed) != len(status.get("completed", [])):
            raise ValueError("resume status contains duplicate completions")
        failure = dict(ended=status.get("ended"), active=status.get("active"),
                       error=status.get("error"))
        # Older sequential queues did not persist failures until resume; keep
        # that compatibility while avoiding duplicate records for parallel
        # queues, which record each failed worker at the failure boundary.
        if not status.get("failures"):
            status.setdefault("failures", []).append(failure)
        status.update(state="running", active=None, error=None, ended=None,
                      resumed=time.time())
    else:
        output.mkdir(parents=True, exist_ok=False)
        completed = set()
        status = dict(started=time.time(), state="running", completed=[], active=None,
                      plan=plan, split=args.split, primary_metric=METRICS[0],
                      secondary_metric=METRICS[1], failures=[])

    def save():
        atomic_write(output / "status.json", json.dumps(status, indent=2) + "\n")

    def command_for(stage, label, config, obj, worker_output=None):
        target = worker_output or (output / label)
        entrypoint = getattr(args, "entrypoint", None) or "run.py"
        return [sys.executable, str(root / entrypoint), "--config", str(config),
                "--data-root", args.data_root, "--output-dir", str(target),
                "--objects", obj, "--device", args.device]

    def execute(stage, label, config, obj, worker_output=None):
        command = command_for(stage, label, config, obj, worker_output)
        log_path = output / f"{label}_{obj}.log"
        if log_path.exists():
            attempt = 2
            while (candidate := output / f"{label}_{obj}.attempt{attempt}.log").exists():
                attempt += 1
            log_path = candidate
        with log_path.open("x") as log:
            subprocess.run(command, cwd=root, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, timeout=args.timeout, check=True)
        return command

    def merge_worker(label, obj, worker_output):
        """Move isolated run artifacts into the stage after a worker succeeds."""
        if not worker_output.is_dir():
            return
        stage_output = output / label

        def merge_tree(source, destination):
            destination.mkdir(parents=True, exist_ok=True)
            for child in sorted(source.iterdir(), key=lambda path: path.name):
                target = destination / child.name
                if child.is_dir():
                    if target.exists() and not target.is_dir():
                        raise FileExistsError(f"parallel artifact collision: {target}")
                    merge_tree(child, target)
                    child.rmdir()
                else:
                    if target.exists():
                        raise FileExistsError(f"parallel artifact collision: {target}")
                    shutil.move(str(child), str(target))

        for child in sorted(worker_output.iterdir(), key=lambda path: path.name):
            if child.name == "summary.tsv":
                continue
            destination = stage_output / child.name
            if child.is_dir() and destination.exists() and destination.is_dir():
                merge_tree(child, destination)
                child.rmdir()
            else:
                if destination.exists():
                    raise FileExistsError(f"parallel artifact collision: {destination}")
                shutil.move(str(child), str(destination))

    def rebuild_parallel_summary(label, objects):
        stage_output = output / label
        summaries = []
        for obj in objects:
            summary = stage_output / ".workers" / obj / "summary.tsv"
            if summary.is_file():
                lines = summary.read_text().splitlines()
                if lines:
                    if not summaries:
                        summaries.append(lines[0])
                    elif lines[0] != summaries[0]:
                        raise ValueError(f"summary.tsv header mismatch: {summary}")
                    summaries.extend(lines[1:])
        if summaries:
            atomic_write(stage_output / "summary.tsv", "\n".join(summaries) + "\n")

    after = getattr(args, "after_status", None)
    if after:
        status.update(state="waiting", predecessor_status_paths=after)
        save()
        try:
            wait_for_statuses(after, args.after_timeout)
        except Exception as error:
            status.update(state="invalid_run", error=str(error), ended=time.time())
            save()
            raise
        status.update(state="running", predecessor_completed_at=time.time())
        save()

    for stage, (label, config) in enumerate(zip(labels, configs)):
        pending = [obj for obj in args.objects if (label, obj) not in completed]
        if workers == 1 or len(pending) <= 1:
            for obj in pending:
                command = command_for(stage, label, config, obj)
                status["active"] = dict(stage=stage, label=label, object=obj, command=command)
                save()
                print(f"START {label} {obj}", flush=True)
                try:
                    execute(stage, label, config, obj)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    status.update(state="invalid_run", error=str(error), ended=time.time())
                    save()
                    raise
                status["completed"].append(dict(stage=stage, label=label, object=obj, ended=time.time()))
                print(f"DONE {label} {obj}", flush=True)
                save()
            continue

        worker_root = output / label / ".workers"
        worker_root.mkdir(parents=True, exist_ok=True)
        active = []
        for obj in pending:
            shard = worker_root / obj
            shard.mkdir(parents=True, exist_ok=True)
            command = command_for(stage, label, config, obj, shard)
            active.append(dict(stage=stage, label=label, object=obj, command=command))
        status["active"] = dict(stage=stage, label=label, workers=active)
        save()
        for item in active:
            print(f"START {label} {item['object']}", flush=True)
        errors = []
        successful = []
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = {
                pool.submit(execute, stage, label, config, item["object"],
                            worker_root / item["object"]): item
                for item in active
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    future.result()
                    successful.append(item["object"])
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    errors.append((item, error))
        for obj in [item["object"] for item in active if item["object"] in successful]:
            merge_worker(label, obj, worker_root / obj)
            status["completed"].append(dict(stage=stage, label=label, object=obj, ended=time.time()))
            print(f"DONE {label} {obj}", flush=True)
        status["completed"].sort(key=lambda item: (item["stage"], args.objects.index(item["object"])))
        if errors:
            ended = time.time()
            status.setdefault("failures", []).extend(
                dict(ended=ended, active=item, error=str(error)) for item, error in errors
            )
            item, error = errors[0]
            status.update(state="invalid_run", error=str(error), ended=time.time(), active=item)
            save()
            raise error
        rebuild_parallel_summary(label, args.objects)
        status["active"] = None
        save()
    status.update(state="complete", active=None, ended=time.time())
    save()


def combine_completed_shards(root, output, inputs, *, recover_partial=False):
    """Collect disjoint completed object shards without inventing another runner.

    Only compact evidence is copied. Original run IDs, configs and manifests
    remain byte-identical; source paths are retained in the combined status.
    Validate every shard before writing, and never replace an unowned output.
    """
    root, output = root.resolve(), output.resolve()
    inputs = [Path(p).resolve() for p in inputs]
    output.relative_to(root)
    if len(set(inputs)) != len(inputs) or not inputs:
        raise ValueError("input shards must be unique and nonempty")
    sources = [str(p.relative_to(root)) for p in inputs]
    owner = output / "shard_sources.json"
    if output.exists() and (not owner.is_file() or json.loads(owner.read_text()) != sources):
        raise ValueError("combined output is not owned by these input shards")
    labels, objects, completed, payloads, states = None, [], [], {}, []
    signatures, summaries, all_seen = {}, {}, set()
    for index, source in enumerate(inputs):
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("combined output and inputs must not overlap")
        state = json.loads((source / "status.json").read_text())
        plan = state.get("plan", {})
        shard_labels, shard_objects = plan.get("labels", []), plan.get("objects", [])
        allowed_states = {"complete", "invalid_run"} if recover_partial else {"complete"}
        if state.get("state") not in allowed_states or not shard_labels or not shard_objects:
            raise ValueError("only complete declared shards can be combined")
        if any(not isinstance(n, str) or Path(n).name != n or n in ("", ".", "..")
               for n in [*shard_labels, *shard_objects]):
            raise ValueError("invalid shard identity")
        if len(set(shard_labels)) != len(shard_labels) or len(set(shard_objects)) != len(shard_objects):
            raise ValueError("duplicate shard identities")
        if labels is None:
            labels = shard_labels
        if (not recover_partial and labels != shard_labels) or (states and state.get("split") != states[0].get("split")):
            raise ValueError("shard labels/splits disagree")
        if recover_partial:
            labels = list(dict.fromkeys([*labels, *shard_labels]))
        if not recover_partial and set(objects) & set(shard_objects):
            raise ValueError("object shards must be disjoint")
        declared = {(label, obj) for label in shard_labels for obj in shard_objects}
        entries = state.get("completed", [])
        expected = {(e["label"], e["object"]) for e in entries}
        if (len(entries) != len(expected) or not expected <= declared
                or (state["state"] == "complete" and expected != declared)):
            raise ValueError("shard completions do not cover the declared plan")
        if all_seen & expected:
            raise ValueError("completed stage/object identities must be disjoint")
        all_seen.update(expected)
        seen = set()
        for label in shard_labels:
            for path in sorted((source / label).rglob("run_manifest.json")):
                manifest = json.loads(path.read_text())
                config = yaml.safe_load(path.with_name("config.yaml").read_text())
                semantic = {k: config.get(k) for k in ("method", "dataset", "backbone", "seed",
                            "protocol", "method_config", "backbone_config", "evaluation", "output")}
                semantic["dataset_config"] = {k: v for k, v in config.get("dataset_config", {}).items()
                                              if k != "objects"}
                signature = json.dumps(semantic, sort_keys=True)
                if label in signatures and signatures[label] != signature:
                    raise ValueError("stage configuration differs across shards")
                signatures[label] = signature
                for obj in manifest["objects"]:
                    identity = (label, obj)
                    if identity not in expected or identity in seen:
                        raise ValueError("duplicate or unexpected shard artifact")
                    seen.add(identity)
                for name in ("metrics.json", "config.yaml", "run_manifest.json"):
                    original = path.with_name(name)
                    target = output / label / f"shard_{index}" / original.relative_to(source / label)
                    payloads[target] = original.read_bytes().decode("utf-8")
            summary = source / label / "summary.tsv"
            if summary.exists():
                lines = summary.read_text().splitlines()
                if lines:
                    if label in summaries and summaries[label][0] != lines[0]:
                        raise ValueError("shard summary headers disagree")
                    summaries.setdefault(label, [lines[0]]).extend(lines[1:])
        if seen != expected:
            raise ValueError("missing completed shard artifacts")
        objects = list(dict.fromkeys([*objects, *shard_objects]))
        completed.extend(dict(entry, source_shard=sources[index]) for entry in entries)
        states.append(state)
    if all_seen != {(label, obj) for label in labels for obj in objects}:
        raise ValueError("recovered shards do not complete the combined declared plan")
    for label, lines in summaries.items():
        payloads[output / label / "summary.tsv"] = "\n".join(lines) + "\n"
    state = dict(state="complete", started=min(s["started"] for s in states),
                 ended=max(s["ended"] for s in states), active=None,
                 plan=dict(labels=labels, objects=objects), completed=completed,
                 split=states[0].get("split", "undeclared"), source_shards=sources,
                 failures=[dict(source_shard=source, failure=failure)
                           for source, s in zip(sources, states) for failure in s.get("failures", [])],
                 source_states=[dict(source_shard=source, state=s["state"],
                                     error=s.get("error"), active=s.get("active"))
                                for source, s in zip(sources, states)])
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(owner, json.dumps(sources, indent=2) + "\n")
    for target, content in payloads.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(target, content)
    atomic_write(output / "status.json", json.dumps(state, indent=2) + "\n")


def _declared_subset(declared, saved):
    """True when every key the declared YAML sets equals the resolved saved value."""
    if isinstance(declared, dict):
        return isinstance(saved, dict) and all(
            key in saved and _declared_subset(value, saved[key]) for key, value in declared.items())
    return declared == saved


def _declared_items(declared, prefix=()):
    if not isinstance(declared, dict):
        return {(prefix, json.dumps(declared, sort_keys=True))}
    return set().union(*(_declared_items(v, (*prefix, k)) for k, v in declared.items())) if declared else set()


def _most_specific(matches, declared):
    """Pick the declared config that refines every other match, e.g. a guard arm over its baseline."""
    items = {label: _declared_items(declared[label]) for label in matches}
    best = [label for label in matches if all(items[other] <= items[label] for other in matches)]
    return best if len(best) == 1 else matches


def _condition_root(data_root):
    """Host-independent dataset identity: path below the DATA directory when present."""
    parts = Path(str(data_root)).parts
    for marker in ("DATA", "data"):
        if marker in parts:
            return "/".join(parts[len(parts) - parts[::-1].index(marker):])
    return str(data_root)


def _resolution_label(metadata, config):
    """Return the recorded single or multi-resolution condition for a CSV row."""
    resolutions = metadata.get("resolutions")
    if resolutions is None:
        backbone = metadata.get("backbone") or {}
        protocol = metadata.get("protocol") or {}
        context = protocol.get("context") or {}
        dataset_config = config.get("dataset_config") or {}
        backbone_config = config.get("backbone_config") or {}
        method_config = config.get("method_config") or {}
        resolutions = (
            backbone.get("resolution")
            or context.get("resolution")
            or dataset_config.get("resolution")
            or backbone_config.get("smaller_edge_size")
            or method_config.get("resolutions")
            or []
        )
    if isinstance(resolutions, (str, int, float)):
        resolutions = [resolutions]
    return "+".join(map(str, resolutions or []))


def _import_provenance(config, manifest, obj):
    """Compare scientific run identity while ignoring rerun and host output details."""
    volatile = ("elapsed", "duration", "started", "ended")
    semantic_config = {key: value for key, value in config.items()
                       if key not in ("data_root", "device", "output_dir", "debug_output", "output")}
    return (semantic_config, _condition_root(config.get("data_root")),
            # ResultsManager hashes the entire resolved config, including the
            # runtime/output fields removed above. The saved config itself is
            # the stronger identity here, so its raw hash is redundant.
            {key: value for key, value in manifest.items()
             if key not in ("run_id", "config_hash", "objects", *volatile)},
            {key: value for key, value in manifest["objects"][obj].items()
             if key not in ("artifacts", *volatile)})


def import_completed_runs(root, output, import_dirs, config_paths, labels, split, objects=None):
    """Register completed runs made outside `run` without rerunning them.

    Each run directory must match exactly one declared config (declared keys
    equal the saved resolved config); when several match, the one refining all
    others wins. All runs must share one dataset root below DATA/, so a
    corruption condition is one import scope. Identical duplicate reruns are
    dropped and recorded; conflicting duplicates are an error. The object scope
    comes from --objects or matching declared config scopes, never observed
    artifacts. Only compact evidence is copied, and an existing output must be
    owned by the same import.
    """
    root, output = root.resolve(), output.resolve()
    import_dirs = [Path(p).resolve() for p in import_dirs]
    config_paths = [Path(p).resolve() for p in config_paths or []]
    labels = labels or [p.stem for p in config_paths]
    if not config_paths or len(labels) != len(config_paths) or len(set(labels)) != len(labels):
        raise ValueError("--import-runs needs one unique label per --configs entry")
    if any(not label or Path(label).name != label or label in (".", "..") for label in labels):
        raise ValueError("labels must be simple directory names")
    if not split:
        raise ValueError("--import-runs requires --split to declare the actual split/label use")
    output.relative_to(root)
    if any(d == output or d in output.parents or output in d.parents for d in import_dirs):
        raise ValueError("import output and sources must not overlap")
    declared = {label: yaml.safe_load(path.read_text()) or {} for label, path in zip(labels, config_paths)}
    sources = dict(import_dirs=[str(d.relative_to(root)) for d in import_dirs],
                   configs=[str(p.relative_to(root)) for p in config_paths], labels=labels, split=split)
    owner = output / "import_sources.json"
    previous_sources = json.loads(owner.read_text()) if owner.is_file() else None
    # Older owned imports recorded their scope only in status.json. Honor that
    # scope on a rerun, but never infer a new import's scope from observed runs.
    owned_status = (json.loads((output / "status.json").read_text())
                    if previous_sources is not None and (output / "status.json").is_file() else None)
    legacy_owned = previous_sources == sources and owned_status is not None
    if objects is None and legacy_owned:
        objects = owned_status["plan"]["objects"]
    if objects is None and previous_sources is not None and owned_status is not None:
        if {key: value for key, value in previous_sources.items() if key != "objects"} == sources:
            objects = previous_sources.get("objects")
    declared_scopes = [config.get("dataset_config", {}).get("objects") for config in declared.values()]
    if objects is None:
        if any(scope is None for scope in declared_scopes) or any(scope != declared_scopes[0] for scope in declared_scopes):
            raise ValueError("import scope is ambiguous: supply --objects or matching dataset_config.objects")
        objects = declared_scopes[0]
    objects = list(objects)
    if (not objects or len(set(objects)) != len(objects)
            or any(not isinstance(obj, str) or not obj or Path(obj).name != obj or obj in (".", "..")
                   for obj in objects)):
        raise ValueError("import objects must be unique simple names")
    if any(scope is not None and scope != objects for scope in declared_scopes):
        raise ValueError("--objects disagrees with declared dataset_config.objects")
    if legacy_owned and objects != owned_status["plan"]["objects"]:
        raise ValueError("import output is not owned by these objects")
    if not legacy_owned:
        sources["objects"] = objects
    if output.exists() and previous_sources != sources:
        raise ValueError("import output is not owned by these sources")
    kept, dropped, data_roots, payloads = {}, [], set(), {}
    for source in import_dirs:
        for path in sorted(source.rglob("run_manifest.json")):
            run_dir = path.parent
            if not (run_dir / "metrics.json").is_file() or not (run_dir / "config.yaml").is_file():
                raise ValueError(f"incomplete run directory: {run_dir}")
            config = yaml.safe_load((run_dir / "config.yaml").read_text())
            matches = _most_specific(
                [label for label in labels if _declared_subset(declared[label], config)], declared)
            if len(matches) != 1:
                raise ValueError(f"{run_dir} matches {len(matches)} declared configs: {matches}")
            label = matches[0]
            data_roots.add(_condition_root(config.get("data_root")))
            manifest = json.loads(path.read_text())
            metrics = json.loads((run_dir / "metrics.json").read_text())
            new, duplicate = [], []
            for obj in manifest["objects"]:
                if obj not in objects:
                    raise ValueError(f"{run_dir} contains undeclared object {obj}")
                if obj not in metrics:
                    raise ValueError(f"{run_dir} lacks metrics for {obj}")
                previous = kept.get((label, obj))
                identity = (metrics[obj], _import_provenance(config, manifest, obj))
                if previous is None:
                    new.append(obj)
                elif previous[1] == identity:
                    duplicate.append(obj)
                else:
                    raise ValueError(f"conflicting duplicate runs for {label}/{obj}: {previous[0]} vs {run_dir}")
            if new and duplicate:
                raise ValueError(f"{run_dir} mixes new and duplicate objects")
            if duplicate:
                dropped.append(str(run_dir.relative_to(root)))
                continue
            target_dir = output / label / f"import_{import_dirs.index(source)}" / run_dir.relative_to(source)
            for obj in new:
                kept[(label, obj)] = (run_dir, (metrics[obj], _import_provenance(config, manifest, obj)))
            for name in ("metrics.json", "config.yaml", "run_manifest.json"):
                payloads[target_dir / name] = (run_dir / name).read_bytes().decode("utf-8")
    if not kept:
        raise ValueError(f"no completed runs found in {import_dirs}")
    # Failed attempts leave config.yaml without a manifest; count them for the audit.
    skipped = sum(not path.with_name("run_manifest.json").is_file()
                  for source in import_dirs for path in source.rglob("config.yaml"))
    if len(data_roots) != 1:
        raise ValueError(f"imported runs span several data roots: {sorted(map(str, data_roots))}")
    completed = [dict(label=label, object=obj, source_run=str(kept[label, obj][0].relative_to(root)),
                      ended=kept[label, obj][0].joinpath("metrics.json").stat().st_mtime)
                 for label in labels for obj in objects if (label, obj) in kept]
    full = len(completed) == len(labels) * len(objects)
    state = dict(state="complete" if full else "partial", imported=True, active=None, split=split,
                 started=min(e["ended"] for e in completed), ended=max(e["ended"] for e in completed),
                 plan=dict(labels=labels, objects=objects, configs=sources["configs"]),
                 completed=completed, data_root=data_roots.pop(), import_sources=sources,
                 duplicates_dropped=dropped, incomplete_run_dirs_skipped=skipped)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(owner, json.dumps(sources, indent=2) + "\n")
    for target, content in payloads.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(target, content)
    atomic_write(output / "status.json", json.dumps(state, indent=2) + "\n")
    return state


def update(root, output, labels=None, objects=None, reference_method=None):
    status = json.loads((output / "status.json").read_text())
    plan = status.get("plan", {})
    labels, objects = labels or plan.get("labels"), objects or plan.get("objects")
    if not labels or not objects:
        raise ValueError("legacy status has no plan: supply --labels and --objects")
    if len(set(labels)) != len(labels) or len(set(objects)) != len(objects):
        raise ValueError("plan labels and objects must be unique")
    if any(not name or Path(name).name != name or name in (".", "..") for name in [*labels, *objects]):
        raise ValueError("plan identities must be simple directory names")
    completed = {(item["label"], item["object"]) for item in status["completed"]}
    if len(completed) != len(status["completed"]) or not completed <= {(s, o) for s in labels for o in objects}:
        raise ValueError("completed identities do not match the declared plan")
    csv_path, md_path = root / LEDGER_CSV, root / "results/comparisons.md"
    original_csv = csv_path.read_bytes().decode("utf-8")
    old_rows = list(csv.DictReader(io.StringIO(original_csv, newline="")))
    relative = str(output.relative_to(root))
    preserved = [r for r in old_rows if not r["source"].startswith(relative + "/")]
    stage_data = {}
    for label in labels:
        data = []
        for path in sorted((output / label).rglob("run_manifest.json")):
            manifest = json.loads(path.read_text())
            config = yaml.safe_load(path.with_name("config.yaml").read_text())
            for obj, metadata in manifest["objects"].items():
                if (label, obj) not in completed:
                    continue
                metrics = json.loads(path.with_name("metrics.json").read_text())[obj]
                coverage = metrics["coverage"]
                if coverage["non_finite_prediction_count"] or coverage["shape_match_count"] != coverage["prediction_map_count"]:
                    raise ValueError(f"invalid prediction coverage: {path}")
                if any(not 0 <= metrics[key] <= 1 for key in METRICS):
                    raise ValueError(f"invalid metrics: {path}")
                data.append((obj, metadata, metrics, manifest, path, config))
        if len(data) != sum(stage == label for stage, _ in completed):
            raise ValueError(f"missing or duplicate completed artifacts: {label}")
        stage_data[label] = data
    baseline = {item[0]: item for item in stage_data[labels[0]]}
    rows, means, matched = [], {}, {}
    for label, data in stage_data.items():
        for obj, metadata, metrics, manifest, path, config in data:
            key = (label, obj)
            matched[key] = label == labels[0]
            if label != labels[0] and obj in baseline:
                original = baseline[obj]
                matched[key] = (
                    all(metadata["protocol"][k] == original[1]["protocol"][k] for k in CONTENT_KEYS)
                    and manifest["code_sha256"] == original[3]["code_sha256"]
                    and all(config.get(k) == original[5].get(k) for k in
                            ("dataset", "backbone", "seed", "protocol", "dataset_config", "backbone_config", "evaluation"))
                    and config["method_config"].get("resolutions") == original[5]["method_config"].get("resolutions")
                    and metadata.get("backbone") == original[1].get("backbone")
                    and manifest.get("environment") == original[3].get("environment")
                    and metrics["metric_contract"] == original[2]["metric_contract"])
            row = dict(row_type="class", method=f"{config['method']}_{label}",
                       scope=output.name + (f"_full{len(objects)}" if len(data) == len(objects) else "_partial"),
                       class_name=obj, n_classes=len(data), backbone=config["backbone"],
                       preprocess=config["dataset_config"]["preprocess"], split=status.get("split", "undeclared"),
                       support_shots=config["protocol"]["shots"], support_policy="see_manifest",
                       seed=config["seed"], support_identity=metadata["protocol"]["support_content_sha256"],
                       training_or_adapter=f"config_hash={manifest['config_hash']};run_id={manifest['run_id']};code_sha256={manifest['code_sha256']};params={metadata.get('trainable_parameters', 'unrecorded')}",
                       resolution=_resolution_label(metadata, config),
                       evaluator_metric_contract=metrics["metric_contract"], delta_AUPRO_vs_SuperAD="N/A",
                       delta_AUROC_vs_SuperAD="N/A", comparison_status="matched_completed_class" if matched[key] else "unmatched_completed_class",
                       source=str(path.with_name("metrics.json").relative_to(root)))
            row.update({k: metrics[k] for k in METRICS[:2]})
            rows.append(row)
        if len(data) == len(objects):
            means[label] = {k: sum(item[2][k] for item in data) / len(data) for k in METRICS}
            macro = rows[-1].copy()
            macro.update(row_type="macro", class_name="ALL", source=str((output / label / "summary.tsv").relative_to(root)),
                         support_identity="per_class_content_sha256_see_class_rows", training_or_adapter=f"stage={label};see_class_rows",
                         comparison_status="matched_full_scope" if all(matched[label, o] for o in objects) else "unmatched_full_scope")
            macro.update({k: means[label][k] for k in METRICS[:2]})
            rows.append(macro)
    total = len(labels) * len(objects)
    updated_at = status.get("ended") or max(
        (item.get("ended", 0) for item in status["completed"]),
        default=0) or status.get("started") or (output / "status.json").stat().st_mtime
    lines = [f"## Latest: {output.name}", "",
             f"Run updated: {datetime.fromtimestamp(updated_at).astimezone().isoformat(timespec='seconds')}. State: `{status['state']}`; completed {len(completed)}/{total}.", "",
             f"Split: `{status.get('split', 'undeclared')}`. Values ×100; macro only for the complete declared class scope. Missing ≠ zero. F1 oracle is diagnostic.", "",
             f"| Stage | Completed | AUPRO@0.05 | pAUROC@0.05 | F1 oracle | Δ AUPRO vs {labels[0]} (pp) |",
             "|---|---:|---:|---:|---:|---:|"]
    for label in labels:
        count = len(stage_data[label])
        if label not in means:
            lines.append(f"| {label} | {count}/{len(objects)} | — | — | — | — |")
            continue
        delta = "—"
        if labels[0] in means and all(matched[label, o] for o in objects):
            delta = f"{(means[label][METRICS[0]] - means[labels[0]][METRICS[0]])*100:+.5f}"
        lines.append(f"| {label} | {count}/{len(objects)} | " + " | ".join(f"{means[label][k]*100:.5f}" for k in METRICS) + f" | {delta} |")
    latest_lines = lines.copy()
    lines += ["", "### Completed class metrics (×100)", "",
              "| Stage | Method | Class | AUPRO@0.05 | pAUROC@0.05 | F1 oracle |",
              "|---|---|---|---:|---:|---:|"]
    for label, data in stage_data.items():
        for obj, _, metrics, _, _, config in data:
            lines.append(f"| {label} | {config['method']} | {obj} | " + " | ".join(f"{metrics[k]*100:.5f}" for k in METRICS) + " |")
    if reference_method:
        audits = []
        for obj, metadata, metrics, _, _, _ in baseline.values():
            references = [r for r in preserved if r["method"] == reference_method and r["row_type"] == "class" and r["class_name"] == obj]
            if len(references) != 1:
                continue
            path = root / references[0]["source"]
            original = json.loads(path.read_text())[obj]
            original_meta = json.loads(path.with_name("run_manifest.json").read_text())["objects"][obj]
            audits.append((all(metadata["protocol"][k] == original_meta["protocol"][k] for k in CONTENT_KEYS),
                           all(abs(metrics[k] - original[k]) < 1e-8 for k in METRICS)))
        lines += ["", f"{labels[0]}/{reference_method} audit: {len(audits)} classes checked; content matched {sum(a[0] for a in audits)}; all metrics within 1e-8: {sum(a[1] for a in audits)}."]
    notices = []
    if not all(matched.values()):
        notices += ["", "Some stages have unmatched/missing baseline conditions; corresponding deltas are withheld. See CSV comparison_status."]
    if status.get("active"):
        active = status["active"]
        active_objects = ([worker["object"] for worker in active["workers"]]
                          if "workers" in active else [active["object"]])
        notices += ["", f"Active/queued: `{active['label']} / {', '.join(active_objects)}`."]
    if status.get("error"):
        notices += ["", f"Execution error: `{status['error']}`. Incomplete runs have no assigned metrics."]
    lines += notices
    if status.get("source_shards"):
        lines += ["", "Combined compact evidence from: " + ", ".join(f"`{p}`" for p in status["source_shards"]) + ". Original run IDs/configs/manifests preserved; compare each class within its execution environment."]
    lines += ["", f"Artifacts: `{relative}`. Config/run/code/parameter/support provenance is recorded in the CSV. No maps/weights/log copies; no new evaluation."]
    report_path = output / "comparison.md"
    report_link = os.path.relpath(report_path, md_path.parent).replace(os.sep, "/")
    latest_lines += notices + ["", f"[Detailed run report](<{report_link}>) · Artifacts: `{relative}`."]
    original_report = report_path.read_text() if report_path.exists() else ""
    if original_report and "<!-- comparison-results:start -->" not in original_report:
        raise ValueError(f"unmanaged comparison report: update only the affected rows manually: {report_path}")
    report = replace_section(original_report or "# Run comparison\n", "comparison-results", "\n".join(lines))
    csv_text, merged = merge_csv(original_csv, relative, rows)
    if csv_text == original_csv and report == original_report:
        return status  # Do not replace a newer suite's summary on an unchanged recollect.
    markdown = md_path.read_text()
    markdown = replace_section(markdown, "comparison-index", comparison_index(merged, markdown))
    markdown = replace_section(markdown, "comparison-latest", "\n".join(latest_lines))
    atomic_write(report_path, report)
    atomic_write(csv_path, csv_text)
    atomic_write(md_path, markdown)
    return status


def collect_recorded_results(root, output, artifact):
    """Register compact external evaluation evidence without fabricating run metrics."""
    import re

    artifact = Path(artifact).resolve()
    artifact.relative_to(output)
    document = json.loads(artifact.read_text())
    csv_path, md_path = root / LEDGER_CSV, root / "results/comparisons.md"
    original_csv = csv_path.read_text()
    fields = csv.DictReader(io.StringIO(original_csv)).fieldnames
    relative = str(output.relative_to(root))
    rows = document["ledger_rows"]
    if not rows:
        raise ValueError("recorded results contain no ledger rows")
    for row in rows:
        if set(row) != set(fields) or row["row_type"] not in ("class", "macro"):
            raise ValueError("recorded result schema does not match the canonical ledger")
        if not row["source"].startswith(str(artifact.relative_to(root))):
            raise ValueError("recorded rows must cite the imported evidence artifact")
        for key in METRICS[:2]:
            if row[key] not in ("", "N/A") and not 0 <= float(row[key]) <= 1:
                raise ValueError("recorded metrics must use the ledger's 0–1 scale")
    csv_text, _ = merge_csv(original_csv, relative, rows)
    markdown = md_path.read_text()
    report_path = output / "comparison.md"
    original_report = report_path.read_text()
    marker = "recorded-results"
    detail = document["report_markdown"]
    report = (replace_section(original_report, marker, detail)
              if f"<!-- {marker}:start -->" in original_report else
              original_report.rstrip() + f"\n\n<!-- {marker}:start -->\n{detail.rstrip()}\n<!-- {marker}:end -->\n")
    if csv_text == original_csv and report == original_report:
        print("ledger: recorded results unchanged")
        return
    # Update only this run's macro lines and scope keys in the existing index.
    start, tail = markdown.split("<!-- comparison-index:start -->", 1)
    index, end = tail.split("<!-- comparison-index:end -->", 1)
    affected_scopes = {r["scope"] for r in rows}
    affected_methods = {r["method"] for r in rows}
    groups = {}
    for line in index.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 3 and re.fullmatch(r"S\d+", cells[0]):
            groups[cells[1]] = cells[0]
    next_id = max((int(g[1:]) for g in groups.values()), default=0) + 1
    for scope in sorted(affected_scopes):
        if scope not in groups:
            groups[scope] = f"S{next_id:02d}"
            next_id += 1
    affected_groups = {groups[s] for s in affected_scopes}
    kept = []
    for line in index.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and cells[0] in affected_groups and (
                len(cells) == 3 or (len(cells) == 7 and cells[1] in affected_methods)):
            continue
        kept.append(line)
    index = "\n".join(kept) + "\n"
    header = "|---|---|---:|---|---|---:|---:|"
    for full in (True, False):
        macros = [r for r in rows if r["row_type"] == "macro" and
                  (str(r["n_classes"]) == "8" and "partial" not in r["scope"]) == full]
        lines = []
        for r in macros:
            values = ["—" if r[k] in ("", "N/A") else f"{float(r[k])*100:.5f}" for k in METRICS[:2]]
            lines.append("| " + " | ".join([groups[r["scope"]], r["method"], str(r["n_classes"]), str(r["seed"]), str(r["support_shots"]), *values]) + " |")
        title = "### Full8" if full else "### 기타 범위 — full8과 분리"
        before, section = index.split(title, 1)
        if lines:
            section = section.replace(header, header + "\n" + "\n".join(lines), 1)
        index = before + title + section
    scope_lines = []
    for scope in sorted(affected_scopes):
        conditions = dict.fromkeys(r["split"] + " / " + r["preprocess"] for r in rows if r["scope"] == scope)
        scope_lines.append(f"| {groups[scope]} | {scope} | {'; '.join(conditions)} |")
    index = index.replace("|---|---|---|", "|---|---|---|\n" + "\n".join(scope_lines), 1)
    markdown = start + "<!-- comparison-index:start -->" + index + "<!-- comparison-index:end -->" + end
    markdown = replace_section(markdown, "comparison-latest", document["latest_markdown"])
    atomic_write(report_path, report)
    atomic_write(csv_path, csv_text)
    atomic_write(md_path, markdown)
    print(f"ledger: {len(rows)} recorded rows registered; unrelated rows preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    runner = subparsers.add_parser("run", help="sequence configs through the existing CLI")
    runner.add_argument("--configs", nargs="+", required=True)
    runner.add_argument("--entrypoint", default="run.py",
                        help="CLI entrypoint; default is the common experiment pipeline")
    runner.add_argument("--data-root", required=True)
    runner.add_argument("--device", default="cuda:0")
    runner.add_argument("--timeout", type=int, default=3600)
    runner.add_argument("--workers", type=int, default=1,
                        help="run independent objects concurrently within each stage")
    runner.add_argument("--after-status", nargs="+",
                        help="start only after all predecessor status files report complete")
    runner.add_argument("--after-timeout", type=float, default=86400,
                        help="maximum predecessor wait in seconds")
    runner.add_argument("--split", required=True, help="declare actual split and development/final exposure")
    runner.add_argument("--resume", action="store_true",
                        help="resume an invalid queue, preserving completed runs and failure history")
    collector = subparsers.add_parser("collect", help="update the cumulative ledger; no experiments")
    collector.add_argument("--recorded-results", help="compact external evaluation JSON with ledger rows and report sections")
    collector.add_argument("--watch", action="store_true")
    collector.add_argument("--max-hours", type=float, default=33)
    collector.add_argument("--reference-method", help="optional historical baseline identity to audit")
    collector.add_argument("--input-dirs", nargs="+", help="combine completed disjoint object shards before collection")
    collector.add_argument("--recover-partial", action="store_true",
                           help="combine stopped partial shards with complementary recovery runs; require full coverage")
    collector.add_argument("--import-runs", nargs="+", help="import ad-hoc completed runs from these directories")
    collector.add_argument("--configs", nargs="+", help="config files for import-runs matching")
    collector.add_argument("--split", help="declare split for imported runs")
    for command in (runner, collector):
        command.add_argument("--output-dir", required=True)
        command.add_argument("--labels", nargs="+")
        command.add_argument("--objects", nargs="+", required=command is runner)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "run":
        return run(args, root)
    output = Path(args.output_dir).resolve()
    if args.recorded_results:
        if args.watch or args.import_runs or args.input_dirs:
            raise ValueError("--recorded-results cannot be combined with other collection modes")
        return collect_recorded_results(root, output, args.recorded_results)
    if args.import_runs:
        if args.watch:
            raise ValueError("--import-runs requires completed runs, not --watch")
        import_completed_runs(root, output, args.import_runs, args.configs, args.labels, args.split, args.objects)
    elif args.input_dirs:
        if args.watch:
            raise ValueError("--input-dirs requires completed shards, not --watch")
        combine_completed_shards(root, output, args.input_dirs, recover_partial=args.recover_partial)
    elif args.recover_partial:
        raise ValueError("--recover-partial requires --input-dirs")
    deadline, previous = time.monotonic() + args.max_hours * 3600, None
    while True:
        path = output / "status.json"
        if path.exists():
            content = path.read_text()
            if content != previous:
                status = update(root, output, args.labels, args.objects, args.reference_method)
                previous = content
                print(f"ledger: {status['state']} {len(status['completed'])} completed", flush=True)
                if status["state"] != "running":
                    return
        elif not args.watch:
            raise FileNotFoundError(path)
        if not args.watch:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("collector deadline; experiment not stopped")
        time.sleep(15)


if __name__ == "__main__":
    main()
