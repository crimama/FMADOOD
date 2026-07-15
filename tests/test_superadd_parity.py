from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from typing import Protocol, cast

import cv2
import numpy as np
import pytest
import torch

import flow_tte.superadd_bank as superadd_bank
import scripts.run_superadd_parity as superadd_runner
from flow_tte.superadd_bank import fit_disk_backed_banks
from flow_tte.superadd_morphology import MorphologyConfig, postprocess_binary
from flow_tte.superadd_outputs import (
    CanonicalMapPaths,
    CategoryRunActiveError,
    CategoryRunPlan,
    MapIdentity,
    SuperADDOutputError,
    canonical_map_paths,
    category_run,
    text_sha256,
    write_category_manifest,
    write_map_artifacts,
)
from flow_tte.superadd_parity import (
    CoresetConfig,
    ManifestContext,
    ModelProvenance,
    SuperADDParityError,
    build_parity_manifest,
    fixed_threshold,
    layerwise_anomaly_map,
    nearest_distance_map,
    partition_training_paths,
    subsample_distance_based,
    subsample_knn_score_rank,
)
from scripts.run_superadd_parity import (
    RunConfig,
    SuperADDRunnerError,
    discover_public_items,
    implementation_sha256,
    load_support_paths,
    main,
    parse_args,
    prepare_run_plan,
)


def _model_provenance() -> ModelProvenance:
    return ModelProvenance(
        model_id="facebook/dinov3-vith16plus-pretrain-lvd1689m",
        revision="c807c9eeea853df70aec4069e6f56b28ddc82acc",
        model_class="DINOv3ViTModel",
        patch_size=16,
        depth=32,
        register_count=4,
        config_sha256="c" * 64,
        resolved_config_sha256="r" * 64,
        weight_sha256="w" * 64,
        transformers_version="4.56.2",
    )


def test_partition_training_paths_uses_official_modulo_eight_split() -> None:
    # Given: paths in the deterministic dataset order used by official training.
    paths = tuple(Path(f"{index:02d}.png") for index in range(18))

    # When: prototypes and threshold images are partitioned.
    partition = partition_training_paths(paths, threshold_fraction=8)

    # Then: every eighth image is held out from the prototype bank.
    assert partition.threshold == (paths[0], paths[8], paths[16])
    assert partition.prototypes == tuple(path for index, path in enumerate(paths) if index % 8)


def test_producer_digest_has_an_explicit_runtime_dependency_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: tuple[Path, ...] = ()

    def capture(paths: tuple[Path, ...]) -> str:
        nonlocal captured
        captured = tuple(paths)
        return "d" * 64

    monkeypatch.setattr(superadd_runner, "files_sha256", capture)

    assert superadd_runner.implementation_sha256() == "d" * 64
    assert {path.name for path in captured} == {
        "darc_backbone.py",
        "run_superadd_parity.py",
        "superadd_bank.py",
        "superadd_inference.py",
        "superadd_morphology.py",
        "superadd_outputs.py",
        "superadd_parity.py",
        "superadd_patching.py",
    }
    assert all("official_superadd_binary" not in path.name for path in captured)


def test_finalization_rejects_mid_run_producer_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(superadd_runner, "implementation_sha256", lambda: "b" * 64)

    with pytest.raises(SuperADDRunnerError, match="changed during execution"):
        superadd_runner._require_unchanged_implementation("a" * 64)  # noqa: SLF001


def test_nearest_distance_map_uses_unsquared_official_cdist_per_dimension() -> None:
    # Given: one two-dimensional query five Euclidean units from its prototype.
    query = torch.tensor([[[3.0, 4.0]]])
    prototypes = torch.tensor([[0.0, 0.0]])

    # When: official unnormalized 1-NN scoring is applied.
    result = nearest_distance_map(query, prototypes, query_chunk_size=1)

    # Then: the unsquared distance is divided by channel dimension.
    torch.testing.assert_close(result, torch.tensor([[2.5]]))


def test_layerwise_anomaly_map_means_layers_and_outputs_native_quarter_size() -> None:
    # Given: two constant layer maps on a 2x2 token grid.
    maps = (torch.full((2, 2), 2.0), torch.full((2, 2), 4.0))

    # When: maps are fused for an 8x12 native image.
    result = layerwise_anomaly_map(maps, native_size=(8, 12))

    # Then: interpolation targets native/4 and layer fusion is an arithmetic mean.
    assert result.shape == (2, 3)
    torch.testing.assert_close(result, torch.full((2, 3), 3.0))


def test_fixed_threshold_flattens_all_calibration_maps() -> None:
    # Given: two equal-sized calibration maps.
    maps = (np.array([[0.0, 1.0]]), np.array([[2.0, 3.0]]))

    # When: a median threshold with factor two is computed.
    result = fixed_threshold(maps, percentile=50.0, factor=2.0)

    # Then: percentile is evaluated over every calibration pixel jointly.
    assert result == 3.0


def test_subsample_distance_based_preserves_bank_below_cap() -> None:
    # Given: a prototype bank already below the official 100k cap.
    features = np.arange(12, dtype=np.float32).reshape(6, 2)

    # When: deterministic coreset construction is requested.
    result = subsample_distance_based(
        features,
        device=torch.device("cpu"),
        rng=np.random.RandomState(42),
        config=CoresetConfig(target_count=10),
    )

    # Then: the bank is retained without an unnecessary lossy pass.
    np.testing.assert_array_equal(result, features)


def test_subsample_distance_based_is_deterministic_above_cap() -> None:
    features = np.arange(80, dtype=np.float32).reshape(40, 2)
    config = CoresetConfig(target_count=10, iterations=2, knn_neighbors=40)

    first = subsample_distance_based(
        features,
        torch.device("cpu"),
        np.random.RandomState(42),
        config,
    )
    second = subsample_distance_based(
        features,
        torch.device("cpu"),
        np.random.RandomState(42),
        config,
    )

    assert first.shape == (10, 2)
    np.testing.assert_array_equal(first, second)


def test_knn_score_rank_retains_lowest_density_scores() -> None:
    features = np.asarray(
        [[0.0], [0.1], [0.2], [5.0], [10.0]],
        dtype=np.float32,
    )

    result = subsample_knn_score_rank(
        torch.from_numpy(features),
        target_count=2,
        knn_neighbors=3,
        query_chunk_size=2,
    )

    np.testing.assert_array_equal(result.numpy(), features[[3, 4]])


def test_knn_score_rank_is_invariant_to_query_chunking() -> None:
    features = torch.arange(120, dtype=torch.float32).reshape(30, 4)

    first = subsample_knn_score_rank(
        features,
        target_count=7,
        knn_neighbors=5,
        query_chunk_size=30,
    )
    second = subsample_knn_score_rank(
        features,
        target_count=7,
        knn_neighbors=5,
        query_chunk_size=4,
    )

    assert len(first) == 7
    assert torch.equal(first, second)


class _BankExtractor:
    def __init__(self, layer: int) -> None:
        self.layer = layer

    def extract(
        self,
        path: Path,
        brightness: float,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, int], bool]:
        del brightness
        offset = float(int(path.stem) * 2)
        grid = torch.tensor(
            [[[[offset + self.layer * 10], [offset + self.layer * 10 + 1]]]],
        )
        return (grid,), (1, 2), True


def test_fit_banks_spools_each_layer_to_disk_and_cleans_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_types = []
    observed_rng = []
    observed_store_counts = []

    def fake_subsample(
        features: np.ndarray,
        _device: torch.device,
        _rng: np.random.RandomState,
        _config: CoresetConfig,
    ) -> np.ndarray:
        observed_types.append(type(features))
        observed_rng.append(float(_rng.rand()))
        observed_store_counts.append(len(tuple(scratch.rglob("*.float32"))))
        return features[::2]

    monkeypatch.setattr(superadd_bank, "subsample_distance_based", fake_subsample)
    scratch = tmp_path / "scratch"
    banks, used_early_exit = fit_disk_backed_banks(
        (Path("0.png"), Path("1.png")),
        tuple(_BankExtractor(layer) for layer in range(4)),
        torch.device("cpu"),
        np.random.RandomState(42),
        scratch,
    )

    assert observed_types == [np.memmap] * 4
    assert observed_store_counts == [1, 1, 1, 1]
    expected_rng = np.random.RandomState(42)
    expected_rng.uniform(0.8, 1.2, size=2)
    np.testing.assert_allclose(observed_rng, expected_rng.rand(4))
    assert used_early_exit is True
    for layer, bank in enumerate(banks):
        torch.testing.assert_close(
            bank,
            torch.tensor([[layer * 10.0], [layer * 10.0 + 2.0]]),
        )
    assert list(scratch.iterdir()) == []


def test_fit_banks_cleans_disk_stores_after_extraction_failure(tmp_path: Path) -> None:
    class FailingExtractor:
        def extract(
            self,
            path: Path,
            brightness: float,
        ) -> tuple[tuple[torch.Tensor, ...], tuple[int, int], bool]:
            if path.stem == "1":
                raise RuntimeError("extract failed")
            return _BankExtractor(0).extract(path, brightness)

    scratch = tmp_path / "scratch"
    with pytest.raises(RuntimeError, match="extract failed"):
        fit_disk_backed_banks(
            (Path("0.png"), Path("1.png")),
            (FailingExtractor(), _BankExtractor(1), _BankExtractor(2), _BankExtractor(3)),
            torch.device("cpu"),
            np.random.RandomState(42),
            scratch,
        )
    assert list(scratch.iterdir()) == []


def test_fit_banks_crosses_explicit_cpu_boundary_before_numpy(tmp_path: Path) -> None:
    class DeviceBoundaryGrid:
        shape = (1, 1, 2, 1)

        def __init__(self) -> None:
            self.on_cpu = False

        def reshape(self, *_shape: int) -> DeviceBoundaryGrid:
            return self

        def detach(self) -> DeviceBoundaryGrid:
            return self

        def to(
            self,
            *,
            device: str,
            dtype: torch.dtype,
        ) -> DeviceBoundaryGrid:
            assert device == "cpu"
            assert dtype == torch.float32
            self.on_cpu = True
            return self

        def contiguous(self) -> DeviceBoundaryGrid:
            return self

        def numpy(self) -> np.ndarray:
            if not self.on_cpu:
                raise RuntimeError("numpy called before CPU transfer")
            return np.array([[1.0], [2.0]], dtype=np.float32)

    class BoundaryExtractor:
        def extract(
            self,
            _path: Path,
            _brightness: float,
        ) -> tuple[tuple[torch.Tensor, ...], tuple[int, int], bool]:
            grid = cast("torch.Tensor", DeviceBoundaryGrid())
            return (grid,), (1, 2), True

    banks, used_early_exit = fit_disk_backed_banks(
        (Path("0.png"),),
        tuple(BoundaryExtractor() for _ in range(4)),
        torch.device("cpu"),
        np.random.RandomState(42),
        tmp_path / "scratch",
    )

    assert used_early_exit is True
    for bank in banks:
        torch.testing.assert_close(bank, torch.tensor([[1.0], [2.0]]))


def test_binary_postprocess_is_separate_from_raw_map() -> None:
    # Given: one high-score pixel and an immutable copy of the raw map.
    raw = np.zeros((7, 7), dtype=np.float32)
    raw[3, 3] = 2.0
    original = raw.copy()

    # When: official-style morphology produces a binary artifact.
    binary = postprocess_binary(
        raw,
        threshold=1.0,
        config=MorphologyConfig(radius=1, angles=1, lower_factor=0.8, erosion=0),
    )

    # Then: binary output is explicit and the raw score map is untouched.
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 255})
    np.testing.assert_array_equal(raw, original)


def test_manifest_records_official_source_and_material_adaptations() -> None:
    # Given: one completed parity resource.
    # When: its audit manifest is built.
    supports = (Path("train/good/000.png"), Path("train/good/001.png"))
    partition = partition_training_paths(supports, 2)
    manifest = build_parity_manifest(
        ManifestContext(
            category="can",
            resource_protocol="Pfull",
            support_paths=supports,
            partition=partition,
            model=_model_provenance(),
            implementation_sha256="i" * 64,
            used_early_exit=True,
        ),
    )

    # Then: source identity, access paths, and adaptation limits are explicit.
    assert manifest["official_commit"] == "44cf25144442fbbc1334ea59d1632327a4376d1a"
    assert manifest["source_sha256"]
    assert manifest["algorithm_contract_matched"] is True
    assert "algorithmically_comparable" not in manifest
    assert manifest["official_runtime_comparable"] is False
    assert "hf_backbone_adapter" in manifest["adaptations"]
    assert manifest["support_paths"] == (
        "train/good/000.png",
        "train/good/001.png",
    )
    assert manifest["prototype_paths"] == ("train/good/001.png",)
    assert manifest["threshold_paths"] == ("train/good/000.png",)
    assert manifest["model"]["revision"] == "c807c9eeea853df70aec4069e6f56b28ddc82acc"


def test_p16_manifest_labels_official_native_split_as_resource_noncomparable() -> None:
    # Given: a P16 support resource passed through official modulo-eight calibration.
    # When: its audit manifest is built.
    supports = tuple(Path(f"{index}.png") for index in range(16))
    manifest = build_parity_manifest(
        ManifestContext(
            category="can",
            resource_protocol="P16-official-native-split",
            support_paths=supports,
            partition=partition_training_paths(supports),
            model=_model_provenance(),
            implementation_sha256="i" * 64,
            used_early_exit=True,
        ),
    )

    # Then: it is not presented as the registered 12/4 P16 protocol.
    assert manifest["resource_protocol"] == "P16-official-native-split"
    assert manifest["resource_comparable"] is False


def test_manifest_rejects_ambiguous_p16_label() -> None:
    supports = tuple(Path(f"{index}.png") for index in range(16))
    context = ManifestContext(
        category="can",
        resource_protocol="P16",
        support_paths=supports,
        partition=partition_training_paths(supports),
        model=_model_provenance(),
        implementation_sha256="i" * 64,
        used_early_exit=True,
    )

    with pytest.raises(SuperADDParityError, match="P16-official-native-split"):
        build_parity_manifest(context)


def test_cli_rejects_p16_and_accepts_explicit_native_split(tmp_path: Path) -> None:
    common = (
        "--data-root",
        str(tmp_path),
        "--category",
        "can",
        "--device",
        "cuda:0",
        "--output-root",
        str(tmp_path / "out"),
        "--resource-protocol",
    )
    with pytest.raises(SystemExit):
        parse_args((*common, "P16"))

    parsed = parse_args(
        (*common, "P16-official-native-split", "--support-manifest", str(tmp_path / "p16.json")),
    )
    assert parsed.resource_protocol == "P16-official-native-split"


def test_cli_resumes_valid_category_before_model_load(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a complete Pfull category and its unchanged dataset inventory.
    data_root = tmp_path / "data"
    train = data_root / "can/train/good"
    public_good = data_root / "can/test_public/good"
    public_bad = data_root / "can/test_public/bad"
    for directory in (train, public_good, public_bad):
        directory.mkdir(parents=True)
    for index in range(8):
        (train / f"{index:03d}.png").write_bytes(b"support")
    (public_good / "good.png").write_bytes(b"public")
    (public_bad / "bad.png").write_bytes(b"public")
    args = (
        "--data-root",
        str(data_root),
        "--category",
        "can",
        "--device",
        "cuda:0",
        "--output-root",
        str(tmp_path / "out"),
        "--resource-protocol",
        "Pfull",
    )
    config = parse_args(args)
    supports = load_support_paths(config)
    partition = partition_training_paths(supports)
    items = discover_public_items(config)
    plan = prepare_run_plan(config, supports, partition, items, implementation_sha256())
    with category_run(plan) as run:
        for paths in plan.expected_maps:
            write_map_artifacts(paths, np.zeros((1, 1)), np.zeros((1, 1)))
        write_category_manifest(run, '{"category":"can"}')

    # When: the real CLI entrypoint receives the identical plan.
    result = main(args)

    # Then: it resumes without importing/loading the multi-gigabyte model.
    assert result == 0
    assert '"status": "resumed_complete"' in capsys.readouterr().out


def test_first_run_rejects_cpu_before_model_loading(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    train = data_root / "can/train/good"
    public = data_root / "can/test_public/good"
    train.mkdir(parents=True)
    public.mkdir(parents=True)
    for index in range(8):
        (train / f"{index:03d}.png").write_bytes(b"support")
    (public / "good.png").write_bytes(b"public")
    args = (
        "--data-root",
        str(data_root),
        "--category",
        "can",
        "--device",
        "cpu",
        "--output-root",
        str(tmp_path / "out"),
        "--resource-protocol",
        "Pfull",
    )

    with pytest.raises(SuperADDRunnerError, match="requires a CUDA device"):
        main(args)


def test_canonical_output_writes_float32_tiff_for_common_evaluator(tmp_path: Path) -> None:
    # Given: one raw map and an independent official binary map.
    identity = MapIdentity(tmp_path, "can", "bad", "001")
    paths = canonical_map_paths(identity)
    raw = np.array([[0.25, 1.5]], dtype=np.float32)
    binary = np.array([[0, 255]], dtype=np.uint8)

    # When: canonical artifacts are persisted.
    write_map_artifacts(paths, raw, binary)

    # Then: evaluator discovery path and floating score precision are retained.
    assert paths.raw == tmp_path / "anomaly_maps/can/test/bad/001.tiff"
    restored = cv2.imread(str(paths.raw), cv2.IMREAD_UNCHANGED)
    assert restored is not None
    assert restored.dtype == np.float32
    np.testing.assert_array_equal(restored, raw)


def _category_plan(tmp_path: Path, category: str = "can") -> CategoryRunPlan:
    paths = canonical_map_paths(MapIdentity(tmp_path, category, "bad", "001"))
    return CategoryRunPlan(tmp_path, category, (paths,), text_sha256(f"spec:{category}"))


def _write_completed_category(plan: CategoryRunPlan) -> None:
    with category_run(plan) as run:
        assert run.resumed is False
        paths = plan.expected_maps[0]
        write_map_artifacts(
            paths,
            np.array([[0.25]], dtype=np.float32),
            np.array([[255]], dtype=np.uint8),
        )
        write_category_manifest(run, json.dumps({"category": plan.category}))


class _ProcessEvent(Protocol):
    def set(self) -> None: ...

    def wait(self, timeout: float) -> bool: ...


def _hold_category_process(
    plan: CategoryRunPlan,
    ready: _ProcessEvent,
    release: _ProcessEvent,
) -> None:
    with category_run(plan) as run:
        ready.set()
        if not release.wait(timeout=15.0):
            raise TimeoutError("test release event timed out")
        paths = plan.expected_maps[0]
        write_map_artifacts(paths, np.zeros((1, 1)), np.zeros((1, 1)))
        write_category_manifest(run, json.dumps({"category": plan.category}))


def _complete_category_process(plan: CategoryRunPlan) -> None:
    _write_completed_category(plan)


def test_category_completion_is_deterministic_and_resumes_without_rewrite(
    tmp_path: Path,
) -> None:
    # Given: one exact category inventory completed under its lease.
    plan = _category_plan(tmp_path)
    _write_completed_category(plan)
    completion = tmp_path / "categories/can/completion.json"
    first_bytes = completion.read_bytes()

    # When: the identical plan is opened again.
    with category_run(plan) as resumed:
        assert resumed.resumed is True

    # Then: completion is reused byte-for-byte and contains both artifact hashes.
    assert completion.read_bytes() == first_bytes
    payload = json.loads(first_bytes)
    assert [entry["kind"] for entry in payload["artifacts"]] == ["raw", "binary"]
    assert all(len(entry["sha256"]) == 64 for entry in payload["artifacts"])


def test_category_lease_rejects_same_category_collision_without_mutation(
    tmp_path: Path,
) -> None:
    # Given: one process-equivalent lease owns the category.
    plan = _category_plan(tmp_path)
    with category_run(plan) as active:
        state = tmp_path / "categories/can/state.json"
        state_before = state.read_bytes()

        # When/Then: a second lease is rejected before touching state or maps.
        with pytest.raises(CategoryRunActiveError, match="already active"), category_run(plan):
            pass
        assert state.read_bytes() == state_before
        paths = plan.expected_maps[0]
        write_map_artifacts(paths, np.zeros((1, 1)), np.zeros((1, 1)))
        write_category_manifest(active, '{"category":"can"}')


def test_category_lease_rejects_cross_process_collision(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    plan = _category_plan(tmp_path)
    ready = context.Event()
    release = context.Event()
    process = context.Process(target=_hold_category_process, args=(plan, ready, release))
    process.start()
    try:
        assert ready.wait(timeout=15.0)
        with pytest.raises(CategoryRunActiveError, match="already active"), category_run(plan):
            pass
    finally:
        release.set()
        process.join(timeout=15.0)
    assert process.exitcode == 0


def test_three_category_processes_complete_under_one_output_root(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    plans = tuple(_category_plan(tmp_path, category) for category in ("can", "rice", "walnuts"))
    processes = tuple(
        context.Process(target=_complete_category_process, args=(plan,)) for plan in plans
    )
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20.0)

    assert [process.exitcode for process in processes] == [0, 0, 0]
    assert all(
        (tmp_path / "categories" / plan.category / "completion.json").is_file()
        for plan in plans
    )


def _verify_stale_cleanup_then_abort(
    plan: CategoryRunPlan,
    stale: Path,
    fabric: Path,
) -> None:
    with category_run(plan):
        assert not stale.exists()
        assert fabric.read_bytes() == b"keep"
        raise RuntimeError("stop after cleanup")


def test_stale_cleanup_removes_only_the_selected_category(tmp_path: Path) -> None:
    # Given: stale can output beside a fabric sentinel sharing the same root.
    plan = _category_plan(tmp_path)
    stale = plan.expected_maps[0].raw
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    fabric = tmp_path / "anomaly_maps/fabric/test/bad/sentinel.tiff"
    fabric.parent.mkdir(parents=True)
    fabric.write_bytes(b"keep")

    # When: can is resumed after an incomplete run.
    with pytest.raises(RuntimeError, match="stop after cleanup"):
        _verify_stale_cleanup_then_abort(plan, stale, fabric)

    # Then: only can receives an incomplete marker.
    state = json.loads((tmp_path / "categories/can/state.json").read_text(encoding="utf-8"))
    assert state["state"] == "incomplete"
    assert fabric.read_bytes() == b"keep"


def _write_extra_inventory(plan: CategoryRunPlan, tmp_path: Path) -> None:
    with category_run(plan) as run:
        paths = plan.expected_maps[0]
        write_map_artifacts(paths, np.zeros((1, 1)), np.zeros((1, 1)))
        extra = canonical_map_paths(MapIdentity(tmp_path, "can", "bad", "extra"))
        write_map_artifacts(extra, np.zeros((1, 1)), np.zeros((1, 1)))
        write_category_manifest(run, '{"category":"can"}')


def test_completion_rejects_extra_category_map_and_marks_incomplete(tmp_path: Path) -> None:
    # Given: expected output plus an unregistered extra map.
    plan = _category_plan(tmp_path)
    with pytest.raises(SuperADDOutputError, match="inventory differs"):
        _write_extra_inventory(plan, tmp_path)

    assert not (tmp_path / "categories/can/completion.json").exists()
    state = json.loads((tmp_path / "categories/can/state.json").read_text(encoding="utf-8"))
    assert state["state"] == "incomplete"


def _write_missing_inventory(plan: CategoryRunPlan) -> None:
    with category_run(plan) as run:
        paths = plan.expected_maps[0]
        write_map_artifacts(paths, np.zeros((1, 1)), np.zeros((1, 1)))
        paths.binary.unlink()
        write_category_manifest(run, '{"category":"can"}')


def test_completion_rejects_missing_category_map(tmp_path: Path) -> None:
    plan = _category_plan(tmp_path)

    with pytest.raises(SuperADDOutputError, match="inventory differs"):
        _write_missing_inventory(plan)

    assert not (tmp_path / "categories/can/completion.json").exists()


def _verify_checksum_cleanup_then_abort(plan: CategoryRunPlan, fabric: Path) -> None:
    with category_run(plan) as run:
        assert run.resumed is False
        assert not plan.expected_maps[0].raw.exists()
        assert fabric.read_text(encoding="utf-8") == "keep"
        raise RuntimeError("verified cleanup")


def test_checksum_mismatch_invalidates_completion_and_cleans_only_category(
    tmp_path: Path,
) -> None:
    # Given: a valid can completion whose raw map is later corrupted.
    plan = _category_plan(tmp_path)
    _write_completed_category(plan)
    plan.expected_maps[0].raw.write_bytes(b"corrupt")
    fabric = tmp_path / "categories/fabric/sentinel"
    fabric.parent.mkdir(parents=True)
    fabric.write_text("keep", encoding="utf-8")

    # When: the completion checksum is revalidated.
    with pytest.raises(RuntimeError, match="verified cleanup"):
        _verify_checksum_cleanup_then_abort(plan, fabric)


def test_category_run_rejects_symlinked_category_without_touching_target(
    tmp_path: Path,
) -> None:
    # Given: the controlled category directory redirects to an unrelated directory.
    victim = tmp_path / "victim"
    victim.mkdir()
    sentinel = victim / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    categories = tmp_path / "categories"
    categories.mkdir()
    (categories / "can").symlink_to(victim, target_is_directory=True)
    plan = _category_plan(tmp_path)

    # When/Then: the run is rejected before lock creation or stale cleanup follows it.
    with pytest.raises(SuperADDOutputError, match="symlink"), category_run(plan):
        pass
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "controlled_root",
    ["categories", "anomaly_maps", "official_binary_maps"],
)
def test_category_run_rejects_symlinked_controlled_ancestor(
    tmp_path: Path,
    controlled_root: str,
) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    sentinel = victim / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    (tmp_path / controlled_root).symlink_to(victim, target_is_directory=True)

    with pytest.raises(SuperADDOutputError, match="symlink"), category_run(
        _category_plan(tmp_path),
    ):
        pass
    assert sentinel.read_text(encoding="utf-8") == "keep"


def _write_partial_then_abort(plan: CategoryRunPlan) -> None:
    with category_run(plan):
        paths = plan.expected_maps[0]
        write_map_artifacts(paths, np.zeros((1, 1)), np.zeros((1, 1)))
        raise RuntimeError("abort")


def test_category_failure_removes_partial_maps_immediately(tmp_path: Path) -> None:
    plan = _category_plan(tmp_path)

    with pytest.raises(RuntimeError, match="abort"):
        _write_partial_then_abort(plan)

    assert not plan.expected_maps[0].raw.exists()
    assert not plan.expected_maps[0].binary.exists()


def test_run_plan_rejects_lexical_parent_escape(tmp_path: Path) -> None:
    escaped = CanonicalMapPaths(
        tmp_path / "anomaly_maps/can/../../victim.tiff",
        tmp_path / "official_binary_maps/can/../../victim.png",
    )

    with pytest.raises(SuperADDOutputError, match="escapes"):
        CategoryRunPlan(tmp_path, "can", (escaped,), "a" * 64)


def test_run_spec_changes_when_input_bytes_change(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    train = data_root / "can/train/good"
    public_good = data_root / "can/test_public/good"
    public_bad = data_root / "can/test_public/bad"
    for directory in (train, public_good, public_bad):
        directory.mkdir(parents=True)
    for index in range(8):
        (train / f"{index:03d}.png").write_bytes(b"support")
    public_path = public_good / "good.png"
    public_path.write_bytes(b"public-v1")
    (public_bad / "bad.png").write_bytes(b"public")
    config = RunConfig(
        data_root,
        "can",
        "cuda:0",
        tmp_path / "out",
        "Pfull",
        None,
    )
    supports = load_support_paths(config)
    partition = partition_training_paths(supports)
    items = discover_public_items(config)
    first = prepare_run_plan(config, supports, partition, items, "i" * 64)

    supports[0].write_bytes(b"support-v2")
    support_changed = prepare_run_plan(config, supports, partition, items, "i" * 64)
    supports[0].write_bytes(b"support")
    public_path.write_bytes(b"public-v2")
    public_changed = prepare_run_plan(config, supports, partition, items, "i" * 64)

    assert first.spec_sha256 != support_changed.spec_sha256
    assert first.spec_sha256 != public_changed.spec_sha256


def test_rediscovered_run_plan_detects_new_support_and_public_members(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    train = data_root / "can/train/good"
    public_good = data_root / "can/test_public/good"
    public_bad = data_root / "can/test_public/bad"
    for directory in (train, public_good, public_bad):
        directory.mkdir(parents=True)
    for index in range(8):
        (train / f"{index:03d}.png").write_bytes(b"support")
    (public_good / "good.png").write_bytes(b"public")
    (public_bad / "bad.png").write_bytes(b"public")
    config = RunConfig(data_root, "can", "cuda:0", tmp_path / "out", "Pfull", None)
    supports = load_support_paths(config)
    initial = prepare_run_plan(
        config,
        supports,
        partition_training_paths(supports),
        discover_public_items(config),
        "i" * 64,
    )

    (train / "008.png").write_bytes(b"new-support")
    (public_bad / "new-bad.png").write_bytes(b"new-public")
    final_supports = load_support_paths(config)
    final = prepare_run_plan(
        config,
        final_supports,
        partition_training_paths(final_supports),
        discover_public_items(config),
        "i" * 64,
    )

    assert final != initial
    assert len(final.expected_maps) == len(initial.expected_maps) + 1


def test_run_spec_changes_with_device_and_torch_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    train = data_root / "can/train/good"
    public = data_root / "can/test_public/good"
    train.mkdir(parents=True)
    public.mkdir(parents=True)
    for index in range(8):
        (train / f"{index:03d}.png").write_bytes(b"support")
    (public / "good.png").write_bytes(b"public")
    cuda_config = RunConfig(data_root, "can", "cuda:0", tmp_path / "out", "Pfull", None)
    supports = load_support_paths(cuda_config)
    partition = partition_training_paths(supports)
    items = discover_public_items(cuda_config)
    cuda_plan = prepare_run_plan(cuda_config, supports, partition, items, "i" * 64)
    cpu_config = RunConfig(data_root, "can", "cpu", tmp_path / "out", "Pfull", None)
    cpu_plan = prepare_run_plan(cpu_config, supports, partition, items, "i" * 64)

    monkeypatch.setattr(torch, "__version__", "different-runtime")
    runtime_plan = prepare_run_plan(cuda_config, supports, partition, items, "i" * 64)

    assert cuda_plan.spec_sha256 != cpu_plan.spec_sha256
    assert cuda_plan.spec_sha256 != runtime_plan.spec_sha256


def _verify_mutation_invalidation(plan: CategoryRunPlan) -> None:
    with category_run(plan) as run:
        assert run.resumed is False
        assert not plan.expected_maps[0].raw.exists()
        raise RuntimeError("verified invalidation")


def test_input_mutation_invalidates_completed_category_resume(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    train = data_root / "can/train/good"
    public = data_root / "can/test_public/good"
    train.mkdir(parents=True)
    public.mkdir(parents=True)
    for index in range(8):
        (train / f"{index:03d}.png").write_bytes(b"support")
    (public / "good.png").write_bytes(b"public")
    config = RunConfig(data_root, "can", "cuda:0", tmp_path / "out", "Pfull", None)
    supports = load_support_paths(config)
    partition = partition_training_paths(supports)
    items = discover_public_items(config)
    original = prepare_run_plan(config, supports, partition, items, "i" * 64)
    _write_completed_category(original)

    supports[0].write_bytes(b"changed-support")
    changed = prepare_run_plan(config, supports, partition, items, "i" * 64)
    with pytest.raises(RuntimeError, match="verified invalidation"):
        _verify_mutation_invalidation(changed)


def test_p16_support_manifest_rejects_path_outside_train_good(tmp_path: Path) -> None:
    train = tmp_path / "data/can/train/good"
    train.mkdir(parents=True)
    supports = []
    for index in range(15):
        path = train / f"{index:03d}.png"
        path.write_bytes(b"support")
        supports.append(str(path))
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"support")
    supports.append(str(outside))
    manifest = tmp_path / "supports.json"
    manifest.write_text(json.dumps({"can": supports}), encoding="utf-8")
    config = RunConfig(
        tmp_path / "data",
        "can",
        "cuda:0",
        tmp_path / "out",
        "P16-official-native-split",
        manifest,
    )

    with pytest.raises(SuperADDRunnerError, match="train/good"):
        load_support_paths(config)
