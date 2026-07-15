from scripts.aggregate_flow_tte_ad2_guided_chunks import metric_value


def test_raw_f1_falls_back_to_unprocessed_f1() -> None:
    assert metric_value({"seg_F1": 0.25}, "seg_F1_raw") == 0.25


def test_explicit_raw_f1_takes_precedence() -> None:
    assert metric_value({"seg_F1": 0.25, "seg_F1_raw": 0.5}, "seg_F1_raw") == 0.5
