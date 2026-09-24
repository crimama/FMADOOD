import pytest
from fmad.pipeline import run_pipeline


def test_pipeline_rejects_legacy_non_streaming_configuration():
    with pytest.raises(ValueError, match="streaming=false"):
        run_pipeline(None, None, None, {"evaluation": {"streaming": False}}, "/tmp/out")


def test_pipeline_rejects_classification_evaluation():
    with pytest.raises(ValueError, match="eval_clf"):
        run_pipeline(None, None, None, {"eval_clf": True}, "/tmp/out")
