import pytest

from fmad.datasets.mvtec_ad2 import MVTecAD2Dataset


@pytest.mark.parametrize("objects", [[], ["typo"], ["can", "can"], "can"])
def test_invalid_object_selection_cannot_silently_change_experiment(objects):
    with pytest.raises(ValueError):
        MVTecAD2Dataset("unused", {"objects": objects})


def test_explicit_object_selection_is_respected():
    dataset = MVTecAD2Dataset("unused", {"objects": ["fabric", "can"]})
    assert [info.name for info in dataset.get_objects()] == ["can", "fabric"]
