import pytest
from fmad.protocol import MethodContext, select_support_paths


def test_context_is_frozen():
    context = MethodContext("can", 3, 448, True, False)
    with pytest.raises(Exception):
        context.seed = 4


def test_support_selection_is_sorted_seeded_and_exact():
    paths = ["/d/c.png", "/d/a.png", "/d/b.png", "/d/d.png"]
    first, _ = select_support_paths(paths, 2, 11)
    second, _ = select_support_paths(list(reversed(paths)), 2, 11)
    assert first == second and len(first) == 2


@pytest.mark.parametrize("shots", [0, 5, True, 1.5])
def test_support_selection_rejects_invalid_shots(shots):
    with pytest.raises(ValueError):
        select_support_paths(["a", "b"], shots, 0)


def test_support_selection_rejects_duplicate_paths():
    with pytest.raises(ValueError, match="duplicates"):
        select_support_paths(["a", "a"], 1, 0)
