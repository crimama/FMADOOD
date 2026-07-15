from __future__ import annotations

import torch
from PIL import Image

from visionad_density.core import DensityStats, combine_score, support_views


def test_support_views_match_official_six_way_augmentation() -> None:
    assert len(support_views(Image.new("RGB", (8, 8)))) == 6


def test_zero_density_is_exact_visionad_cosine_search() -> None:
    support = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    query = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    stats = DensityStats.fit(support)
    score = combine_score(query, support, stats, 0.0)
    torch.testing.assert_close(score, torch.tensor([0.0, 1.0]))


def test_density_extension_only_adds_nonnegative_penalty() -> None:
    support = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    query = torch.tensor([[8.0, 8.0]])
    stats = DensityStats.fit(support)
    base = combine_score(query, support, stats, 0.0)
    extended = combine_score(query, support, stats, 0.25)
    assert extended.item() > base.item()
