"""P0/P1/P2: blind-region normal feature prediction, without a reference bank.

Mask pixels BEFORE the frozen encoder. Only masked features/coordinates and
visible-pixel statistics enter the predictor; unmasked features are detached
targets. Dense tensors live for one view or query, never across fitting steps.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fmad.methods.base import BaseMethod
from fmad.methods.PSMA_MR import (
    _apply_photometric, _feature_matrix, _read_rgb,
    _sample_photometric_transform, _synthetic_anomalies,
)
from fmad.registry import METHOD_REGISTRY


def tile_masks(grid, rows, columns, tile, guard):
    """Nonoverlapping scored tiles; only their input masks include a guard."""
    height, width = grid
    if rows > height or columns > width or not 0 <= tile < rows * columns:
        raise ValueError("mask layout must fit the patch grid")
    row, col = divmod(tile, columns)
    y0, y1 = row * height // rows, (row + 1) * height // rows
    x0, x1 = col * width // columns, (col + 1) * width // columns
    target = torch.zeros(grid, dtype=torch.bool)
    hidden = torch.zeros_like(target)
    target[y0:y1, x0:x1] = True
    hidden[max(0, y0-guard):min(height, y1+guard),
           max(0, x0-guard):min(width, x1+guard)] = True
    return target, hidden


def pixel_mask(mask, tensor):
    return F.interpolate(mask.to(device=tensor.device, dtype=torch.float32)[None, None],
                         size=tensor.shape[-2:], mode="nearest")[0, 0].bool()


def hide_region(tensor, hidden):
    """Fill using visible pixels only; even style cannot read the hidden region."""
    pixels = pixel_mask(hidden, tensor)
    visible = tensor[:, ~pixels]
    if not visible.shape[1]:
        raise ValueError("guard mask leaves no visible context")
    mean = visible.mean(1)
    std = visible.std(1, unbiased=False)
    masked = tensor.clone()
    masked[:, pixels] = mean[:, None]
    return masked, torch.cat((mean, std))


def coordinates(grid, device):
    y, x = torch.meshgrid(torch.linspace(-1, 1, grid[0], device=device),
                          torch.linspace(-1, 1, grid[1], device=device), indexing="ij")
    return torch.stack((y, x), -1).reshape(-1, 2)


class ContextPredictor(nn.Module):
    """Small cross-attention predictor, with no unmasked target shortcut."""
    def __init__(self, dimension, hidden, heads, layers, pool_grid, uncertainty,
                 variance_min, variance_max, mean, variance):
        super().__init__()
        self.dimension, self.pool_grid = dimension, pool_grid
        self.uncertainty = uncertainty
        self.variance_min, self.variance_max = variance_min, variance_max
        self.input = nn.Sequential(nn.Linear(dimension, hidden), nn.LayerNorm(hidden))
        self.position = nn.Linear(2, hidden)
        self.style = nn.Linear(6, hidden)
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
            for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2*hidden),
                          nn.GELU(), nn.Linear(2*hidden, hidden)) for _ in range(layers)])
        self.mean_head = nn.Linear(hidden, dimension)
        # A scalar per patch limits variance-head capacity in the normal16 regime.
        self.variance_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.mean_head.weight)
        with torch.no_grad():
            self.mean_head.bias.copy_(mean)
        nn.init.zeros_(self.variance_head.weight)
        nn.init.zeros_(self.variance_head.bias)
        self.variance_head.requires_grad_(uncertainty)
        self.register_buffer("base_variance", variance.detach().clone().clamp_min(1e-4))

    def forward(self, masked_features, grid, selected, style):
        pool = (min(self.pool_grid, grid[0]), min(self.pool_grid, grid[1]))
        memory = F.adaptive_avg_pool2d(
            masked_features.T.reshape(1, self.dimension, *grid), pool).flatten(2)[0].T
        memory = self.input(memory) + self.position(coordinates(pool, memory.device))
        query = self.position(coordinates(grid, memory.device)[selected])
        query = query + self.style(style.to(memory.device))[None]
        for attention, norm, ffn in zip(self.attention, self.norm, self.ffn):
            delta, _ = attention(norm(query)[None], memory[None], memory[None],
                                 need_weights=False)
            query = query + delta[0]
            query = query + ffn(query)
        mean = self.mean_head(query)
        if self.uncertainty:
            factor = torch.exp(torch.clamp(self.variance_head(query),
                               math.log(self.variance_min), math.log(self.variance_max)))
        else:
            factor = torch.ones_like(mean[:, :1])
        return mean, self.base_variance * factor


def normal_loss(mean, variance, target):
    return .5 * (((target - mean).square() / variance) + variance.log()).mean()


def prediction_score(mean, variance, target):
    return ((target - mean).square() / variance).mean(-1)


def synthetic_margin(mean, variance, normal, anomaly, coverage, margin):
    """Synthetic supervision cannot learn to shrink/inflate the variance head."""
    variance = variance.detach()
    loss = F.relu(margin + prediction_score(mean, variance, normal)
                  - prediction_score(mean, variance, anomaly))
    return (loss * coverage).sum() / coverage.sum().clamp_min(1e-8)


@METHOD_REGISTRY.register("masked_prediction")
class MaskedPrediction(BaseMethod):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.device = torch.device(backbone.device)
        self.uncertainty = config.get("uncertainty_normalization", False)
        self.synthetic = config.get("synthetic_supervision", False)
        self.objective = config.get("prediction_objective", "gaussian")
        if self.objective not in ("gaussian", "cosine"):
            raise ValueError("unknown prediction_objective")
        if self.objective == "cosine" and (self.uncertainty or self.synthetic):
            raise ValueError("cosine baseline excludes uncertainty and synthetic supervision")
        if type(self.uncertainty) is not bool or type(self.synthetic) is not bool:
            raise ValueError("P ablation flags must be booleans")
        if self.synthetic and not self.uncertainty:
            raise ValueError("synthetic_supervision requires uncertainty_normalization")
        self.resolutions = tuple(int(r) for r in config.get("resolutions", [448, 672]))
        self.steps = int(config.get("steps", 128))
        self.rows = int(config.get("mask_rows", 2))
        self.columns = int(config.get("mask_columns", 4))
        self.guard = int(config.get("guard_tokens", 1))
        self.train_tokens = int(config.get("train_tokens", 128))
        self.target_layer = int(config.get("target_layer", 11))
        self.context_layer = int(config.get("context_layer", 23))
        self.hidden = int(config.get("hidden_dim", 128))
        self.heads = int(config.get("attention_heads", 4))
        self.layers = int(config.get("predictor_layers", 2))
        self.pool_grid = int(config.get("context_pool_grid", 8))
        self.variance_min = float(config.get("variance_min_factor", .1))
        self.variance_max = float(config.get("variance_max_factor", 10.))
        self.margin = float(config.get("synthetic_margin", .2))
        self.synthetic_weight = float(config.get("synthetic_weight", .1))
        self.learning_rate = float(config.get("learning_rate", .001))
        if (not self.resolutions or min(*self.resolutions, self.steps, self.rows, self.columns,
                self.train_tokens, self.hidden, self.heads, self.layers, self.pool_grid) <= 0
                or self.guard < 1 or self.hidden % self.heads):
            raise ValueError("invalid predictor dimensions, grid or step budget")
        if (not all(np.isfinite(v) for v in (self.variance_min, self.variance_max,
                    self.margin, self.synthetic_weight, self.learning_rate))
                or not 0 < self.variance_min <= 1 <= self.variance_max
                or min(self.margin, self.synthetic_weight, self.learning_rate) <= 0):
            raise ValueError("invalid variance bounds or loss/optimizer constants")
        feature_layers = tuple(backbone.feature_layers)
        if self.target_layer not in feature_layers or self.context_layer not in feature_layers:
            raise ValueError("target/context layers must be explicitly requested from backbone")
        self.target_index = feature_layers.index(self.target_layer)
        self.context_index = feature_layers.index(self.context_layer)
        self.predictor = None
        self.history = []
        self.feature_calls = 0
        self.support_count = 0

    def features(self, tensor, grid):
        with torch.no_grad():
            outputs = self.backbone.extract_features(tensor)
        self.feature_calls += 1
        target, _ = _feature_matrix(outputs[self.target_index], grid)
        context, _ = _feature_matrix(outputs[self.context_index], grid)
        if target.shape != context.shape:
            raise ValueError("target/context feature shapes must match")
        # Fixed sqrt(D) rescales unit embeddings; it does not learn a representation.
        target = F.normalize(target.to(self.device), dim=-1) * math.sqrt(target.shape[1])
        context = F.normalize(context.to(self.device), dim=-1)
        return target, context

    def _initialize(self, images):
        total = square = None
        count = 0
        for image in images:
            for resolution in self.resolutions:
                tensor, grid = self.backbone.prepare_image(image, resolution=resolution)
                target, _ = self.features(tensor, grid)
                current = target.double()
                total = current.sum(0) if total is None else total + current.sum(0)
                square = current.square().sum() if square is None else square + current.square().sum()
                count += len(current)
        mean = (total / count).float()
        variance = (square / (count * len(mean)) - (total / count).square().mean()).float()
        self.predictor = ContextPredictor(len(mean), self.hidden, self.heads, self.layers,
                            self.pool_grid, self.uncertainty, self.variance_min,
                            self.variance_max, mean, variance).to(self.device)

    def _anomaly(self, image, rng, tile, family_index):
        row, col = divmod(tile, self.columns)
        height, width = image.shape[:2]
        y0, y1 = row*height//self.rows, (row+1)*height//self.rows
        x0, x1 = col*width//self.columns, (col+1)*width//self.columns
        changed, _, family = _synthetic_anomalies(image[y0:y1, x0:x1], rng, [])[family_index]
        anomaly = image.copy()
        anomaly[y0:y1, x0:x1] = changed
        return anomaly, family

    def fit(self, support_paths, context):
        if context.masking or context.rotation:
            raise ValueError("P uses no_mask_no_rotation preprocessing; internal blind masks are explicit")
        if len(support_paths) < 2 or len(set(support_paths)) != len(support_paths):
            raise ValueError("at least two unique normal supports required")
        torch.manual_seed(context.seed)
        rng = np.random.default_rng(context.seed)
        sampling = torch.Generator(device=self.device).manual_seed(context.seed + 19)
        images = [_read_rgb(p) for p in support_paths]
        self.support_count = len(images)
        self.history, self.feature_calls = [], 0
        self._initialize(images)
        parameters = [p for p in self.predictor.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate, weight_decay=0.)
        self.predictor.train()
        for step in range(self.steps):
            source = step % len(images)
            tile = (step // len(images)) % (self.rows*self.columns)
            image = images[source]
            anomaly, family = ((image, "disabled") if self.objective == "cosine" else
                               self._anomaly(image, rng, tile, (source + tile) % 2))
            transform = _sample_photometric_transform(rng, 1.)
            digest = hashlib.sha256(image.tobytes() + anomaly.tobytes()
                      + json.dumps([source, tile, transform]).encode()).hexdigest()
            optimizer.zero_grad(set_to_none=True)
            normal_values, synthetic_values, variances, coverages = [], [], [], []
            for resolution in self.resolutions:
                for normal_view, anomaly_view in ((image, anomaly),
                        (_apply_photometric(image, transform), _apply_photometric(anomaly, transform))):
                    tensor, grid = self.backbone.prepare_image(normal_view, resolution=resolution)
                    abnormal, abnormal_grid = self.backbone.prepare_image(anomaly_view, resolution=resolution)
                    if grid != abnormal_grid:
                        raise ValueError("paired view grids differ")
                    scored, hidden = tile_masks(grid, self.rows, self.columns, tile, self.guard)
                    # Remove resize interpolation spillover beyond the scored tile.
                    pixels = pixel_mask(scored, tensor)
                    abnormal = torch.where(pixels[None], abnormal, tensor)
                    changed = (abnormal-tensor).abs().amax(0) > 1e-7
                    coverage = F.interpolate(changed.float()[None, None], size=grid, mode="area").flatten()
                    masked, style = hide_region(tensor, hidden)
                    target, _ = self.features(tensor, grid)
                    abnormal_target = None
                    if self.objective != "cosine":
                        abnormal_target, _ = self.features(abnormal, grid)
                    _, context_features = self.features(masked, grid)
                    selected = scored.flatten().nonzero().flatten().to(self.device)
                    # Identical normal-token sampling across all three P variants.
                    selected = selected[torch.randperm(len(selected), device=self.device,
                                        generator=sampling)[:self.train_tokens]]
                    mean, variance = self.predictor(context_features, grid, selected, style)
                    nll = ((1 - F.cosine_similarity(mean, target[selected], dim=-1)).mean()
                           if self.objective == "cosine" else normal_loss(mean, variance, target[selected]))
                    weights = coverage.to(self.device)[selected]
                    contrast = (mean.new_zeros(()) if abnormal_target is None else
                                synthetic_margin(mean, variance, target[selected],
                                abnormal_target[selected], weights, self.margin))
                    loss = nll + (self.synthetic_weight * contrast if self.synthetic else 0.)
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite masked prediction loss")
                    (loss / (2 * len(self.resolutions))).backward()
                    normal_values.append(float(nll.detach()))
                    synthetic_values.append(float(contrast.detach()))
                    variances.append(float(variance.detach().mean()))
                    coverages.append(float(weights.detach().sum()))
                    del target, abnormal_target, context_features, mean, variance, loss, nll, contrast
            gradient = nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            optimizer.step()
            nll, contrast = float(np.mean(normal_values)), float(np.mean(synthetic_values))
            row = dict(loss=nll + (self.synthetic_weight*contrast if self.synthetic else 0.),
                       normal_nll=nll, synthetic_margin_loss=contrast,
                       mean_variance=float(np.mean(variances)), gradient_norm=float(gradient),
                       source=source, tile=tile, synthetic_family=family,
                       synthetic_coverage=float(np.sum(coverages)), episode_sha256=digest,
                       feature_extract_calls=self.feature_calls)
            self.history.append(row)
            self.log_training(step, **row)
        self.fit_feature_calls = self.feature_calls
        self.predictor.eval()

    def predict(self, test_image_path):
        if self.predictor is None:
            raise RuntimeError("fit must precede prediction")
        image = _read_rgb(test_image_path)
        total = None
        with torch.no_grad():
            for resolution in self.resolutions:
                tensor, grid = self.backbone.prepare_image(image, resolution=resolution)
                target, _ = self.features(tensor, grid)
                scores = torch.empty(len(target), device=self.device)
                for tile in range(self.rows*self.columns):
                    scored, hidden = tile_masks(grid, self.rows, self.columns, tile, self.guard)
                    masked, style = hide_region(tensor, hidden)
                    _, context_features = self.features(masked, grid)
                    selected = scored.flatten().nonzero().flatten().to(self.device)
                    mean, variance = self.predictor(context_features, grid, selected, style)
                    scores[selected] = (1 - F.cosine_similarity(mean, target[selected], dim=-1)
                                        if self.objective == "cosine" else
                                        prediction_score(mean, variance, target[selected]))
                native = F.interpolate(scores.reshape(1, 1, *grid), size=image.shape[:2],
                                       mode="bilinear", align_corners=False)[0, 0]
                total = native if total is None else total + native
            result = (total / len(self.resolutions)).cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("nonfinite masked prediction map")
        return result, float(result.max())

    def object_metadata(self):
        return dict(backbone_frozen=True, support_count=self.support_count,
                    prediction_objective=self.objective,
                    trainable_parameters=sum(p.numel() for p in self.predictor.parameters()
                                             if p.requires_grad) if self.predictor else 0,
                    steps_completed=len(self.history), resolutions=list(self.resolutions),
                    target_layer=self.target_layer, context_layer=self.context_layer,
                    uncertainty_normalization=self.uncertainty, synthetic_supervision=self.synthetic,
                    bank_sizes={}, mask_layout=[self.rows, self.columns], guard_tokens=self.guard,
                    mask_applied="prepared_pixels_before_frozen_encoder_visible_mean_fill",
                    predictor_inputs="masked_context_position_visible_rgb_statistics_only",
                    feature_target="same_view_l2_normalized_times_sqrt_dimension",
                    variance_model="bounded_scalar_per_patch_times_support_pooled_variance",
                    variance_bounds=[self.variance_min, self.variance_max],
                    fit_feature_extract_calls=getattr(self, "fit_feature_calls", 0),
                    inference_feature_calls_per_image=len(self.resolutions)*(1+self.rows*self.columns),
                    score_definition=("mean_resolution_cosine_prediction_error" if self.objective == "cosine"
                                      else "mean_resolution_variance_normalized_prediction_error"),
                    synthetic_location="native_tile_then_clip_prepared_pixels_to_scored_tile",
                    synthetic_variance_gradient=False,
                    last_training=self.history[-1] if self.history else None)
