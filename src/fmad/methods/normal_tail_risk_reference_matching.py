"""Fit 12 positive calibration multipliers using normal-source LOO tail risk.

No query data or labels enter this method. Spatial probes use matching relative
image coordinates across resolutions and exclude their entire source bank.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import optimize, sparse

from fmad.methods.reference_matching_fast import _masked_loo_distances
from fmad.methods.statistical_reference_matching import StatisticalReferenceMatching
from fmad.registry import METHOD_REGISTRY


def empirical_cvar(values, alpha=.95):
    x = np.asarray(values, np.float64)
    index=max(0,min(len(x)-1,int(np.ceil(alpha*len(x)))-1))
    threshold = float(np.partition(x,index)[index])
    return float(threshold + np.maximum(x-threshold, 0.).mean()/(1-alpha))


def fit_tail_risk(probes, gamma=.125, alpha=.95, fixed_layer=True, excess=False,
                  bound=8.):
    """Convex empirical CVaR minimization over bounded positive multipliers.

    Probes are [normal image, aligned coordinate, resolution, layer], already
    Q95-normalized then bag-averaged. Equal probes/image give equal image weight.
    This is empirical risk optimization, not an FPR/independence guarantee.
    """
    z = np.asarray(probes, dtype=np.float64)
    if z.ndim != 4 or z.shape[-1] != 3 or z.shape[0] < 2 or not np.isfinite(z).all():
        raise ValueError('tail risk requires >=2 aligned normal source images')
    if not 0 < alpha < 1 or not bound >= 1 or gamma <= 0:
        raise ValueError('invalid tail-risk parameters')
    if excess:
        z = np.maximum(z-1., 0.)
    r, l = z.shape[-2:]
    x = (z * np.array([gamma,.5,.5]) / r).reshape(-1,r*l)
    n, d = x.shape
    baseline = empirical_cvar(x.sum(axis=1), alpha)
    if baseline <= 1e-12 and np.max(np.abs(x)) <= 1e-12:
        return np.ones((r,l)), dict(objective='empirical_CVaR',alpha=alpha,
            signal='positive_Q95_excess' if excess else 'Q95_normalized_score',
            fixed_layer_total=fixed_layer,baseline_risk=baseline,fitted_risk=baseline,
            source_count=z.shape[0],aligned_probes_per_image=z.shape[1],observations=d,
            multiplier_bounds=[1/bound,bound],solver='identity_zero_signal',
            fallback='zero_signal_retains_Q95',source='normal_support_source_LOO_only',
            statistical_coverage_claimed=False)
    # Variables: d weights, one VaR threshold, n nonnegative excess slacks.
    objective = np.r_[np.zeros(d), 1., np.full(n,1/((1-alpha)*n))]
    inequality = sparse.hstack([sparse.csr_matrix(x),
        sparse.csr_matrix(-np.ones((n,1))), -sparse.eye(n)],format='csr')
    constraints = np.zeros((l if fixed_layer else 1, d+1+n))
    if fixed_layer:
        for li in range(l):
            constraints[li, np.arange(li,d,l)] = 1.
        targets = np.full(l,float(r))
    else:
        constraints[0,:d]=1.
        targets=np.array([float(d)])
    result = optimize.linprog(objective, A_ub=inequality, b_ub=np.zeros(n),
        A_eq=sparse.csr_matrix(constraints), b_eq=targets,
        bounds=[(1/bound,bound)]*d+[(None,None)]+[(0.,None)]*n, method='highs',
        options={'time_limit':120.})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('tail-risk LP failed: '+result.message)
    weights=result.x[:d]
    fitted=empirical_cvar(x@weights,alpha)
    if fitted > baseline + 1e-6*max(1.,abs(baseline)):
        raise RuntimeError('tail-risk solver returned risk worse than feasible Q95')
    return weights.reshape(r,l), dict(objective='empirical_CVaR', alpha=alpha,
        signal='positive_Q95_excess' if excess else 'Q95_normalized_score',
        fixed_layer_total=fixed_layer, baseline_risk=baseline, fitted_risk=fitted,
        source_count=z.shape[0], aligned_probes_per_image=z.shape[1],
        observations=d, multiplier_bounds=[1/bound,bound], solver='scipy_highs_LP',
        iterations=int(result.nit), source='normal_support_source_LOO_only',
        statistical_coverage_claimed=False)


@METHOD_REGISTRY.register('normal_tail_risk_reference_matching')
class NormalTailRiskReferenceMatching(StatisticalReferenceMatching):
    def __init__(self, backbone, config):
        self.tail_risk_mode=config.get('tail_risk_mode','excess_fixed_layer')
        modes={'off','score_free','score_fixed_layer','excess_free','excess_fixed_layer'}
        if self.tail_risk_mode not in modes:
            raise ValueError('unknown tail_risk_mode')
        if (config.get('statistical_normalization','q95')!='q95'
                or not config.get('fast_reuse_calibration_bank',False)
                or config.get('bank_selection','random')!='random'
                or config.get('bank_outlier_trim',0.)!=0.):
            raise ValueError('normal tail risk requires unchanged random banks and source-LOO Q95')
        self.probe_side=int(config.get('tail_probe_side',16))
        if self.probe_side < 2:
            raise ValueError('tail_probe_side must be >=2')
        self._probe_fitting=False
        super().__init__(backbone,config)
        if len(self.layers)!=3 or self.layer_fusion_shallow_idx!=0:
            raise ValueError('tail risk requires shallow-first three-layer fusion')

    def _extract(self,image,resolution,pool=None):
        layers,grid=super()._extract(image,resolution,pool)
        if self._probe_fitting:
            h,w=grid
            # Fixed image-relative cell centers. The nearest feature token is
            # used at each resolution; no interpolation of normalized features.
            yy,xx=np.meshgrid((np.arange(self.probe_side)+.5)/self.probe_side,
                              (np.arange(self.probe_side)+.5)/self.probe_side,indexing='ij')
            indices=(np.minimum((yy*h).astype(int),h-1)*w+
                     np.minimum((xx*w).astype(int),w-1)).ravel()
            self._probe_features.setdefault(resolution,[]).append([
                layer[torch.as_tensor(indices,device=layer.device)].detach().cpu()
                for layer in layers])
        return layers,grid

    def _fit_scales_for_resolution(self,resolution):
        scales=super()._fit_scales_for_resolution(resolution)
        records=self.records[resolution]
        cached=self._fast_full_bank_cache
        if cached is None or cached[0]!=resolution or cached[1]!=id(records):
            raise RuntimeError('tail risk needs the matching source-LOO bank cache')
        bank=cached[2]
        owners=torch.as_tensor(bank['owners'],device=self.device,dtype=torch.long)
        sources=[]
        for source,features in enumerate(self._probe_features[resolution]):
            columns=[]
            for li,query in enumerate(features):
                distance=_masked_loo_distances(query.to(self.device),bank['layers'][li],
                    owners,source,self.chunk,self.knn_k)
                columns.append((distance/float(scales[li])).detach().cpu().numpy())
            sources.append(np.stack(columns,-1))
        normalized=np.stack(sources)
        self._probe_bags.setdefault(resolution,[]).append(normalized)
        if len(self._probe_bags[resolution])==self.bank_bags:
            self._probe_features.pop(resolution)
        return scales

    def fit(self,support_paths,context):
        if len(support_paths)<2:
            raise ValueError('normal tail risk requires >=2 normal supports')
        self._probe_features={}
        self._probe_bags={}
        self._probe_fitting=True
        try:
            super().fit(support_paths,context)
        finally:
            self._probe_fitting=False
        if self._probe_features:
            raise RuntimeError('incomplete aligned probe collection')
        # [source,point,resolution,layer]; only normal support features used.
        probes=np.stack([np.mean(self._probe_bags[r],axis=0) for r in self.resolutions],axis=2)
        if self.tail_risk_mode=='off':
            weights=np.ones((len(self.resolutions),len(self.layers)))
            audit=dict(source='normal_support_source_LOO_only', mode='off',
                       source_count=len(support_paths))
        else:
            weights,audit=fit_tail_risk(probes,gamma=self.layer_fusion_gamma,
                fixed_layer=self.tail_risk_mode.endswith('fixed_layer'),
                excess=self.tail_risk_mode.startswith('excess'))
        self.normal_tail_risk_audit=dict(audit,mode=self.tail_risk_mode,
            multipliers=weights.tolist(),resolutions=list(self.resolutions),
            layers=list(self.layers),probe_coordinates='nearest_token_to_fixed_normalized_cell_centers',
            spatial_probe_count=self.probe_side**2, changes_fusion_coefficients=True)
        self.normal_tail_risk_audit['effective_calibration_degrees_of_freedom']=(
            0 if self.tail_risk_mode=='off' else
            len(self.resolutions)*len(self.layers)-(
                len(self.layers) if self.tail_risk_mode.endswith('fixed_layer') else 1))
        for ri,r in enumerate(self.resolutions):
            for bag in self.bag_tails[r]:
                for li,t in enumerate(bag):
                    t['statistical']['fusion_multiplier']=float(weights[ri,li])
        self._probe_bags={}

    def object_metadata(self):
        metadata=super().object_metadata()
        metadata.update(normal_tail_risk_calibration=self.normal_tail_risk_audit,
            normalization='normal_support_LOO_Q95_plus_'+self.tail_risk_mode,
            score_normalization='normal_tail_risk_'+self.tail_risk_mode,
            score_definition='normal_only_aligned_tail_risk_12_scale_calibration')
        return metadata
