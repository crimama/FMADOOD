"""Normal-support directional density calibration on fixed ScopeAD banks.

A bank is an equal-weight mixture of von Mises-Fisher kernels on the unit
sphere. Concentrations are fitted by source-held-out normal likelihood, shared
across random bags at each resolution/layer. No anomalies, query features or
labels are used for fitting. vMF mixtures are established directional statistics,
not a new density family (Banerjee et al., JMLR 2005).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy import optimize, special

from fmad.methods.statistical_reference_matching import StatisticalReferenceMatching
from fmad.methods.reference_matching_fast import _build_shared_bag_union
from fmad.registry import METHOD_REGISTRY


def log_vmf_uniform_ratio_constant(dimension, concentration):
    """log(C_d(kappa) / C_d(0)), with a stable small-kappa Bessel series."""
    if dimension < 2 or concentration < 0 or not math.isfinite(concentration):
        raise ValueError('invalid vMF dimension or concentration')
    if concentration == 0:
        return 0.
    nu = dimension / 2. - 1.
    bessel = special.ive(nu, concentration)
    if bessel > 0 and math.isfinite(bessel):
        return float(nu * math.log(concentration / 2.)
                     - special.gammaln(dimension / 2.)
                     - math.log(bessel) - concentration)
    # I_nu(k)=(k/2)^nu/Gamma(nu+1) * sum_j (k²/4)^j/(j!*(nu+1)_j).
    # ive can underflow for high dimension and small k; positive series avoids it.
    term, total = 0., 0.
    for j in range(1, 10000):
        term += 2*math.log(concentration/2.) - math.log(j) - math.log(nu+j)
        total = float(np.logaddexp(total, term))
        if term < total - 36. and j > concentration/2.:
            return -total
    raise RuntimeError('vMF Bessel series failed to converge')


def fit_directional_concentration(similarities, dimension):
    """Source-LOO conditional log-likelihood; each group is weighted equally."""
    groups=[np.asarray(x,dtype=np.float64) for x in similarities]
    if not groups or any(x.ndim!=2 or not x.size or not np.isfinite(x).all() for x in groups):
        raise ValueError('invalid source-held-out similarity groups')
    def nll(log_kappa):
        kappa=math.exp(log_kappa)
        logc=log_vmf_uniform_ratio_constant(dimension,kappa)
        return -float(np.mean([logc + (special.logsumexp(kappa*x,axis=1)
                                      - math.log(x.shape[1])).mean() for x in groups]))
    bounds=(math.log(1e-3),math.log(1e7))
    result=optimize.minimize_scalar(nll,bounds=bounds,method='bounded',
                                    options={'xatol':1e-5,'maxiter':120})
    if not result.success or not math.isfinite(result.fun):
        raise RuntimeError('vMF concentration fitting failed')
    candidates=[(0.,0.),(math.exp(result.x),float(result.fun))]
    candidates.extend((math.exp(x),nll(x)) for x in bounds)
    kappa,risk=min(candidates,key=lambda x:x[1])
    return kappa,dict(concentration=kappa,dimension=dimension,normal_heldout_nll=risk,
        uniform_nll=0.,source_bag_groups=len(groups),
        group_query_counts=[len(x) for x in groups],
        reference_token_counts=[x.shape[1] for x in groups],
        optimizer='bounded_log_kappa_scalar_likelihood',numerical_kappa_bounds=[1e-3,1e7],
        hit_numerical_bound=kappa in [1e-3,1e7],uniform_selected=kappa==0.,
        objective='equal_source_bag_mean_vMF_mixture_LOO_log_likelihood')


@METHOD_REGISTRY.register('directional_reference_matching')
class DirectionalReferenceMatching(StatisticalReferenceMatching):
    def __init__(self,backbone,config):
        self.directional_score=config.get('directional_score','mixture')
        if self.directional_score not in {'mixture','concentration_nn'}:
            raise ValueError('directional_score must be mixture or concentration_nn')
        super().__init__(backbone,config)
        if (self.statistical_normalization!='q95' or self.calibration_audit
                or not self.fast_reuse_calibration_bank or self.knn_k!=1
                or self.local_reliability!='off' or self.noise_guard_action=='lowres_pool'
                or self.image_residual_tau_q is not None):
            raise ValueError('directional calibration requires fixed cosine1NN Q95 comparison, no pooled/image/map audit')
        self._directional_groups={}

    def _fit_scales_for_resolution(self,resolution):
        scales=super()._fit_scales_for_resolution(resolution)
        bank=self._fast_full_bank_cache[2]
        owners=torch.as_tensor(bank['owners'],device=self.device)
        groups=self._directional_groups.setdefault(resolution,[[] for _ in self.layers])
        for record in self.records[resolution]:
            keep=owners!=record['source']
            for li,query in enumerate(record['layers']):
                similarities=query.to(self.device) @ bank['layers'][li][keep].T
                groups[li].append(similarities.cpu().numpy())
        return scales

    def fit(self,support_paths,context):
        if len(support_paths)<2:
            raise ValueError('directional calibration requires at least two supports')
        self._directional_groups={}
        super().fit(support_paths,context)
        self.directional_parameters={}
        for resolution,groups in self._directional_groups.items():
            params=[]
            for li,values in enumerate(groups):
                dimension=self.bag_banks[resolution][0]['layers'][li].shape[1]
                kappa,audit=fit_directional_concentration(values,dimension)
                audit.update(log_uniform_ratio_constant=log_vmf_uniform_ratio_constant(dimension,kappa),
                             layer=int(self.layers[li]),resolution=int(resolution))
                params.append(audit)
            self.directional_parameters[resolution]=params
        self._directional_groups={}
        self._directional_bank_resolution={id(banks):r for r,banks in self.bag_banks.items()}

    def _score_tokens_impl(self,layers,grid,bank,*,normalize=True):
        if not normalize:
            return super()._score_tokens_impl(layers,grid,bank,normalize=normalize)
        resolution=self._directional_bank_resolution[id(bank)]
        per_bag=[[] for _ in bank]
        for li,query in enumerate(layers):
            values=[b['layers'][li] for b in bank]
            key=tuple(id(x) for x in values)
            if key not in self._shared_union_cache:
                self._shared_union_cache[key]=_build_shared_bag_union(
                    values,[b['token_keys'] for b in bank],self.device)
            union,members=self._shared_union_cache[key]
            p=self.directional_parameters[resolution][li]
            parts=[[] for _ in bank]
            for start in range(0,len(query),self.chunk):
                distances=1.-query[start:start+self.chunk] @ union.T
                for bi,indices in enumerate(members):
                    d=distances.index_select(1,indices)
                    if self.directional_score=='mixture':
                        score=(-torch.logsumexp(-p['concentration']*d,dim=1)
                               + math.log(d.shape[1]) - p['concentration']
                               - p['log_uniform_ratio_constant'])
                    else:
                        score=p['concentration']*d.amin(1)
                    parts[bi].append(score)
            for bi in range(len(bank)):
                per_bag[bi].append(torch.cat(parts[bi]))
        return torch.stack([self._fuse_layers(scores) for scores in per_bag]).mean(0)

    def object_metadata(self):
        m=super().object_metadata()
        m.update(normalization='source_LOO_vMF_mixture_likelihood',
            score_normalization='directional_'+self.directional_score,
            score_definition=('negative_normal_mixture_log_density_ratio_to_uniform_sphere'
                              if self.directional_score=='mixture' else
                              'vMF_concentration_times_cosine1NN_distance'),
            directional_score=self.directional_score,
            directional_parameters={str(k):v for k,v in self.directional_parameters.items()},
            fitted_parameter_count=len(self.resolutions)*len(self.layers),
            query_used_for_fit=False,fit_source='normal_support_only',
            query_matching_change='soft_kernel_mixture' if self.directional_score=='mixture' else 'none',
            likelihood_ratio_claim=self.directional_score=='mixture',
            original_q95_parameters_role='comparison_audit_only_not_prediction_divisors',
            numerical_constant_policy='exact_vMF_normalizer_in_mixture_score',
            concentration_pooling='equal_source_and_bag_within_resolution_layer',
            statistical_coverage_claimed=False)
        return m
