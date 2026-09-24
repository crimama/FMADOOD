# Relational Tail (RGTA)

RGTA is a D3-preserving ablation family. F1 restores the clean keep objective
and adds standardized score-level teacher relation preservation, a bounded
surrogate for normal-neighbourhood structure. F2 cycles weak synthetic blends
at strengths 0.35/0.65/1.0 and gates synthetic-positive ranking with a detached
reliability score, so weak synthetic defects are not treated as universally
correct. Its positive sampler is half low-score hard positives and half random
positives, with reliability indexed alongside each selected token. F3 projects conflicting anomaly gradients in
score-leaf space before the existing exact VJP replay; it is explicitly not a
parameter-space PCGrad claim. All stages use the common CLI and keep dense maps
and checkpoints disabled.
