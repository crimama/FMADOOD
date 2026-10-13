# Analysis and Rationale

## Evidence being tested

The completed SOVM diagnostic did not validate its learned signed-code velocity
score: macro pAUROC was `0.674283`, it lost to its endpoint baselines, and code
selection collapsed. It nevertheless produced one unusually strong causal
signal: the N3 *spatial-position roll* improved over the candidate by
`0.145896` pAUROC (95% CI `[0.114077, 0.177550]`; 4/4 object wins). N3 did
not shuffle source identity. Separately, the unavailable same-source oracle
left substantial headroom. This rejects that specific velocity/code
formulation, not the hypothesis that normal cross-image part correspondence
makes support more useful.

## Changed hypothesis

For a query patch with a benign nuisance shift, a support source representing
the same normal part should be retrievable without learning or adaptation. Its
feature displacement should be largely represented by a tangent estimated from
normal support transformations. A defect should retain residual energy
orthogonal to that tangent. The candidate therefore scores the orthogonal
residual, not the direction or magnitude of transport itself.

The full-shot ablation is informative because an increased normal-source pool
can approximate an unavailable same source while evaluation remains exactly
paired with the fixed 16-source probe. It is a support-budget diagnostic, not a
claim that full training support is an allowed few-shot benchmark regime.

## Why this construction is deliberately narrow

- The graph is same-position cross-source rather than global all-patch nearest
  neighbour. It combines the positive N3 spatial evidence with the separately
  observed same-source-oracle headroom, while avoiding a quadratic-in-patches
  full-shot search.
- DINO layer 5 is a frozen retrieval key and layer 11 is a frozen score value,
  preventing the value residual from selecting its own explanation.
- Tangents are normal-only. Synthetic anomalies enter only a separately
  reported margin gate after the frozen geometric score is formed.
- Query features, graph state, tangent bases, and support memory remain
  immutable. This preserves a clean interface for a later TTA branch without
  silently performing TTA here.

## Falsifiers

The branch is not supported if raw retrieved distance or a global tangent
absorbs the candidate, if graph/tangent/position nulls behave equivalently, or
if the apparent improvement exists only after replacing the query key or anchor
with unavailable clean held information. A synthetic-margin gain that survives
neither mask shuffle nor pair-label swap is likewise not method evidence.
