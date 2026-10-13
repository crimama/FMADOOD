# Fundamental Canonical Transport Architecture

Date: 2026-08-16  
Status: design-stage; not experiment validated

## Decision

The next admissible architecture is FACT-RN, an offline intervention-learned support-conditioned canonical representation. It replaces post-RN gating and reranking with a shared canonical space computed before anomaly measurement.

## Why a claim-class change is required

Target support alone has repeatedly failed to identify unseen nuisance versus anomaly. Consequently, another support-only residual, tangent, covariance projection, or global condition is a closed-branch retread. FACT-RN must learn the factorization across held-object normal episodes and explicit nuisance interventions, then remain frozen at deployment.

## Architecture

1. Frozen DINO multi-layer patch encoder.
2. Permutation-invariant support-set encoder.
3. Probabilistic part-conditioned canonical frames.
4. Invertible `(content, nuisance)` factorizer.
5. Support-conditioned continuous canonical transport.
6. Bi-Lipschitz local non-contraction and cycle constraints.
7. One exact distance in canonical content space; no raw-score fusion or gate.

## First gate

Before TESTpublic scaling, a held-object assay must jointly pass support-LOO normal-tail reduction, held-family nuisance correction, raw-RN neighbor retention, local/broad defect retention, and part/intervention/capacity hard-null margins. Failure closes the architecture before a benchmark run.
