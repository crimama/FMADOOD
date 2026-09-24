# Sequential follow-up: trace matching

User authorized the sequential follow-up on 2026-09-21. Stage E tests whether
protected PG_D outperforms unprotected contraction with the same contraction
trace. This is an ablation, with no novelty claim or parameter search.

From the identical normal support estimate U,V as in PG_D. Define
P=(I-VV^T)U and E=sqrt(||P||_F^2/||U||_F^2) U. Use the existing strength 0.5,
final normalization and refitted support MAD. Absent protection gives identity,
matching D. The operator trace is matched per layer/resolution; eigenvalue
spectrum and nuisance-energy suppression are not matched.

Budget: 16 additional evaluations, full8, seed0, nested SD_support_1/16,
frozen DINOv2-L, original pipeline, test_public_seen_development. Compare E
against completed PG_A/B/C/D within each class. Primary shift_1 AUPRO@0.05,
regular safety and shift_1 pAUROC secondary. Save metrics/manifests/logs;
maps/weights off. Verify matched traces and unchanged evidence before inference
about direction selection. D-E >=0.5pp and positive in >=5/8 classes at both
shots is the fixed informative-effect criterion, with positive leave-largest-
gain-out means. Otherwise directional benefit remains unsupported under this
control; no sweep. This does not override the prior D-versus-QR1 failure.

Reuse local can/fabric and dsba3 class/device assignment from the completed
four-GPU suite. Existing containers/runtimes only. Failures stop the affected
shard; repair execution faults without changing method parameters.

After collection, assess feasibility of a single query-correction confidence
hypothesis using normal-only split-patch agreement and existing evidence.
Record its exact rule before the next candidate comparison; no labels in fitting
or prediction, and no automatic multi-seed expansion. Carry the sequence through
the evidence-based decision, documenting any unsupported hypothesis explicitly.
