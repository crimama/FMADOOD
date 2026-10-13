# Support-Orbit Velocity Mismatch (SOVM)

Date: 2026-08-11

## Claim

A normal support-trained nuisance velocity field may be more useful as a
**differential reference** than as a support-correction operator. A patch is
anomalous when no support anchor and no allowed signed nuisance code can explain
the geometric feature-space path toward that patch with a normal velocity.

## Evidence

- SO-CFM correction failed: guarded coverage was `0.520364`, tail reduction was
  `-0.003347`, and the discrete orbit bank was stronger on all four objects.
- The old synthetic guard was inactive: terminal hinge loss was zero in all 16
  object-fold fits.
- Flow Mismatching supplies a distinct operator: compare a learned normal
  velocity with the geometric velocity toward the observation, rather than
  reconstructing or transporting the observation.

## Boundary

- MVTec AD2 single-image, frozen support-16 `train/good` only.
- Four-object leave-one-support-image-out synthetic mechanism assay.
- No `test_public`, `test_private`, ground truth, filename condition token,
  query-time fitting, memory update, or support correction.
- The primary score enumerates all eight signed nuisance codes. The true code
  and the clean source of a pseudo-query are oracle ceilings only.
- Maximum verdict is `PROMISING_DIAGNOSTIC`; no benchmark or deployable method
  claim is available from this stage.

## Positioning

This is not another SO-CFM correction arm, query-view persistence rule, or
kNN score fusion. The new evidence is path-wise vector-field disagreement. It
is killed if raw/orbit/flow endpoint distance or condition-destroyed fields
explain the same signal.
