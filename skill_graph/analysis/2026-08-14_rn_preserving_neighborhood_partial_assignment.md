# RNC-PA: Regional Normal Correspondence Partial Assignment

Date: 2026-08-14
Status: `DESIGN_STAGE / CONDITIONAL_KEEP`

## Decision

The highest-priority next method is RNC-PA. Exact RN stays as a lower-bounded
unary term; the new evidence is whether a 3x3 query region admits one coherent
partial assignment to a single normal support source under a fixed relative
translation. Unmatched rows are explicit rather than forced onto normal memory.

## Why This Follows the Evidence

- TQRN closed support-tangent quotienting: B0 RN beat M0 on all eight objects,
  and M0 was almost indistinguishable from tangent shuffle.
- SPCT's surviving clue was source/part correspondence, not tangent direction.
- Stable-key and fixed-position retrieval failed, so RNC-PA preserves full
  descriptors and uses relative, not absolute, geometry.
- Region completion was absorbed by morphology, so the method acts inside
  correspondence rather than after the score map.

## Frozen Design

- exact final-layer RN top-5 edges;
- fixed 3x3 query window;
- at most four candidate support sources;
- one-to-one partial assignment within a +/-2 relative-translation window;
- support-LOO distance scale and dustbin increment;
- center min-marginal score never below exact top-1 RN.

## Mandatory Nulls and Kill Rule

Preserve the exact unary edge multiset while shuffling source labels, relative
offsets, or query neighborhoods. Kill the method if it does not separate
bad/good pixels inside narrow B0-distance bins, if a structural null absorbs its
gain, or if normal FPR/object no-harm fails.

## Claim Boundary

This is a method design, not a result. A compact strong/control diagnostic is
next. Only a positive structural-null gate licenses all-eight same-support
TESTpublic comparison with RN-FMLK and SuperAD-16.
