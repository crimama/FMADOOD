# Few-shot × shift × adaptation diagnosis, preregistered 2026-09-20

Purpose: test the problem framing, not propose a novel method. H1: synthetic
shift raises clean-normal false positives more at1-shot than16-shot. H2: global
query correction reduces false positives but can reduce defect detection.
H3: excluding the scored spatial block from correction fitting mitigates that
tradeoff. These hypotheses may fail; do not force a geometry narrative.

Budget: full8 AD2 seen-development, seed0, shots1/16, QR0/no correction,
QR1/global correction, QR3/spatially excluded correction:48 class evaluations.
Here QR3 has robust_trim=false, so ONLY spatial exclusion differs from QR1;
it is not the historical QR3 configuration. Frozen DINOv2-L layers11/23,
448/672 resolutions,256 tokens/source, one bank, same original R1 MAD pipeline
for every arm. This diagnostic comparator differs from RSG3; no direct claims
against CM/LR macro results. Identical fixed support manifests within a shot;
1-shot is the first element of the seeded16 support pool. No parameter search.

After standard public evaluation, opt-in evaluation-only diagnostics predict
16 additional normal training images excluded from the entire support16 pool.
They are the first16 sorted remaining paths. Each arm's pixel95 threshold uses
at most4096 uniformly spaced pixels/image. This is an ADDITIONAL normal
evaluation-calibration set, not method fitting or a strict one-image deployment
budget claim. Calibration is fixed across shots/arms. No calibration labels or
query labels/ground truth are passed to methods. Methods remain query-stateless.

Diagnostic queries: first8 sorted good and first8 sorted bad public images,
each original and brightness gain0.7 (round-to-nearest uint8, no geometric
change). Normal-image mean pixel FPR and defect-mask mean pixel TPR at the
clean-normal threshold are primary diagnostics; clean/shift normal and defect
score means are also saved. Brightness is SYNTHETIC, original public images are
the paired reference, not certified ID. No inference about private acquisition
shift or unseen generalization. Labels only define evaluator strata/masks.
Per-image scalar records, calibration paths/hash and thresholds are retained in
run_manifest.json; temporary transformed images are automatically removed.
No persistent maps or weights. Original image bytes remain unchanged.

Analysis fixed before execution: per class/arm/shot compute shift-minus-original
normal FPR and defect TPR. H1 interaction = FPR increase at1shot minus16shot.
For H2 compare QR1−QR0 normal FPR and defect TPR separately under each condition;
do not call uniform score shrinkage defect preservation. H3 compares QR3−QR1.
Report all8 classes, equal-class macro, both shots, paired query records, and
calibration uncertainty. One seed/one corruption is exploratory, not significance
or novelty evidence. A follow-up method is justified only if H1 interaction is
positive in >=5/8 classes, AND QR1 exhibits lower shifted normal FPR together
with lower defect TPR than QR0 in >=4/8 classes at1shot. Otherwise revise framing
instead of inventing a correction module. No automatic replication in this budget.

Execution: existing generic runner/collector, local can/fabric, dsba3 remaining
six objects on existing assigned UUIDs. Existing containers only; remote local
snapshot, no commits/pushes. Methods and common inference unchanged; the sole
pipeline addition is an opt-in evaluation hook after standard metrics. Existing
standard metric contract remains mvtec_ad2_float16_streaming_v1.
