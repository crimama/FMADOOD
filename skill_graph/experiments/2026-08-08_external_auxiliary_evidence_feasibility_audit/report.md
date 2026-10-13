# External Auxiliary Evidence Feasibility Audit

Date: 2026-08-08

Verdict: `NO_METADATA / DESIGN_KILL_METADATA_ORACLE / NEXT_FROZEN_IMAGE_ONLY_PRIOR_FEASIBILITY`

## 1. Motivation

The previous branch, `nuisance-orbit anomaly persistence`, was killed as a closed-branch retread:
it reprocessed transformed query score maps without adding a new anomaly-aligned evidence source.
The non-overlapping question is therefore:

> Is there a deployable external auxiliary evidence source in the current MVTec AD2 single-image
> repo/data that can help distinguish normal distribution shift from defect-induced shift?

Claim/Evidence/Boundary/Positioning:

- Claim: an external evidence source may be required before choosing Flow, Flow Matching, Diffusion,
  or DiT parameterizations.
- Evidence sought: per-image acquisition state, sensor/camera/exposure metadata, paired normal capture,
  or a frozen general prior independent of benchmark answers.
- Boundary: MVTec AD2 single-image, no test/public/private/GT access for method fitting, no query update.
- Positioning: feasibility audit only; no method claim and no GPU benchmark.

Negative-evidence intake:

- internal CLS/register, moment, support-bank, paired-loss, query-persistence, density/object-prior
  branches are not external evidence and are mostly closed or diagnostic-only.
- filename/split/suffix metadata is not a missing-data blocker; it is an oracle channel and must be
  killed as forbidden.

## 2. Implementable Design

The smallest implementable diagnostic is a read-only feasibility audit.

- `target_dataset`: MVTec AD2 single-image
- protocol default remote root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- local evidence root used here: `/home/hun/Volume/DATA/mvtec_ad_2`
- object set: all eight public AD2 objects:
  `can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `vial`, `wallplugs`, `walnuts`
- method family: external evidence/source feasibility, not a detector
- SuperAD baseline source: `N/A`, because no method metric is produced
- primary metric schema: `N/A`, because no candidate score map is evaluated
- SuperAD metric contract retained for future method claims: `seg_AUROC_0.05`, `seg_F1`,
  map/claim coverage, same split/reference/metric

Pre-registered gates:

| Candidate evidence | Gate | Verdict rule |
| --- | --- | --- |
| file/split/suffix condition token | must be deployment-safe and not reveal benchmark answer | fail -> `DESIGN_KILL_METADATA_ORACLE` |
| per-image metadata sidecar | manifest/CSV/JSON/XML/camera/exposure/sensor available for deployment | absent -> `NO_METADATA` |
| embedded image metadata | EXIF/PNG fields differ by acquisition state without using labels | absent/uninformative -> `NO_METADATA` |
| paired acquisition | must preserve single-image task | fail -> forbidden unless user reauthorizes task switch |
| frozen general prior | named prior, weights, license, image-only output, anomaly-preservation assay | pass -> next feasibility audit only |

No code, GPU run, or SuperAD comparison is allowed at this stage.

## 3. Evaluation Alignment

This design tests the motivating claim because it asks whether the missing identifiability signal exists
outside the already failed support/query score families.

SuperAD comparability:

- candidate method: none
- candidate metrics: none
- same-condition SuperAD comparison: `N/A`
- reason: a feasibility audit cannot claim better anomaly detection performance.

Protocol alignment:

- The target remains MVTec AD2 single-image.
- No switch to M2AD, multi-view, active observation, controlled observation, or paired capture is made.
- Public filename condition tokens are treated as forbidden benchmark-oracle channels, not as deployable
  metadata.

Precedent conflicts checked:

- `2026-08-08_nuisance_orbit_anomaly_persistence_design_audit`: query-view persistence closed.
- `2026-08-07_layer7_fused_split_interface_stage_minus1h`: split-interface direction closed.
- `2026-08-07_conditional_normalization_transport_stage_minus1e`: moment/memory transport closed.
- `2026-08-04_flowtte_paired_loss_diagnostic`: alignment improves some normal-shift signal but absorbs defects.
- `2026-08-05_flowtte_semantic_atlas_metric`: geometry safety alone is insufficient.

## 4. Code Modification / Creation

No experiment code was modified or created.

Documentation artifacts created:

- `skill_graph/analysis/2026-08-08_external_auxiliary_evidence_feasibility_audit.md`
- `skill_graph/experiments/2026-08-08_external_auxiliary_evidence_feasibility_audit/report.md`

Lifecycle indexes updated:

- `tasks/todo.md`
- `skill_graph/log.md`
- `skill_graph/index.md`

## 5. Added Code Evaluation

No added code exists, so `py_compile`, unit tests, lsp diagnostics, and CLI tests are not applicable.

Read-only evidence checks performed:

- local data root existence and object split structure
- image counts by object/split
- non-image sidecar search
- representative filename inspection
- representative PNG info and EXIF inspection with Python/Pillow
- documentation link and markdown target verification
- `git diff --check`

## 6. Remote Experiment Execution

No remote execution was performed.

Reason:

- This audit does not produce anomaly maps or metric claims.
- The metadata branch is killed at design stage.
- The frozen-prior branch requires a separately named prior and preregistered anomaly-preservation assay.

Remote/local roots:

- protocol default remote root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- local evidence root: `/home/hun/Volume/DATA/mvtec_ad_2`
- this session did not have `/home/hunim/Volume/DATA/mvtec_ad_2` mounted

## 7. Evaluation Results and Analysis

### 7.1 Dataset and split evidence

All eight public AD2 objects were present locally.

| object | train | validation | test_public | test_private | test_private_mixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| can | 412 | 46 | 252 | 321 | 321 |
| fabric | 387 | 43 | 246 | 314 | 314 |
| fruit_jelly | 263 | 37 | 140 | 255 | 255 |
| rice | 313 | 35 | 222 | 277 | 277 |
| sheet_metal | 137 | 19 | 204 | 142 | 142 |
| vial | 291 | 41 | 245 | 276 | 276 |
| wallplugs | 293 | 33 | 240 | 232 | 232 |
| walnuts | 432 | 48 | 240 | 228 | 228 |

Non-image sidecars under the dataset root were only:

- `/home/hun/Volume/DATA/mvtec_ad_2/readme.txt`
- `/home/hun/Volume/DATA/mvtec_ad_2/license.txt`

No per-image manifest, CSV, JSON, XML, camera log, exposure log, or sensor metadata file was found.

### 7.2 Filename and split tokens

Representative `can/test_public/bad` filenames expose:

- `000_regular.png`
- `000_overexposed.png`
- `000_underexposed.png`
- `000_shift_1.png`
- `000_shift_2.png`
- `000_shift_3.png`

Private split filenames expose:

- `test_private/000_regular.png`
- `test_private_mixed/000_mixed.png`

These are not acceptable auxiliary variables. They reveal benchmark condition/split information and are
image-dependent oracle channels. A method using them would answer "which benchmark bucket is this image in?"
rather than solve deployable few-shot AD under unknown distribution shift.

Verdict for this branch: `FORBIDDEN / DESIGN_KILL_METADATA_ORACLE`.

### 7.3 Embedded metadata

Python/Pillow inspection of representative PNGs showed the same generic metadata across conditions:

```text
format=PNG, mode=RGB
info={'dpi': (299.9994, 299.9994), 'gamma': 0.45455}
exif_len=0
```

This held for sampled train/validation/public/private/private_mixed images including `regular`,
`overexposed`, `underexposed`, and `shift_*` examples.

Verdict for this branch: `NO_METADATA`.

### 7.4 Validation good and paired acquisition

Validation good images are regular normal images, not an acquisition-state variable. Using them as extra
support changes the 16-shot protocol and does not solve shift-defect identifiability.

Paired normal capture would be a different task setting. Under the current MVTec AD2 single-image contract,
paired acquisition is forbidden unless the user explicitly reauthorizes a task switch to paired/multi-view or
controlled-observation data.

Verdict for this branch: forbidden under current scope.

### 7.5 Internal signals are not external evidence

Previously tested internal signals remain insufficient:

- CLS/register/global task code: conditioning worsened held shifted-normal inflation.
- paired loss: normal alignment appeared but defect retention failed.
- SemanticAtlas/geometry: preserved defect/neighbor geometry but had negligible shift effect.
- conditional smoothness and shared direction: did not become a safe nuisance selector.
- support-bank and query-orbit persistence: closed as score/map aggregation retreads.
- density/object priors: calibration or foreground priors, not acquisition evidence.

Therefore, the next non-overlapping path is not a new aggregation or module family. It is a named frozen
general prior feasibility audit.

### 7.6 Frozen general prior feasibility

A frozen general image-only prior remains open only as a feasibility question. It must not consume AD2
filename/split/condition tokens, paired captures, public labels, private labels, or query-specific benchmark
answers.

Required contract:

- named prior and version
- weights source and local availability
- license and research-use compatibility
- input preprocessing and output schema
- image-only output independent from AD2 path tokens
- no AD2 test tuning
- anomaly-preservation assay for local and broad synthetic defects
- hard-stop if the output mostly proxies condition labels or suppresses defect evidence

Flow Matching, Diffusion, and DiT are not promoted. They remain parameterization choices that can be considered
only after a valid evidence source is found.

Strict method claim now: no.

Small next diagnostic justified: yes, but only `frozen general image-only prior feasibility`; not metadata,
filename, paired acquisition, or score-fusion retuning.

## 8. Conclusion

Verdict: `NO_METADATA / DESIGN_KILL_METADATA_ORACLE / NEXT_FROZEN_IMAGE_ONLY_PRIOR_FEASIBILITY`

This audit closes the metadata branch in two different ways:

1. **No deployable metadata found**: no per-image acquisition sidecar or embedded EXIF-like signal exists in the
   current local dataset tree.
2. **Visible condition tokens are forbidden**: public `regular/overexposed/underexposed/shift_*` and private
   `regular/mixed` suffixes are benchmark-oracle channels, not usable deployment metadata.

Paired acquisition is also not allowed under the current single-image task. It is not an unblock contract.

The only allowed next branch is a separately preregistered frozen general image-only prior feasibility audit.
That branch must name the prior, weights, license, output schema, and anomaly-preservation assay before any
code or GPU benchmark. If no such prior is available, the architecture search remains blocked.

## 9. Post-Conclusion Storage Cleanup

No dense outputs were generated.

- remote result root: none
- local pullback root: none
- `anomaly_maps/`: none created
- cleanup verification: no GPU run and no result root exist for this audit

