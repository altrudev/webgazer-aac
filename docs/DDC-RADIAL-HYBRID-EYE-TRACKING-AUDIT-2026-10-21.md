# DDC Radial — webcam AAC capture audit and hybrid architecture (2026-10-21)

## Scope and provenance

- Repository: `altrudev/webgazer-aac`, audited main `0289bb4ada8dbf81bffd21f654ad75a13f0f0c0a`.
- Deployment: `https://altru.dev/webgazer-aac/test`, with distinct hosted guided-calibration, holdout-validation and diagnostic JavaScript.
- Reproducible checks: `npm test` 13/13 PASS, `node --check` on all four hosted scripts PASS. These are **not** hardware acceptance.
- Webcam evidence: user-exported v2 diagnostic results, median five-target errors 150–222 px (latest 222.44 px), approximately 30 observed video frames/second, and consistent initial gaze-acquisition delay. No retained camera frames or landmarks.
- Safety: prohibit gaze-driven AAC activation until quality gates pass; no automatic telemetry or video uploads.

## Verified source-level findings

**P0 — stale training pairs.** Hosted `webgazer-aac-guided.js` uses a recent *gaze prediction callback* (<180ms) to authorize `webgazer.recordScreenPosition` from a timer, not a proven current eye-patch timestamp. Its `accepted` counter increments whenever that function does not throw, even though WebGazer's `recordScreenPosition` returns without storing if `latestEyeFeatures` is absent. Thus the diagnostic counter is attempts, not confirmed examples.

**P0 — incompatible model lifecycle.** Guided reset calls `webgazer.clearData()` but the AAC adapter maintains its own `_regressions`, PCA bases and eye-feature buffers. Regression training and the upstream lifecycle are not transactionally bound. Source audit does not establish that the custom model is both trained and evaluated on same feature schema and reset boundary.

**P1 — inadequately isolated holdout.** The hosted validation disables explicit test clicks, but there is no central training gate applied to all upstream and adapter learning operations. Report states this limitation itself. A separate UI does not prove holdout isolation.

**P1 — insufficient calibration coverage.** Nine guided training points are positioned within one internal test panel, not across the entire display; that imposes extrapolation when using gaze outside the panel. Three clickable targets remain accessible and can change model data.

**P1 — limited data quality signals.** The adapter validates patch object presence, and normalized resampling now rejects malformed dimensions, but captures no explicit eye-region quality, face size, pupil/iris visibility, pose and crop consistency metadata in diagnostics.

**P1 — tracking versus evidence rate.** Prediction callbacks are not camera-frame ground truth. Current export gives only aggregate five-target errors and no model identifier/build SHA. A 30fps video stream does not imply reliable gaze estimates.

**P2 — deployment drift.** Hosted lab scripts are maintained separately from repository files; review/test gates for repo `main` do not cover the live lab. The lab needs source-controlled assets, a release manifest and checksum verification.

## Recommended multi-solution hybrid — gated ensemble, local-first

1. **Capture A: WebGazer baseline.** Keep upstream FaceMesh eye patches + ridge predictor as control. Explicitly suppress mouse auto-learning after begin, and use only supervised calibration points with verified current features.
2. **Capture B: geometry and iris.** Evaluate MediaPipe **Face Landmarker Tasks Vision** (478 landmark model), iris centers relative to inner/outer canthi, eyelid opening, pupil-to-eye-width ratios, binocular agreement and face transformation matrix/head-pose. These are *features*, not directly screen gaze. Confirm actual model bundle, version, browser runtime and license before adoption.
3. **Capture C: normalized appearance.** Keep normalized left/right 60×40 eye crops, contrast normalization and a modest regularized appearance-to-screen mapping. Do not use unfitted PCA/random projections as authoritative gaze.
4. **Pose-aware fusion.** Combine geometry and appearance predictions only after separately collecting quality-weighted supervised labels. Compare each component on frozen, withheld test points, track error by screen zone and head pose. Degrade to no prediction when eye quality or agreement fails; do **not** average two bad models.
5. **Temporal state.** A time-stamped acquisition/track/lost/recovered state machine, smooth only trusted predictions, reset smoothing after blink or loss. No dwell activation until calibration quality, age and confidence gates pass.
6. **Calibration.** Whole-viewport 3×3 or 4×4 targets with dwell fixation, settle duration, paired frame/time ID, minimum eye-quality score, confirmed accepted-count, robust outlier rejection. Hard freeze model and feature normalization during validation; retain camera processing.
7. **Diagnostics.** Local metrics for frame age, face/eyes availability, head-pose bins, crop quality, raw A/B predictions, fusion disagreement, per-target median/P90, and latency. Only export counts, error summaries and version hashes, not coordinates, images, landmarks, or biometric templates.
8. **Control alternative.** Offer intentional switch/mouse/touch input as an **independent** backup AAC modality; never silently substitute mouse coordinates into the claimed eye-gaze channel.

## Experiment matrix (run same camera, lighting, targets and session protocol)

| Arm | Tracker/features | Predictor | Supervised training |
|---|---|---|---|
| A | Upstream FaceMesh eye patch | WebGazer ridge | Same calibrated points |
| B | Face Landmarker iris + pose | Regularized geometry model | Same calibrated points |
| C | Resampled eye appearance | Ridge/robust regression | Same calibrated points |
| D | A+B+C quality-gated | Confidence/variance-based fusion | Frozen, withheld validation |

Compare whole-viewport median error, P90 error, per-zone worst case, re-acquisition, valid **independent-frame** percentage, latency, CPU and memory. Benchmarks must be measured, not projected. **Fail closed for activation** if calibration incomplete, face/eye confidence low, model disagreement large, tracking stale or dwell interrupted. Continue rendering UI and offering other communication input modes.

## Execution gates

- G0: bring hosted page/scripts into repo, pinned vendor manifests and CI-independent DDC `repo.verify`.
- G1: test-first stale-pair rejection; verified paired frame ID; actual accepted-count; reset upstream + adapter and lock learning during validation.
- G2: optional browser camera/landmark pipeline with visualized **local-only** quality metadata and no automatic dispatch.
- G3: implement geometry/iris candidate in isolation, license and runtime budget review.
- G4: A/B/C/D within-subject evaluation with explicit positive and negative fixtures, signed reproducible evidence, genuine webcam results.
- G5: supervised AAC selection with accessible cancellation and confidence/dwell gates; requires independent user acceptance.

## Terminology

- Eye patch = cropped eye-region image tensor (not a gaze estimate).
- Iris feature = iris center relative to eye corners/eyelids.
- Head pose = camera-relative facial rotation/translation.
- Calibration = supervised feature-to-display-coordinate fit.
- Holdout = validation samples unavailable to all model updates.
- Fusion = combining independently evaluated predictions with reliability weights.
- Fail closed = do not activate AAC actions when quality evidence is inadequate.

## Third-party and license review

WebGazer code is GPL-3.0-or-later; retain notices and audit distribution/combination implications before bundling with proprietary source. MediaPipe Tasks Vision components, model assets and their licenses need independent package/model-specific checking. This document licenses no vendor component and grants no implied right to use model weights.
