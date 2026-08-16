# webgazer-aac v2 architecture

## Design objective

The v2 architecture treats gaze interaction as a chain of evidence-bearing transitions rather than a single `(x, y, confidence)` output.

```text
camera frame
  → WebGazer eyeFeatures
  → normalized eye patches
  → feature-space projection
  → gaze estimate
  → temporal quality assessment
  → target candidate resolution
  → dwell/intent decision
  → selected target
  → weighted adaptation evidence
```

Each transition has different semantics. Collapsing them into one confidence score creates circular reasoning and makes failures difficult to diagnose.

## Core distinctions

### Gaze estimate

An `(x, y)` estimate produced by the active regressor and temporal filter. It is a prediction, not truth.

### Gaze stability

A temporal heuristic based on motion and innovation. High stability means the recent signal is internally steady; it does not prove that the estimate is spatially accurate.

### Tracking quality

A frame-level quality score combining feature availability and stability. This replaces the ambiguous v1 `confidence` terminology. The old `confidence` property remains only as a compatibility alias.

### Target confidence

A separate score produced by `GazeTargetResolver`. It reflects how strongly the current gaze estimate supports one actionable target relative to alternatives.

### Confirmed target

The application-level target that completed the dwell/selection process. This is stronger evidence than the gaze coordinate itself because it incorporates target geometry and the interaction decision.

### Ground truth / supervision

A coordinate used to train or evaluate the gaze model. Supervision must identify its source and evidence weight.

## Evidence taxonomy

v2 uses explicit evidence classes:

| Source | Default weight | Interpretation |
|---|---:|---|
| `calibration` | 1.00 | explicit calibration target |
| `explicit` | 1.00 | externally confirmed screen coordinate |
| `confirmed-click` | 0.85 | explicit user selection with current eye evidence |
| `dwell-selection` | 0.35 | selected target anchor after dwell completion |
| `inferred` | 0.15 | weak/non-independent coordinate evidence |

Weights affect adaptation and drift evidence. They do not convert weak evidence into ground truth.

## Avoiding circular evidence

The v1 anti-pattern was conceptually:

```text
model predicts (x, y)
  → dwell completes at (x, y)
  → feed (x, y) back as ground truth
  → model appears to agree with its own prediction
```

v2 uses:

```text
model predicts gaze
  → resolver selects target candidate
  → dwell confirms target
  → target anchor becomes low-weight supervision
```

The target's center, or an application-supplied `data-gaze-x` / `data-gaze-y` anchor, is therefore the label. The gaze prediction is never recycled as independent truth by the built-in dwell path.

## PCA transition invariant

A fitted PCA basis changes the coordinates of the feature space. Therefore:

```text
features encoded before PCA fit ≠ features encoded after PCA fit
```

v2 retains normalized raw calibration patches during calibration. `fitUserBasis()` performs:

1. PCA fit from those normalized patches;
2. deletion/replacement of old regression feature vectors;
3. regeneration of every training vector in the fitted basis;
4. regression refit on demand.

This invariant is covered by the automated test suite.

## WebGazer integration boundary

WebGazer 3.5.3 includes `eyeFeatures` on the data delivered to its gaze listener. v2 consumes that exact feature object.

It does not call `tracker.getEyePatches()` from the listener because current FaceMesh patch extraction is asynchronous and requires video/canvas/dimension arguments. Reusing `data.eyeFeatures` also keeps AAC processing aligned with the same frame used by WebGazer's prediction.

## Target resolution

`GazeTargetResolver` normalizes DOM interaction targets by:

- climbing nested DOM content to the actionable ancestor;
- supporting explicit `[data-gaze-target]` controls;
- considering common interactive HTML/ARIA roles;
- expanding target geometry for noisy gaze;
- comparing multiple candidates rather than only using `elementFromPoint()`;
- adding mild hysteresis for the previously selected candidate;
- caching the target set briefly to reduce repeated DOM queries.

The resolver produces `targetConfidence`, which is intentionally separate from `trackingQuality`.

## Adaptive dwell

`AdaptiveDwellController` modifies the dwell requirement based on interaction risk. Lower quality, smaller targets, poorer target confidence, or denser target layouts can increase dwell time.

The controller is deterministic and local. It is a heuristic, not an inferred neurological or clinical state.

## Drift monitoring

The watchdog evaluates the current model against newly confirmed supervision **before** the new label is added to the model. This avoids measuring the model after it has already fit the observation being evaluated.

Residual evidence is weighted according to provenance. Pixel RMSE is diagnostic only and should not be presented as a clinical accuracy guarantee.

## Persistence boundary

v2 snapshots carry three independent version concepts:

- `libraryVersion`: implementation release;
- `schemaVersion`: storage structure;
- `featureVersion`: semantic identity of the feature pipeline.

A library patch release can therefore remain compatible with stored calibration if its schema and feature pipeline are unchanged.

Viewport changes mark a snapshot `_validationRequired` instead of silently treating it as equivalent. The caller may run a short validation and explicitly apply the snapshot afterward.

## Local-first boundary

The AAC layer itself:

- makes no network requests;
- contains no telemetry;
- stores calibration only through local IndexedDB when requested;
- exposes diagnostics to the host application rather than uploading them.

The host application's WebGazer distribution and webcam/runtime configuration remain separate deployment concerns.
