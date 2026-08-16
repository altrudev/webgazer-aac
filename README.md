# webgazer-aac

**Local-first AAC reliability and gaze interaction layer for WebGazer.js**

Built by [ALTRU.dev](https://altru.dev) — Code for Humanity · GPLv3

`webgazer-aac` is a dependency-free browser layer for turning WebGazer gaze estimates into safer, more stable interaction signals for augmentative and alternative communication (AAC) interfaces.

Version 2.0 is a correctness-focused redesign. It does **not** claim medical-device status, clinical validation, or calibrated statistical confidence. Its job is narrower: preserve the browser-only deployment model while making calibration, dwell selection, adaptive feedback, drift detection, target resolution, and diagnostics more defensible and testable.

## Why v2

The 1.x implementation added useful ideas, but review against WebGazer 3.5.3 exposed three important correctness risks:

1. **Eye-feature integration:** current WebGazer already emits `data.eyeFeatures` with each gaze prediction. v2 consumes that exact evidence instead of attempting to call tracker patch extraction again without the required arguments.
2. **PCA feature-space consistency:** fitting a new per-user PCA basis after calibration changes the feature space. v2 keeps normalized raw calibration patches long enough to rebuild all regression samples after PCA fitting.
3. **Evidence-safe adaptation:** a model's own gaze coordinate is not independent ground truth. v2 uses the selected target anchor for dwell feedback and weights different evidence sources explicitly.

These changes are why this is a major version rather than a 1.x patch.

## Architecture

```text
WebGazer 3.5.3 gaze frame
        │
        ├── eyeFeatures ──> fixed-size patch normalization ──> per-user PCA
        │                                                   │
        │                                                   ├── polynomial regressor
        │                                                   └── RBF regressor
        │                                                         │
        └── upstream x/y fallback <──────────────────────── ensemble prediction
                                                                  │
                                                motion / blink / Kalman filtering
                                                                  │
                                                   trackingQuality + gazeStability
                                                                  │
                                                       GazeTargetResolver
                                                                  │
                                                 targetConfidence + adaptive dwell
                                                                  │
                                                      confirmed target selection
                                                                  │
                            target anchor ──> weighted drift evidence + recalibration
```

The key distinction is:

**gaze estimate ≠ tracking quality ≠ target confidence ≠ confirmed target ≠ ground truth**

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the evidence model.

## Quick start

Load WebGazer first, then this file:

```html
<script src="webgazer.js"></script>
<script src="webgazer-aac.js"></script>
<script>
  webgazerAAC
    .install()
    .enableAdaptiveRecalibration()
    .enableDriftWatchdog();

  const dwell = webgazerAAC.createDwellTimer({
    dwellMs: 800,
    minTrackingQuality: 0.3,
    minTargetConfidence: 0.35,
  });

  webgazer.setGazeListener((gaze) => {
    if (!gaze) return; // blink / unavailable frame
    dwell.updateFromGaze(gaze);
  });

  webgazer.begin();
</script>
```

For best target resolution, mark AAC controls explicitly:

```html
<button data-gaze-target>Yes</button>
<button data-gaze-target>No</button>
<button data-gaze-target>Help</button>
```

The resolver also recognizes common actionable elements such as buttons, links, form controls, ARIA buttons/options/menu items, and focusable elements.

## Calibration flow

During calibration, keep using WebGazer's `recordScreenPosition()` or your existing calibration UI. The v2 interceptor preserves upstream behavior and records the same label locally.

At the end of calibration:

```js
const result = webgazerAAC.fitUserBasis();
console.log(result);
// { left: true, right: true, samples: ..., rebuilt: true }

await webgazerAAC.saveCalibration();
```

`fitUserBasis()` performs a two-pass transition:

1. fit the per-user PCA basis from normalized calibration patches;
2. rebuild every polynomial and RBF training sample in that fitted basis.

That rebuild is a v2 invariant. Training vectors encoded with the fallback basis are never silently mixed with inference vectors encoded by the fitted basis.

## Calibration persistence

v2 separates library, storage-schema, and feature-space versions:

```js
const snap = webgazerAAC.getCalibrationSnapshot();

snap.libraryVersion; // "2.0.0"
snap.schemaVersion;  // 2
snap.featureVersion; // feature-pipeline identifier
```

Restore on startup:

```js
const saved = await webgazerAAC.loadCalibration();

if (!saved) {
  // no saved calibration
} else if (saved._incompatibleReason) {
  // incompatible schema/feature pipeline: recalibrate
} else if (saved._validationRequired) {
  // viewport changed materially: run a short validation first
  // then, if acceptable:
  webgazerAAC.applyCalibrationSnapshot(saved);
} else {
  // compatible snapshot was applied automatically
}
```

v1.x calibration snapshots are intentionally not treated as v2-compatible because the feature pipeline changed.

### Configure storage before install

```js
webgazerAAC.configureStore({
  dbName: 'my-aac-app',
  storeName: 'gaze-calibration',
  profileKey: 'user-1',
});

webgazerAAC.install();
```

Unlike v1.3, `install()` does not overwrite a previously configured store.

## Tracking quality vs target confidence

v2 deprecates the idea that one number should be called simply `confidence`.

Every gaze result can include:

```js
{
  x,
  y,
  vx,
  vy,
  isSaccade,
  isBlink,
  gazeStability,       // temporal stability heuristic, 0..1
  trackingQuality,     // quality of this gaze frame, 0..1
  eyeFeaturesAvailable // whether WebGazer eyeFeatures were available
}
```

`confidence` remains as a backward-compatible alias of `trackingQuality`, but new code should use the explicit name.

Target resolution produces a separate `targetConfidence` based on candidate geometry and competition between nearby controls.

## AAC target resolver

Resolve a target directly:

```js
const result = webgazerAAC.resolveTarget(gaze.x, gaze.y);

if (result) {
  console.log(result.element);
  console.log(result.confidence);
}
```

Or create a custom resolver:

```js
const resolver = webgazerAAC.createTargetResolver({
  selector: '[data-gaze-target]',
  expansionPx: 48,
  maxDistancePx: 180,
  hysteresisBonus: 0.22,
});
```

The resolver:

- canonicalizes nested DOM content to the actionable ancestor;
- uses expanded target geometry rather than requiring pixel-perfect hits;
- compares multiple nearby candidates;
- applies target hysteresis to reduce rapid switching;
- caches the target set briefly so it does not query the full DOM every gaze frame.

## Adaptive dwell

Adaptive dwell is enabled by default for timers created through `webgazerAAC.createDwellTimer()`.

```js
const timer = webgazerAAC.createDwellTimer({
  dwellMs: 800,             // baseline
  minTrackingQuality: 0.3,
  minTargetConfidence: 0.35,
});
```

Dwell duration can increase when:

- tracking quality is weak;
- target confidence is low;
- the target is small;
- the interaction is crowded/ambiguous;
- recent correction history indicates accidental selections.

The built-in controller is deterministic and local; it does not require a cloud model.

## Evidence-safe adaptive recalibration

When dwell completes, v2 does **not** feed the gaze coordinate back as truth.

Instead:

```text
predicted gaze → resolved target → confirmed selection → target anchor → low-weight adaptation
```

Built-in evidence weights distinguish sources such as calibration, explicit/confirmed input, dwell selections, and inferred coordinates.

The legacy `recordDwellHitXY(x, y)` API still exists, but arbitrary coordinates are treated as low-weight inferred evidence unless the caller explicitly supplies stronger provenance.

## Drift watchdog

```js
webgazerAAC.enableDriftWatchdog({
  warnThreshold: 120,
  critThreshold: 220,
  minEvidence: 5,
});

document.addEventListener('webgazer-aac:drift-warning', event => {
  console.log(event.detail);
});
```

The watchdog evaluates the model **before** adding the newly confirmed label, then updates a weighted residual history. Dwell-derived target evidence is lower-weight than explicit calibration evidence.

```js
webgazerAAC.getDriftRmse();
webgazerAAC.resetDriftWatchdog();
```

Pixel RMSE is a diagnostic signal, not a clinical accuracy claim.

## Diagnostics

```js
console.table(webgazerAAC.getDiagnostics());
```

Diagnostics include:

- effective gaze-listener FPS;
- eye-feature coverage;
- fallback frame count;
- blink and saccade frame counts;
- explicit and adaptive evidence counts;
- calibration sample count;
- current tracking quality;
- drift RMSE and drift state;
- active regression mode;
- PCA fitted state.

For browser frame-clock diagnostics:

```js
webgazerAAC.attachVideoClock(videoElement);
```

When `requestVideoFrameCallback()` is available, the diagnostics record presented frames and frame metadata separately from gaze-listener timing.

## Main API

### Setup

```js
webgazerAAC.install(options?)
webgazerAAC.setRegression('ensemble' | 'polynomial' | 'rbf' | 'ridge')
webgazerAAC.getRegressionMode()
```

### Calibration

```js
webgazerAAC.fitUserBasis()
webgazerAAC.resetCalibrationPatches()
webgazerAAC.isPCAFitted()
webgazerAAC.recordGroundTruth(x, y, options?)
```

### Adaptive feedback

```js
webgazerAAC.enableAdaptiveRecalibration()
webgazerAAC.disableAdaptiveRecalibration()
webgazerAAC.recordConfirmedSelection(element, options?)
webgazerAAC.recordDwellHit(element, options?)       // compatibility helper
webgazerAAC.recordDwellHitXY(x, y, options?)       // low-weight by default
```

### Targets and dwell

```js
webgazerAAC.resolveTarget(x, y, root?)
webgazerAAC.createTargetResolver(options?)
webgazerAAC.createDwellTimer(options?)
```

### Smoothing and quality

```js
webgazerAAC.setKalmanParams(processNoise, measurementNoise)
webgazerAAC.resetSmoother()
webgazerAAC.getTrackingQuality()
webgazerAAC.getConfidence() // deprecated alias
```

### Persistence

```js
webgazerAAC.configureStore(options)
await webgazerAAC.saveCalibration()
await webgazerAAC.loadCalibration()
await webgazerAAC.clearCalibration()
await webgazerAAC.isStorageAvailable()
webgazerAAC.getCalibrationSnapshot()
webgazerAAC.applyCalibrationSnapshot(snapshot)
```

### Diagnostics

```js
webgazerAAC.getDiagnostics()
webgazerAAC.attachVideoClock(videoElement)
webgazerAAC.detachVideoClock()
```

## Tests

No dependencies or build step are required:

```bash
npm test
```

or:

```bash
node test/webgazer-aac.test.js
```

The v2 regression suite checks the key invariants rather than only exercising individual classes. See [`docs/VALIDATION.md`](docs/VALIDATION.md).

## Compatibility

v2 is designed around **WebGazer 3.5.3**, the final planned upstream release.

Expected integration order:

1. load WebGazer;
2. load `webgazer-aac.js`;
3. call `webgazerAAC.install()`;
4. register your gaze listener;
5. call `webgazer.begin()`.

The library remains a classic single-file browser script and also exports through `module.exports` for Node-based testing.

## Privacy

`webgazer-aac` contains no telemetry, analytics, network requests, account system, or cloud inference. Calibration state is stored locally through IndexedDB when persistence is enabled.

Webcam access and any upstream WebGazer model/runtime assets remain governed by the browser and the WebGazer deployment chosen by the host application.

## Validation status

This repository is an engineering project for assistive interaction. It has unit/invariant tests, but it is **not** a medical device and has not been clinically validated. Real-user AAC evaluation remains necessary before making claims about communication speed, accuracy, fatigue, accessibility outcomes, or suitability for a particular person's needs.

See [`docs/VALIDATION.md`](docs/VALIDATION.md) for the validation boundary and proposed benchmark protocol.

## License

GPLv3, compatible with upstream WebGazer.

## Contributing

Issues and pull requests are welcome. Feedback from AAC users, caregivers, occupational therapists, speech-language pathologists, accessibility engineers, rehab engineers, and special-education practitioners is especially valuable.
