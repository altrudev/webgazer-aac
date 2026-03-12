# webgazer-aac

**Accessibility-first eye tracking enhancement layer for [WebGazer.js](https://webgazer.cs.brown.edu/)**

Built by [ALTRU.dev](https://altru.dev) — Code for Humanity · GPLv3

---

webgazer-aac is a single-file drop-in that loads after WebGazer and replaces its regression, smoothing, and gaze output pipeline with a significantly more accurate and robust stack. It was built specifically for **AAC (augmentative and alternative communication)** use cases where tracking reliability directly affects a user's ability to communicate.

No bundler. No new dependencies. No data leaves the device.

---

## What it does

| Layer | Component | What it replaces / adds |
|---|---|---|
| Preprocessing | CLAHE contrast normalisation | Raw patch → better features in poor light |
| Features | Per-user PCA basis | Fixed random projection → user-fitted eigenvectors |
| Regression | Polynomial + RBF + Ensemble | WebGazer's ridge regression |
| Smoothing | 4-state Kalman filter | EMA velocity smoother |
| Signal | Blink detection | — (new) |
| Signal | Saccade suppression | — (new) |
| CPU | Frame cache | — (new, saves ~30% on stable gaze) |
| Session | Drift watchdog | — (new) |
| UX | Confidence-gated dwell timer | — (new) |
| Persistence | IndexedDB calibration store | — (new, survives page reload) |
| Feedback | Adaptive recalibration | — (new) |

---

## Quick start

```html
<!-- 1. Load WebGazer first -->
<script src="webgazer.js"></script>

<!-- 2. Then load webgazer-aac -->
<script src="webgazer-aac.js"></script>

<!-- 3. Install and go -->
<script>
  webgazerAAC
    .install()
    .enableAdaptiveRecalibration()
    .enableDriftWatchdog();

  webgazer.begin();
</script>
```

### With calibration persistence (recommended)

```js
webgazerAAC.install().enableAdaptiveRecalibration().enableDriftWatchdog();

// On page load — restore previous calibration if available
const saved = await webgazerAAC.loadCalibration();
if (saved) {
  console.log('Calibration restored', saved._screenMismatch ? '(screen size changed)' : '');
  // Skip calibration UI
} else {
  // Show calibration UI, then at the end:
  webgazerAAC.fitUserBasis();
  await webgazerAAC.saveCalibration();
}

webgazer.begin();
```

### With confidence-gated dwell

```js
const timer = webgazerAAC.createDwellTimer({ dwellMs: 800, minConfidence: 0.3 });

webgazer.setGazeListener((data) => {
  if (!data) return; // blink — dwell timer freezes automatically
  const el = document.elementFromPoint(data.x, data.y);
  timer.update(el, data.x, data.y, data.confidence, data.isSaccade);
});

// Listen for completion
document.addEventListener('webgazer-aac:dwell-complete', e => {
  console.log('Dwell completed at', e.detail.x, e.detail.y);
});
```

### Drift detection

```js
webgazerAAC.enableDriftWatchdog({
  warnThreshold: 120,   // px RMSE
  critThreshold: 220,
});

document.addEventListener('webgazer-aac:drift-warning', e => {
  showRecalibrationNudge(`Gaze accuracy dropping (${e.detail.rmse}px error)`);
});

document.addEventListener('webgazer-aac:drift-critical', e => {
  showRecalibrationPrompt();
});
```

---

## API reference

### Setup

```js
webgazerAAC.install()                         // intercept WebGazer, wire pipeline
webgazerAAC.setRegression(mode)               // 'ensemble' (default) | 'polynomial' | 'rbf' | 'ridge'
webgazerAAC.enableAdaptiveRecalibration()     // dwell hits feed back as training samples
webgazerAAC.disableAdaptiveRecalibration()
```

### Calibration

```js
webgazerAAC.fitUserBasis()                    // fit per-user PCA at calibration end
webgazerAAC.resetCalibrationPatches()         // clear patches before re-calibration
webgazerAAC.isPCAFitted()                     // → boolean
```

### Persistence

```js
await webgazerAAC.saveCalibration(options?)   // → Promise<boolean>
await webgazerAAC.loadCalibration(options?)   // → Promise<snapshot | null>
await webgazerAAC.clearCalibration(options?)  // → Promise<boolean>
await webgazerAAC.isStorageAvailable()        // → Promise<boolean> (false in private browsing)

webgazerAAC.getCalibrationSnapshot()          // → plain object (serialisable)
webgazerAAC.applyCalibrationSnapshot(snap)    // → boolean
webgazerAAC.configureStore({ dbName, storeName, profileKey })
```

### Dwell timer

```js
const timer = webgazerAAC.createDwellTimer({
  dwellMs:       800,   // ms to complete a dwell (default 800)
  minConfidence: 0.25,  // Kalman confidence gate (default 0.25)
  holdAfterMs:   1200,  // re-fire lockout after completion (default 1200)
});

timer.update(element, x, y, confidence, isSaccade, isBlink?)  // → progress 0–1
timer.reset()
timer.progress  // current progress 0–1
```

**Events** (fire on the target element, bubble):
- `webgazer-aac:dwell-progress` — `{ progress, confidence, x, y, frozen }`
- `webgazer-aac:dwell-complete` — `{ x, y }`
- `webgazer-aac:dwell-cancel` — `{ reason, x, y }`

### Drift watchdog

```js
webgazerAAC.enableDriftWatchdog({
  warnThreshold: 120,   // RMSE px for warning (default 120)
  critThreshold: 220,   // RMSE px for critical (default 220)
  minSamples:    8,     // samples before RMSE is trusted (default 8)
  onWarn:        fn,    // optional callback in addition to CustomEvent
  onCritical:    fn,
})
webgazerAAC.disableDriftWatchdog()
webgazerAAC.resetDriftWatchdog()              // call after re-calibration
webgazerAAC.getDriftRmse()                    // → current RMSE in px
```

**Events** (fire on `document`):
- `webgazer-aac:drift-warning` — `{ rmse, level, sampleCount, timestamp }`
- `webgazer-aac:drift-critical` — `{ rmse, level, sampleCount, timestamp }`

### Kalman

```js
webgazerAAC.setKalmanParams(processNoise, measurementNoise)
// processNoise: larger = more responsive (default 8)
// measurementNoise: larger = smoother but laggier (default 50)
webgazerAAC.resetSmoother()
```

### Gaze listener output

```js
webgazer.setGazeListener((data, elapsed) => {
  if (data === null) return; // blink in progress

  data.x          // smoothed gaze X (px)
  data.y          // smoothed gaze Y (px)
  data.confidence // Kalman confidence 0–1
  data.isSaccade  // boolean — fast eye movement in progress
});
```

---

## Exposed classes

All classes are available for direct use or testing:

```js
const {
  PolynomialRegression,
  RBFRegression,
  EnsembleRegression,
  KalmanFilter,
  BlinkDetector,
  SaccadeDetector,
  FrameCache,
  PCABasis,
  DriftWatchdog,
  DwellTimer,
  CalibrationStore,
  _MemoryBackend,    // in-memory IDB mock for testing
  AdaptiveRecalibrator,
} = webgazerAAC;
```

---

## Running the tests

No dependencies, no build step — pure Node.js:

```bash
node test/webgazer-aac-test.js
```

Expected output: all tests passing, including async `CalibrationStore` tests printed after the sync summary.

---

## Compatibility

- Loads after `webgazer.js` as a plain `<script>` tag
- No bundler, no transpiler, no npm install
- Works with any WebGazer version that exposes `setGazeListener` and `recordScreenPosition`
- Tested in Chrome, Edge, Firefox
- IndexedDB persistence degrades gracefully in private browsing (save/load return `false`/`null`)

---

## Connection to JelloOS

webgazer-aac is the eye tracking engine that powers hands-free access to [JelloOS](https://altru.dev) — a free, privacy-first PWA platform built for users who depend on assistive technology.

For users who cannot use a keyboard or pointer, webgazer-aac turns any device with a webcam into a full gaze-input interface: no specialist hardware, no subscription, no data leaving the device.

---

## License

GPLv3 — same as upstream [brownhci/WebGazer](https://github.com/brownhci/WebGazer).

Free to use, modify, and redistribute under the same terms. If you build something with it, we'd love to hear about it.

---

## Contributing

Issues and PRs welcome. If you work in AAC, rehab engineering, or special education and have real-world feedback on what would make this more useful — that's especially valuable. Open an issue or reach out via [altru.dev](https://altru.dev).
