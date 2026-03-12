# Changelog

## [1.3.0] — 2026-03-12

### Added

**Confidence-gated dwell timer** (`DwellTimer` class)
- Tracks gaze fixation on DOM elements; progress advances only when Kalman confidence meets the threshold
- Progress **freezes** (does not reset) during low confidence, blinks, and saccades — resumes when gaze stabilises
- Progress resets to 0 only when gaze leaves the element entirely
- Fires `webgazer-aac:dwell-progress`, `webgazer-aac:dwell-complete`, and `webgazer-aac:dwell-cancel` CustomEvents on the target element
- On completion, automatically calls `recordDwellHitXY()` so the drift watchdog and adaptive recalibrator both receive the signal
- Configurable: `dwellMs`, `minConfidence`, `holdAfterMs`
- Create via `webgazerAAC.createDwellTimer(options)` — wired to the instance automatically

**IndexedDB calibration persistence** (`CalibrationStore` class)
- Saves and restores full calibration state across sessions: regression datasets, fitted PCA bases, Kalman noise params, screen dimensions
- Async API: `saveCalibration()`, `loadCalibration()`, `clearCalibration()`, `isStorageAvailable()`
- Version-checked on load — stale snapshots (wrong `aacVersion`) return `null` rather than silently corrupting state
- Screen-size mismatch surfaced as `_screenMismatch: true` on the snapshot; caller decides whether to trust a cross-resolution calibration
- Graceful fallback in private browsing (IDB unavailable) — `save` returns `false`, `load` returns `null`
- `_MemoryBackend` class exposed for zero-IDB unit testing
- Multi-profile support via `profileKey` option

### API additions

```js
// Dwell timer
const timer = webgazerAAC.createDwellTimer({ dwellMs, minConfidence, holdAfterMs });
timer.update(element, x, y, confidence, isSaccade, isBlink?)  // → progress 0–1
timer.reset()

// Calibration persistence
await webgazerAAC.saveCalibration(options?)
await webgazerAAC.loadCalibration(options?)
await webgazerAAC.clearCalibration(options?)
await webgazerAAC.isStorageAvailable()
webgazerAAC.getCalibrationSnapshot()          // → serialisable plain object
webgazerAAC.applyCalibrationSnapshot(snap)    // → boolean
webgazerAAC.configureStore({ dbName, storeName, profileKey })
```

Events fired on the dwell target element:
```js
'webgazer-aac:dwell-progress'   // { progress, confidence, x, y, frozen }
'webgazer-aac:dwell-complete'   // { x, y }
'webgazer-aac:dwell-cancel'     // { reason, x, y }
```

### Upgrade from v1.2.0

```html
<!-- Was -->
<script src="webgazer-aac.js"></script>
<script>
  webgazerAAC.install().enableAdaptiveRecalibration().enableDriftWatchdog();
</script>

<!-- Now — add persistence and dwell timer -->
<script src="webgazer-aac.js"></script>
<script>
  webgazerAAC.install().enableAdaptiveRecalibration().enableDriftWatchdog();

  // Restore previous calibration on load
  webgazerAAC.loadCalibration().then(saved => {
    if (!saved) {
      // run calibration UI, then:
      webgazerAAC.fitUserBasis();
      webgazerAAC.saveCalibration();
    }
    webgazer.begin();
  });

  // Replace manual dwell logic with:
  const timer = webgazerAAC.createDwellTimer({ dwellMs: 800 });
  webgazer.setGazeListener((data) => {
    if (!data) return;
    const el = document.elementFromPoint(data.x, data.y);
    timer.update(el, data.x, data.y, data.confidence, data.isSaccade);
  });
</script>
```

Everything else is backwards-compatible.

---

## [1.2.0] — 2026-03-12

### Added

**Drift watchdog** (`DriftWatchdog` class)
- Detects when the regression model silently degrades mid-session using an independent ground-truth residual signal
- Every confirmed gaze position (calibration click, dwell hit, `recordScreenPosition`) is compared against the current prediction; residuals are tracked as an exponential-decay-weighted RMSE
- Two configurable thresholds: `warnThreshold` (default 120px) and `critThreshold` (default 220px)
- Hysteretic: won't re-fire the same level until RMSE cools below 80% of the threshold — prevents event storms
- `minSamples` gate (default 8) prevents false alerts during model warm-up
- Fires `webgazer-aac:drift-warning` and `webgazer-aac:drift-critical` CustomEvents on `document`
- Optional `onWarn` / `onCritical` callbacks in addition to CustomEvents
- Distinct from ensemble error tracking (`_errPoly`/`_errRbf`), which only compares models against each other — the watchdog uses independent ground-truth and catches both models drifting together

### API additions

```js
webgazerAAC.enableDriftWatchdog(options?)   // { warnThreshold, critThreshold, minSamples, onWarn, onCritical }
webgazerAAC.disableDriftWatchdog()
webgazerAAC.resetDriftWatchdog()            // call after re-calibration
webgazerAAC.getDriftRmse()                  // → current weighted RMSE in px
```

Events fired on `document`:
```js
'webgazer-aac:drift-warning'    // { rmse, level, sampleCount, timestamp }
'webgazer-aac:drift-critical'   // { rmse, level, sampleCount, timestamp }
```

### Wiring

`recordScreenPosition` and `recordDwellHitXY` both now feed the drift watchdog automatically when it is enabled. No changes required in application code.

### Upgrade from v1.1.0

```html
<!-- Was -->
<script src="webgazer-aac.js"></script>
<script>webgazerAAC.install().enableAdaptiveRecalibration();</script>

<!-- Now — optionally add drift watchdog -->
<script src="webgazer-aac.js"></script>
<script>
  webgazerAAC.install().enableAdaptiveRecalibration().enableDriftWatchdog();

  document.addEventListener('webgazer-aac:drift-warning', e => {
    console.warn('Gaze drift detected — RMSE:', e.detail.rmse, 'px');
  });
</script>
```

Everything else is backwards-compatible.

---

## [1.1.0] — 2026-03-12

### Added

**Per-user PCA basis** (`PCABasis` class)
- Eye patch features are now projected onto a basis fitted from the user's own calibration patches, rather than a fixed random projection
- Call `webgazerAAC.fitUserBasis()` at the end of your calibration sequence
- Uses power iteration in sample space (n×n, not dim×dim) so it runs in ~5–20ms even on slow devices
- Falls back gracefully to the random projection if insufficient patches were collected

**CLAHE contrast normalisation**
- Adaptive histogram equalisation applied to every eye patch before feature extraction
- Dramatically improves feature quality in poor or uneven lighting without blowing out highlights
- No new dependencies — pure JS canvas math

**Kalman filter** (replaces EMA velocity smoother)
- 4-state filter: [x, y, vx, vy] — tracks position and velocity simultaneously
- Separates process noise (real head movement) from measurement noise (regression jitter)
- Measurement update is skipped during blinks and saccades, allowing the filter to coast on its own prediction
- Tunable via `webgazerAAC.setKalmanParams(processNoise, measurementNoise)`
- Exposes `vx`, `vy` velocity components in every prediction

**Blink detection** (`BlinkDetector` class)
- Measures mean eye-patch brightness per frame; triggers when brightness drops below an adaptive threshold
- Outputs `null` during blinks so dwell timers don't advance falsely
- Includes a configurable post-blink lockout (default 80ms) to absorb reopening jitter

**Saccade suppression** (`SaccadeDetector` class)
- Uses Kalman velocity estimate to detect fast eye movements (default threshold: 600 px/sec)
- Suppresses Kalman measurement update during saccades; prediction coasts instead
- Passes `isSaccade: true` flag in prediction for downstream gating

**Ensemble regression** (`EnsembleRegression` class)
- Blends polynomial and RBF predictions, weighted inversely by each model's rolling RMSE
- Whichever model has lower recent error gets more weight — adapts over the session
- Falls back to whichever model is ready if only one has sufficient calibration data
- Now the default regression mode (was `polynomial` in v1.0.0)

**Frame cache** (`FrameCache` class)
- Skips regression inference when gaze velocity is below threshold (default: 15 px/sec)
- Reuses the last prediction for up to 50ms during stable fixation
- Saves 20–40% CPU on typical hardware; more on mobile/low-power devices

### Changed

- Default regression mode: `polynomial` → `ensemble`
- `PATCH_COMPONENTS` increased from 8 to 10 per eye
- `VelocitySmoother` replaced by `KalmanFilter` — same API surface via `smooth(x, y, isBlink, isSaccade)`
- `recordScreenPosition` intercept now also calls `EnsembleRegression.trackError()` to keep error weights current
- `AdaptiveRecalibrator.maxHitsPerSession` increased from 200 → 500

### API additions

```js
webgazerAAC.fitUserBasis()               // fit PCA after calibration
webgazerAAC.resetCalibrationPatches()    // clear stored patches before re-calibration
webgazerAAC.setKalmanParams(Q, R)        // tune Kalman noise parameters
webgazerAAC.isPCAFitted()                // → boolean
webgazerAAC.setRegression('ensemble')   // new default mode
```

Prediction objects now include:
```js
{ x, y, confidence, isSaccade }   // isBlink frames return null instead
```

### Upgrade from v1.0.0

```html
<!-- Was -->
<script src="webgazer-aac.js"></script>
<script>webgazerAAC.install().enableAdaptiveRecalibration();</script>

<!-- Now — add fitUserBasis() at calibration end -->
<script src="webgazer-aac.js"></script>
<script>
  webgazerAAC.install().enableAdaptiveRecalibration();
  // ... after user completes calibration:
  webgazerAAC.fitUserBasis();
</script>
```

Everything else is backwards-compatible. Existing `setRegression('polynomial')` / `'rbf'` / `'ridge'` calls still work.

---

## [1.0.0] — 2026-03-12

Initial release. Polynomial regression, RBF regression, velocity-aware EMA smoother, adaptive recalibration via dwell hits, prediction confidence score.
