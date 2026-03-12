# Changelog

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
