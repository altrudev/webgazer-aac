# Changelog

## [2.0.0] — 2026-08-16

### Correctness

- Reworked the WebGazer integration to consume `data.eyeFeatures` from the same WebGazer 3.5.3 prediction frame instead of re-invoking tracker patch extraction.
- Added fixed-size eye-patch normalization so variable FaceMesh crop sizes feed a stable feature pipeline.
- Fixed the PCA feature-space transition: calibration patches are retained in normalized raw form, the per-user PCA basis is fitted, and every regression sample is rebuilt in the fitted basis.
- Removed circular dwell evidence from the built-in path. Dwell completion now uses the selected target's anchor as low-weight supervision instead of feeding the predicted gaze coordinate back as truth.
- Drift residuals are evaluated before the newly confirmed label is added to the regression model.
- Kept polynomial and RBF training evidence synchronized so switching local regression modes does not activate a stale secondary model.
- `configureStore()` state now survives `install()`.

### AAC interaction

- Added `GazeTargetResolver` with actionable-ancestor canonicalization, expanded target geometry, candidate competition, target hysteresis, and short-lived DOM target caching.
- Added `targetConfidence`, separate from gaze tracking quality.
- Added `AdaptiveDwellController` and integrated it with `DwellTimer`.
- Added `DwellTimer.updateFromGaze()` for direct resolver + dwell operation.
- Added `recordConfirmedSelection()` as the evidence-safe adaptive feedback path.
- Retained `recordDwellHit()` and `recordDwellHitXY()` compatibility helpers; XY-only evidence is deliberately low-weight by default.

### Signal semantics

- Replaced the ambiguous primary `confidence` concept with explicit `gazeStability` and `trackingQuality` fields.
- Retained `confidence` as a deprecated compatibility alias of `trackingQuality`.
- Added explicit evidence provenance/weighting for calibration, explicit input, confirmed click, dwell selection, and inferred coordinates.

### Persistence

- Introduced independent `libraryVersion`, `schemaVersion`, and `featureVersion` fields.
- Moved default storage to a v2 object store.
- Feature-space incompatibility is surfaced explicitly rather than silently applied.
- Material viewport changes mark a snapshot `_validationRequired`; such snapshots are returned without automatic activation.

### Diagnostics

- Added `getDiagnostics()` for frame cadence, eye-feature coverage, fallbacks, blink/saccade frames, evidence counts, calibration state, tracking quality, and drift state.
- Added optional `attachVideoClock(video)` support using `requestVideoFrameCallback()` when available.

### Tests and repository cleanup

- Replaced inconsistent 1.x test files with one canonical zero-dependency v2 invariant suite.
- Added `package.json` with `npm test`.
- Added architecture and validation-boundary documentation.
- Removed stale duplicate README/test artifacts.

### Compatibility

- Targeted to WebGazer 3.5.3, the final planned upstream release.
- v1.x calibration snapshots are not feature-compatible with v2 and require recalibration.

---

## 1.x history

### [1.3.0] — 2026-03-12

Added confidence-gated dwell and IndexedDB calibration persistence.

### [1.2.0] — 2026-03-12

Added drift watchdog and residual events.

### [1.1.0] — 2026-03-12

Added per-user PCA, contrast normalization, Kalman smoothing, blink detection, saccade suppression, ensemble regression, and frame caching.

### [1.0.0] — 2026-03-12

Initial accessibility-focused WebGazer enhancement release.
