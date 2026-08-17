# Validation boundary and test plan

## Current status

webgazer-aac v2 is engineering software for assistive interaction. The repository has deterministic code-level tests, but it has **not** been clinically validated and is not represented as a medical device.

Passing the automated suite means the implemented invariants hold under the supplied test fixtures. It does not establish real-world AAC effectiveness.

## Automated v2 invariants

The current suite verifies:

1. variable-sized eye crops normalize to one fixed feature dimension;
2. the WebGazer listener consumes emitted `data.eyeFeatures`;
3. gaze output distinguishes `trackingQuality` from the legacy confidence alias;
4. per-user PCA fitting rebuilds all training samples in the new feature basis;
5. calibration labels survive that rebuild;
6. intercepted `recordScreenPosition()` preserves upstream behavior;
7. explicit/confirmed evidence is accounted for separately;
8. nested DOM content canonicalizes to an actionable gaze target;
9. geometric target resolution chooses the expected AAC control;
10. dwell adaptation uses the selected target anchor, not the gaze prediction, as supervision;
11. adaptive dwell becomes more conservative under poorer quality/ambiguity;
12. calibration persistence separates schema and feature versions;
13. incompatible feature spaces are rejected explicitly;
14. preconfigured stores survive idempotent install;
15. diagnostics expose quality and drift as separate signals.

The integration audit additionally checks fallback operation without eye features, regression-mode switching, pre-training drift evaluation, viewport-change validation, and `requestVideoFrameCallback()` lifecycle behavior.

Run:

```bash
npm test
```

No third-party test framework is required.

### Current automated result — 2026-08-16

The exact v2 runtime committed on `agent/webgazer-aac-v2` as Git blob `471e37e4dcd0dff4c7fa20f1e81e8aac4b29463c` was exercised with both test groups:

- invariant suite: **22 passed, 0 failed**;
- integration audit: **10 passed, 0 failed**;
- combined: **32 passed, 0 failed**.

This is code-level evidence only. Browser/webcam and human interaction validation remain open release gates below.

## Browser validation still required

Before calling v2 production-ready for a specific AAC deployment, test at minimum:

- Chrome / Chromium;
- Edge;
- Firefox;
- Safari where WebGazer itself is supported by the chosen deployment;
- laptop and desktop webcams at multiple resolutions;
- front-facing mobile/tablet cameras where the host interface supports them;
- low light, side light, glasses, head movement, partial face loss, and re-entry;
- viewport resize, zoom, orientation change, and display scaling;
- multi-target AAC layouts with different target sizes and spacing.

## Functional AAC metrics

Do not evaluate only pixel error. Record both estimator and interaction outcomes.

Recommended measures:

### Gaze-estimator measures

- median and 90th-percentile point error on held-out validation targets;
- drift over session time;
- eye-feature availability;
- effective gaze-listener FPS;
- blink/face-loss interruption time;
- post-blink recovery time;
- target-confidence calibration.

### Interaction measures

- intended-target selection rate;
- false-selection rate;
- correction/undo rate;
- median time to selection;
- dwell cancellation rate;
- target-switch oscillation rate;
- selections per minute for representative boards;
- performance by target size and target density.

### Human factors

With appropriate consent and study design:

- perceived fatigue;
- comfort;
- frustration;
- calibration burden;
- preference versus baseline access method;
- ability to recover independently from drift or tracking loss.

## Comparative benchmark

A useful engineering benchmark should compare at least:

1. upstream WebGazer 3.5.3 baseline;
2. webgazer-aac v2 with fixed dwell;
3. webgazer-aac v2 with target resolver + adaptive dwell.

Use the same participant/session/order controls and retain the raw measurement definitions. Avoid claiming a percentage improvement until the benchmark protocol and data are published.

## Frame-clock diagnostics

Listener callback timing is not identical to camera-frame timing. When supported, attach the actual video element:

```js
webgazerAAC.attachVideoClock(videoElement);
```

This lets diagnostics retain `requestVideoFrameCallback()` metadata alongside gaze-listener cadence. Treat any latency number according to the clock that produced it; do not label output-callback timing as camera-to-prediction latency without evidence.

## Release gate

A v2 release should not be promoted beyond experimental/beta status until:

- automated invariants pass;
- browser integration is exercised with WebGazer 3.5.3;
- no unhandled errors occur across calibration/save/load/recalibration flows;
- target resolver behavior is tested on a representative AAC board;
- viewport-change validation behavior is confirmed;
- performance and interaction metrics are collected on real hardware.
