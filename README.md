# webgazer-aac

**Accessibility-first enhancement layer for [WebGazer.js](https://webgazer.cs.brown.edu/)**

By [ALTRU.dev](https://altru.dev) — Code for Humanity · GPLv3

---

WebGazer.js is the best browser-based eye tracker available, but its default ridge regression is linear — it can't model the non-linear distortion that makes gaze inaccurate at screen corners and edges. For AAC (augmentative and alternative communication) users who depend on gaze to communicate, this matters.

`webgazer-aac` is a drop-in patch layer that improves accuracy without replacing WebGazer entirely.

## What it improves

| Feature | Upstream WebGazer | webgazer-aac |
|---|---|---|
| Regression model | Linear ridge | Polynomial (degree 2) or RBF |
| Corner accuracy | Poor | Significantly improved |
| Smoothing | Fixed lerp 0.25 | Velocity-aware EMA |
| Calibration drift | Not addressed | Adaptive recalibration via dwell hits |
| Prediction confidence | Not exposed | 0–1 score per prediction |

## Installation

```html
<!-- 1. Load WebGazer as normal -->
<script src="https://unpkg.com/webgazer@2.1.0/dist/webgazer.js"></script>

<!-- 2. Load webgazer-aac immediately after -->
<script src="https://unpkg.com/webgazer-aac/webgazer-aac.js"></script>

<!-- 3. Install the patch -->
<script>
  webgazerAAC.install();
  // optionally enable continuous recalibration:
  webgazerAAC.enableAdaptiveRecalibration();
</script>
```

Or via npm:
```bash
npm install webgazer-aac
```

## Usage

### Basic

```js
// Install once after WebGazer loads
webgazerAAC.install();

// Start WebGazer as normal — nothing else changes
await webgazer
  .setGazeListener((data) => {
    if (!data) return;
    console.log(data.x, data.y, data.confidence); // confidence is new
  })
  .begin();
```

### Switch regression model

```js
webgazerAAC.setRegression('polynomial'); // default — best for most cases
webgazerAAC.setRegression('rbf');        // better for uneven calibration grids
webgazerAAC.setRegression('ridge');      // passthrough to WebGazer's original
```

### Adaptive recalibration

When your app knows the user successfully activated a UI element via gaze dwell, record it:

```js
// Enable background recalibration
webgazerAAC.enableAdaptiveRecalibration();

// In your dwell handler — pass the DOM element
function onDwellFired(element) {
  webgazerAAC.recordDwellHit(element);
  // ... rest of your dwell logic
}

// Or pass explicit coordinates
webgazerAAC.recordDwellHitXY(targetX, targetY);
```

Every confirmed dwell hit is a free labeled training sample. Over a session the model continuously self-corrects as the user's head position drifts.

### Confidence score

```js
webgazer.setGazeListener((data) => {
  if (!data) return;
  const { x, y, confidence } = data;
  // confidence: 1.0 = stable gaze, 0.0 = fast movement / just started
  // Use to delay dwell activation when confidence is low
  if (confidence > 0.6) checkDwell(x, y);
});
```

## Calibration tips for AAC users

1. **Light the face from the front.** A lamp behind the camera (not behind the user) is ideal.
2. **Stay 50–70 cm from camera.** Too close = barrel distortion. Too far = too few eye pixels.
3. **Use 9+ calibration points.** Click each point 3 times without moving your head.
4. **Enable adaptive recalibration** for long sessions — it compensates for gradual head drift.
5. **Use large dwell targets** — even with this patch, browser eye tracking has ~80–150px error radius. Design targets accordingly.

## Accuracy expectations

On typical hardware (1080p laptop webcam, good lighting, 9-point calibration):

| Model | Typical error radius |
|---|---|
| WebGazer ridge (original) | ~150–200px |
| webgazer-aac polynomial | ~90–130px |
| webgazer-aac RBF | ~80–120px |
| webgazer-aac + adaptive recalibration (10 min session) | ~60–100px |

These are estimates — results vary significantly with lighting, camera quality, and user.

## Architecture

`webgazer-aac.js` is a single self-contained file with no dependencies beyond WebGazer itself. It patches WebGazer by intercepting:

- `webgazer.setGazeListener()` — wraps the callback to run our regression + smoother
- `webgazer.recordScreenPosition()` — feeds calibration data into our model in parallel

The original WebGazer ridge regression continues to run underneath, providing a fallback until our model has enough calibration data (minimum ~6 points).

## License

GPLv3 — same as upstream WebGazer.

If your organisation's valuation is under $1,000,000, you may use this under LGPLv3 (same upstream exception). For other licensing, open an issue.

## Credits

Built on top of [WebGazer.js](https://github.com/brownhci/WebGazer) by Brown University (2016–2026).
Polynomial/RBF regression, velocity smoother, and adaptive recalibration by ALTRU.dev.

---

*Part of the [JelloOS](https://altru.dev) project — eye-tracking AAC for people who communicate with their gaze.*
