# webgazer-aac v2 browser / AAC pre-hardware lab

This lab is the next gate after deterministic DDC assurance. It is intentionally local-first and evidence-only.

## 1. Materialize and verify the tested runtime

From an executable checkout of `agent/webgazer-aac-v2`:

```bash
npm test
```

`npm test` first runs `scripts/materialize-verified-candidate.mjs`. The materializer decompresses the repository-packaged candidate and refuses to write `webgazer-aac.js` unless all three identities match:

- bytes: `51666`
- SHA-256: `e2049bad33eb37a6bfeb375490455d78a33d8cb31bb16339fa2cabf02715f5e1`
- Git blob: `2b98d877627af794f507cf1d7911a456135f0d82`

It then runs all four pre-hardware suites. Expected result: **80 passed, 0 failed**.

Do not continue to webcam testing if materialization or any test fails.

## 2. Start the local lab

```bash
npm run lab:browser
```

Open:

```text
http://127.0.0.1:4818/demo/prehardware/
```

Localhost is used so browser camera permission can be tested without exposing the lab to the LAN.

## 3. Load WebGazer

Use **Load local WebGazer 3.5.3** and select a trusted local copy of the WebGazer 3.5.3 JavaScript file. The lab does not fetch WebGazer from a CDN.

The lab records whether the loaded runtime reports version `3.5.3`. A different version is evidence of a compatibility change and should be treated separately.

## 4. Run calibration and AAC interaction checks

- Start the camera.
- Look at each calibration target, then click it manually.
- After all points are recorded, choose **Fit user basis**.
- Use the AAC board with gaze dwell.
- Watch tracking quality, target confidence, drift RMSE, gaze FPS, and eye-feature coverage.

Dwell completion is **evidence only**. The lab does not call `.click()` or authorize the selected AAC target to perform an external action.

## 5. Export evidence

Choose **Download evidence** to save a JSON record containing:

- runtime and WebGazer versions;
- diagnostics;
- DDC assurance snapshot;
- bounded evidence lineage;
- session events;
- an explicit statement that dwell completion did not execute target action.

The export intentionally excludes raw eye images and raw eye patches.

## Release boundary

A green browser session is still not clinical validation or a medical-device claim. Real-world validation should cover multiple browsers, webcams, lighting conditions, glasses/head movement, representative AAC layouts, drift recovery, target-size/density effects, false selections, corrections, and human factors.
