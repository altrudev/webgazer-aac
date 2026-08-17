# webgazer-aac v2 browser / AAC pre-hardware lab

This lab is the next gate after deterministic DDC assurance. It is intentionally local-first, evidence-only, and does not promote experimental gaze work into the canonical runtime.

## 1. Materialize and verify the canonical runtime

From an executable checkout of `agent/webgazer-aac-v2`:

```bash
npm test
```

`npm test` first runs `scripts/materialize-verified-candidate.mjs`. The materializer refuses promotion unless the packaged candidate reproduces the verified canonical runtime identity:

- bytes: `51666`
- SHA-256: `e2049bad33eb37a6bfeb375490455d78a33d8cb31bb16339fa2cabf02715f5e1`
- Git blob: `2b98d877627af794f507cf1d7911a456135f0d82`

The gate then runs the canonical invariant/integration/DDC/long-session suites plus the experimental capture-fusion and detector-isolation tests. Do not continue to webcam testing if materialization or any test fails.

## 2. Start the browser lab

```bash
npm run lab:browser
```

The server binds to `127.0.0.1:4818`. In a Codespace, use the forwarded HTTPS port URL. The server independently verifies the locally built WebGazer 3.5.3 artifact and MediaPipe asset tree before serving either one.

## 3. Start verified WebGazer first

1. Choose **Use verified Codespace WebGazer 3.5.3**.
2. Start the camera.
3. Wait until the lab session reads **RUNNING**.

The experimental refined detector is deliberately not loaded in the main page. Only after the WebGazer session is running does the lab load `capture-fusion.js`; that module starts a same-origin hidden frame containing the separate refined FaceMesh/Emscripten/WASM runtime. This prevents the second legacy MediaPipe runtime from sharing `Module` or virtual-filesystem globals with WebGazer.

The isolated frame copies each webcam frame into a canvas in its own JavaScript realm before FaceMesh inference. No raw camera frame, eye image, or landmark array is persisted.

## 4. Establish capture quality

The **Experimental Capture Fusion** panel reports:

- left/right eye reliability;
- bilateral agreement;
- glare/clipping and eye aperture effects;
- head-pose/centering/distance checks;
- video-frame interval and refined-inference latency;
- frames skipped while the refined detector is busy.

Select the validation cohort manually: `Unknown`, `No eyewear`, `Glasses`, or `Contacts`. This is evidence metadata only; the lab does not infer eyewear.

The positioning coach requires acceptable centering, distance, head pose, lighting symmetry, and per-eye reliability to remain stable for 1.5 seconds before dense calibration is enabled. **Start pursuit anyway** is an explicit experimental bypass and is recorded in evidence.

## 5. Preserve the canonical WebGazer/PCA calibration gate

The canonical AAC path still requires deliberate calibration and a successful PCA fit in the current session:

- click each of the 9 calibration points twice (18 deliberate samples);
- choose **Fit user basis**;
- only after the current-session fit may dwell/adaptation/drift be authorized by the assurance guard.

The experimental iris engine does not weaken or replace this authority boundary.

## 6. Run dense smooth-pursuit calibration

Once the positioning coach reads **READY**, choose **Start 18 s smooth-pursuit calibration** and follow the moving marker with your eyes.

The marker follows a bounded Lissajous path across the screen. Refined iris/head-pose measurements are paired to actual presented webcam frames using `requestVideoFrameCallback()` when available. Frames with inadequate combined reliability are rejected rather than forced into the model.

The experimental feature vector contains eye-local iris coordinates from both eyes, fused iris coordinates, head yaw/pitch/roll, eye aperture, distance scale, and bilateral agreement. One-Euro filtering is applied in feature space, not to final screen coordinates.

## 7. Model selection and evaluation

After dense calibration the lab fits and compares three experimental mappings:

- standardized linear ridge;
- standardized degree-2 polynomial ridge;
- standardized RBF kernel ridge regression.

The RBF model uses a bounded deterministic training set to keep browser fitting tractable. Model choice is evidence-driven: the lab selects the lowest held-out RMSE rather than assuming RBF is always superior.

Evidence includes:

- held-out RMSE / median / p95 error for each model;
- leave-region-out spatial error for the selected model;
- the untouched WebGazer screen prediction as a baseline;
- model kernel / regularization / RBF gamma diagnostics;
- capture and inference timing.

A separate **Shared isolated iris diagnostic** reuses the same detector for CENTER / LEFT / RIGHT / UP / DOWN measurements. It does not start another FaceMesh runtime.

## 8. AAC interaction checks

After the current-session PCA fit, use the AAC board with gaze dwell. Watch tracking quality, target confidence, drift RMSE, gaze FPS, eye-feature coverage, and experimental capture reliability.

Dwell completion is **evidence only**. The lab does not call `.click()` or authorize the selected AAC target to perform an external action. When experimental capture reliability is poor, the experimental predictor should withhold or lower confidence instead of treating a low-quality eye frame as valid evidence.

## 9. Export evidence

Use the three evidence exports as applicable:

- **Download evidence** — canonical browser/AAC session evidence;
- **Download fusion evidence** — experimental dense calibration, capture quality, timing, eyewear cohort and predictor comparison;
- **Download eye-motion evidence** — direct directional iris-motion diagnostic.

The exports intentionally exclude raw eye images, raw eye patches and raw landmark arrays.

## Release boundary

The refined capture / pursuit / kernel-ridge work is experimental. It must not replace or modify the verified `webgazer-aac.js` artifact merely because a browser run appears better. Promotion requires repeatable evidence across users and conditions, explicit regression tests, and the existing artifact-identity gate.

A green browser session is not clinical validation or a medical-device claim. Representative validation should cover multiple browsers, webcams, camera positions, bright/dim lighting, glasses, contacts, one-eye degradation, head movement, drift, target-size/density effects, false selections, corrections, and human factors.
