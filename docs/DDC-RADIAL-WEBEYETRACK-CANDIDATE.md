# DDC Radial — WebEyeTrack replacement spike

## Current incident
Camera ~30 FPS yet predicted gaze frequently moves in the opposite direction or does not follow eye movements. Do not silently flip axes or treat WebGazer confidence as measured accuracy. Live AAC activation remains disabled.

## Candidate selection
**Primary candidate:** RedForestAI/WebEyeTrack, upstream 75fbd2f (2026-08-31), JS npm package `webeyetrack` v0.0.2 in checked repository. MIT license (retain copyright and notices). Repository: https://github.com/RedForestAI/WebEyeTrack; paper: https://arxiv.org/abs/2508.19544.

Source inspection confirms browser worker support, TF.js BlazeGaze weights (must be explicitly hosted), MediaPipe Face Landmarker features, head pose, normalized gaze coordinates and on-device few-shot personalization. Dependencies include @mediapipe/tasks-vision, @tensorflow/tfjs, mathjs, ml-matrix and worker-loader. This is a researched candidate **not yet installed in the AAC lab**.

## Coordinate contract (must be tested)
WebEyeTrack `normPog` is nominally in [-0.5,0.5] on each axis, with (0,0) centered, X positive RIGHT, Y positive DOWN. For CSS viewport W × H, expected mapping is `x=(normX+0.5)*W; y=(normY+0.5)*H`; no mirrored-video visual presentation may change this mapping implicitly. Clip only for visualization; do not silently convert invalid/out-of-range predictions into valid gaze. Verify contract independently with a synthetic dataset before UI wiring.

## Isolated comparison plan
1. Vendor pin: package version and upstream hash, transitive dependency manifests, license and model-weight licenses. Check original pre-trained model weights are redistributable before bundling.
2. Build WebEyeTrack in an **isolated browser test page** on a local/VPS hosted origin; do not replace the working Altru.dev lab yet.
3. One webcam stream, local only, strict explicit user permission. Record only timestamps, model availability and summary errors, no frames, coordinates or iris template export.
4. Test true gaze movement left/right/up/down with mouse parked and head stationary; then repeat with controlled head translation/rotation.
5. Display independent WebEyeTrack point-of-gaze next to existing WebGazer and eye-geometry indicators. Never average predictions without confidence and empirical holdout evidence.
6. Nine-point whole-viewport gaze calibration from camera-frame features, not from gaze callback; use withheld validation targets, per-target medians/P90, accuracy by region, frame latency and invalid detection.
7. Enable dwell display only with verified stable predictions and fail-closed threshold. Keep keyboard/touch/switch AAC interaction available.
8. Use measured results on the user's actual Logitech webcam to select the engine; benchmark CPU usage and browser responsiveness.

## Risks
- Paper metrics do not guarantee 720p Logitech webcam results.
- Model weight origin and license require explicit inspection before public deployment.
- Webcam preview may be mirrored while screen gaze coordinate system is not.
- The current separate MediaPipe geometry branch may lack iris landmarks and cannot itself calculate gaze.
- WebEyeTrack 0.0.2 is early-stage: validate worker bootstrapping, model assets, browser compatibility, API changes and cleanup before selection.

## Acceptance gates
G1 pinned candidate + model license audit PASS. G2 browser worker/model load PASS. G3 no-mouse directional tests and calibration holdout PASS. G4 median/P90, drift and head-motion performance documented. G5 no implicit AAC selection, local-only data handling and fallback modality verified.

No gate above has been claimed to pass from repository inspection alone.
