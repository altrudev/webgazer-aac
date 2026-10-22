`measurement-assurance.js` exports `measureFrame` through CommonJS and the browser
`AACMeasurementAssurance` global. It has no dependencies or retained frame state.

Pass `{landmarks, width, height, frameTimeMs, now}`. Landmarks must be an array of
478 MediaPipe normalized `{x, y, z?}` points. Width and height are camera image
pixel dimensions. Both times must use the same monotonic millisecond clock.
The caller must supply the actual source frame timestamp, not receipt time;
this function cannot detect replayed content with a newly assigned timestamp.

The `available` flag means both eyes passed geometry gates. `quality` contains
frame age and per-eye measurements: corner width, mean eyelid gap projected onto
the corner baseline normal, gap/width openness, and an all-four-rim circle fit
with pixel radius, RMS radial residual, and residual/radius. Left uses corners
33/133, lid pairs 160/144 and 158/153, rim 469–472; right uses corners 362/263,
lid pairs 385/380 and 387/373, rim 474–477. Coordinates are unmirrored source
landmark indices. Iris centers 468/473 are not used to fit the rims.

Optional gates default to `maxAgeMs: 180`, `minEyeWidthPx: 2`,
`minIrisRadiusPx: 1`, `maxRelativeResidual: 0.2`. These are engineering defaults,
not validated clinical thresholds. Closed eyelids produce zero openness, not
an inferred blink classification. Invalid input returns `available: false` and
an explicit `reason`; geometry rejections preserve available per-eye features.

Quality features are not calibrated confidence probabilities or screen gaze
accuracy estimates. Every result has `screenGaze: null` and
`qualifiedForDwell: false`. This module does not alter predictions, calibrate
screen coordinates, or authorize AAC activation.
