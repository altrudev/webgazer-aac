'use strict';

let fakeNow = 0;
global.performance = { now: () => fakeNow };
global.window = global;
global.innerWidth = 1280;
global.innerHeight = 720;
global.devicePixelRatio = 1;
global.CustomEvent = class CustomEvent { constructor(type, init) { this.type = type; this.detail = init && init.detail; } };
global.document = { dispatchEvent() { return true; }, querySelectorAll() { return []; } };
global.ImageData = class ImageData { constructor(data, width, height) { this.data = data; this.width = width; this.height = height; } };
global.HTMLCanvasElement = class HTMLCanvasElement {};

const fakeWebgazer = {
  _callback: null,
  setGazeListener(callback) { this._callback = callback; return this; },
  recordScreenPosition() { return this; },
  setRegression() { return this; },
  clearGazeListener() { this._callback = null; return this; },
};
global.webgazer = fakeWebgazer;

const aac = require('../webgazer-aac.js');
let passed = 0;
let failed = 0;
function assert(condition, name) {
  if (condition) { console.log('✓', name); passed++; }
  else { console.error('✗', name); failed++; }
}

console.log('\nwebgazer-aac bounded long-session simulation');

aac.install();
let delivered = 0;
fakeWebgazer.setGazeListener(data => { if (data) delivered++; });
for (let i = 0; i < 20000; i++) {
  fakeNow += 16.67;
  fakeWebgazer._callback({ x: (i * 17) % 1280, y: (i * 11) % 720, eyeFeatures: null }, fakeNow);
}
const d = aac.getDiagnostics();
assert(delivered === 20000, '20,000 fallback gaze frames complete without queue growth or exception');
assert((d.gazeFrames || d.frames) >= 20000, 'diagnostics retain scalar long-session frame accounting');

const normalized = { left: new Float32Array(960), right: new Float32Array(960) };
for (let i = 0; i < 520; i++) {
  normalized.left[i % 960] = (i % 13) / 13;
  normalized.right[i % 960] = (i % 17) / 17;
  aac.recordGroundTruth(i % 1280, (i * 3) % 720, {
    normalized,
    source: 'calibration',
    evidenceWeight: 1,
    includeInCalibration: true,
  });
}
assert(aac._trainingRecords.length <= 500, 'long-session raw-normalized training retention is capped at 500');
assert(aac._regressions.polynomial.samples.length <= 220, 'long-session polynomial retention is capped at 220');
assert(aac._regressions.rbf.features.length <= 120, 'long-session RBF retention is capped at 120');
assert(aac._calibrationRaw.length <= 300, 'long-session calibration retention is capped at 300');
assert(aac._evidenceLineage.length <= 500, 'long-session evidence lineage is capped at 500');

const before = fakeWebgazer.setGazeListener;
aac.uninstall();
assert(aac._installed === false && fakeWebgazer.setGazeListener !== before, 'long-session teardown restores interception boundary');
assert(aac._lastNormalized === null && aac._lastEyeFeatures === null, 'long-session teardown drops transient eye evidence');

const report = aac.getAssuranceSnapshot();
assert(report.ok === true, 'post-session DDC assurance invariants remain green');

console.log(`\n${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
