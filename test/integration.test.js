'use strict';

let fakeNow = 1000;
global.performance = { now: () => fakeNow };
global.window = global;
global.innerWidth = 1000;
global.innerHeight = 600;
global.devicePixelRatio = 1;
global.CustomEvent = class CustomEvent {
  constructor(type, init) { this.type = type; this.detail = init && init.detail; }
};
global.document = {
  events: [],
  dispatchEvent(e) { this.events.push(e); return true; },
  querySelectorAll() { return []; },
};
global.ImageData = class ImageData {
  constructor(data, width, height) { this.data = data; this.width = width; this.height = height; }
};
global.HTMLCanvasElement = class HTMLCanvasElement {};

function makeEyeFeatures(seed = 1) {
  const makePatch = s => {
    const data = new Uint8ClampedArray(20 * 10 * 4);
    for (let i = 0; i < data.length; i += 4) {
      const value = (s * 31 + i) % 200 + 30;
      data[i] = value;
      data[i + 1] = value;
      data[i + 2] = value;
      data[i + 3] = 255;
    }
    return new ImageData(data, 20, 10);
  };
  return { left: { patch: makePatch(seed) }, right: { patch: makePatch(seed + 1) } };
}

const fakeWebgazer = {
  _callback: null,
  mode: null,
  setGazeListener(callback) { this._callback = callback; return this; },
  recordScreenPosition() { return this; },
  setRegression(mode) { this.mode = mode; return this; },
};
global.webgazer = fakeWebgazer;

const aac = require('../webgazer-aac.js');
let passed = 0;
let failed = 0;
function assert(condition, name) {
  if (condition) { console.log('✓', name); passed++; }
  else { console.error('✗', name); failed++; }
}

console.log('\nwebgazer-aac v2 integration tests');

aac.install();
let output = null;
fakeWebgazer.setGazeListener(data => { output = data; });
fakeNow += 16;
fakeWebgazer._callback({ x: 250, y: 180, eyeFeatures: null }, 16);
assert(output && output.eyeFeaturesAvailable === false, 'fallback frame remains usable without eyeFeatures');
assert(aac.getDiagnostics().fallbackFrames === 1, 'fallback diagnostics increments');

aac.setRegression('ridge');
assert(aac.getRegressionMode() === 'ridge' && fakeWebgazer.mode === 'ridge', 'ridge mode delegates upstream');
aac.setRegression('ensemble');
assert(aac.getRegressionMode() === 'ensemble', 'ensemble mode restores AAC regressor');

aac.resetCalibrationPatches();
for (let i = 0; i < 14; i++) {
  const normalized = aac._normalizeEyeFeatures(makeEyeFeatures(i + 10));
  aac.recordGroundTruth(50 + (i % 4) * 250, 50 + Math.floor(i / 4) * 150, {
    normalized,
    source: 'calibration',
    includeInCalibration: true,
    evidenceWeight: 1,
  });
}
aac.fitUserBasis();
aac.enableDriftWatchdog({ minEvidence: 1, warnThreshold: 10, critThreshold: 20 });
const normalized = aac._normalizeEyeFeatures(makeEyeFeatures(10));
const before = aac._regressions.polynomial.getData().length;
aac.recordGroundTruth(999, 599, { normalized, source: 'explicit', evidenceWeight: 1 });
assert(aac._regressions.polynomial.getData().length === before + 1, 'ground truth trains after evaluation');
assert(aac.getDriftRmse() > 0, 'drift watchdog evaluates pre-training model');

(async () => {
  const backend = new aac._MemoryBackend();
  const store = new aac.CalibrationStore({ backend });
  const snapshot = aac.getCalibrationSnapshot();
  snapshot.viewport = { width: 700, height: 400, dpr: 1 };
  await store.save(snapshot);
  const loaded = await store.load();
  assert(loaded._validationRequired === true, 'material viewport change requires validation');

  let callbackCount = 0;
  let cancelled = false;
  const video = {
    requestVideoFrameCallback(callback) { this.callback = callback; return ++callbackCount; },
    cancelVideoFrameCallback() { cancelled = true; },
  };
  assert(aac.attachVideoClock(video) === true, 'video clock attaches');
  video.callback(0, { mediaTime: 1, presentedFrames: 1 });
  assert(aac.getDiagnostics().videoFrames === 1, 'video clock records frame');
  aac.detachVideoClock();
  assert(cancelled, 'video clock cancels callback');

  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed) process.exit(1);
})().catch(error => {
  console.error(error);
  process.exit(1);
});
