'use strict';

let fakeNow = 1000;
global.performance = { now: () => fakeNow };
global.window = global;
global.innerWidth = 1280;
global.innerHeight = 720;
global.devicePixelRatio = 2;
global.CustomEvent = class CustomEvent {
  constructor(type, init) { this.type = type; this.detail = init && init.detail; this.bubbles = !!(init && init.bubbles); }
};
global.ImageData = class ImageData {
  constructor(data, width, height) { this.data = data; this.width = width; this.height = height; }
};
global.HTMLCanvasElement = class HTMLCanvasElement {};
global.document = {
  events: [],
  dispatchEvent(e) { this.events.push(e); return true; },
  querySelectorAll() { return []; },
};

function makePatch(seed, w = 22, h = 12, brightness = 120) {
  const data = new Uint8ClampedArray(w * h * 4);
  let s = seed >>> 0;
  for (let i = 0; i < data.length; i += 4) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const jitter = (s % 80) - 40;
    const v = Math.max(0, Math.min(255, brightness + jitter));
    data[i] = v; data[i + 1] = Math.max(0, v - 8); data[i + 2] = Math.max(0, v - 15); data[i + 3] = 255;
  }
  return new ImageData(data, w, h);
}

function makeEyeFeatures(seed, brightness) {
  return {
    left: { patch: makePatch(seed, 21 + (seed % 4), 11 + (seed % 3), brightness) },
    right: { patch: makePatch(seed + 999, 23 + (seed % 3), 12 + (seed % 2), brightness) },
  };
}

const fakeWebgazer = {
  version: '3.5.3',
  _callback: null,
  _recorded: [],
  setGazeListener(cb) { this._callback = cb; return this; },
  recordScreenPosition(x, y, eventType) { this._recorded.push({ x, y, eventType }); return this; },
  setRegression() { return this; },
};
global.webgazer = fakeWebgazer;

const aac = require('../webgazer-aac.js');
let passed = 0, failed = 0;
function assert(cond, name) {
  if (cond) { console.log('✓', name); passed++; }
  else { console.error('✗', name); failed++; }
}
function approx(a, b, eps = 1e-6) { return Math.abs(a - b) <= eps; }

console.log('\nwebgazer-aac v2.0.0 tests');

{
  const n1 = aac._normalizeEyeFeatures(makeEyeFeatures(1));
  const n2 = aac._normalizeEyeFeatures(makeEyeFeatures(2));
  assert(n1 && n1.left.length === 40 * 24 && n1.right.length === 40 * 24, 'normalizes variable eye patches to fixed feature dimensions');
  assert(n2 && n2.left.length === n1.left.length, 'different crop dimensions share the same normalized shape');
}

{
  aac.install();
  let received = null;
  fakeWebgazer.setGazeListener(data => { received = data; });
  fakeNow += 16;
  fakeWebgazer._callback({ x: 300, y: 200, eyeFeatures: makeEyeFeatures(10) }, 16);
  assert(aac._lastEyeFeatures && aac._lastEyeFeatures.left, 'listener captures WebGazer data.eyeFeatures directly');
  assert(received && received.eyeFeaturesAvailable === true, 'gaze output reports eye feature availability');
  assert(received && typeof received.trackingQuality === 'number', 'gaze output exposes trackingQuality');
  assert(received && received.confidence === received.trackingQuality, 'legacy confidence remains a compatibility alias');
}

{
  aac.resetCalibrationPatches();
  for (let i = 0; i < 16; i++) {
    const normalized = aac._normalizeEyeFeatures(makeEyeFeatures(100 + i, 95 + i * 4));
    aac.recordGroundTruth(80 + (i % 4) * 340, 70 + Math.floor(i / 4) * 180, {
      normalized,
      source: 'calibration',
      evidenceWeight: 1,
      includeInCalibration: true,
    });
  }
  const before = aac._regressions.polynomial.getData().map(x => x.features[1]);
  const fit = aac.fitUserBasis();
  const afterData = aac._regressions.polynomial.getData();
  const after = afterData.map(x => x.features[1]);
  const changed = after.some((x, i) => Math.abs(x - before[i]) > 1e-6);
  assert(fit.left && fit.right && fit.rebuilt, 'fits per-user PCA and reports a training rebuild');
  assert(afterData.length === 16, 'PCA rebuild preserves all calibration labels');
  assert(changed, 'training feature vectors are regenerated in the fitted PCA basis');
}

{
  const countBefore = fakeWebgazer._recorded.length;
  aac._lastNormalized = aac._normalizeEyeFeatures(makeEyeFeatures(250));
  fakeWebgazer.recordScreenPosition(500, 300, 'click');
  assert(fakeWebgazer._recorded.length === countBefore + 1, 'recordScreenPosition preserves upstream behavior');
  assert(aac._diagnostics.explicitSamples > 0, 'explicit/confirmed samples are counted separately');
}

function fakeElement(name, rect) {
  return {
    name,
    disabled: false,
    dataset: {},
    _events: [],
    getAttribute() { return null; },
    getBoundingClientRect() { return Object.assign({}, rect); },
    dispatchEvent(e) { this._events.push(e); return true; },
    closest() { return this; },
  };
}

{
  const parent = fakeElement('YES', { left: 100, top: 100, right: 300, bottom: 220, width: 200, height: 120 });
  const child = { closest() { return parent; } };
  const resolver = new aac.GazeTargetResolver();
  const r1 = resolver.resolveElement(child, 170, 150);
  assert(r1 && r1.element === parent, 'nested DOM content resolves to a canonical gaze target');

  const no = fakeElement('NO', { left: 600, top: 100, right: 800, bottom: 220, width: 200, height: 120 });
  const root = { querySelectorAll() { return [parent, no]; } };
  const r2 = resolver.resolve(180, 150, root);
  assert(r2 && r2.element === parent && r2.confidence > 0.5, 'resolver selects the geometrically likely AAC target');
}

{
  const target = fakeElement('HELLO', { left: 100, top: 80, right: 300, bottom: 180, width: 200, height: 100 });
  aac._lastNormalized = aac._normalizeEyeFeatures(makeEyeFeatures(301));
  let captured = null;
  const oldRecal = aac._recalibrator;
  aac._recalibrator = {
    recordNormalized(_n, x, y, w) { captured = { x, y, w }; return true; },
    enable() {}, disable() {}, regression: aac._regression,
  };
  const timer = aac.createDwellTimer({ dwellMs: 100, holdAfterMs: 0, adaptive: false, minTrackingQuality: 0.1, minTargetConfidence: 0.1 });
  timer.update(target, 1000, 650, 0.95, false, false, 0.95);
  fakeNow += 120;
  timer.update(target, 1000, 650, 0.95, false, false, 0.95);
  assert(captured && approx(captured.x, 200) && approx(captured.y, 130), 'dwell feedback labels the selected target centre rather than gaze x/y');
  assert(captured && (!approx(captured.x, 1000) || !approx(captured.y, 650)), 'predicted gaze is not recycled as independent ground truth');
  aac._recalibrator = oldRecal;
}

{
  const ctl = new aac.AdaptiveDwellController({ baseMs: 800, minMs: 400, maxMs: 1400 });
  const good = ctl.getDwellMs({ trackingQuality: 0.95, targetConfidence: 0.95, targetArea: 12000 });
  const poor = ctl.getDwellMs({ trackingQuality: 0.3, targetConfidence: 0.35, targetArea: 1200, crowding: 0.8 });
  assert(poor > good, 'adaptive dwell increases dwell time for uncertain/crowded targets');
}

(async () => {
  {
    const backend = new aac._MemoryBackend();
    const store = new aac.CalibrationStore({ backend, profileKey: 'u1' });
    const snapshot = aac.getCalibrationSnapshot();
    assert(snapshot.schemaVersion === 2 && typeof snapshot.featureVersion === 'string', 'calibration snapshot separates schema and feature versions');
    await store.save(snapshot);
    const loaded = await store.load();
    assert(loaded && !loaded._incompatibleReason, 'compatible calibration snapshot round-trips');

    const bad = Object.assign({}, snapshot, { featureVersion: 'old-feature-space' });
    await store.save(bad);
    const rejected = await store.load();
    assert(rejected && rejected._incompatibleReason === 'feature-version', 'incompatible feature spaces are explicitly rejected');
  }

  {
    const custom = new aac.CalibrationStore({ backend: new aac._MemoryBackend(), profileKey: 'custom' });
    aac._store = custom;
    aac.install();
    assert(aac._store === custom, 'install is idempotent and preserves configured calibration store');
  }

  {
    const d = aac.getDiagnostics();
    assert(d.version === '2.0.0' && typeof d.eyeFeatureCoverage === 'number', 'diagnostics report version and eye-feature coverage');
    assert(Object.prototype.hasOwnProperty.call(d, 'trackingQuality') && Object.prototype.hasOwnProperty.call(d, 'driftRmse'), 'diagnostics expose quality and drift separately');
  }

  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed) process.exit(1);
})().catch(err => {
  console.error(err);
  process.exit(1);
});
