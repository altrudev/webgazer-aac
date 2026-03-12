global.window      = { innerWidth: 1920, innerHeight: 1080 };
global.performance = { now: Date.now.bind(Date) };
global.ImageData   = class { constructor(d,w,h){this.data=d;this.width=w;this.height=h;} };
global.HTMLCanvasElement = class {};

require('../webgazer-aac.js');
const {
  PolynomialRegression, RBFRegression, EnsembleRegression,
  KalmanFilter, BlinkDetector, SaccadeDetector, FrameCache, PCABasis,
  DriftWatchdog, DwellTimer, CalibrationStore, _MemoryBackend,
} = webgazerAAC;

let passed = 0, failed = 0;
function assert(cond, label) {
  if (cond) { console.log('  ✓', label); passed++; }
  else       { console.error('  ✗', label); failed++; }
}

function makePatches(seed, brightness) {
  brightness = brightness != null ? brightness : 128;
  const size = 60 * 40;
  const mk = (s) => {
    const d = new Uint8ClampedArray(size * 4);
    for (let i = 0; i < size * 4; i += 4) {
      const v = brightness + ((s * 1664525 + i * 22695477 + 1013904223) & 0x7f) - 64;
      const c = Math.max(0, Math.min(255, v));
      d[i]=c; d[i+1]=Math.floor(c*0.8); d[i+2]=Math.floor(c*0.6); d[i+3]=255;
    }
    return new ImageData(d, 60, 40);
  };
  return { left: { patch: mk(seed) }, right: { patch: mk(seed+1) } };
}

function addNinePoints(reg) {
  const pts = [[100,100],[960,100],[1820,100],[100,540],[960,540],[1820,540],[100,980],[960,980],[1820,980]];
  for (let i = 0; i < pts.length; i++) reg.addData(makePatches(i+10), pts[i][0], pts[i][1]);
}

// ── Per-user PCA ─────────────────────────────────────────────────────────────
console.log('\nPer-user PCA basis:');
{
  const basis = new PCABasis(100, 5, 0x1234);
  assert(!basis.fitted, 'starts unfitted');
  assert(!basis.fit([[1,2,3]]), 'rejects insufficient patches');
  const patches = Array.from({length:14}, (_,i) =>
    Array.from({length:100}, (_,j) => Math.sin(i*j*0.1)*128+128));
  assert(basis.fit(patches), 'fits with 14 patches');
  assert(basis.fitted, 'marks fitted=true');
  const proj = basis.project(patches[0]);
  assert(Array.isArray(proj) && proj.length === 5, 'projects to correct dim');
  assert(proj.every(v => !isNaN(v)), 'projection values finite');
}

// ── Polynomial ───────────────────────────────────────────────────────────────
console.log('\nPolynomial regression:');
{
  const reg = new PolynomialRegression();
  assert(reg.predict(makePatches(1)) === null, 'null before enough data');
  addNinePoints(reg);
  const pred = reg.predict(makePatches(10));
  assert(pred !== null, 'predicts with 9 samples');
  assert(pred.x >= 0 && pred.x <= 1920, 'x clamped');
  assert(pred.y >= 0 && pred.y <= 1080, 'y clamped');
  const d = reg.getData(); assert(d.length > 0, 'getData returns samples');
  const r2 = new PolynomialRegression(); r2.setData(d);
  const p2 = r2.predict(makePatches(10));
  assert(p2 !== null && Math.abs(p2.x - pred.x) < 2, 'setData roundtrip consistent');
}

// ── RBF ──────────────────────────────────────────────────────────────────────
console.log('\nRBF regression:');
{
  const reg = new RBFRegression();
  assert(reg.predict(makePatches(1)) === null, 'null before enough data');
  addNinePoints(reg);
  const pred = reg.predict(makePatches(20));
  assert(pred !== null && !isNaN(pred.x), 'predicts with 9 samples');
  assert(pred.x >= 0 && pred.x <= 1920, 'x clamped');
}

// ── Ensemble ─────────────────────────────────────────────────────────────────
console.log('\nEnsemble regression:');
{
  const poly = new PolynomialRegression(), rbf = new RBFRegression();
  const ens = new EnsembleRegression(poly, rbf);
  addNinePoints(ens);
  const pred = ens.predict(makePatches(10));
  assert(pred !== null && !isNaN(pred.x), 'ensemble predicts');
  ens.trackError(makePatches(10), pred.x+50, pred.y+50);
  assert(ens._errPoly > 0 || ens._errRbf > 0, 'error tracking works');
  const ens2 = new EnsembleRegression(new PolynomialRegression(), new RBFRegression());
  addNinePoints(ens2.poly);
  assert(ens2.predict(makePatches(10)) !== null, 'fallback to poly if rbf empty');
}

// ── Kalman ────────────────────────────────────────────────────────────────────
console.log('\nKalman filter:');
{
  const kf = new KalmanFilter({ processNoise: 8, measurementNoise: 50 });
  const r1 = kf.smooth(500, 500, false, false);
  assert(r1.x === 500 && r1.y === 500, 'init at first measurement');
  assert(r1.confidence === 0, 'confidence 0 on init');
  let last;
  for (let i = 0; i < 30; i++) last = kf.smooth(500 + Math.random()*2, 500 + Math.random()*2, false, false);
  assert(last.confidence > 0.4, 'stable gaze builds confidence');
  const preBlink = { x: last.x, y: last.y };
  const br = kf.smooth(800, 800, true, false);
  assert(Math.abs(br.x - preBlink.x) < 60, 'blink suppresses update');
  kf.reset();
  assert(kf.smooth(100, 100, false, false).x === 100, 'reset clears state');
  for (let i = 0; i < 5; i++) kf.smooth(100+i*50, 100+i*30, false, false);
  const rv = kf.smooth(400, 250, false, false);
  assert(typeof rv.vx === 'number', 'outputs velocity');
}

// ── Blink ─────────────────────────────────────────────────────────────────────
console.log('\nBlink detector:');
{
  const bd = new BlinkDetector({ windowSize: 10, blinkThresh: 0.55, lockoutMs: 0 });
  for (let i = 0; i < 12; i++) bd.update(makePatches(i, 180));
  assert(!bd.update(makePatches(20, 170)), 'bright patch not a blink');
  assert(bd.update(makePatches(99, 10)), 'dark patch detected as blink');
}

// ── Saccade ───────────────────────────────────────────────────────────────────
console.log('\nSaccade detector:');
{
  const sd = new SaccadeDetector(600);
  assert(!sd.isSaccade(0, 0), 'zero velocity not saccade');
  assert(!sd.isSaccade(300, 200), 'moderate velocity not saccade');
  assert(sd.isSaccade(700, 0), 'high velocity is saccade');
}

// ── Frame cache ───────────────────────────────────────────────────────────────
console.log('\nFrame cache:');
{
  const fc = new FrameCache({ minSpeed: 15, maxReuseMs: 50 });
  assert(!fc.shouldSkip(0, 0), 'no skip before first store');
  fc.store({ x: 500, y: 400 });
  assert(fc.shouldSkip(5, 5), 'slow gaze hits cache');
  assert(!fc.shouldSkip(700, 200), 'fast gaze bypasses cache');
  assert(fc.get().x === 500, 'get returns stored value');
}

// ── Drift watchdog ────────────────────────────────────────────────────────────
console.log('\nDrift watchdog:');
{
  // Use a polynomial regression as the model under watch
  const reg = new PolynomialRegression();
  addNinePoints(reg);

  const wd = new DriftWatchdog(reg, {
    windowSize:    20,
    decayAlpha:    0.2,
    minSamples:    4,
    warnThreshold: 100,
    critThreshold: 200,
  });

  // ── starts disabled ──────────────────────────────────────────────────────
  assert(!wd._enabled, 'starts disabled');
  assert(wd.rmse === 0, 'rmse is 0 before enable');

  // ── enable ───────────────────────────────────────────────────────────────
  wd.enable();
  assert(wd._enabled, 'enable() sets enabled');

  // ── small residuals — no critical alert before minSamples ────────────────
  // Before enough samples accumulate, rmse should stay 0 → no events
  let warnFired = false, critFired = false;
  wd.onWarn     = () => { warnFired = true; };
  wd.onCritical = () => { critFired = true; };

  // Feed only (minSamples - 1) samples — watchdog should stay silent (rmse=0)
  for (let i = 0; i < 3; i++) {
    wd.record(makePatches(10 + i), 960, 540);
  }
  assert(wd.sampleCount === 3, 'sampleCount increments');
  assert(wd.rmse === 0, 'rmse stays 0 below minSamples');
  assert(!critFired && !warnFired, 'no events before minSamples reached');

  // ── large residuals — warning fires ──────────────────────────────────────
  // Feed deliberately wrong ground-truth so the residual is huge
  for (let i = 0; i < 8; i++) {
    wd.record(makePatches(10), 0, 0); // reg predicts ~960,540 → residual ~1075px
  }
  assert(warnFired || critFired, 'high-residual samples fire an alert');

  // ── reset clears state ────────────────────────────────────────────────────
  wd.reset();
  assert(wd.rmse === 0, 'reset() clears rmse');
  assert(wd.sampleCount === 0, 'reset() clears sampleCount');
  assert(wd._lastLevel === 'ok', 'reset() clears lastLevel');

  // ── disable suppresses events ─────────────────────────────────────────────
  wd.disable();
  let firedWhileDisabled = false;
  wd.onCritical = () => { firedWhileDisabled = true; };
  for (let i = 0; i < 10; i++) wd.record(makePatches(10), 0, 0);
  assert(!firedWhileDisabled, 'disabled watchdog does not fire events');
  assert(wd.sampleCount === 0, 'disabled watchdog does not accumulate samples');

  // ── null regression is handled gracefully ─────────────────────────────────
  const wdNull = new DriftWatchdog(null, { minSamples: 1 });
  wdNull.enable();
  wdNull.record(makePatches(1), 500, 500); // should not throw
  assert(wdNull.sampleCount === 0, 'null regression skips record gracefully');

  // ── regression with no data returns null, handled gracefully ─────────────
  const emptyReg = new PolynomialRegression();
  const wdEmpty  = new DriftWatchdog(emptyReg, { minSamples: 1 });
  wdEmpty.enable();
  wdEmpty.record(makePatches(1), 500, 500); // predict() returns null → skip
  assert(wdEmpty.sampleCount === 0, 'no-data regression skips record gracefully');

  // ── hysteresis: single below-threshold sample after crisis doesn't refire ─
  const reg2 = new PolynomialRegression();
  addNinePoints(reg2);
  const wd2 = new DriftWatchdog(reg2, {
    minSamples: 2, decayAlpha: 0.5,
    warnThreshold: 50, critThreshold: 150,
  });
  wd2.enable();
  let warnCount = 0;
  wd2.onWarn = () => { warnCount++; };
  // Spike to warning level
  for (let i = 0; i < 4; i++) wd2.record(makePatches(10), 0, 0);
  const firstCount = warnCount;
  // Same level again — should NOT re-fire (hysteresis)
  for (let i = 0; i < 4; i++) wd2.record(makePatches(10), 0, 0);
  assert(warnCount === firstCount, 'same-level event not re-fired (hysteresis)');
}

// ── DwellTimer ────────────────────────────────────────────────────────────────
console.log('\nDwellTimer:');
{
  // Minimal DOM element stub
  const makeEl = () => {
    const events = [];
    return {
      _events: events,
      dispatchEvent(e) { events.push({ type: e.type, detail: e.detail }); return true; },
    };
  };

  const dt = new DwellTimer({ dwellMs: 200, minConfidence: 0.3, holdAfterMs: 0 });
  assert(dt.progress === 0, 'starts at 0 progress');

  // ── null gaze (blink) freezes without reset ───────────────────────────────
  const el = makeEl();
  // Seed a partial progress by calling with the element first
  dt.update(el, 500, 300, 0.9, false, false);
  // Advance time manually by driving many updates through a fake clock
  // We simulate ~120ms worth of frames at high confidence in 3 big steps
  const _origNow = performance.now;
  let _fakeNow = _origNow();
  performance.now = () => _fakeNow;

  // Frame 1: enter element
  dt.update(el, 500, 300, 0.9, false, false);
  _fakeNow += 60; // +60ms
  dt.update(el, 500, 300, 0.9, false, false); // progress ~0.3
  const progBefore = dt.progress;
  assert(progBefore > 0, 'progress advances with high confidence');

  // Frame 2: blink — progress must freeze, not reset
  dt.update(null, 500, 300, 0.9, false, true);
  assert(dt.progress === progBefore, 'blink freezes progress');

  // Frame 3: saccade — also freezes
  _fakeNow += 60;
  dt.update(el, 500, 300, 0.9, true, false);
  assert(dt.progress === progBefore, 'saccade freezes progress');

  // Frame 4: low confidence — also freezes
  _fakeNow += 60;
  dt.update(el, 500, 300, 0.1, false, false);
  assert(dt.progress === progBefore, 'low confidence freezes progress');

  // Frame 5: resume high confidence — progress resumes
  _fakeNow += 60;
  dt.update(el, 500, 300, 0.9, false, false);
  assert(dt.progress > progBefore, 'progress resumes after freeze');

  // ── element change triggers cancel event ──────────────────────────────────
  const el2 = makeEl();
  dt.reset();
  dt.update(el, 500, 300, 0.9, false, false);
  _fakeNow += 40;
  dt.update(el, 500, 300, 0.9, false, false);
  const progressBeforeLeave = dt.progress;
  dt.update(el2, 600, 400, 0.9, false, false); // switch element
  const cancelEvents = el._events.filter(e => e.type === 'webgazer-aac:dwell-cancel');
  assert(cancelEvents.length > 0, 'cancel event fires on element change');
  assert(dt.progress === 0, 'progress resets on element change');

  // ── completion fires complete event and resets ────────────────────────────
  dt.reset();
  const el3 = makeEl();
  dt.update(el3, 500, 300, 0.9, false, false); // seed _lastT
  // Need multiple 100ms steps to exceed dwellMs=200 (clamped at 100ms per frame)
  _fakeNow += 90; dt.update(el3, 500, 300, 0.9, false, false); // +45% = 45%
  _fakeNow += 90; dt.update(el3, 500, 300, 0.9, false, false); // +45% = 90%
  _fakeNow += 90; dt.update(el3, 500, 300, 0.9, false, false); // +45% → hits 100%
  const completeEvents = el3._events.filter(e => e.type === 'webgazer-aac:dwell-complete');
  assert(completeEvents.length > 0, 'complete event fires at 100%');
  assert(dt.progress === 0, 'progress resets after completion');

  // ── progress events carry expected fields ─────────────────────────────────
  dt.reset();
  const el4 = makeEl();
  _fakeNow += 10;
  dt.update(el4, 500, 300, 0.8, false, false);
  _fakeNow += 30;
  dt.update(el4, 500, 300, 0.8, false, false);
  const progEvt = el4._events.find(e => e.type === 'webgazer-aac:dwell-progress');
  assert(progEvt && typeof progEvt.detail.progress === 'number', 'progress event has progress field');
  assert(progEvt && typeof progEvt.detail.confidence === 'number', 'progress event has confidence field');
  assert(progEvt && typeof progEvt.detail.frozen === 'boolean', 'progress event has frozen flag');

  // ── aacRef wiring calls recordDwellHitXY ─────────────────────────────────
  let dwellHitCalled = false;
  const fakeAac = { recordDwellHitXY: () => { dwellHitCalled = true; } };
  const timerWired = new DwellTimer({ dwellMs: 50, holdAfterMs: 0, aacRef: fakeAac });
  const el5 = makeEl();
  timerWired.update(el5, 100, 100, 0.9, false, false);
  _fakeNow += 100;
  timerWired.update(el5, 100, 100, 0.9, false, false);
  assert(dwellHitCalled, 'completion calls aacRef.recordDwellHitXY()');

  performance.now = _origNow;
}

// ── CalibrationStore (memory backend) ─────────────────────────────────────────
console.log('\nCalibrationStore:');
{
  const backend = new _MemoryBackend();
  const store   = new CalibrationStore({ backend });

  // available() with mock backend
  store.available().then(ok => assert(ok, 'available() resolves true with memory backend'));

  // ── save + load roundtrip ─────────────────────────────────────────────────
  // Train a regression so there's real data to snapshot
  const reg = new PolynomialRegression();
  addNinePoints(reg);
  // Manually prime webgazerAAC state for snapshot (without install())
  webgazerAAC._regressions = { polynomial: reg, rbf: new RBFRegression(), ensemble: new EnsembleRegression(reg, new RBFRegression()) };
  webgazerAAC._kalman.Q = 12;
  webgazerAAC._kalman.R = 75;

  const snap = webgazerAAC.getCalibrationSnapshot();
  assert(snap.aacVersion === '1.3.0', 'snapshot has correct version');
  assert(snap.kalman.Q === 12 && snap.kalman.R === 75, 'snapshot captures Kalman params');
  assert(Array.isArray(snap.regressions.polynomial), 'snapshot includes polynomial data');
  assert(snap.pca && typeof snap.pca.left === 'object', 'snapshot includes PCA state');
  assert(typeof snap.screenWidth === 'number', 'snapshot includes screen dimensions');

  // Save then load
  store.save(snap).then(ok => {
    assert(ok, 'save() resolves true');
    return store.load();
  }).then(loaded => {
    assert(loaded !== null, 'load() returns saved snapshot');
    assert(loaded.aacVersion === '1.3.0', 'loaded snapshot has correct version');
    assert(loaded.kalman.Q === 12, 'loaded Kalman Q matches');
    assert(loaded.regressions.polynomial.length === snap.regressions.polynomial.length,
      'loaded regression data length matches');
  });

  // ── version mismatch returns null ─────────────────────────────────────────
  const backend2 = new _MemoryBackend();
  const store2   = new CalibrationStore({ backend: backend2 });
  backend2.save('default', { aacVersion: '0.9.0', data: 'stale' }).then(() =>
    store2.load()
  ).then(result => assert(result === null, 'old version snapshot returns null'));

  // ── clear removes data ────────────────────────────────────────────────────
  const backend3 = new _MemoryBackend();
  const store3   = new CalibrationStore({ backend: backend3 });
  store3.save(snap).then(() => store3.clear()).then(() => store3.load()).then(result =>
    assert(result === null, 'clear() removes stored data')
  );

  // ── applyCalibrationSnapshot restores Kalman params ──────────────────────
  webgazerAAC._kalman.Q = 8;  // reset
  webgazerAAC._kalman.R = 50;
  const applied = webgazerAAC.applyCalibrationSnapshot(snap);
  assert(applied === true, 'applyCalibrationSnapshot returns true');
  assert(webgazerAAC._kalman.Q === 12, 'applyCalibrationSnapshot restores Q');
  assert(webgazerAAC._kalman.R === 75, 'applyCalibrationSnapshot restores R');

  // ── apply with wrong version returns false ────────────────────────────────
  assert(webgazerAAC.applyCalibrationSnapshot({ aacVersion: '0.1.0' }) === false,
    'applyCalibrationSnapshot rejects wrong version');

  // ── apply null returns false ──────────────────────────────────────────────
  assert(webgazerAAC.applyCalibrationSnapshot(null) === false,
    'applyCalibrationSnapshot handles null gracefully');
}

// ── New exports on webgazerAAC ────────────────────────────────────────────────
console.log('\nNew v1.3.0 exports:');
{
  assert(typeof DwellTimer        === 'function', 'DwellTimer exported');
  assert(typeof CalibrationStore  === 'function', 'CalibrationStore exported');
  assert(typeof _MemoryBackend    === 'function', '_MemoryBackend exported');
  assert(webgazerAAC.version      === '1.3.0',    'version is 1.3.0');
}

console.log(`\n${'─'.repeat(44)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
