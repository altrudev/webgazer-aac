global.window      = { innerWidth: 1920, innerHeight: 1080 };
global.performance = { now: Date.now.bind(Date) };
global.ImageData   = class { constructor(d,w,h){this.data=d;this.width=w;this.height=h;} };
global.HTMLCanvasElement = class {};

require('../webgazer-aac-v1.1.0.js');
const {
  PolynomialRegression, RBFRegression, EnsembleRegression,
  KalmanFilter, BlinkDetector, SaccadeDetector, FrameCache, PCABasis,
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

console.log(`\n${'─'.repeat(44)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
