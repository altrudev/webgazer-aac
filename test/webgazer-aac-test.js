/**
 * Basic unit tests for webgazer-aac regression modules.
 * Run with: node test/regression.test.js
 * No test framework needed — pure Node.js.
 */

// Minimal browser shims
global.window = { innerWidth: 1920, innerHeight: 1080 };
global.performance = { now: Date.now.bind(Date) };
global.ImageData = class ImageData {
  constructor(data, w, h) { this.data = data; this.width = w; this.height = h; }
};

// Load the module
require('../webgazer-aac.js');
const {
  PolynomialRegression,
  RBFRegression,
  VelocitySmoother,
} = webgazerAAC;

let passed = 0, failed = 0;

function assert(condition, label) {
  if (condition) { console.log('  ✓', label); passed++; }
  else           { console.error('  ✗', label); failed++; }
}

function makeEyePatches(seed) {
  // Deterministic fake eye patch data
  const size = 60 * 40;
  const makeData = (s) => {
    const d = new Uint8ClampedArray(size * 4);
    for (let i = 0; i < size * 4; i += 4) {
      const v = ((s * 1664525 + i * 22695477 + 1013904223) & 0xff);
      d[i] = v; d[i+1] = v * 0.8; d[i+2] = v * 0.6; d[i+3] = 255;
    }
    return new ImageData(d, 60, 40);
  };
  return {
    left:  { patch: makeData(seed) },
    right: { patch: makeData(seed + 1) },
  };
}

// ── Polynomial regression tests ──────────────────────────────────────────────
console.log('\nPolynomial Regression:');
{
  const reg = new PolynomialRegression();

  // Before enough data — should return null
  reg.addData(makeEyePatches(1), 100, 200);
  assert(reg.predict(makeEyePatches(1)) === null, 'returns null with < 6 samples');

  // Add enough samples
  const points = [
    [100, 100], [960, 100], [1820, 100],
    [100, 540], [960, 540], [1820, 540],
    [100, 980], [960, 980], [1820, 980],
  ];
  for (let i = 0; i < points.length; i++) {
    reg.addData(makeEyePatches(i + 10), points[i][0], points[i][1]);
  }

  const pred = reg.predict(makeEyePatches(10));
  assert(pred !== null, 'returns prediction with 9 samples');
  assert(typeof pred.x === 'number' && !isNaN(pred.x), 'prediction x is a number');
  assert(typeof pred.y === 'number' && !isNaN(pred.y), 'prediction y is a number');
  assert(pred.x >= 0 && pred.x <= 1920, 'prediction x clamped to viewport width');
  assert(pred.y >= 0 && pred.y <= 1080, 'prediction y clamped to viewport height');

  // getData / setData roundtrip
  const data = reg.getData();
  assert(Array.isArray(data) && data.length > 0, 'getData returns stored samples');
  const reg2 = new PolynomialRegression();
  reg2.setData(data);
  const pred2 = reg2.predict(makeEyePatches(10));
  assert(pred2 !== null, 'setData → predict works');
  assert(Math.abs(pred2.x - pred.x) < 1, 'setData roundtrip x consistent');
  assert(Math.abs(pred2.y - pred.y) < 1, 'setData roundtrip y consistent');
}

// ── RBF regression tests ─────────────────────────────────────────────────────
console.log('\nRBF Regression:');
{
  const reg = new RBFRegression();

  reg.addData(makeEyePatches(1), 100, 200);
  assert(reg.predict(makeEyePatches(1)) === null, 'returns null with < 4 samples');

  for (let i = 0; i < 9; i++) {
    reg.addData(makeEyePatches(i + 20), i * 200 + 100, i * 100 + 50);
  }

  const pred = reg.predict(makeEyePatches(20));
  assert(pred !== null, 'returns prediction with 9 samples');
  assert(!isNaN(pred.x) && !isNaN(pred.y), 'prediction is numeric');
  assert(pred.x >= 0 && pred.x <= 1920, 'x clamped');
  assert(pred.y >= 0 && pred.y <= 1080, 'y clamped');
}

// ── Velocity smoother tests ──────────────────────────────────────────────────
console.log('\nVelocity Smoother:');
{
  const s = new VelocitySmoother();

  const r1 = s.smooth(500, 500);
  assert(r1.x === 500 && r1.y === 500, 'first call returns raw values');
  assert(r1.confidence === 0, 'first call confidence is 0');

  // Stable gaze — confidence should rise
  let last;
  for (let i = 0; i < 20; i++) last = s.smooth(500 + Math.random() * 2, 500 + Math.random() * 2);
  assert(last.confidence > 0.5, 'stable gaze builds confidence');

  // Fast movement — confidence drops
  const jump = s.smooth(1800, 900);
  assert(jump.confidence < last.confidence, 'fast movement reduces confidence');

  // Reset
  s.reset();
  const after = s.smooth(100, 100);
  assert(after.x === 100 && after.y === 100, 'reset clears state');
}

// ── Summary ──────────────────────────────────────────────────────────────────
console.log(`\n${'─'.repeat(40)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
