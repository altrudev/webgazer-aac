'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

let fakeNow = 1000;
global.performance = { now: () => fakeNow };
global.window = global;
global.innerWidth = 1280;
global.innerHeight = 720;
global.devicePixelRatio = 1;
global.CustomEvent = class CustomEvent { constructor(type, init) { this.type = type; this.detail = init && init.detail; } };
global.document = { dispatchEvent() { return true; }, querySelectorAll() { return []; } };
global.ImageData = class ImageData { constructor(data, width, height) { this.data = data; this.width = width; this.height = height; } };
global.HTMLCanvasElement = class HTMLCanvasElement {};

const originalSet = function (callback) { this._callback = callback; return this; };
const originalRecord = function (x, y, eventType) { this._recorded.push({ x, y, eventType }); return this; };
const fakeWebgazer = {
  version: '3.5.3',
  _callback: null,
  _recorded: [],
  clearCount: 0,
  setGazeListener: originalSet,
  recordScreenPosition: originalRecord,
  setRegression() { return this; },
  clearGazeListener() { this._callback = null; this.clearCount++; return this; },
};
global.webgazer = fakeWebgazer;

const aac = require('../webgazer-aac.js');
let passed = 0;
let failed = 0;
function assert(condition, name) {
  if (condition) { console.log('✓', name); passed++; }
  else { console.error('✗', name); failed++; }
}

function hasRawEyeMaterial(value, seen = new Set()) {
  const raw = new Set(['patch', 'eyefeatures', 'imagedata', 'rawimage', 'rawframe', 'canvas']);
  if (value == null || typeof value !== 'object') return false;
  if (seen.has(value)) return false;
  seen.add(value);
  for (const key of Object.keys(value)) {
    if (raw.has(String(key).toLowerCase())) return true;
    if (hasRawEyeMaterial(value[key], seen)) return true;
  }
  return false;
}

console.log('\nwebgazer-aac DDC pre-hardware assurance tests');

const contract = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'ddc', 'pre-hardware-assurance.json'), 'utf8'));
const adapter = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'ddc', 'product-adapter.json'), 'utf8'));
const workOrder = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'ddc', 'PRE-HARDWARE-WORK-ORDER.json'), 'utf8'));

assert(contract.protocol === 'ddc-platform/pre-hardware-assurance', 'DDC assurance contract declares protocol');
assert(contract.authority.principle === 'observation-does-not-imply-authority', 'observation/authority distinction is explicit');
assert(contract.authority.forbiddenPromotions.includes('predicted-coordinate->ground-truth'), 'circular ground-truth promotion is forbidden');
assert(contract.transitions.every(t => t.authority && t.transition && Array.isArray(t.verification) && t.verification.length && Array.isArray(t.conserved) && t.conserved.length), 'every transition declares authority, verification, and conserved dimensions');
assert(contract.privacy.persistRawEyeImages === false && contract.privacy.persistNormalizedEyePatches === false, 'privacy contract forbids persisted eye material');
assert(contract.resourceBounds.installMustBeReversible === true, 'reversibility is a release invariant');

assert(adapter.protocol === 'ddc-platform/product-adapter' && adapter.version === '0.1', 'adapter matches DDC product-adapter protocol/version');
assert(adapter.authority.networkDefault === 'deny', 'adapter denies network authority by default');
assert(adapter.operations.act === false, 'pre-hardware adapter cannot activate a real AAC target');

assert(workOrder.schema === 'ddc-platform/work-order/0.1' && workOrder.id.startsWith('wo_'), 'DDC Work Order declares canonical schema/id');
assert(workOrder.authority.network === 'deny-by-default' && workOrder.authority.publish === false, 'DDC Work Order remains fail-closed for external authority');
assert(workOrder.epistemicIntegrity.mode === 'high-assurance' && workOrder.epistemicIntegrity.generatorMaySolelyVerifyConsequentialClaims === false, 'DDC Work Order requires high-assurance independent verification');
assert(workOrder.operationCapabilities.every(cap => cap.external === false && ['observe','simulate'].includes(cap.class)), 'DDC Work Order permits only local observe/simulate capabilities');
assert(workOrder.details.nonClaims.some(x => /clinical/i.test(x)) && workOrder.details.nonClaims.some(x => /DDC providers executed/i.test(x)), 'DDC Work Order preserves explicit non-claims');
const runtimeBytes = fs.readFileSync(path.join(__dirname, '..', 'webgazer-aac.js'));
const runtimeSha256 = crypto.createHash('sha256').update(runtimeBytes).digest('hex');
assert(workOrder.declarations[0].candidateSha256 === runtimeSha256, 'DDC Work Order is bound to the exact candidate runtime SHA-256');
assert(['describe','prepare','health','observe','collect','stop'].every(k => adapter.operations[k] === true), 'adapter declares required DDC operations');

const weights = aac.EVIDENCE_WEIGHTS;
assert(weights.calibration === 1 && weights.explicit === 1 && weights['confirmed-click'] > weights['dwell-selection'] && weights['dwell-selection'] > weights.inferred, 'runtime evidence ordering matches DDC contract');

const beforePatchedSet = fakeWebgazer.setGazeListener;
aac.install();
const patchedSet = fakeWebgazer.setGazeListener;
assert(patchedSet !== beforePatchedSet, 'install creates a bounded WebGazer interception surface');

const synthetic = { left: new Float32Array(960), right: new Float32Array(960) };

aac.resetCalibrationPatches();
for (let i = 0; i < 14; i++) {
  const n = { left: new Float32Array(960), right: new Float32Array(960) };
  n.left[i] = 0.5 + i / 50; n.right[i + 20] = -0.5 + i / 60;
  aac.recordGroundTruth(50 + i * 20, 80 + i * 10, { normalized: n, source: 'calibration', includeInCalibration: true });
}
const inferred = { left: new Float32Array(960), right: new Float32Array(960) };
inferred.left[100] = 0.91; inferred.right[200] = -0.73;
aac.recordGroundTruth(777, 333, { normalized: inferred, source: 'inferred', evidenceWeight: 0.15, includeInCalibration: false });
const allTrainingBeforePca = aac._trainingRecords.length;
const pcaResult = aac.fitUserBasis();
assert(pcaResult.rebuilt && aac._regressions.polynomial.samples.length === allTrainingBeforePca, 'PCA transition rebuilds calibration and non-calibration training evidence');
assert(aac._regressions.polynomial.getData().some(row => row.source === 'inferred'), 'non-calibration provenance survives PCA re-encoding');

aac.resetCalibrationPatches();
for (let i = 0; i < 520; i++) {
  synthetic.left[i % 960] = (i % 31) / 31;
  synthetic.right[i % 960] = (i % 29) / 29;
  aac.recordGroundTruth(i % 1280, (i * 7) % 720, {
    normalized: synthetic,
    source: 'calibration',
    evidenceWeight: 1,
    includeInCalibration: true,
    observation: 'ddc-bounded-fixture'
  });
}
assert(aac._trainingRecords.length <= contract.resourceBounds.trainingRecordsMax, 'raw-normalized training evidence remains within declared bound');
assert(aac._regressions.polynomial.samples.length <= contract.resourceBounds.polynomialSamplesMax, 'polynomial evidence remains within declared bound');
assert(aac._regressions.rbf.features.length <= contract.resourceBounds.rbfSamplesMax, 'RBF evidence remains within declared bound');
assert(aac._calibrationRaw.length <= contract.resourceBounds.calibrationRecordsMax, 'calibration evidence remains within declared bound');
assert(aac._evidenceLineage.length <= contract.resourceBounds.evidenceLineageMax, 'evidence lineage remains within declared bound');

const lineage = aac.getEvidenceLineage();
const last = lineage[lineage.length - 1];
assert(contract.evidence.requiredLineage.every(key => Object.prototype.hasOwnProperty.call(last, key)), 'runtime lineage carries all DDC-required fields');
assert(aac._regressions.polynomial.getData().every(row => row.source === 'calibration'), 'regression persistence preserves evidence provenance');

const snapshot = aac.getCalibrationSnapshot();
assert(!hasRawEyeMaterial(snapshot), 'persisted calibration snapshot excludes raw eye material');

aac._lastEyeFeatures = { sensitive: true };
aac._lastNormalized = synthetic;
aac.uninstall();
assert(aac._installed === false && fakeWebgazer.setGazeListener !== patchedSet, 'uninstall restores the WebGazer method boundary');
assert(fakeWebgazer.clearCount === 1, 'uninstall clears active gaze listener when upstream supports it');
assert(aac._lastEyeFeatures === null && aac._lastNormalized === null, 'uninstall clears transient eye evidence');

(async () => {
  aac.install();
  const backend = new aac._MemoryBackend();
  aac._store = new aac.CalibrationStore({ backend, profileKey: 'assurance' });
  const n = { left: new Float32Array(960), right: new Float32Array(960) };
  aac.recordGroundTruth(100, 100, { normalized: n, source: 'explicit', includeInCalibration: true });
  await aac.saveCalibration();
  aac._lastEyeFeatures = { sensitive: true };
  aac._lastNormalized = n;
  const cleared = await aac.clearAllCalibration();
  const loaded = await aac._store.load();
  assert(cleared === true && loaded == null, 'full clear removes persisted calibration');
  assert(aac._trainingRecords.length === 0 && aac._calibrationRaw.length === 0 && aac._regressions.polynomial.samples.length === 0 && aac._regressions.rbf.features.length === 0, 'full clear removes in-memory calibration/model evidence');
  assert(aac._evidenceLineage.length === 0, 'full clear removes in-memory evidence lineage');
  assert(aac._lastEyeFeatures === null && aac._lastNormalized === null, 'full clear removes transient eye evidence');

  const report = aac.getAssuranceSnapshot();
  assert(report.ok === true && report.protocol === contract.protocol, 'runtime emits a green DDC assurance snapshot');

  const source = fs.readFileSync(path.join(__dirname, '..', 'webgazer-aac.js'), 'utf8');
  const forbiddenNetwork = [/(^|[^A-Za-z])fetch\s*\(/, /XMLHttpRequest\s*\(/, /new\s+WebSocket\s*\(/, /sendBeacon\s*\(/];
  assert(forbiddenNetwork.every(re => !re.test(source)), 'runtime introduces no network primitive');
  assert(!/\.click\s*\(/.test(source), 'gaze runtime never directly invokes target click authority');

  aac.uninstall();
  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed) process.exit(1);
})().catch(error => {
  console.error(error);
  process.exit(1);
});
