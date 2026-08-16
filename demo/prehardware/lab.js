'use strict';

const $ = id => document.getElementById(id);
const REQUIRED_SAMPLES_PER_POINT = 2;
const REQUIRED_DISTINCT_DWELL_TARGETS = 3;
const MIN_EYE_FEATURE_COVERAGE = 0.95;
const MIN_EYE_ROI_VALIDITY = 0.90;

const LEFT_EYE = [466,388,387,386,385,384,398,263,249,390,373,374,380,381,382,362];
const RIGHT_EYE = [246,161,160,159,158,157,173,33,7,163,144,145,153,154,155,133];

const state = {
  started: false,
  events: [],
  dwell: null,
  lastResolved: null,
  timer: null,
  webgazerIdentity: null,
  calibrationCounts: new Map(),
  distinctDwellTargets: new Set(),
  lastValidation: null,
  lastLiveValidation: null,
  geometry: {
    frames: 0,
    validRoiFrames: 0,
    last: null,
    lastPose: null,
  },
  motion: {
    last: null,
    emaSpeed: 0,
    highCount: 0,
    lowCount: 0,
    active: false,
    frames: 0,
    rawSaccadeFrames: 0,
    filteredSaccadeFrames: 0,
  },
};

const logEl = $('log');
const board = $('aacBoard');
const gazeDot = $('gazeDot');

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function hypot2(a, b) { return Math.hypot(a, b); }
function deepCopy(value) { return value == null ? value : JSON.parse(JSON.stringify(value)); }

function record(type, detail = {}) {
  const entry = { t: new Date().toISOString(), type, detail };
  state.events.push(entry);
  if (state.events.length > 1000) state.events.shift();
  logEl.textContent = state.events.slice(-80).map(e => `${e.t}  ${e.type}  ${JSON.stringify(e.detail)}`).join('\n');
  logEl.scrollTop = logEl.scrollHeight;
}

function setSession(text) {
  $('sessionState').textContent = text;
  updateValidationState();
}

function errorEvidence(error) {
  return {
    name: error && error.name ? String(error.name) : null,
    message: String(error && (error.message || error) || 'unknown error'),
    stack: error && error.stack ? String(error.stack) : null,
    webgazerReady: !!(window.webgazer && typeof window.webgazer.isReady === 'function' && window.webgazer.isReady()),
    videoElementPresent: !!document.getElementById('webgazerVideoFeed'),
    videoContainerPresent: !!document.getElementById('webgazerVideoContainer'),
  };
}

function setGate(id, pass, text) {
  const strong = $(id);
  if (!strong) return;
  strong.textContent = text;
  const article = strong.closest('article');
  if (article) {
    article.classList.toggle('gate-pass', !!pass);
    article.classList.toggle('gate-fail', !pass);
  }
}

function getDiagnosticsSafe() {
  try { return window.webgazerAAC ? window.webgazerAAC.getDiagnostics() : null; }
  catch (_) { return null; }
}

function geometrySnapshot() {
  const g = state.geometry;
  const m = state.motion;
  return {
    eyeRoiValidity: g.frames ? g.validRoiFrames / g.frames : 0,
    geometryFrames: g.frames,
    validRoiFrames: g.validRoiFrames,
    pose: g.last ? deepCopy(g.last.pose) : null,
    eyeRois: g.last ? deepCopy(g.last.rois) : null,
    alignment: g.last ? deepCopy(g.last.alignment) : null,
    rawSaccadeRate: m.frames ? m.rawSaccadeFrames / m.frames : 0,
    filteredSaccadeRate: m.frames ? m.filteredSaccadeFrames / m.frames : 0,
    motionFrames: m.frames,
  };
}

function validationSnapshot({ requireRunning = true } = {}) {
  const d = getDiagnosticsSafe() || {};
  const geometry = geometrySnapshot();
  const identityVerified = !!(state.webgazerIdentity && state.webgazerIdentity.verified && state.webgazerIdentity.version === '3.5.3');
  const mediaPipeVerified = !!(identityVerified && state.webgazerIdentity.mediaPipeAvailable && state.webgazerIdentity.mediaPipeSha256);
  const pcaFitted = !!(window.webgazerAAC && window.webgazerAAC.isPCAFitted && window.webgazerAAC.isPCAFitted());
  const eyeFeatureCoverage = Number(d.eyeFeatureCoverage || 0);
  const driftLevel = d.driftLevel || 'disabled';
  const driftRmse = Number(d.driftRmse || 0);
  const distinctTargets = Array.from(state.distinctDwellTargets).sort();
  const reasons = [];
  if (!identityVerified) reasons.push('WebGazer identity not verified');
  if (!mediaPipeVerified) reasons.push('MediaPipe assets not verified');
  if (requireRunning && !state.started) reasons.push('session not running');
  if (eyeFeatureCoverage < MIN_EYE_FEATURE_COVERAGE) reasons.push(`eye-feature coverage below ${Math.round(MIN_EYE_FEATURE_COVERAGE * 100)}%`);
  if (geometry.eyeRoiValidity < MIN_EYE_ROI_VALIDITY) reasons.push(`eye ROI validity below ${Math.round(MIN_EYE_ROI_VALIDITY * 100)}%`);
  if (!pcaFitted) reasons.push('PCA basis not fitted');
  if (driftLevel !== 'ok') reasons.push(`drift state is ${driftLevel}`);
  if (distinctTargets.length < REQUIRED_DISTINCT_DWELL_TARGETS) reasons.push(`need ${REQUIRED_DISTINCT_DWELL_TARGETS} distinct dwell targets`);
  return {
    ready: reasons.length === 0,
    capturedAt: new Date().toISOString(),
    reasons,
    criteria: {
      identityVerified,
      mediaPipeVerified,
      sessionRunning: state.started,
      minEyeFeatureCoverage: MIN_EYE_FEATURE_COVERAGE,
      eyeFeatureCoverage,
      minEyeRoiValidity: MIN_EYE_ROI_VALIDITY,
      eyeRoiValidity: geometry.eyeRoiValidity,
      pcaFitted,
      driftLevel,
      driftRmse,
      requiredDistinctDwellTargets: REQUIRED_DISTINCT_DWELL_TARGETS,
      distinctDwellTargets: distinctTargets,
      distinctDwellTargetCount: distinctTargets.length,
      dwellExecutesAction: false,
      rawSaccadeRate: geometry.rawSaccadeRate,
      filteredSaccadeRate: geometry.filteredSaccadeRate,
    },
    geometry,
  };
}

function updateValidationState() {
  const current = validationSnapshot();
  state.lastValidation = current;
  if (state.started) state.lastLiveValidation = validationSnapshot({ requireRunning: true });
  const shown = state.started ? current : (state.lastLiveValidation || current);

  setGate('identityGate', current.criteria.identityVerified, current.criteria.identityVerified ? 'VERIFIED' : 'NOT VERIFIED');
  setGate('mediaPipeGate', current.criteria.mediaPipeVerified, current.criteria.mediaPipeVerified ? 'VERIFIED' : 'NOT VERIFIED');
  setGate('sessionGate', current.criteria.sessionRunning, current.criteria.sessionRunning ? 'RUNNING' : 'NOT RUNNING');
  setGate('pcaGate', shown.criteria.pcaFitted, shown.criteria.pcaFitted ? 'FITTED' : 'NOT FITTED');
  setGate('targetGate', shown.criteria.distinctDwellTargetCount >= REQUIRED_DISTINCT_DWELL_TARGETS,
    `${shown.criteria.distinctDwellTargetCount} / ${REQUIRED_DISTINCT_DWELL_TARGETS}`);
  setGate('readinessGate', shown.ready, shown.ready ? (state.started ? 'READY' : 'LAST LIVE READY') : (state.started ? 'NOT READY' : 'LAST LIVE NOT READY'));
  $('readinessReason').textContent = shown.ready ? 'All browser-beta gates satisfied.' : shown.reasons.join(' · ');
  $('fitState').textContent = shown.criteria.pcaFitted ? 'PCA FITTED' : 'PCA NOT FITTED';
  return current;
}

function resetGeometryAndMotion() {
  state.geometry = { frames: 0, validRoiFrames: 0, last: null, lastPose: null };
  state.motion = { last: null, emaSpeed: 0, highCount: 0, lowCount: 0, active: false, frames: 0, rawSaccadeFrames: 0, filteredSaccadeFrames: 0 };
  $('eyeRoiValidity').textContent = '0%';
  $('poseState').textContent = 'waiting';
  $('poseRoll').textContent = '0°';
  $('rawSaccadeRate').textContent = '0%';
  $('filteredSaccadeRate').textContent = '0%';
  $('alignmentState').textContent = 'waiting';
}

function point(positions, index) {
  const p = positions && positions[index];
  return Array.isArray(p) && Number.isFinite(p[0]) && Number.isFinite(p[1]) ? { x: p[0], y: p[1], z: Number(p[2]) || 0 } : null;
}

function boundsForIndices(positions, indices, width, height) {
  const pts = indices.map(i => point(positions, i)).filter(Boolean);
  if (pts.length < Math.max(6, Math.floor(indices.length * 0.6))) return null;
  let minX = Math.min(...pts.map(p => p.x));
  let maxX = Math.max(...pts.map(p => p.x));
  let minY = Math.min(...pts.map(p => p.y));
  let maxY = Math.max(...pts.map(p => p.y));
  const rawW = Math.max(1, maxX - minX);
  const rawH = Math.max(1, maxY - minY);
  const marginX = Math.max(3, rawW * 0.18);
  const marginY = Math.max(3, rawH * 0.45);
  const unclamped = { left: minX - marginX, top: minY - marginY, right: maxX + marginX, bottom: maxY + marginY };
  const rect = {
    left: clamp(unclamped.left, 0, width),
    top: clamp(unclamped.top, 0, height),
    right: clamp(unclamped.right, 0, width),
    bottom: clamp(unclamped.bottom, 0, height),
  };
  rect.width = rect.right - rect.left;
  rect.height = rect.bottom - rect.top;
  const unclampedArea = Math.max(1, (unclamped.right - unclamped.left) * (unclamped.bottom - unclamped.top));
  const visibleArea = Math.max(0, rect.width * rect.height);
  rect.visibleFraction = visibleArea / unclampedArea;
  rect.valid = rect.width >= 10 && rect.height >= 6 && rect.visibleFraction >= 0.9;
  return rect;
}

function derivePose(positions) {
  const rightOuter = point(positions, 33);
  const leftOuter = point(positions, 263);
  const nose = point(positions, 1);
  const forehead = point(positions, 10);
  const chin = point(positions, 152);
  if (!rightOuter || !leftOuter || !nose) return null;
  const dx = leftOuter.x - rightOuter.x;
  const dy = leftOuter.y - rightOuter.y;
  const interocular = Math.max(1, hypot2(dx, dy));
  const eyeMid = { x: (leftOuter.x + rightOuter.x) / 2, y: (leftOuter.y + rightOuter.y) / 2 };
  const rollDeg = Math.atan2(dy, dx) * 180 / Math.PI;
  const yawProxy = (nose.x - eyeMid.x) / interocular;
  let pitchProxy = 0;
  if (forehead && chin) {
    const faceHeight = Math.max(1, hypot2(chin.x - forehead.x, chin.y - forehead.y));
    pitchProxy = (nose.y - eyeMid.y) / faceHeight;
  }
  return { rollDeg, yawProxy, pitchProxy, interocularPx: interocular, eyeMid };
}

function classifyRelativePose(pose) {
  if (!pose) return { label: 'unavailable', usable: false, hint: 'Face landmarks unavailable.' };
  const yaw = pose.yawProxy;
  const roll = pose.rollDeg;
  const lateral = Math.abs(yaw) > 0.20;
  const tilted = Math.abs(roll) > 16;
  const moderate = Math.abs(yaw) > 0.12 || Math.abs(roll) > 9;
  let label = 'centered';
  if (lateral && tilted) label = 'oblique';
  else if (lateral) label = yaw > 0 ? 'lateral-right' : 'lateral-left';
  else if (tilted) label = roll > 0 ? 'rolled-right' : 'rolled-left';
  else if (moderate) label = 'moderate-angle';
  const usable = Math.abs(yaw) < 0.34 && Math.abs(roll) < 28 && pose.interocularPx >= 45;
  const hint = usable
    ? (moderate ? 'Off-axis geometry is usable; keep both eyes unobstructed and calibrate from this same camera position.' : 'Eye geometry is well positioned for calibration.')
    : 'Move or tilt the camera/user until both eyes are fully visible and the face is less oblique.';
  return { label, usable, hint };
}

function deriveGeometry() {
  if (!window.webgazer || typeof window.webgazer.getTracker !== 'function') return null;
  let tracker = null;
  let positions = null;
  try {
    tracker = window.webgazer.getTracker();
    positions = tracker && typeof tracker.getPositions === 'function' ? tracker.getPositions() : null;
  } catch (_) { return null; }
  if (!positions || positions.length < 264) return null;
  const video = document.getElementById('webgazerVideoFeed');
  const width = video && (video.videoWidth || video.width) || 640;
  const height = video && (video.videoHeight || video.height) || 480;
  const left = boundsForIndices(positions, LEFT_EYE, width, height);
  const right = boundsForIndices(positions, RIGHT_EYE, width, height);
  const pose = derivePose(positions);
  const valid = !!(left && right && left.valid && right.valid && pose);
  const alignment = classifyRelativePose(pose);
  const result = { valid, rois: { left, right }, pose, alignment, frame: { width, height }, landmarkCount: positions.length };
  state.geometry.frames++;
  if (valid) state.geometry.validRoiFrames++;
  state.geometry.last = result;
  return result;
}

function ensureEyeOverlay() {
  const container = document.getElementById('webgazerVideoContainer');
  const video = document.getElementById('webgazerVideoFeed');
  if (!container || !video) return null;
  let canvas = document.getElementById('aacEyeOverlay');
  if (!canvas) {
    canvas = document.createElement('canvas');
    canvas.id = 'aacEyeOverlay';
    container.appendChild(canvas);
  }
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  canvas.style.width = video.style.width || `${video.clientWidth || 320}px`;
  canvas.style.height = video.style.height || `${video.clientHeight || 240}px`;
  return canvas;
}

function drawEyeGeometry(geometry) {
  const canvas = ensureEyeOverlay();
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!geometry || !geometry.rois) return;
  ctx.lineWidth = Math.max(2, canvas.width / 320);
  ctx.strokeStyle = geometry.valid ? '#111827' : '#7f1d1d';
  for (const rect of [geometry.rois.left, geometry.rois.right]) {
    if (!rect) continue;
    ctx.strokeRect(rect.left, rect.top, rect.width, rect.height);
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    ctx.beginPath();
    ctx.moveTo(cx - 5, cy); ctx.lineTo(cx + 5, cy);
    ctx.moveTo(cx, cy - 5); ctx.lineTo(cx, cy + 5);
    ctx.stroke();
  }
}

function poseMotionScore(current, previous) {
  if (!current || !previous) return 0;
  const yaw = Math.abs(current.yawProxy - previous.yawProxy) * 3;
  const roll = Math.abs(current.rollDeg - previous.rollDeg) / 20;
  const center = hypot2(current.eyeMid.x - previous.eyeMid.x, current.eyeMid.y - previous.eyeMid.y) / Math.max(40, current.interocularPx);
  return clamp(yaw + roll + center, 0, 3);
}

function filterMotion(gaze, geometry) {
  const m = state.motion;
  const now = performance.now();
  m.frames++;
  if (gaze.isSaccade) m.rawSaccadeFrames++;

  if (!m.last) {
    m.last = { x: gaze.x, y: gaze.y, t: now };
    state.geometry.lastPose = geometry && geometry.pose || null;
    return false;
  }

  const dtMs = now - m.last.t;
  const dt = clamp(dtMs / 1000, 0.008, 0.2);
  const speed = hypot2(gaze.x - m.last.x, gaze.y - m.last.y) / dt;
  m.last = { x: gaze.x, y: gaze.y, t: now };
  m.emaSpeed = m.emaSpeed ? m.emaSpeed * 0.72 + speed * 0.28 : speed;

  const pose = geometry && geometry.pose || null;
  const poseMotion = poseMotionScore(pose, state.geometry.lastPose);
  if (pose) state.geometry.lastPose = pose;
  const quality = clamp(Number(gaze.trackingQuality || 0), 0, 1);
  const enterThreshold = 1700 + (1 - quality) * 900 + poseMotion * 500;
  const exitThreshold = enterThreshold * 0.48;
  const validDt = dtMs >= 10 && dtMs <= 160;
  const high = validDt && quality >= 0.35 && m.emaSpeed >= enterThreshold;
  const low = !validDt || quality < 0.25 || m.emaSpeed <= exitThreshold;

  if (high) { m.highCount++; m.lowCount = 0; }
  else if (low) { m.lowCount++; m.highCount = 0; }
  else { m.highCount = Math.max(0, m.highCount - 1); m.lowCount = Math.max(0, m.lowCount - 1); }

  if (!m.active && m.highCount >= 3) m.active = true;
  if (m.active && m.lowCount >= 3) m.active = false;
  if (m.active) m.filteredSaccadeFrames++;
  return m.active;
}

function updateGeometryUi() {
  const snap = geometrySnapshot();
  $('eyeRoiValidity').textContent = `${Math.round(snap.eyeRoiValidity * 100)}%`;
  $('rawSaccadeRate').textContent = `${Math.round(snap.rawSaccadeRate * 100)}%`;
  $('filteredSaccadeRate').textContent = `${Math.round(snap.filteredSaccadeRate * 100)}%`;
  const pose = snap.pose;
  const alignment = snap.alignment;
  $('poseState').textContent = alignment ? alignment.label : 'waiting';
  $('poseRoll').textContent = pose ? `${pose.rollDeg.toFixed(1)}°` : '0°';
  $('alignmentState').textContent = alignment ? (alignment.usable ? 'usable' : 'adjust') : 'waiting';
  $('alignmentHint').textContent = alignment ? alignment.hint : 'FaceMesh geometry will report relative camera/head placement after tracking begins.';
}

async function cleanPartialSession() {
  clearInterval(state.timer); state.timer = null;
  try { if (window.webgazer && typeof window.webgazer.stopVideo === 'function') window.webgazer.stopVideo(); } catch (_) {}
  try { if (window.webgazer && typeof window.webgazer.end === 'function') await window.webgazer.end(); } catch (_) {}
  try { if (window.webgazerAAC && window.webgazerAAC._installed) window.webgazerAAC.uninstall(); } catch (_) {}
  state.started = false;
  state.dwell = null;
  gazeDot.hidden = true;
  $('stopBtn').disabled = true;
  $('startBtn').disabled = !(window.webgazer && state.webgazerIdentity && state.webgazerIdentity.verified);
  setSession('failed');
}

function runtimeStatus() {
  const aac = window.webgazerAAC;
  $('runtimeBadge').textContent = aac ? `AAC ${aac.version} · ${aac.featureVersion || aac._featureVersion || 'feature-v2'}` : 'AAC runtime missing';
}
runtimeStatus();
resetGeometryAndMotion();
updateValidationState();

function finishWebGazerLoad(source, identity = null) {
  if (!window.webgazer) throw new Error('Loaded script did not expose window.webgazer');
  const runtimeVersion = window.webgazer.version || null;
  const verifiedVersion = identity && identity.verified ? identity.version : runtimeVersion;
  const matchesExpected = verifiedVersion === '3.5.3';
  state.webgazerIdentity = identity || { source, verified: matchesExpected, version: runtimeVersion, expectedVersion: '3.5.3', bundleBytes: null, bundleSha256: null };
  $('wgStatus').textContent = matchesExpected ? `${verifiedVersion} verified` : `${verifiedVersion || 'unknown'} (expected 3.5.3)`;
  $('startBtn').disabled = !matchesExpected;
  $('useCodespaceBtn').disabled = true;
  record('webgazer-loaded', { runtimeVersion, verifiedVersion, expected: '3.5.3', matchesExpected, source, identity: state.webgazerIdentity });
  updateValidationState();
}

async function loadScriptUrl(url, source, identity = null) {
  if (state.started) return record('load-refused', { reason: 'stop-session-first' });
  $('wgStatus').textContent = 'loading…';
  try {
    const script = document.createElement('script');
    script.src = `${url}?t=${Date.now()}`;
    script.async = true;
    const done = new Promise((resolve, reject) => {
      script.onload = resolve;
      script.onerror = () => reject(new Error(`Could not load ${url}`));
    });
    document.head.appendChild(script);
    await done;
    finishWebGazerLoad(source, identity);
  } catch (error) {
    $('wgStatus').textContent = 'load failed';
    $('useCodespaceBtn').disabled = false;
    record('webgazer-load-error', { ...errorEvidence(error), source });
  }
}

async function getCodespaceIdentity() {
  const response = await fetch('/__webgazer__/status', { cache: 'no-store' });
  let status = null;
  try { status = await response.json(); } catch (_) {}
  if (!response.ok || !status || !status.verified) throw new Error(status && status.reason || `Codespace WebGazer identity check failed (${response.status})`);
  return status;
}

async function checkCodespaceBuild() {
  try {
    const status = await getCodespaceIdentity();
    state.webgazerIdentity = status;
    $('useCodespaceBtn').disabled = false;
    record('codespace-webgazer-verified', {
      version: status.version,
      expectedVersion: status.expectedVersion,
      packageName: status.packageName,
      bundleBytes: status.bundleBytes,
      bundleSha256: status.bundleSha256,
      mediaPipeFiles: status.mediaPipeFiles,
      mediaPipeBytes: status.mediaPipeBytes,
      mediaPipeSha256: status.mediaPipeSha256,
      source: status.source,
    });
    updateValidationState();
  } catch (error) {
    state.webgazerIdentity = null;
    $('useCodespaceBtn').disabled = true;
    record('codespace-webgazer-status-error', errorEvidence(error));
    updateValidationState();
  }
}
checkCodespaceBuild();

$('useCodespaceBtn').addEventListener('click', async () => {
  try {
    const identity = await getCodespaceIdentity();
    await loadScriptUrl('/__webgazer__/webgazer.js', 'codespace-build', identity);
  } catch (error) {
    $('wgStatus').textContent = 'identity failed';
    $('useCodespaceBtn').disabled = true;
    record('webgazer-identity-error', { ...errorEvidence(error), source: 'codespace-build' });
  }
});

const points = [
  [0.1,0.12],[0.5,0.12],[0.9,0.12],
  [0.1,0.5],[0.5,0.5],[0.9,0.5],
  [0.1,0.88],[0.5,0.88],[0.9,0.88],
];

function calibrationProgress() {
  const counts = Array.from(state.calibrationCounts.values());
  const completedPoints = counts.filter(v => v >= REQUIRED_SAMPLES_PER_POINT).length;
  const totalSamples = counts.reduce((a, v) => a + v, 0);
  const readyToFit = completedPoints === points.length;
  $('fitBtn').disabled = !readyToFit;
  $('fitBtn').classList.toggle('attention', readyToFit && !(window.webgazerAAC && window.webgazerAAC.isPCAFitted()));
  $('fitBtn').textContent = readyToFit ? 'FIT USER BASIS — REQUIRED' : 'Fit user basis';
  $('calibrationStatus').textContent = readyToFit
    ? `${totalSamples} samples captured. Fit the PCA basis before validation can pass.`
    : `${completedPoints}/${points.length} points complete · ${totalSamples}/${points.length * REQUIRED_SAMPLES_PER_POINT} samples. Click each point twice while looking directly at it.`;
}

points.forEach(([nx, ny], index) => {
  state.calibrationCounts.set(index, 0);
  const p = document.createElement('button');
  p.className = 'calibrationPoint';
  p.type = 'button';
  p.dataset.nx = nx;
  p.dataset.ny = ny;
  p.dataset.pointIndex = String(index);
  p.setAttribute('aria-label', `Calibration point ${Math.round(nx*100)} ${Math.round(ny*100)}; requires two samples`);
  p.addEventListener('click', () => {
    if (!state.started || !window.webgazer) return;
    const r = p.getBoundingClientRect();
    const x = r.left + r.width/2, y = r.top + r.height/2;
    window.webgazer.recordScreenPosition(x, y, 'click');
    const next = Math.min(REQUIRED_SAMPLES_PER_POINT, (state.calibrationCounts.get(index) || 0) + 1);
    state.calibrationCounts.set(index, next);
    p.classList.toggle('done', next >= REQUIRED_SAMPLES_PER_POINT);
    record('calibration-sample', { point: index + 1, sample: next, requiredPerPoint: REQUIRED_SAMPLES_PER_POINT, x: Math.round(x), y: Math.round(y), geometry: state.geometry.last ? { pose: state.geometry.last.pose, alignment: state.geometry.last.alignment } : null });
    calibrationProgress();
    updateValidationState();
  });
  $('calibrationGrid').appendChild(p);
});

$('wgFile').addEventListener('change', event => {
  event.preventDefault();
  event.target.value = '';
  record('load-refused', { reason: 'arbitrary-upload-disabled-in-assurance-mode' });
});

function wireBoardEvidence() {
  board.querySelectorAll('[data-gaze-target]').forEach(button => {
    button.addEventListener('webgazer-aac:dwell-progress', event => {
      const p = Number(event.detail && event.detail.progress) || 0;
      button.style.setProperty('--dwell', p);
      button.classList.toggle('active', p > 0);
    });
    button.addEventListener('webgazer-aac:dwell-cancel', event => {
      button.style.setProperty('--dwell', 0);
      button.classList.remove('active');
      record('dwell-cancel', { target: button.textContent.trim(), reason: event.detail && event.detail.reason });
    });
    button.addEventListener('webgazer-aac:dwell-complete', event => {
      button.style.setProperty('--dwell', 0);
      button.classList.remove('active');
      const target = button.textContent.trim();
      state.distinctDwellTargets.add(target);
      button.classList.add('validated');
      $('lastSelection').textContent = `Evidence: ${target}`;
      record('dwell-complete-evidence-only', {
        target,
        trackingQuality: event.detail && event.detail.trackingQuality,
        targetConfidence: event.detail && event.detail.targetConfidence,
        requiredMs: event.detail && event.detail.requiredMs,
        actionExecuted: false,
        distinctTargetCount: state.distinctDwellTargets.size,
        geometry: state.geometry.last ? { pose: state.geometry.last.pose, alignment: state.geometry.last.alignment } : null,
      });
      updateValidationState();
    });
  });
}
wireBoardEvidence();

$('startBtn').addEventListener('click', async () => {
  if (!window.webgazer || !window.webgazerAAC || state.started) return;
  if (!state.webgazerIdentity || !state.webgazerIdentity.verified || state.webgazerIdentity.version !== '3.5.3') {
    record('session-start-refused', { reason: 'webgazer-identity-not-verified', identity: state.webgazerIdentity });
    return;
  }
  setSession('starting');
  state.distinctDwellTargets.clear();
  state.lastLiveValidation = null;
  resetGeometryAndMotion();
  board.querySelectorAll('[data-gaze-target]').forEach(button => button.classList.remove('validated'));
  try {
    window.webgazerAAC.install().disableAdaptiveRecalibration().disableDriftWatchdog().resetDriftWatchdog();
    state.dwell = window.webgazerAAC.createDwellTimer({ dwellMs: 800, minTrackingQuality: 0.3, minTargetConfidence: 0.35 });
    window.webgazer.setGazeListener(gaze => {
      if (!gaze) return;
      const geometry = deriveGeometry();
      drawEyeGeometry(geometry);
      const filteredSaccade = filterMotion(gaze, geometry);
      const geometryQuality = geometry && geometry.valid ? (geometry.alignment && geometry.alignment.usable ? 1 : 0.75) : 0.35;
      const stableGaze = {
        ...gaze,
        isSaccade: filteredSaccade,
        trackingQuality: clamp(Number(gaze.trackingQuality || 0) * geometryQuality, 0, 1),
        rawIsSaccade: !!gaze.isSaccade,
        poseAware: true,
      };
      gazeDot.hidden = false;
      gazeDot.style.left = `${stableGaze.x}px`;
      gazeDot.style.top = `${stableGaze.y}px`;
      const resolved = window.webgazerAAC.resolveTarget(stableGaze.x, stableGaze.y, board);
      state.lastResolved = resolved;
      $('targetConfidence').textContent = resolved ? resolved.confidence.toFixed(2) : '0.00';
      if (window.webgazerAAC.isPCAFitted()) state.dwell.updateFromGaze(stableGaze, board);
    });
    await window.webgazer.begin(() => record('webgazer-begin-onfail', { stage: 'camera-or-init' }));
    state.started = true;
    $('startBtn').disabled = true;
    $('stopBtn').disabled = false;
    setSession('running');
    record('session-start', { runtime: window.webgazerAAC.version, webgazerIdentity: state.webgazerIdentity, poseAwareValidationLayer: '0.1' });
    calibrationProgress();
    updateDiagnostics();
    state.timer = setInterval(updateDiagnostics, 250);
  } catch (error) {
    record('session-start-error', errorEvidence(error));
    await cleanPartialSession();
  }
});

async function stopSession() {
  if (!state.started) return;
  const liveBeforeStop = validationSnapshot({ requireRunning: true });
  state.lastLiveValidation = deepCopy(liveBeforeStop);
  record('last-live-validation-capture', liveBeforeStop);
  clearInterval(state.timer); state.timer = null;
  try { if (window.webgazer && typeof window.webgazer.stopVideo === 'function') window.webgazer.stopVideo(); } catch (_) {}
  try { if (window.webgazer && typeof window.webgazer.end === 'function') await window.webgazer.end(); } catch (_) {}
  try { window.webgazerAAC.uninstall(); } catch (_) {}
  state.started = false;
  state.dwell = null;
  gazeDot.hidden = true;
  $('startBtn').disabled = !(window.webgazer && state.webgazerIdentity && state.webgazerIdentity.verified);
  $('stopBtn').disabled = true;
  setSession('stopped');
  record('session-stop', { diagnostics: getDiagnosticsSafe(), lastLiveValidation: state.lastLiveValidation, geometry: geometrySnapshot() });
  updateValidationState();
}
$('stopBtn').addEventListener('click', stopSession);

$('fitBtn').addEventListener('click', async () => {
  const result = window.webgazerAAC.fitUserBasis();
  const fitted = !!(result && result.rebuilt && window.webgazerAAC.isPCAFitted());
  $('calibrationStatus').textContent = fitted ? `PCA rebuilt from ${result.samples} calibration samples. Post-fit drift baseline reset.` : 'Not enough valid calibration evidence to fit PCA.';
  $('fitBtn').classList.remove('attention');
  record('fit-user-basis', { ...result, fitted });
  if (fitted) {
    window.webgazerAAC.resetDriftWatchdog().enableDriftWatchdog().enableAdaptiveRecalibration();
    record('drift-baseline-reset', { reason: 'PCA feature basis changed; pre-fit residuals are not comparable to post-fit residuals' });
    const saved = await window.webgazerAAC.saveCalibration();
    record('calibration-save', { saved: !!saved });
  }
  updateDiagnostics();
  updateValidationState();
});

$('resetBtn').addEventListener('click', async () => {
  const cleared = await window.webgazerAAC.clearAllCalibration();
  state.calibrationCounts.forEach((_, key) => state.calibrationCounts.set(key, 0));
  state.distinctDwellTargets.clear();
  state.lastLiveValidation = null;
  document.querySelectorAll('.calibrationPoint').forEach(p => p.classList.remove('done'));
  board.querySelectorAll('[data-gaze-target]').forEach(button => button.classList.remove('validated'));
  $('fitBtn').disabled = true;
  $('fitBtn').classList.remove('attention');
  $('fitState').textContent = 'PCA NOT FITTED';
  $('calibrationStatus').textContent = 'Calibration cleared.';
  resetGeometryAndMotion();
  record('calibration-clear', { cleared: !!cleared });
  updateValidationState();
});

function updateDiagnostics() {
  if (!window.webgazerAAC) return;
  const d = window.webgazerAAC.getDiagnostics();
  $('quality').textContent = Number(d.trackingQuality || 0).toFixed(2);
  $('drift').textContent = Number(d.driftRmse || 0).toFixed(1);
  $('fps').textContent = Number(d.effectiveListenerFps || d.effectiveGazeFps || d.gazeFps || 0).toFixed(1);
  $('coverage').textContent = `${Math.round(Number(d.eyeFeatureCoverage || 0) * 100)}%`;
  updateGeometryUi();
  updateValidationState();
}

$('downloadBtn').addEventListener('click', () => {
  const currentValidation = validationSnapshot();
  const evidence = {
    schema: 'webgazer-aac/browser-lab-evidence/0.3',
    exportedAt: new Date().toISOString(),
    runtimeVersion: window.webgazerAAC && window.webgazerAAC.version,
    webgazerIdentity: state.webgazerIdentity,
    webgazerRuntimeVersionProperty: window.webgazer && window.webgazer.version || null,
    diagnostics: getDiagnosticsSafe(),
    validation: currentValidation,
    lastLiveValidation: state.lastLiveValidation,
    poseAwareValidation: {
      version: '0.1',
      experimental: true,
      canonicalRuntimeModified: false,
      geometry: geometrySnapshot(),
      eyeLandmarkRois: { leftIndices: LEFT_EYE, rightIndices: RIGHT_EYE },
      motionFilter: { enterBasePxPerSec: 1700, enterConsecutiveFrames: 3, exitRatio: 0.48, exitConsecutiveFrames: 3 },
      note: 'FaceMesh establishes face geometry; tighter eye ROIs and pose-aware hysteresis are evaluated in the lab before promotion into the canonical runtime.',
    },
    calibrationProtocol: {
      points: points.length,
      requiredSamplesPerPoint: REQUIRED_SAMPLES_PER_POINT,
      capturedPerPoint: Array.from(state.calibrationCounts.entries()).map(([index, samples]) => ({ point: index + 1, samples })),
    },
    representativeTargets: Array.from(state.distinctDwellTargets).sort(),
    assurance: window.webgazerAAC && window.webgazerAAC.getAssuranceSnapshot && window.webgazerAAC.getAssuranceSnapshot(),
    lineage: window.webgazerAAC && window.webgazerAAC.getEvidenceLineage ? window.webgazerAAC.getEvidenceLineage() : [],
    events: state.events,
    authority: { dwellCompletionExecutedAction: false },
  };
  const blob = new Blob([JSON.stringify(evidence, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `webgazer-aac-evidence-${Date.now()}.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
  record('evidence-export', { events: state.events.length, liveStatePreserved: !!state.lastLiveValidation });
});

window.addEventListener('beforeunload', () => {
  if (state.started && window.webgazerAAC) {
    try { window.webgazerAAC.uninstall(); } catch (_) {}
  }
});
