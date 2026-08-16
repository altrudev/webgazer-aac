'use strict';

const $ = id => document.getElementById(id);
const REQUIRED_SAMPLES_PER_POINT = 2;
const REQUIRED_DISTINCT_DWELL_TARGETS = 3;
const MIN_EYE_FEATURE_COVERAGE = 0.95;
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
};
const logEl = $('log');
const board = $('aacBoard');
const gazeDot = $('gazeDot');

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

function validationSnapshot() {
  const d = getDiagnosticsSafe() || {};
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
  if (!state.started) reasons.push('session not running');
  if (eyeFeatureCoverage < MIN_EYE_FEATURE_COVERAGE) reasons.push(`eye-feature coverage below ${Math.round(MIN_EYE_FEATURE_COVERAGE * 100)}%`);
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
      pcaFitted,
      driftLevel,
      driftRmse,
      requiredDistinctDwellTargets: REQUIRED_DISTINCT_DWELL_TARGETS,
      distinctDwellTargets: distinctTargets,
      distinctDwellTargetCount: distinctTargets.length,
      dwellExecutesAction: false,
    },
  };
}

function updateValidationState() {
  const v = validationSnapshot();
  state.lastValidation = v;
  if (state.started) state.lastLiveValidation = JSON.parse(JSON.stringify(v));
  setGate('identityGate', v.criteria.identityVerified, v.criteria.identityVerified ? 'VERIFIED' : 'NOT VERIFIED');
  setGate('mediaPipeGate', v.criteria.mediaPipeVerified, v.criteria.mediaPipeVerified ? 'VERIFIED' : 'NOT VERIFIED');
  setGate('sessionGate', v.criteria.sessionRunning, v.criteria.sessionRunning ? 'RUNNING' : 'NOT RUNNING');
  setGate('pcaGate', v.criteria.pcaFitted, v.criteria.pcaFitted ? 'FITTED' : 'NOT FITTED');
  setGate('targetGate', v.criteria.distinctDwellTargetCount >= REQUIRED_DISTINCT_DWELL_TARGETS,
    `${v.criteria.distinctDwellTargetCount} / ${REQUIRED_DISTINCT_DWELL_TARGETS}`);
  setGate('readinessGate', v.ready, v.ready ? 'READY' : 'NOT READY');
  $('readinessReason').textContent = v.ready ? 'All browser-beta gates satisfied.' : v.reasons.join(' · ');
  $('fitState').textContent = v.criteria.pcaFitted ? 'PCA FITTED' : 'PCA NOT FITTED';
  return v;
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
updateValidationState();

function finishWebGazerLoad(source, identity = null) {
  if (!window.webgazer) throw new Error('Loaded script did not expose window.webgazer');
  const runtimeVersion = window.webgazer.version || null;
  const verifiedVersion = identity && identity.verified ? identity.version : runtimeVersion;
  const matchesExpected = verifiedVersion === '3.5.3';
  state.webgazerIdentity = identity || {
    source,
    verified: matchesExpected,
    version: runtimeVersion,
    expectedVersion: '3.5.3',
    bundleBytes: null,
    bundleSha256: null,
  };
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
    record('calibration-sample', { point: index + 1, sample: next, requiredPerPoint: REQUIRED_SAMPLES_PER_POINT, x: Math.round(x), y: Math.round(y) });
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
  board.querySelectorAll('[data-gaze-target]').forEach(button => button.classList.remove('validated'));
  try {
    window.webgazerAAC.install().enableAdaptiveRecalibration().enableDriftWatchdog();
    state.dwell = window.webgazerAAC.createDwellTimer({ dwellMs: 800, minTrackingQuality: 0.3, minTargetConfidence: 0.35 });
    window.webgazer.setGazeListener(gaze => {
      if (!gaze) return;
      gazeDot.hidden = false;
      gazeDot.style.left = `${gaze.x}px`;
      gazeDot.style.top = `${gaze.y}px`;
      const resolved = window.webgazerAAC.resolveTarget(gaze.x, gaze.y, board);
      state.lastResolved = resolved;
      $('targetConfidence').textContent = resolved ? resolved.confidence.toFixed(2) : '0.00';
      state.dwell.updateFromGaze(gaze, board);
    });
    await window.webgazer.begin(() => record('webgazer-begin-onfail', { stage: 'camera-or-init' }));
    state.started = true;
    $('startBtn').disabled = true;
    $('stopBtn').disabled = false;
    setSession('running');
    record('session-start', { runtime: window.webgazerAAC.version, webgazerIdentity: state.webgazerIdentity });
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
  const liveValidation = updateValidationState();
  state.lastLiveValidation = JSON.parse(JSON.stringify(liveValidation));
  record('last-live-validation-capture', state.lastLiveValidation);
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
  record('session-stop', { diagnostics: window.webgazerAAC.getDiagnostics(), lastLiveValidation: state.lastLiveValidation });
}
$('stopBtn').addEventListener('click', stopSession);

$('fitBtn').addEventListener('click', async () => {
  const result = window.webgazerAAC.fitUserBasis();
  const fitted = !!result.rebuilt;
  $('fitBtn').classList.toggle('attention', !fitted);
  $('fitBtn').textContent = fitted ? 'PCA basis fitted' : 'FIT USER BASIS — RETRY';
  $('fitBtn').disabled = fitted;
  $('calibrationStatus').textContent = fitted
    ? `PCA fitted and regression rebuilt from ${result.samples} calibration samples.`
    : `PCA fit failed with ${result.samples} calibration samples. Keep your gaze steady and recapture calibration.`;
  record('fit-user-basis', { ...result, fitted });
  if (fitted) {
    const saved = await window.webgazerAAC.saveCalibration();
    record('calibration-save', { saved: !!saved });
  }
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
  $('fitBtn').textContent = 'Fit user basis';
  $('calibrationStatus').textContent = 'Calibration cleared. Click each point twice while looking directly at it.';
  record('calibration-clear', { cleared: !!cleared });
  updateValidationState();
});

function updateDiagnostics() {
  if (!window.webgazerAAC) return;
  const d = window.webgazerAAC.getDiagnostics();
  $('quality').textContent = Number(d.trackingQuality || 0).toFixed(2);
  $('drift').textContent = `${Number(d.driftRmse || 0).toFixed(1)} · ${d.driftLevel || 'disabled'}`;
  $('fps').textContent = Number(d.effectiveListenerFps || d.effectiveGazeFps || d.gazeFps || 0).toFixed(1);
  $('coverage').textContent = `${Math.round(Number(d.eyeFeatureCoverage || 0) * 100)}%`;
  updateValidationState();
}

$('downloadBtn').addEventListener('click', () => {
  const validation = updateValidationState();
  const evidence = {
    schema: 'webgazer-aac/browser-lab-evidence/0.3',
    exportedAt: new Date().toISOString(),
    runtimeVersion: window.webgazerAAC && window.webgazerAAC.version,
    webgazerIdentity: state.webgazerIdentity,
    webgazerRuntimeVersionProperty: window.webgazer && window.webgazer.version || null,
    diagnostics: window.webgazerAAC && window.webgazerAAC.getDiagnostics(),
    validation,
    lastLiveValidation: state.lastLiveValidation,
    calibrationProtocol: {
      points: points.length,
      requiredSamplesPerPoint: REQUIRED_SAMPLES_PER_POINT,
      capturedPerPoint: Array.from(state.calibrationCounts.entries()).map(([point, samples]) => ({ point: point + 1, samples })),
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
  record('evidence-export', {
    events: state.events.length,
    browserBetaReady: validation.ready,
    lastLiveBrowserBetaReady: !!(state.lastLiveValidation && state.lastLiveValidation.ready),
    representativeTargets: evidence.representativeTargets,
  });
});

window.addEventListener('beforeunload', () => { if (state.started) window.webgazerAAC.uninstall(); });
