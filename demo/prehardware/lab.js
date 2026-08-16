'use strict';

const $ = id => document.getElementById(id);
const state = { started: false, events: [], dwell: null, lastResolved: null, timer: null, webgazerIdentity: null };
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

function setSession(text) { $('sessionState').textContent = text; }

function runtimeStatus() {
  const aac = window.webgazerAAC;
  $('runtimeBadge').textContent = aac ? `AAC ${aac.version} · ${aac.featureVersion || aac._featureVersion || 'feature-v2'}` : 'AAC runtime missing';
}
runtimeStatus();

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
  $('wgStatus').textContent = matchesExpected
    ? `${verifiedVersion} verified`
    : `${verifiedVersion || 'unknown'} (expected 3.5.3)`;
  $('startBtn').disabled = !matchesExpected;
  $('useCodespaceBtn').disabled = true;
  record('webgazer-loaded', {
    runtimeVersion,
    verifiedVersion,
    expected: '3.5.3',
    matchesExpected,
    source,
    identity: state.webgazerIdentity,
  });
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
    record('webgazer-load-error', { message: String(error.message || error), source });
  }
}

async function getCodespaceIdentity() {
  const response = await fetch('/__webgazer__/status', { cache: 'no-store' });
  let status = null;
  try { status = await response.json(); } catch (_) {}
  if (!response.ok || !status || !status.verified) {
    throw new Error(status && status.reason || `Codespace WebGazer identity check failed (${response.status})`);
  }
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
      source: status.source,
    });
  } catch (error) {
    state.webgazerIdentity = null;
    $('useCodespaceBtn').disabled = true;
    record('codespace-webgazer-status-error', { message: String(error.message || error) });
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
    record('webgazer-identity-error', { message: String(error.message || error), source: 'codespace-build' });
  }
});

const points = [
  [0.1,0.12],[0.5,0.12],[0.9,0.12],
  [0.1,0.5],[0.5,0.5],[0.9,0.5],
  [0.1,0.88],[0.5,0.88],[0.9,0.88],
];
for (const [nx, ny] of points) {
  const p = document.createElement('button');
  p.className = 'calibrationPoint';
  p.type = 'button';
  p.dataset.nx = nx;
  p.dataset.ny = ny;
  p.setAttribute('aria-label', `Calibration point ${Math.round(nx*100)} ${Math.round(ny*100)}`);
  p.addEventListener('click', () => {
    if (!state.started || !window.webgazer) return;
    const r = p.getBoundingClientRect();
    const x = r.left + r.width/2, y = r.top + r.height/2;
    window.webgazer.recordScreenPosition(x, y, 'click');
    p.classList.add('done');
    record('calibration-sample', { x: Math.round(x), y: Math.round(y) });
    const done = document.querySelectorAll('.calibrationPoint.done').length;
    $('calibrationStatus').textContent = `${done}/${points.length} points recorded.`;
    $('fitBtn').disabled = done < points.length;
  });
  $('calibrationGrid').appendChild(p);
}

$('wgFile').addEventListener('change', async event => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  if (state.started) return record('load-refused', { reason: 'stop-session-first' });
  try {
    const source = await file.text();
    const script = document.createElement('script');
    script.textContent = source;
    document.head.appendChild(script);
    finishWebGazerLoad('uploaded-file');
  } catch (error) {
    $('wgStatus').textContent = 'load failed';
    record('webgazer-load-error', { message: String(error.message || error), source: 'uploaded-file' });
  }
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
      $('lastSelection').textContent = `Evidence: ${button.textContent.trim()}`;
      record('dwell-complete-evidence-only', {
        target: button.textContent.trim(),
        trackingQuality: event.detail && event.detail.trackingQuality,
        targetConfidence: event.detail && event.detail.targetConfidence,
        requiredMs: event.detail && event.detail.requiredMs,
        actionExecuted: false,
      });
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
    await window.webgazer.begin();
    state.started = true;
    $('startBtn').disabled = true;
    $('stopBtn').disabled = false;
    $('wgFile').disabled = true;
    setSession('running');
    record('session-start', { runtime: window.webgazerAAC.version, webgazerIdentity: state.webgazerIdentity });
    state.timer = setInterval(updateDiagnostics, 250);
  } catch (error) {
    record('session-start-error', { message: String(error.message || error) });
  }
});

async function stopSession() {
  if (!state.started) return;
  clearInterval(state.timer); state.timer = null;
  try { if (window.webgazer && typeof window.webgazer.end === 'function') await window.webgazer.end(); } catch (_) {}
  try { window.webgazerAAC.uninstall(); } catch (_) {}
  state.started = false;
  gazeDot.hidden = true;
  $('startBtn').disabled = !(window.webgazer && state.webgazerIdentity && state.webgazerIdentity.verified);
  $('stopBtn').disabled = true;
  $('wgFile').disabled = false;
  setSession('stopped');
  record('session-stop', { diagnostics: window.webgazerAAC.getDiagnostics() });
}
$('stopBtn').addEventListener('click', stopSession);

$('fitBtn').addEventListener('click', async () => {
  const result = window.webgazerAAC.fitUserBasis();
  $('calibrationStatus').textContent = result.rebuilt ? `PCA rebuilt from ${result.samples} calibration samples.` : 'Not enough valid calibration evidence to fit PCA.';
  record('fit-user-basis', result);
  if (result.rebuilt) {
    const saved = await window.webgazerAAC.saveCalibration();
    record('calibration-save', { saved: !!saved });
  }
});

$('resetBtn').addEventListener('click', async () => {
  const cleared = await window.webgazerAAC.clearAllCalibration();
  document.querySelectorAll('.calibrationPoint').forEach(p => p.classList.remove('done'));
  $('fitBtn').disabled = true;
  $('calibrationStatus').textContent = 'Calibration cleared.';
  record('calibration-clear', { cleared: !!cleared });
});

function updateDiagnostics() {
  if (!window.webgazerAAC) return;
  const d = window.webgazerAAC.getDiagnostics();
  $('quality').textContent = Number(d.trackingQuality || 0).toFixed(2);
  $('drift').textContent = Number(d.driftRmse || 0).toFixed(1);
  $('fps').textContent = Number(d.effectiveGazeFps || d.gazeFps || 0).toFixed(1);
  $('coverage').textContent = `${Math.round(Number(d.eyeFeatureCoverage || 0) * 100)}%`;
}

$('downloadBtn').addEventListener('click', () => {
  const evidence = {
    schema: 'webgazer-aac/browser-lab-evidence/0.1',
    exportedAt: new Date().toISOString(),
    runtimeVersion: window.webgazerAAC && window.webgazerAAC.version,
    webgazerIdentity: state.webgazerIdentity,
    webgazerRuntimeVersionProperty: window.webgazer && window.webgazer.version || null,
    diagnostics: window.webgazerAAC && window.webgazerAAC.getDiagnostics(),
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
  record('evidence-export', { events: state.events.length });
});

window.addEventListener('beforeunload', () => { if (state.started) window.webgazerAAC.uninstall(); });
