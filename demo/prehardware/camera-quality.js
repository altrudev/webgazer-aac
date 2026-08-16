'use strict';

(function () {
  const DURATION_MS = 8000;
  const SAMPLE_STRIDE = 4;
  const profiles = { A: null, B: null };
  let active = null;
  let patched = false;
  let centerMarker = null;

  function el(tag, attrs = {}, text = '') {
    const node = document.createElement(tag);
    Object.entries(attrs).forEach(([key, value]) => {
      if (key === 'class') node.className = value;
      else node.setAttribute(key, value);
    });
    if (text) node.textContent = text;
    return node;
  }

  function pct(v) { return `${(Math.max(0, Math.min(1, v || 0)) * 100).toFixed(1)}%`; }
  function num(v, digits = 2) { return Number.isFinite(v) ? v.toFixed(digits) : '—'; }
  function mean(values) { return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0; }
  function stdev(values) {
    if (values.length < 2) return 0;
    const m = mean(values);
    return Math.sqrt(values.reduce((a, v) => a + (v - m) * (v - m), 0) / values.length);
  }
  function percentile(values, q) {
    if (!values.length) return 0;
    const a = values.slice().sort((x, y) => x - y);
    const pos = Math.max(0, Math.min(a.length - 1, Math.round((a.length - 1) * q)));
    return a[pos];
  }

  function readEye(eye) {
    if (!eye) return null;
    const src = eye.patch || eye;
    if (!src || !src.data || !src.width || !src.height) return null;
    const d = src.data;
    const rgba = d.length >= src.width * src.height * 4;
    const luminance = [];
    let bright = 0, clipped = 0, dark = 0, sampled = 0;
    const step = rgba ? 4 * SAMPLE_STRIDE : SAMPLE_STRIDE;
    for (let i = 0; i < d.length; i += step) {
      let y;
      if (rgba) {
        const r = d[i] || 0, g = d[i + 1] || 0, b = d[i + 2] || 0;
        y = 0.299 * r + 0.587 * g + 0.114 * b;
      } else y = Number(d[i]) || 0;
      luminance.push(y);
      if (y >= 235) bright++;
      if (y >= 250) clipped++;
      if (y <= 8) dark++;
      sampled++;
    }
    if (!sampled) return null;
    const p10 = percentile(luminance, 0.10);
    const p90 = percentile(luminance, 0.90);
    return {
      meanLuma: mean(luminance),
      specularFraction: bright / sampled,
      clippedFraction: clipped / sampled,
      darkFraction: dark / sampled,
      contrast: (p90 - p10) / 255,
      centerX: Number(eye.imagex || 0) + Number(eye.width || src.width || 0) / 2,
      centerY: Number(eye.imagey || 0) + Number(eye.height || src.height || 0) / 2,
      width: Number(eye.width || src.width || 0),
      height: Number(eye.height || src.height || 0),
    };
  }

  function observe(gaze) {
    if (!active) return;
    active.frames++;
    const f = gaze && gaze.eyeFeatures;
    if (!f || !f.left || !f.right) return;
    const left = readEye(f.left);
    const right = readEye(f.right);
    if (!left || !right) return;
    active.eyeFrames++;
    const both = [left, right];
    active.glare.push(mean(both.map(x => x.specularFraction)));
    active.clipped.push(mean(both.map(x => x.clippedFraction)));
    active.dark.push(mean(both.map(x => x.darkFraction)));
    active.contrast.push(mean(both.map(x => x.contrast)));
    active.luma.push(mean(both.map(x => x.meanLuma)));
    active.leftX.push(left.centerX); active.leftY.push(left.centerY);
    active.rightX.push(right.centerX); active.rightY.push(right.centerY);
    active.scale.push((left.width + right.width + left.height + right.height) / 4);
  }

  function summarize(run) {
    const eyeCentersJitter = mean([
      Math.hypot(stdev(run.leftX), stdev(run.leftY)),
      Math.hypot(stdev(run.rightX), stdev(run.rightY)),
    ]);
    const scaleMean = mean(run.scale) || 1;
    const coverage = run.frames ? run.eyeFrames / run.frames : 0;
    const glareP95 = percentile(run.glare, 0.95);
    const clippedP95 = percentile(run.clipped, 0.95);
    const contrastMedian = percentile(run.contrast, 0.50);
    const score = Math.max(0, Math.min(100,
      coverage * 40 +
      (1 - Math.min(1, glareP95 / 0.12)) * 20 +
      (1 - Math.min(1, clippedP95 / 0.05)) * 10 +
      (1 - Math.min(1, eyeCentersJitter / 8)) * 20 +
      Math.min(1, contrastMedian / 0.35) * 10
    ));
    return {
      durationMs: Date.now() - run.startedAt,
      frames: run.frames,
      eyeFeatureFrames: run.eyeFrames,
      eyeFeatureCoverage: coverage,
      glareSpecularP95: glareP95,
      clippedHighlightP95: clippedP95,
      darkFractionP95: percentile(run.dark, 0.95),
      contrastMedian,
      meanLuminance: mean(run.luma),
      eyeCenterJitterPx: eyeCentersJitter,
      eyeScaleCv: stdev(run.scale) / scaleMean,
      comparisonScore: score,
      interpretation: 'Heuristic camera-position comparison only; not a clinical or gaze-accuracy score.'
    };
  }

  function renderProfile(label) {
    const r = profiles[label];
    const out = document.getElementById(`cameraResult${label}`);
    if (!out) return;
    if (!r) { out.textContent = 'not measured'; return; }
    out.textContent = `score ${num(r.comparisonScore, 0)} · eyes ${pct(r.eyeFeatureCoverage)} · glare ${pct(r.glareSpecularP95)} · clipped ${pct(r.clippedHighlightP95)} · jitter ${num(r.eyeCenterJitterPx, 1)} px · contrast ${pct(r.contrastMedian)}`;
  }

  function renderRecommendation() {
    const out = document.getElementById('cameraRecommendation');
    if (!out) return;
    if (!profiles.A || !profiles.B) {
      out.textContent = 'Measure both positions under the same room and screen lighting.';
      return;
    }
    const a = profiles.A, b = profiles.B;
    const winner = a.comparisonScore >= b.comparisonScore ? 'A' : 'B';
    const w = profiles[winner];
    const other = profiles[winner === 'A' ? 'B' : 'A'];
    const difference = Math.abs(w.comparisonScore - other.comparisonScore);
    out.textContent = difference < 4
      ? 'A and B are effectively close. Prefer the position with visibly less pupil/corneal reflection and the more comfortable head posture.'
      : `Position ${winner} is the better current candidate by the local comparison (${num(w.comparisonScore, 0)} vs ${num(other.comparisonScore, 0)}). Recheck after changing room or screen lighting.`;
  }

  function stopMeasurement(label) {
    if (!active || active.label !== label) return;
    profiles[label] = summarize(active);
    active = null;
    if (centerMarker) centerMarker.hidden = true;
    const a = document.getElementById('measureCameraA');
    const b = document.getElementById('measureCameraB');
    if (a) a.disabled = false;
    if (b) b.disabled = false;
    renderProfile(label);
    renderRecommendation();
  }

  function startMeasurement(label) {
    if (!window.webgazer || !document.getElementById('webgazerVideoFeed')) {
      const out = document.getElementById('cameraRecommendation');
      if (out) out.textContent = 'Start the verified camera session first.';
      return;
    }
    active = {
      label, startedAt: Date.now(), frames: 0, eyeFrames: 0,
      glare: [], clipped: [], dark: [], contrast: [], luma: [],
      leftX: [], leftY: [], rightX: [], rightY: [], scale: []
    };
    if (centerMarker) centerMarker.hidden = false;
    const a = document.getElementById('measureCameraA');
    const b = document.getElementById('measureCameraB');
    if (a) a.disabled = true;
    if (b) b.disabled = true;
    const out = document.getElementById('cameraRecommendation');
    if (out) out.textContent = `Measuring position ${label} for 8 seconds. Look at the center marker and keep your head naturally still.`;
    setTimeout(() => stopMeasurement(label), DURATION_MS);
  }

  function patchWebGazer() {
    if (patched || !window.webgazer || typeof window.webgazer.setGazeListener !== 'function') return;
    const original = window.webgazer.setGazeListener.bind(window.webgazer);
    window.webgazer.setGazeListener = function (listener) {
      return original(function (gaze, elapsed) {
        try { observe(gaze); } catch (_) {}
        return listener(gaze, elapsed);
      });
    };
    patched = true;
  }

  function mount() {
    if (document.getElementById('cameraQualityPanel')) return;
    const panel = el('section', { id: 'cameraQualityPanel', class: 'panel' });
    const head = el('div', { class: 'panelHead' });
    const titleWrap = el('div');
    titleWrap.appendChild(el('p', { class: 'eyebrow' }, 'CAMERA / GLARE COMPARISON'));
    titleWrap.appendChild(el('h2', {}, 'Choose camera position from evidence'));
    head.appendChild(titleWrap);
    panel.appendChild(head);

    const controls = el('div', { class: 'controls' });
    const a = el('button', { id: 'measureCameraA', type: 'button' }, 'Measure position A');
    const b = el('button', { id: 'measureCameraB', type: 'button' }, 'Measure position B');
    const dl = el('button', { id: 'downloadCameraComparison', type: 'button' }, 'Download camera comparison');
    controls.append(a, b, dl);
    panel.appendChild(controls);

    const grid = el('div', { class: 'cameraComparisonGrid' });
    const cardA = el('article'); cardA.append(el('span', {}, 'Position A'), el('strong', { id: 'cameraResultA' }, 'not measured'));
    const cardB = el('article'); cardB.append(el('span', {}, 'Position B'), el('strong', { id: 'cameraResultB' }, 'not measured'));
    grid.append(cardA, cardB);
    panel.appendChild(grid);
    panel.appendChild(el('p', { id: 'cameraRecommendation', class: 'hint' }, 'Measure both positions under the same room and screen lighting.'));
    panel.appendChild(el('p', { class: 'hint' }, 'Glare is estimated from bright/clipped pixels inside WebGazer eye patches. This is a comparative quality signal, not proof that a highlight lies on the pupil. No eye images or raw patches are saved.'));

    const validation = document.querySelector('.validationGrid');
    if (validation && validation.parentNode) validation.parentNode.insertBefore(panel, validation);
    else document.querySelector('main').appendChild(panel);

    const style = el('style');
    style.textContent = '.cameraComparisonGrid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:8px}.cameraComparisonGrid article{border:1px solid #e2e8f0;border-radius:12px;padding:14px}.cameraComparisonGrid span{display:block;font-size:12px;color:#64748b}.cameraComparisonGrid strong{display:block;font-size:14px;margin-top:5px;line-height:1.35}.cameraCenterMarker{position:fixed;left:50%;top:50%;width:24px;height:24px;border:4px solid #111827;border-radius:50%;transform:translate(-50%,-50%);z-index:5000;pointer-events:none;background:rgba(255,255,255,.7)}@media(max-width:900px){.cameraComparisonGrid{grid-template-columns:1fr}}';
    document.head.appendChild(style);

    centerMarker = el('div', { class: 'cameraCenterMarker', 'aria-hidden': 'true' });
    centerMarker.hidden = true;
    document.body.appendChild(centerMarker);

    a.addEventListener('click', () => startMeasurement('A'));
    b.addEventListener('click', () => startMeasurement('B'));
    dl.addEventListener('click', () => {
      const evidence = {
        schema: 'webgazer-aac/camera-quality-comparison/0.1',
        exportedAt: new Date().toISOString(),
        localOnly: true,
        rawEyeMaterialPersisted: false,
        profiles,
        recommended: profiles.A && profiles.B ? (profiles.A.comparisonScore >= profiles.B.comparisonScore ? 'A' : 'B') : null,
      };
      const blob = new Blob([JSON.stringify(evidence, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `webgazer-camera-comparison-${Date.now()}.json`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    });
  }

  mount();
  const poll = setInterval(() => {
    patchWebGazer();
    if (patched) clearInterval(poll);
  }, 100);

  window.webgazerCameraQuality = {
    getProfiles: () => JSON.parse(JSON.stringify(profiles)),
    startA: () => startMeasurement('A'),
    startB: () => startMeasurement('B'),
  };
})();
