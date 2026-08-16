/**
 * webgazer-aac.js v2.0.0
 * Local-first AAC reliability and gaze interaction layer for WebGazer.js 3.5.3.
 *
 * GPLv3 — ALTRU.dev / Code for Humanity
 *
 * Design invariants:
 *   - consume the exact WebGazer data.eyeFeatures used for each prediction
 *   - never mix regression vectors encoded in different PCA feature spaces
 *   - never recycle the model's gaze coordinate as independent dwell ground truth
 *   - keep tracking quality, target confidence, and supervision provenance separate
 *   - perform no network requests or telemetry
 */
(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.webgazerAAC = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (global) {
  'use strict';

  const LIBRARY_VERSION = '2.0.0';
  const SCHEMA_VERSION = 2;
  const FEATURE_VERSION = 'wg-aac-v2:norm40x24:pca10:poly2';
  const NORM_W = 40;
  const NORM_H = 24;
  const PATCH_DIM = NORM_W * NORM_H;
  const PCA_COMPONENTS = 10;
  const RIDGE_LAMBDA = 1e-3;

  const EVIDENCE_WEIGHTS = Object.freeze({
    calibration: 1.0,
    explicit: 1.0,
    'confirmed-click': 0.85,
    'dwell-selection': 0.35,
    inferred: 0.15,
  });

  function perfNow() {
    return global.performance && typeof global.performance.now === 'function'
      ? global.performance.now()
      : Date.now();
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function finite(v, fallback) { return Number.isFinite(v) ? v : fallback; }
  function viewportWidth() { return finite(global.innerWidth, 0); }
  function viewportHeight() { return finite(global.innerHeight, 0); }
  function devicePixelRatio() { return finite(global.devicePixelRatio, 1) || 1; }

  function dispatch(target, name, detail) {
    if (!target || typeof target.dispatchEvent !== 'function' || typeof global.CustomEvent === 'undefined') return;
    try { target.dispatchEvent(new global.CustomEvent(name, { detail, bubbles: true })); } catch (_) {}
  }

  function patchPixels(patchLike) {
    if (!patchLike) return null;
    const patch = patchLike.patch || patchLike;
    try {
      if (patch.data && patch.width && patch.height) {
        return { data: patch.data, width: patch.width, height: patch.height };
      }
      if (typeof global.HTMLCanvasElement !== 'undefined' && patch instanceof global.HTMLCanvasElement) {
        const ctx = patch.getContext('2d', { willReadFrequently: true });
        const id = ctx.getImageData(0, 0, patch.width, patch.height);
        return { data: id.data, width: id.width, height: id.height };
      }
    } catch (_) {}
    return null;
  }

  function grayscalePatch(patchLike) {
    const p = patchPixels(patchLike);
    if (!p || !p.width || !p.height) return null;
    const out = new Float32Array(p.width * p.height);
    if (p.data.length >= out.length * 4) {
      for (let i = 0, j = 0; j < out.length; i += 4, j++) {
        out[j] = 0.299 * p.data[i] + 0.587 * p.data[i + 1] + 0.114 * p.data[i + 2];
      }
    } else {
      for (let i = 0; i < out.length; i++) out[i] = finite(p.data[i], 0);
    }
    return { grey: out, width: p.width, height: p.height };
  }

  function resizeBilinear(src, sw, sh, dw, dh) {
    const out = new Float32Array(dw * dh);
    if (!src || !sw || !sh) return out;
    for (let y = 0; y < dh; y++) {
      const sy = ((y + 0.5) * sh / dh) - 0.5;
      const y0 = clamp(Math.floor(sy), 0, sh - 1);
      const y1 = clamp(y0 + 1, 0, sh - 1);
      const fy = clamp(sy - y0, 0, 1);
      for (let x = 0; x < dw; x++) {
        const sx = ((x + 0.5) * sw / dw) - 0.5;
        const x0 = clamp(Math.floor(sx), 0, sw - 1);
        const x1 = clamp(x0 + 1, 0, sw - 1);
        const fx = clamp(sx - x0, 0, 1);
        const a = src[y0 * sw + x0] * (1 - fx) + src[y0 * sw + x1] * fx;
        const b = src[y1 * sw + x0] * (1 - fx) + src[y1 * sw + x1] * fx;
        out[y * dw + x] = a * (1 - fy) + b * fy;
      }
    }
    return out;
  }

  function normalizeContrast(values) {
    let sum = 0;
    for (let i = 0; i < values.length; i++) sum += values[i];
    const mean = sum / Math.max(1, values.length);
    let ss = 0;
    for (let i = 0; i < values.length; i++) {
      const d = values[i] - mean;
      ss += d * d;
    }
    const sd = Math.sqrt(ss / Math.max(1, values.length)) || 1;
    const out = new Float32Array(values.length);
    for (let i = 0; i < values.length; i++) out[i] = clamp((values[i] - mean) / (sd * 3), -1, 1);
    return out;
  }

  function normalizeEyeFeatures(eyeFeatures) {
    if (!eyeFeatures || !eyeFeatures.left || !eyeFeatures.right) return null;
    const l = grayscalePatch(eyeFeatures.left);
    const r = grayscalePatch(eyeFeatures.right);
    if (!l || !r) return null;
    return {
      left: normalizeContrast(resizeBilinear(l.grey, l.width, l.height, NORM_W, NORM_H)),
      right: normalizeContrast(resizeBilinear(r.grey, r.width, r.height, NORM_W, NORM_H)),
    };
  }

  function meanBrightness(eyeFeatures) {
    if (!eyeFeatures || !eyeFeatures.left || !eyeFeatures.right) return null;
    const l = grayscalePatch(eyeFeatures.left);
    const r = grayscalePatch(eyeFeatures.right);
    if (!l || !r) return null;
    let sum = 0;
    let count = 0;
    for (let i = 0; i < l.grey.length; i++) { sum += l.grey[i]; count++; }
    for (let i = 0; i < r.grey.length; i++) { sum += r.grey[i]; count++; }
    return count ? sum / count : null;
  }

  function cloneNormalized(n) {
    if (!n) return null;
    return { left: Float32Array.from(n.left), right: Float32Array.from(n.right) };
  }

  function serializeNormalized(n) {
    if (!n) return null;
    return { left: Array.from(n.left), right: Array.from(n.right) };
  }

  function deserializeNormalized(n) {
    if (!n || !n.left || !n.right) return null;
    return { left: Float32Array.from(n.left), right: Float32Array.from(n.right) };
  }

  function makeFallbackBasis(dim, k, seed) {
    let s = seed >>> 0;
    const basis = [];
    const rand = () => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return (s / 0xffffffff) - 0.5;
    };
    for (let c = 0; c < k; c++) {
      const v = new Float64Array(dim);
      let mag = 0;
      for (let i = 0; i < dim; i++) { v[i] = rand(); mag += v[i] * v[i]; }
      mag = Math.sqrt(mag) || 1;
      for (let i = 0; i < dim; i++) v[i] /= mag;
      basis.push(Array.from(v));
    }
    return basis;
  }

  function PCABasis(dim, k, seed) {
    this.dim = dim;
    this.k = k;
    this.seed = seed >>> 0;
    this.fitted = false;
    this.mean = null;
    this.basis = makeFallbackBasis(dim, k, this.seed);
  }

  PCABasis.prototype.reset = function () {
    this.fitted = false;
    this.mean = null;
    this.basis = makeFallbackBasis(this.dim, this.k, this.seed);
  };

  PCABasis.prototype.fit = function (patches) {
    if (!Array.isArray(patches) || patches.length < this.k + 2) return false;
    const n = patches.length;
    const P = patches.map(p => {
      const v = new Float64Array(this.dim);
      const len = Math.min(this.dim, p.length || 0);
      for (let i = 0; i < len; i++) v[i] = finite(Number(p[i]), 0);
      return v;
    });
    const mean = new Float64Array(this.dim);
    for (const p of P) for (let d = 0; d < this.dim; d++) mean[d] += p[d] / n;
    for (const p of P) for (let d = 0; d < this.dim; d++) p[d] -= mean[d];

    const S = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        let dot = 0;
        for (let d = 0; d < this.dim; d++) dot += P[i][d] * P[j][d];
        S[i][j] = S[j][i] = dot / n;
      }
    }
    const M = S.map(row => Float64Array.from(row));
    const basis = [];
    for (let e = 0; e < this.k; e++) {
      let u = Float64Array.from({ length: n }, (_, i) => Math.sin((i + 1) * 1.618 + e * 2.17));
      for (let iter = 0; iter < 45; iter++) {
        const next = new Float64Array(n);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) next[i] += M[i][j] * u[j];
        let mag = 0;
        for (let i = 0; i < n; i++) mag += next[i] * next[i];
        mag = Math.sqrt(mag);
        if (mag < 1e-12) break;
        for (let i = 0; i < n; i++) next[i] /= mag;
        u = next;
      }
      const v = new Float64Array(this.dim);
      let vmag = 0;
      for (let d = 0; d < this.dim; d++) {
        let x = 0;
        for (let i = 0; i < n; i++) x += P[i][d] * u[i];
        v[d] = x;
        vmag += x * x;
      }
      vmag = Math.sqrt(vmag);
      if (vmag < 1e-12) {
        const fallback = makeFallbackBasis(this.dim, 1, this.seed + e * 977)[0];
        basis.push(fallback);
        continue;
      }
      for (let d = 0; d < this.dim; d++) v[d] /= vmag;
      basis.push(Array.from(v));
      let lambda = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) lambda += u[i] * M[i][j] * u[j];
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i][j] -= lambda * u[i] * u[j];
    }
    this.mean = mean;
    this.basis = basis;
    this.fitted = true;
    return true;
  };

  PCABasis.prototype.project = function (values) {
    const v = values || [];
    const out = new Array(this.basis.length);
    for (let b = 0; b < this.basis.length; b++) {
      const basis = this.basis[b];
      let dot = 0;
      for (let i = 0; i < this.dim; i++) {
        const x = finite(Number(v[i]), 0) - (this.mean ? this.mean[i] : 0);
        dot += basis[i] * x;
      }
      out[b] = dot;
    }
    return out;
  };

  const LEFT_PCA = new PCABasis(PATCH_DIM, PCA_COMPONENTS, 0xdeadbeef);
  const RIGHT_PCA = new PCABasis(PATCH_DIM, PCA_COMPONENTS, 0xcafebabe);

  function featureVector(normalized) {
    if (!normalized || !normalized.left || !normalized.right) return null;
    const components = LEFT_PCA.project(normalized.left).concat(RIGHT_PCA.project(normalized.right));
    const features = [1];
    for (const c of components) features.push(c);
    for (let i = 0; i < components.length; i++) {
      for (let j = i; j < components.length; j++) features.push(components[i] * components[j]);
    }
    return features;
  }

  function gaussianSolve(A, b) {
    const n = b.length;
    const M = Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(n + 1);
      for (let j = 0; j < n; j++) row[j] = finite(Number(A[i][j]), 0);
      row[n] = finite(Number(b[i]), 0);
      return row;
    });
    for (let col = 0; col < n; col++) {
      let pivot = col;
      let max = Math.abs(M[col][col]);
      for (let r = col + 1; r < n; r++) {
        const v = Math.abs(M[r][col]);
        if (v > max) { max = v; pivot = r; }
      }
      if (pivot !== col) { const tmp = M[col]; M[col] = M[pivot]; M[pivot] = tmp; }
      if (Math.abs(M[col][col]) < 1e-10) M[col][col] += 1e-8;
      const div = M[col][col];
      if (Math.abs(div) < 1e-14) continue;
      for (let j = col; j <= n; j++) M[col][j] /= div;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = M[r][col];
        if (!f) continue;
        for (let j = col; j <= n; j++) M[r][j] -= f * M[col][j];
      }
    }
    return Array.from({ length: n }, (_, i) => finite(M[i][n], 0));
  }

  function ridgeSolve(X, y, weights, lambda) {
    if (!X.length) return null;
    const m = X[0].length;
    const XtX = Array.from({ length: m }, () => new Float64Array(m));
    const Xty = new Float64Array(m);
    for (let k = 0; k < X.length; k++) {
      const row = X[k];
      const w = Math.max(1e-8, weights ? finite(weights[k], 1) : 1);
      for (let i = 0; i < m; i++) {
        const xi = row[i] * w;
        Xty[i] += xi * y[k];
        for (let j = i; j < m; j++) XtX[i][j] += xi * row[j];
      }
    }
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < i; j++) XtX[i][j] = XtX[j][i];
      XtX[i][i] += lambda;
    }
    return gaussianSolve(XtX, Xty);
  }

  function PolynomialRegression() {
    this.xSamples = [];
    this.ySamples = [];
    this.weights = [];
    this.sources = [];
    this.betaX = null;
    this.betaY = null;
    this._dirty = false;
    this.name = 'polynomial';
  }
  PolynomialRegression.prototype.clear = function () {
    this.xSamples = []; this.ySamples = []; this.weights = []; this.sources = [];
    this.betaX = this.betaY = null; this._dirty = false;
  };
  PolynomialRegression.prototype.addFeatures = function (features, x, y, weight, source) {
    if (!features) return false;
    this.xSamples.push(Array.from(features));
    this.ySamples.push([x, y]);
    this.weights.push(weight == null ? 1 : weight);
    this.sources.push(source || 'explicit');
    if (this.xSamples.length > 300) {
      this.xSamples.shift(); this.ySamples.shift(); this.weights.shift(); this.sources.shift();
    }
    this._dirty = true;
    return true;
  };
  PolynomialRegression.prototype.addNormalized = function (normalized, x, y, weight, source) {
    return this.addFeatures(featureVector(normalized), x, y, weight, source);
  };
  PolynomialRegression.prototype.addData = function (eyeFeatures, x, y, weight) {
    return this.addNormalized(normalizeEyeFeatures(eyeFeatures), x, y, weight, 'explicit');
  };
  PolynomialRegression.prototype.setData = function (data) {
    this.clear();
    for (const d of data || []) {
      if (!d || !d.features || !d.screenPos) continue;
      this.addFeatures(d.features, d.screenPos[0], d.screenPos[1], d.weight, d.source);
    }
  };
  PolynomialRegression.prototype.getData = function () {
    return this.xSamples.map((features, i) => ({
      features: Array.from(features),
      screenPos: Array.from(this.ySamples[i]),
      weight: this.weights[i],
      source: this.sources[i],
    }));
  };
  PolynomialRegression.prototype._fit = function () {
    if (this.xSamples.length < 6) { this.betaX = this.betaY = null; this._dirty = false; return; }
    this.betaX = ridgeSolve(this.xSamples, this.ySamples.map(p => p[0]), this.weights, RIDGE_LAMBDA);
    this.betaY = ridgeSolve(this.xSamples, this.ySamples.map(p => p[1]), this.weights, RIDGE_LAMBDA);
    this._dirty = false;
  };
  PolynomialRegression.prototype.predictNormalized = function (normalized) {
    if (this._dirty) this._fit();
    if (!this.betaX || !this.betaY) return null;
    const f = featureVector(normalized);
    if (!f) return null;
    let x = 0, y = 0;
    const m = Math.min(f.length, this.betaX.length);
    for (let i = 0; i < m; i++) { x += f[i] * this.betaX[i]; y += f[i] * this.betaY[i]; }
    return { x: clamp(x, 0, viewportWidth() || x), y: clamp(y, 0, viewportHeight() || y) };
  };
  PolynomialRegression.prototype.predict = function (eyeFeatures) {
    return this.predictNormalized(normalizeEyeFeatures(eyeFeatures));
  };

  function RBFRegression() {
    this.features = [];
    this.targets = [];
    this.weights = [];
    this.sources = [];
    this.alphaX = null;
    this.alphaY = null;
    this.gamma = 1;
    this._dirty = false;
    this.name = 'rbf';
  }
  RBFRegression.prototype.clear = function () {
    this.features = []; this.targets = []; this.weights = []; this.sources = [];
    this.alphaX = this.alphaY = null; this._dirty = false;
  };
  RBFRegression.prototype._sqDist = function (a, b) {
    let s = 0;
    const m = Math.min(a.length, b.length);
    for (let i = 0; i < m; i++) { const d = a[i] - b[i]; s += d * d; }
    return s;
  };
  RBFRegression.prototype.addFeatures = function (features, x, y, weight, source) {
    if (!features) return false;
    this.features.push(Array.from(features));
    this.targets.push([x, y]);
    this.weights.push(weight == null ? 1 : weight);
    this.sources.push(source || 'explicit');
    if (this.features.length > 120) {
      this.features.shift(); this.targets.shift(); this.weights.shift(); this.sources.shift();
    }
    this._dirty = true;
    return true;
  };
  RBFRegression.prototype.addNormalized = function (normalized, x, y, weight, source) {
    return this.addFeatures(featureVector(normalized), x, y, weight, source);
  };
  RBFRegression.prototype.addData = function (eyeFeatures, x, y, weight) {
    return this.addNormalized(normalizeEyeFeatures(eyeFeatures), x, y, weight, 'explicit');
  };
  RBFRegression.prototype.setData = function (data) {
    this.clear();
    for (const d of data || []) {
      if (!d || !d.features || !d.screenPos) continue;
      this.addFeatures(d.features, d.screenPos[0], d.screenPos[1], d.weight, d.source);
    }
  };
  RBFRegression.prototype.getData = function () {
    return this.features.map((features, i) => ({
      features: Array.from(features), screenPos: Array.from(this.targets[i]),
      weight: this.weights[i], source: this.sources[i],
    }));
  };
  RBFRegression.prototype._fit = function () {
    const n = this.features.length;
    if (n < 4) { this.alphaX = this.alphaY = null; this._dirty = false; return; }
    const dists = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) dists.push(this._sqDist(this.features[i], this.features[j]));
    dists.sort((a, b) => a - b);
    const median = dists[Math.floor(dists.length / 2)] || 1;
    this.gamma = 1 / (2 * median + 1e-12);
    const K = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) K[i][j] = Math.exp(-this.gamma * this._sqDist(this.features[i], this.features[j]));
      K[i][i] += RIDGE_LAMBDA / Math.max(0.05, this.weights[i]);
    }
    this.alphaX = gaussianSolve(K, this.targets.map(t => t[0]));
    this.alphaY = gaussianSolve(K, this.targets.map(t => t[1]));
    this._dirty = false;
  };
  RBFRegression.prototype.predictNormalized = function (normalized) {
    if (this._dirty) this._fit();
    if (!this.alphaX) return null;
    const f = featureVector(normalized);
    if (!f) return null;
    let x = 0, y = 0;
    for (let i = 0; i < this.features.length; i++) {
      const k = Math.exp(-this.gamma * this._sqDist(f, this.features[i]));
      x += this.alphaX[i] * k;
      y += this.alphaY[i] * k;
    }
    return { x: clamp(x, 0, viewportWidth() || x), y: clamp(y, 0, viewportHeight() || y) };
  };
  RBFRegression.prototype.predict = function (eyeFeatures) {
    return this.predictNormalized(normalizeEyeFeatures(eyeFeatures));
  };

  function EnsembleRegression(poly, rbf) {
    this.poly = poly || new PolynomialRegression();
    this.rbf = rbf || new RBFRegression();
    this._errPoly = 0;
    this._errRbf = 0;
    this._alpha = 0.08;
    this.name = 'ensemble';
  }
  EnsembleRegression.prototype.clear = function () { this.poly.clear(); this.rbf.clear(); this._errPoly = this._errRbf = 0; };
  EnsembleRegression.prototype.addNormalized = function (n, x, y, w, source) {
    this.poly.addNormalized(n, x, y, w, source);
    this.rbf.addNormalized(n, x, y, w, source);
    return true;
  };
  EnsembleRegression.prototype.addData = function (eyeFeatures, x, y, w) {
    const n = normalizeEyeFeatures(eyeFeatures);
    return n ? this.addNormalized(n, x, y, w, 'explicit') : false;
  };
  EnsembleRegression.prototype.setData = function (data) { this.poly.setData(data); this.rbf.setData(data); };
  EnsembleRegression.prototype.getData = function () { return this.poly.getData(); };
  EnsembleRegression.prototype.predictNormalized = function (n) {
    const p = this.poly.predictNormalized(n);
    const r = this.rbf.predictNormalized(n);
    if (!p && !r) return null;
    if (!p) return r;
    if (!r) return p;
    const ep = this._errPoly > 0 ? this._errPoly : 1;
    const er = this._errRbf > 0 ? this._errRbf : 1;
    const wp = 1 / ep, wr = 1 / er, sum = wp + wr;
    return { x: (p.x * wp + r.x * wr) / sum, y: (p.y * wp + r.y * wr) / sum };
  };
  EnsembleRegression.prototype.predict = function (eyeFeatures) {
    return this.predictNormalized(normalizeEyeFeatures(eyeFeatures));
  };
  EnsembleRegression.prototype.trackErrorNormalized = function (n, x, y) {
    const p = this.poly.predictNormalized(n);
    const r = this.rbf.predictNormalized(n);
    if (p) {
      const e = Math.hypot(p.x - x, p.y - y);
      this._errPoly = this._errPoly ? this._errPoly * (1 - this._alpha) + e * this._alpha : e;
    }
    if (r) {
      const e = Math.hypot(r.x - x, r.y - y);
      this._errRbf = this._errRbf ? this._errRbf * (1 - this._alpha) + e * this._alpha : e;
    }
  };
  EnsembleRegression.prototype.trackError = function (eyeFeatures, x, y) {
    const n = normalizeEyeFeatures(eyeFeatures);
    if (n) this.trackErrorNormalized(n, x, y);
  };

  function KalmanFilter(options) {
    options = options || {};
    this.Q = options.processNoise == null ? 8 : options.processNoise;
    this.R = options.measurementNoise == null ? 50 : options.measurementNoise;
    this.reset();
  }
  KalmanFilter.prototype.reset = function () {
    this.x = null;
    this.P = [1000,0,0,0, 0,1000,0,0, 0,0,100,0, 0,0,0,100];
    this.lastT = null;
  };
  KalmanFilter.prototype._mul4 = function (A, B) {
    const C = new Array(16).fill(0);
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) for (let k = 0; k < 4; k++) C[r*4+c] += A[r*4+k] * B[k*4+c];
    return C;
  };
  KalmanFilter.prototype._predict = function (dtMs) {
    const s = dtMs / 1000;
    this.x[0] += this.x[2] * s;
    this.x[1] += this.x[3] * s;
    const F = [1,0,s,0, 0,1,0,s, 0,0,1,0, 0,0,0,1];
    const Ft = [1,0,0,0, 0,1,0,0, s,0,1,0, 0,s,0,1];
    this.P = this._mul4(this._mul4(F, this.P), Ft);
    this.P[0] += this.Q; this.P[5] += this.Q; this.P[10] += this.Q * 0.1; this.P[15] += this.Q * 0.1;
  };
  KalmanFilter.prototype._update = function (mx, my) {
    const S00 = this.P[0] + this.R, S01 = this.P[1], S10 = this.P[4], S11 = this.P[5] + this.R;
    const det = S00 * S11 - S01 * S10;
    if (Math.abs(det) < 1e-12) return;
    const i00 = S11 / det, i01 = -S01 / det, i10 = -S10 / det, i11 = S00 / det;
    const PH = [this.P[0],this.P[1], this.P[4],this.P[5], this.P[8],this.P[9], this.P[12],this.P[13]];
    const K = new Array(8);
    for (let r = 0; r < 4; r++) {
      K[r*2] = PH[r*2] * i00 + PH[r*2+1] * i10;
      K[r*2+1] = PH[r*2] * i01 + PH[r*2+1] * i11;
    }
    const dx = mx - this.x[0], dy = my - this.x[1];
    for (let r = 0; r < 4; r++) this.x[r] += K[r*2] * dx + K[r*2+1] * dy;
    const I_KH = [
      1-K[0], -K[1], 0, 0,
      -K[2], 1-K[3], 0, 0,
      -K[4], -K[5], 1, 0,
      -K[6], -K[7], 0, 1,
    ];
    this.P = this._mul4(I_KH, this.P);
  };
  KalmanFilter.prototype.smooth = function (mx, my, suppressMeasurement) {
    const now = perfNow();
    if (!this.x) {
      this.x = [mx, my, 0, 0];
      this.lastT = now;
      return { x: mx, y: my, vx: 0, vy: 0, innovation: 0, stability: 0.5 };
    }
    const dt = clamp(now - this.lastT, 1, 120);
    this.lastT = now;
    this._predict(dt);
    const innovationBefore = Math.hypot(mx - this.x[0], my - this.x[1]);
    if (!suppressMeasurement) this._update(mx, my);
    const speed = Math.hypot(this.x[2], this.x[3]);
    const stability = clamp((1 - speed / 900) * (1 - innovationBefore / 350), 0, 1);
    return { x: this.x[0], y: this.x[1], vx: this.x[2], vy: this.x[3], innovation: innovationBefore, stability };
  };

  function BlinkDetector(options) {
    options = options || {};
    this.windowSize = options.windowSize == null ? 30 : options.windowSize;
    this.threshold = options.threshold == null ? 0.55 : options.threshold;
    this.lockoutMs = options.lockoutMs == null ? 80 : options.lockoutMs;
    this.history = [];
    this.lockoutEnd = 0;
  }
  BlinkDetector.prototype.reset = function () { this.history = []; this.lockoutEnd = 0; };
  BlinkDetector.prototype.update = function (eyeFeatures) {
    const b = meanBrightness(eyeFeatures);
    if (b == null) return false;
    const now = perfNow();
    if (now < this.lockoutEnd) return true;
    const baseline = this.history.length
      ? this.history.reduce((a, v) => a + v, 0) / this.history.length
      : b;
    const blink = this.history.length >= 5 && b < baseline * this.threshold;
    if (!blink) {
      this.history.push(b);
      if (this.history.length > this.windowSize) this.history.shift();
    } else {
      this.lockoutEnd = now + this.lockoutMs;
    }
    return blink;
  };

  function SaccadeDetector(threshold) { this.threshold = threshold == null ? 600 : threshold; }
  SaccadeDetector.prototype.isSaccade = function (vx, vy) { return Math.hypot(vx || 0, vy || 0) > this.threshold; };

  function DriftWatchdog(owner, options) {
    options = options || {};
    this.owner = owner || null;
    this.warnThreshold = options.warnThreshold == null ? 120 : options.warnThreshold;
    this.critThreshold = options.critThreshold == null ? 220 : options.critThreshold;
    this.minEvidence = options.minEvidence == null ? 5 : options.minEvidence;
    this.decay = options.decay == null ? 0.88 : options.decay;
    this.onWarn = options.onWarn || null;
    this.onCritical = options.onCritical || null;
    this.enabled = false;
    this.reset();
  }
  DriftWatchdog.prototype.enable = function () { this.enabled = true; return this; };
  DriftWatchdog.prototype.disable = function () { this.enabled = false; return this; };
  DriftWatchdog.prototype.reset = function () { this.weightedSS = 0; this.weightSum = 0; this.rmse = 0; this.state = 'ok'; };
  DriftWatchdog.prototype.record = function (normalized, x, y, evidenceWeight) {
    if (!this.enabled || !this.owner || !normalized) return null;
    const pred = this.owner._predictRegression(normalized);
    if (!pred) return null;
    const w = clamp(evidenceWeight == null ? 1 : evidenceWeight, 0.01, 1);
    const residual = Math.hypot(pred.x - x, pred.y - y);
    this.weightedSS = this.weightedSS * this.decay + residual * residual * w;
    this.weightSum = this.weightSum * this.decay + w;
    this.rmse = this.weightSum >= this.minEvidence ? Math.sqrt(this.weightedSS / this.weightSum) : 0;
    const previous = this.state;
    if (this.rmse >= this.critThreshold) this.state = 'critical';
    else if (this.rmse >= this.warnThreshold) this.state = 'warning';
    else if (this.rmse < this.warnThreshold * 0.8) this.state = 'ok';
    if (this.state !== previous && this.state !== 'ok') {
      const detail = { rmse: Math.round(this.rmse), state: this.state, evidence: this.weightSum, timestamp: Date.now() };
      const doc = global.document;
      dispatch(doc, this.state === 'critical' ? 'webgazer-aac:drift-critical' : 'webgazer-aac:drift-warning', detail);
      const cb = this.state === 'critical' ? this.onCritical : this.onWarn;
      if (typeof cb === 'function') try { cb(detail); } catch (_) {}
    }
    return residual;
  };

  function AdaptiveRecalibrator(owner) {
    this.owner = owner;
    this.enabled = false;
    this.hitCount = 0;
    this.maxHitsPerSession = 500;
  }
  AdaptiveRecalibrator.prototype.enable = function () { this.enabled = true; return this; };
  AdaptiveRecalibrator.prototype.disable = function () { this.enabled = false; return this; };
  AdaptiveRecalibrator.prototype.recordNormalized = function (normalized, x, y, weight, source) {
    if (!this.enabled || !normalized || this.hitCount >= this.maxHitsPerSession) return false;
    this.owner.recordGroundTruth(x, y, {
      normalized,
      source: source || 'dwell-selection',
      evidenceWeight: weight,
      includeInCalibration: false,
      adaptive: true,
    });
    this.hitCount++;
    return true;
  };

  const DEFAULT_TARGET_SELECTOR = [
    '[data-gaze-target]', 'button', 'a[href]', 'input:not([type="hidden"])', 'select', 'textarea',
    '[role="button"]', '[role="option"]', '[role="menuitem"]', '[role="tab"]', '[role="switch"]',
    '[tabindex]:not([tabindex="-1"])',
  ].join(',');

  function rectOf(el) {
    if (!el || typeof el.getBoundingClientRect !== 'function') return null;
    try {
      const r = el.getBoundingClientRect();
      const left = finite(r.left, 0), top = finite(r.top, 0);
      const width = finite(r.width, finite(r.right, left) - left);
      const height = finite(r.height, finite(r.bottom, top) - top);
      return { left, top, right: finite(r.right, left + width), bottom: finite(r.bottom, top + height), width, height };
    } catch (_) { return null; }
  }

  function isDisabledTarget(el) {
    if (!el) return true;
    if (el.disabled) return true;
    if (typeof el.getAttribute === 'function') {
      const aria = el.getAttribute('aria-disabled');
      if (aria === 'true') return true;
    }
    return false;
  }

  function GazeTargetResolver(options) {
    options = options || {};
    this.selector = options.selector || DEFAULT_TARGET_SELECTOR;
    this.expansionPx = options.expansionPx == null ? 48 : options.expansionPx;
    this.maxDistancePx = options.maxDistancePx == null ? 180 : options.maxDistancePx;
    this.hysteresisBonus = options.hysteresisBonus == null ? 0.22 : options.hysteresisBonus;
    this.cacheMs = options.cacheMs == null ? 120 : options.cacheMs;
    this._lastTarget = null;
    this._cacheRoot = null;
    this._cacheAt = -Infinity;
    this._cacheTargets = [];
  }
  GazeTargetResolver.prototype._targets = function (root) {
    root = root || global.document;
    const now = perfNow();
    if (root === this._cacheRoot && now - this._cacheAt <= this.cacheMs) return this._cacheTargets;
    let list = [];
    try { list = Array.from(root && root.querySelectorAll ? root.querySelectorAll(this.selector) : []); } catch (_) {}
    this._cacheRoot = root; this._cacheAt = now; this._cacheTargets = list.filter(el => !isDisabledTarget(el));
    return this._cacheTargets;
  };
  GazeTargetResolver.prototype.invalidate = function () { this._cacheAt = -Infinity; this._cacheTargets = []; };
  GazeTargetResolver.prototype.resolveElement = function (element, x, y) {
    if (!element) return null;
    let target = null;
    try { target = typeof element.closest === 'function' ? element.closest(this.selector) : element; } catch (_) { target = element; }
    if (!target || isDisabledTarget(target)) return null;
    const rect = rectOf(target);
    if (!rect) return null;
    const cx = rect.left + rect.width / 2, cy = rect.top + rect.height / 2;
    const distance = Math.hypot((x == null ? cx : x) - cx, (y == null ? cy : y) - cy);
    this._lastTarget = target;
    return { element: target, confidence: 1, targetConfidence: 1, rect, distance, crowding: 0 };
  };
  GazeTargetResolver.prototype.resolve = function (x, y, root) {
    const candidates = this._targets(root);
    if (!candidates.length) return null;
    const scored = [];
    for (const el of candidates) {
      const r = rectOf(el);
      if (!r || r.width <= 0 || r.height <= 0) continue;
      const ex = this.expansionPx;
      const dx = x < r.left - ex ? (r.left - ex - x) : x > r.right + ex ? (x - r.right - ex) : 0;
      const dy = y < r.top - ex ? (r.top - ex - y) : y > r.bottom + ex ? (y - r.bottom - ex) : 0;
      const outsideDistance = Math.hypot(dx, dy);
      if (outsideDistance > this.maxDistancePx) continue;
      const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
      const centerDistance = Math.hypot(x - cx, y - cy);
      const inside = x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
      const scale = Math.max(40, Math.min(180, Math.sqrt(r.width * r.height)));
      let score = inside ? 1.2 : Math.exp(-(outsideDistance * outsideDistance) / (2 * scale * scale));
      score *= 0.85 + 0.15 * Math.exp(-centerDistance / Math.max(1, scale * 2));
      if (el === this._lastTarget) score *= 1 + this.hysteresisBonus;
      scored.push({ element: el, rect: r, score: Math.max(1e-9, score), distance: centerDistance });
    }
    if (!scored.length) return null;
    scored.sort((a, b) => b.score - a.score);
    const best = scored[0];
    const total = scored.reduce((s, c) => s + c.score, 0);
    const confidence = clamp(best.score / Math.max(best.score, total), 0, 1);
    const near = scored.filter(c => c !== best && c.distance < this.maxDistancePx).length;
    const crowding = clamp(near / 4, 0, 1);
    this._lastTarget = best.element;
    return {
      element: best.element,
      confidence,
      targetConfidence: confidence,
      rect: best.rect,
      distance: best.distance,
      crowding,
      candidateCount: scored.length,
    };
  };

  function AdaptiveDwellController(options) {
    options = options || {};
    this.baseMs = options.baseMs == null ? 800 : options.baseMs;
    this.minMs = options.minMs == null ? 400 : options.minMs;
    this.maxMs = options.maxMs == null ? 1400 : options.maxMs;
    this.correctionPressure = 0;
  }
  AdaptiveDwellController.prototype.noteCorrection = function () { this.correctionPressure = clamp(this.correctionPressure + 0.15, 0, 1); };
  AdaptiveDwellController.prototype.noteSuccess = function () { this.correctionPressure = clamp(this.correctionPressure - 0.04, 0, 1); };
  AdaptiveDwellController.prototype.getDwellMs = function (ctx) {
    ctx = ctx || {};
    const q = clamp(finite(ctx.trackingQuality, 0.5), 0, 1);
    const tc = clamp(finite(ctx.targetConfidence, 0.5), 0, 1);
    const area = Math.max(1, finite(ctx.targetArea, 4000));
    const crowding = clamp(finite(ctx.crowding, 0), 0, 1);
    const qualityPenalty = (1 - q) * 0.45;
    const confidencePenalty = (1 - tc) * 0.5;
    const sizePenalty = clamp((5000 - area) / 5000, 0, 1) * 0.35;
    const crowdingPenalty = crowding * 0.35;
    const correctionPenalty = this.correctionPressure * 0.3;
    const highQualityDiscount = q > 0.85 && tc > 0.85 && area > 8000 ? 0.18 : 0;
    const factor = 1 + qualityPenalty + confidencePenalty + sizePenalty + crowdingPenalty + correctionPenalty - highQualityDiscount;
    return clamp(Math.round(this.baseMs * factor), this.minMs, this.maxMs);
  };

  function targetAnchor(element) {
    if (!element) return null;
    const r = rectOf(element);
    if (!r) return null;
    let x = r.left + r.width / 2;
    let y = r.top + r.height / 2;
    const ds = element.dataset || {};
    if (ds.gazeX != null && Number.isFinite(Number(ds.gazeX))) x = Number(ds.gazeX);
    if (ds.gazeY != null && Number.isFinite(Number(ds.gazeY))) y = Number(ds.gazeY);
    if (typeof element.getAttribute === 'function') {
      const ax = element.getAttribute('data-gaze-x');
      const ay = element.getAttribute('data-gaze-y');
      if (ax != null && Number.isFinite(Number(ax))) x = Number(ax);
      if (ay != null && Number.isFinite(Number(ay))) y = Number(ay);
    }
    return { x, y, rect: r };
  }

  function DwellTimer(options) {
    options = options || {};
    this.dwellMs = options.dwellMs == null ? 800 : options.dwellMs;
    this.minTrackingQuality = options.minTrackingQuality == null
      ? (options.minConfidence == null ? 0.25 : options.minConfidence)
      : options.minTrackingQuality;
    this.minTargetConfidence = options.minTargetConfidence == null ? 0.25 : options.minTargetConfidence;
    this.holdAfterMs = options.holdAfterMs == null ? 1200 : options.holdAfterMs;
    this.adaptive = options.adaptive !== false;
    this._aacRef = options.aacRef || null;
    this._resolver = options.resolver || (this._aacRef && this._aacRef._targetResolver) || new GazeTargetResolver();
    this._controller = options.controller || new AdaptiveDwellController({ baseMs: this.dwellMs });
    this.reset();
  }
  DwellTimer.prototype.reset = function () {
    this._target = null; this._progress = 0; this._lastT = null; this._holdUntil = 0; this._requiredMs = this.dwellMs;
  };
  Object.defineProperty(DwellTimer.prototype, 'progress', { get: function () { return this._progress; } });
  DwellTimer.prototype.update = function (element, x, y, trackingQuality, isSaccade, isBlink, targetConfidence, meta) {
    const now = perfNow();
    if (element == null || isBlink) { this._lastT = now; return this._progress; }
    if (element !== this._target) {
      if (this._target && this._progress > 0) dispatch(this._target, 'webgazer-aac:dwell-cancel', { reason: 'target-changed', x, y });
      this._target = element; this._progress = 0; this._lastT = now;
      const r = rectOf(element);
      const area = r ? r.width * r.height : 0;
      const tc0 = targetConfidence == null ? 1 : targetConfidence;
      this._requiredMs = this.adaptive
        ? this._controller.getDwellMs({ trackingQuality, targetConfidence: tc0, targetArea: area, crowding: meta && meta.crowding })
        : this.dwellMs;
      return 0;
    }
    const dt = this._lastT == null ? 0 : clamp(now - this._lastT, 0, 150);
    this._lastT = now;
    const tc = targetConfidence == null ? 1 : targetConfidence;
    const frozen = !!isSaccade || trackingQuality < this.minTrackingQuality || tc < this.minTargetConfidence || now < this._holdUntil;
    if (!frozen) this._progress = clamp(this._progress + dt / Math.max(1, this._requiredMs), 0, 1);
    dispatch(element, 'webgazer-aac:dwell-progress', {
      progress: this._progress, trackingQuality, targetConfidence: tc, x, y, frozen, requiredMs: this._requiredMs,
    });
    if (this._progress >= 1 && now >= this._holdUntil) {
      this._holdUntil = now + this.holdAfterMs;
      this._progress = 0;
      dispatch(element, 'webgazer-aac:dwell-complete', { x, y, trackingQuality, targetConfidence: tc });
      if (this._aacRef) this._aacRef.recordConfirmedSelection(element, { source: 'dwell-selection', evidenceWeight: EVIDENCE_WEIGHTS['dwell-selection'] });
      this._controller.noteSuccess();
    }
    return this._progress;
  };
  DwellTimer.prototype.updateFromGaze = function (gaze, root) {
    if (!gaze) { this._lastT = perfNow(); return this._progress; }
    const result = this._resolver.resolve(gaze.x, gaze.y, root);
    if (!result) {
      if (this._target && this._progress > 0) dispatch(this._target, 'webgazer-aac:dwell-cancel', { reason: 'no-target', x: gaze.x, y: gaze.y });
      this._target = null; this._progress = 0; this._lastT = perfNow();
      return 0;
    }
    return this.update(result.element, gaze.x, gaze.y, gaze.trackingQuality == null ? gaze.confidence : gaze.trackingQuality,
      !!gaze.isSaccade, !!gaze.isBlink, result.confidence, result);
  };

  function _MemoryBackend() { this._store = Object.create(null); }
  _MemoryBackend.prototype.save = function (key, value) { this._store[key] = JSON.parse(JSON.stringify(value)); return Promise.resolve(true); };
  _MemoryBackend.prototype.load = function (key) { const v = this._store[key]; return Promise.resolve(v ? JSON.parse(JSON.stringify(v)) : null); };
  _MemoryBackend.prototype.clear = function (key) { delete this._store[key]; return Promise.resolve(true); };

  function CalibrationStore(options) {
    options = options || {};
    this.dbName = options.dbName || 'webgazer-aac';
    this.storeName = options.storeName || 'calibrations';
    this.profileKey = options.profileKey || 'default';
    this._backend = options.backend || null;
    this._db = null;
  }
  CalibrationStore.prototype._check = function (snap) {
    if (!snap) return null;
    if (snap.schemaVersion !== SCHEMA_VERSION) { snap._incompatibleReason = 'schema-version'; return snap; }
    if (snap.featureVersion !== FEATURE_VERSION) { snap._incompatibleReason = 'feature-version'; return snap; }
    const vw = viewportWidth(), vh = viewportHeight(), dpr = devicePixelRatio();
    const sizeChanged = snap.viewport && (Math.abs((snap.viewport.width || 0) - vw) > Math.max(32, vw * 0.05) || Math.abs((snap.viewport.height || 0) - vh) > Math.max(32, vh * 0.05));
    const dprChanged = snap.viewport && Math.abs((snap.viewport.dpr || 1) - dpr) > 0.25;
    if (sizeChanged || dprChanged) snap._validationRequired = true;
    return snap;
  };
  CalibrationStore.prototype.available = function () {
    if (this._backend) return Promise.resolve(true);
    if (typeof global.indexedDB === 'undefined') return Promise.resolve(false);
    return this._open().then(() => true).catch(() => false);
  };
  CalibrationStore.prototype.save = function (snap) {
    if (this._backend) return this._backend.save(this.profileKey, snap);
    return this._open().then(db => new Promise((resolve, reject) => {
      try {
        const req = db.transaction(this.storeName, 'readwrite').objectStore(this.storeName).put(snap, this.profileKey);
        req.onsuccess = () => resolve(true); req.onerror = e => reject(e.target.error);
      } catch (e) { reject(e); }
    }));
  };
  CalibrationStore.prototype.load = function () {
    if (this._backend) return this._backend.load(this.profileKey).then(s => this._check(s));
    return this._open().then(db => new Promise((resolve, reject) => {
      try {
        const req = db.transaction(this.storeName, 'readonly').objectStore(this.storeName).get(this.profileKey);
        req.onsuccess = e => resolve(this._check(e.target.result)); req.onerror = e => reject(e.target.error);
      } catch (e) { reject(e); }
    }));
  };
  CalibrationStore.prototype.clear = function () {
    if (this._backend) return this._backend.clear(this.profileKey);
    return this._open().then(db => new Promise((resolve, reject) => {
      try {
        const req = db.transaction(this.storeName, 'readwrite').objectStore(this.storeName).delete(this.profileKey);
        req.onsuccess = () => resolve(true); req.onerror = e => reject(e.target.error);
      } catch (e) { reject(e); }
    }));
  };
  CalibrationStore.prototype._open = function () {
    if (this._db) return Promise.resolve(this._db);
    if (typeof global.indexedDB === 'undefined') return Promise.reject(new Error('IndexedDB unavailable'));
    return new Promise((resolve, reject) => {
      try {
        const req = global.indexedDB.open(this.dbName, 1);
        req.onupgradeneeded = e => {
          const db = e.target.result;
          if (!db.objectStoreNames.contains(this.storeName)) db.createObjectStore(this.storeName);
        };
        req.onsuccess = e => { this._db = e.target.result; resolve(this._db); };
        req.onerror = e => reject(e.target.error);
      } catch (e) { reject(e); }
    });
  };

  const api = {
    version: LIBRARY_VERSION,
    libraryVersion: LIBRARY_VERSION,
    schemaVersion: SCHEMA_VERSION,
    featureVersion: FEATURE_VERSION,
    EVIDENCE_WEIGHTS,

    _installed: false,
    _origSetGazeListener: null,
    _origRecordScreenPosition: null,
    _lastEyeFeatures: null,
    _lastNormalized: null,
    _lastResult: null,
    _trackingQuality: 0,
    _gazeStability: 0,
    _trainingRecords: [],
    _calibrationRecords: [],
    _store: null,
    _currentMode: 'ensemble',
    _videoClock: null,
    _diagnostics: {
      frames: 0, eyeFeatureFrames: 0, fallbackFrames: 0, blinkFrames: 0, saccadeFrames: 0,
      explicitSamples: 0, adaptiveSamples: 0, calibrationSamples: 0,
      firstFrameAt: 0, lastFrameAt: 0, videoFrames: 0, lastVideoMetadata: null,
    },

    _normalizeEyeFeatures: normalizeEyeFeatures,

    _initModels() {
      if (this._regressions) return;
      const poly = new PolynomialRegression();
      const rbf = new RBFRegression();
      this._regressions = { polynomial: poly, rbf, ensemble: new EnsembleRegression(poly, rbf) };
      this._regression = this._regressions.ensemble;
      this._kalman = new KalmanFilter();
      this._blink = new BlinkDetector();
      this._saccade = new SaccadeDetector();
      this._recalibrator = new AdaptiveRecalibrator(this);
      this._watchdog = new DriftWatchdog(this);
      this._targetResolver = new GazeTargetResolver();
    },

    install(options) {
      this._initModels();
      options = options || {};
      if (this._installed) return this;
      const wg = global.webgazer;
      if (!wg || typeof wg.setGazeListener !== 'function' || typeof wg.recordScreenPosition !== 'function') {
        throw new Error('[webgazer-aac] WebGazer must be loaded before install()');
      }
      const self = this;
      this._origSetGazeListener = wg.setGazeListener.bind(wg);
      this._origRecordScreenPosition = wg.recordScreenPosition.bind(wg);

      wg.setGazeListener = function (callback) {
        return self._origSetGazeListener(function (data, elapsedTime) {
          self._diagnostics.frames++;
          const t = perfNow();
          if (!self._diagnostics.firstFrameAt) self._diagnostics.firstFrameAt = t;
          self._diagnostics.lastFrameAt = t;
          if (!data) { callback(null, elapsedTime); return; }

          const eyeFeatures = data.eyeFeatures || null;
          const normalized = normalizeEyeFeatures(eyeFeatures);
          self._lastEyeFeatures = eyeFeatures;
          self._lastNormalized = normalized;
          if (normalized) self._diagnostics.eyeFeatureFrames++; else self._diagnostics.fallbackFrames++;

          const blink = eyeFeatures ? self._blink.update(eyeFeatures) : false;
          if (blink) self._diagnostics.blinkFrames++;

          let raw = normalized ? self._predictRegression(normalized) : null;
          if (!raw) raw = { x: finite(data.x, 0), y: finite(data.y, 0) };

          const priorVx = self._lastResult ? self._lastResult.vx : 0;
          const priorVy = self._lastResult ? self._lastResult.vy : 0;
          const saccade = self._saccade.isSaccade(priorVx, priorVy);
          if (saccade) self._diagnostics.saccadeFrames++;

          const filtered = self._kalman.smooth(raw.x, raw.y, blink || saccade);
          const featureFactor = normalized ? 1 : 0.45;
          const trackingQuality = clamp(filtered.stability * 0.8 + featureFactor * 0.2, 0, 1);
          self._lastResult = filtered;
          self._trackingQuality = trackingQuality;
          self._gazeStability = filtered.stability;

          if (blink) { callback(null, elapsedTime); return; }
          callback({
            x: filtered.x, y: filtered.y, vx: filtered.vx, vy: filtered.vy,
            isSaccade: saccade, isBlink: false,
            gazeStability: filtered.stability,
            trackingQuality,
            confidence: trackingQuality,
            eyeFeaturesAvailable: !!normalized,
          }, elapsedTime);
        });
      };

      wg.recordScreenPosition = function (x, y, eventType) {
        const result = self._origRecordScreenPosition(x, y, eventType);
        const source = eventType === 'click' ? 'confirmed-click' : 'explicit';
        const includeInCalibration = eventType !== 'move';
        self.recordGroundTruth(x, y, {
          normalized: self._lastNormalized,
          source,
          evidenceWeight: EVIDENCE_WEIGHTS[source],
          includeInCalibration,
        });
        return result;
      };

      if (options.regression) this.setRegression(options.regression);
      this._installed = true;
      return this;
    },

    _predictRegression(normalized) {
      if (!normalized || this._currentMode === 'ridge' || !this._regression) return null;
      try { return this._regression.predictNormalized(normalized); } catch (_) { return null; }
    },

    _addTrainingRecord(record) {
      this._trainingRecords.push(record);
      if (this._trainingRecords.length > 500) this._trainingRecords.shift();
      if (record.includeInCalibration) {
        this._calibrationRecords.push(record);
        if (this._calibrationRecords.length > 300) this._calibrationRecords.shift();
      }
      this._regressions.ensemble.addNormalized(record.normalized, record.x, record.y, record.weight, record.source);
    },

    recordGroundTruth(x, y, options) {
      this._initModels();
      options = options || {};
      const normalized = options.normalized || this._lastNormalized;
      if (!normalized || !Number.isFinite(x) || !Number.isFinite(y)) return false;
      const source = options.source || 'explicit';
      const weight = clamp(options.evidenceWeight == null ? (EVIDENCE_WEIGHTS[source] == null ? 0.5 : EVIDENCE_WEIGHTS[source]) : options.evidenceWeight, 0.01, 1);

      if (this._watchdog) this._watchdog.record(normalized, x, y, weight);
      if (this._regressions && this._regressions.ensemble) this._regressions.ensemble.trackErrorNormalized(normalized, x, y);

      const record = {
        normalized: cloneNormalized(normalized), x, y, weight, source,
        includeInCalibration: !!options.includeInCalibration,
      };
      this._addTrainingRecord(record);
      if (options.adaptive) this._diagnostics.adaptiveSamples++;
      else this._diagnostics.explicitSamples++;
      if (record.includeInCalibration) this._diagnostics.calibrationSamples++;
      return true;
    },

    fitUserBasis() {
      this._initModels();
      const left = this._calibrationRecords.map(r => r.normalized.left);
      const right = this._calibrationRecords.map(r => r.normalized.right);
      const lOk = LEFT_PCA.fit(left);
      const rOk = RIGHT_PCA.fit(right);
      let rebuilt = false;
      if (lOk && rOk) {
        this._regressions.polynomial.clear();
        this._regressions.rbf.clear();
        for (const r of this._trainingRecords) this._regressions.ensemble.addNormalized(r.normalized, r.x, r.y, r.weight, r.source);
        rebuilt = true;
      }
      return { left: lOk, right: rOk, samples: this._calibrationRecords.length, rebuilt };
    },

    resetCalibrationPatches() {
      this._initModels();
      this._trainingRecords = [];
      this._calibrationRecords = [];
      this._regressions.polynomial.clear();
      this._regressions.rbf.clear();
      LEFT_PCA.reset(); RIGHT_PCA.reset();
      this._diagnostics.explicitSamples = 0;
      this._diagnostics.adaptiveSamples = 0;
      this._diagnostics.calibrationSamples = 0;
      if (this._watchdog) this._watchdog.reset();
      return this;
    },

    isPCAFitted() { return LEFT_PCA.fitted && RIGHT_PCA.fitted; },

    setRegression(mode) {
      this._initModels();
      mode = mode || 'ensemble';
      if (mode === 'ridge') {
        this._currentMode = 'ridge'; this._regression = null;
        if (global.webgazer && typeof global.webgazer.setRegression === 'function') try { global.webgazer.setRegression('ridge'); } catch (_) {}
        return this;
      }
      if (!this._regressions[mode]) throw new Error('[webgazer-aac] unknown regression: ' + mode);
      this._currentMode = mode;
      this._regression = this._regressions[mode];
      return this;
    },
    getRegressionMode() { return this._currentMode; },

    enableAdaptiveRecalibration() { this._initModels(); this._recalibrator.enable(); return this; },
    disableAdaptiveRecalibration() { this._initModels(); this._recalibrator.disable(); return this; },

    recordConfirmedSelection(element, options) {
      this._initModels();
      options = options || {};
      const anchor = targetAnchor(element);
      const normalized = options.normalized || this._lastNormalized;
      if (!anchor || !normalized || !this._recalibrator) return false;
      const source = options.source || 'dwell-selection';
      const weight = options.evidenceWeight == null ? (EVIDENCE_WEIGHTS[source] || EVIDENCE_WEIGHTS['dwell-selection']) : options.evidenceWeight;
      return this._recalibrator.recordNormalized(normalized, anchor.x, anchor.y, weight, source);
    },
    recordDwellHit(element, options) { return this.recordConfirmedSelection(element, Object.assign({ source: 'dwell-selection' }, options || {})); },
    recordDwellHitXY(x, y, options) {
      this._initModels();
      options = options || {};
      const normalized = options.normalized || this._lastNormalized;
      if (!normalized) return false;
      const source = options.source || 'inferred';
      const weight = options.evidenceWeight == null ? (EVIDENCE_WEIGHTS[source] || EVIDENCE_WEIGHTS.inferred) : options.evidenceWeight;
      if (this._recalibrator && this._recalibrator.enabled) return this._recalibrator.recordNormalized(normalized, x, y, weight, source);
      return this.recordGroundTruth(x, y, { normalized, source, evidenceWeight: weight, includeInCalibration: false, adaptive: true });
    },

    createTargetResolver(options) { return new GazeTargetResolver(options); },
    resolveTarget(x, y, root) { this._initModels(); return this._targetResolver.resolve(x, y, root); },
    createDwellTimer(options) {
      this._initModels();
      return new DwellTimer(Object.assign({}, options || {}, { aacRef: this, resolver: (options && options.resolver) || this._targetResolver }));
    },

    enableDriftWatchdog(options) {
      this._initModels();
      if (options) {
        for (const k of ['warnThreshold','critThreshold','minEvidence','decay','onWarn','onCritical']) if (options[k] != null) this._watchdog[k] = options[k];
      }
      this._watchdog.enable(); return this;
    },
    disableDriftWatchdog() { this._initModels(); this._watchdog.disable(); return this; },
    resetDriftWatchdog() { this._initModels(); this._watchdog.reset(); return this; },
    getDriftRmse() { this._initModels(); return this._watchdog.rmse || 0; },

    setKalmanParams(processNoise, measurementNoise) {
      this._initModels();
      if (processNoise != null) this._kalman.Q = processNoise;
      if (measurementNoise != null) this._kalman.R = measurementNoise;
      return this;
    },
    resetSmoother() { this._initModels(); this._kalman.reset(); this._lastResult = null; return this; },
    getTrackingQuality() { return this._trackingQuality; },
    getConfidence() { return this._trackingQuality; },

    configureStore(options) { this._store = new CalibrationStore(options || {}); return this; },
    getCalibrationSnapshot() {
      this._initModels();
      return {
        libraryVersion: LIBRARY_VERSION,
        schemaVersion: SCHEMA_VERSION,
        featureVersion: FEATURE_VERSION,
        timestamp: Date.now(),
        viewport: { width: viewportWidth(), height: viewportHeight(), dpr: devicePixelRatio() },
        kalman: { Q: this._kalman.Q, R: this._kalman.R },
        pca: {
          left: LEFT_PCA.fitted ? { fitted: true, mean: Array.from(LEFT_PCA.mean), basis: LEFT_PCA.basis } : { fitted: false },
          right: RIGHT_PCA.fitted ? { fitted: true, mean: Array.from(RIGHT_PCA.mean), basis: RIGHT_PCA.basis } : { fitted: false },
        },
        regressions: {
          polynomial: this._regressions.polynomial.getData(),
          rbf: this._regressions.rbf.getData(),
        },
      };
    },
    applyCalibrationSnapshot(snapshot) {
      this._initModels();
      if (!snapshot || snapshot.schemaVersion !== SCHEMA_VERSION || snapshot.featureVersion !== FEATURE_VERSION) return false;
      if (snapshot.kalman) { if (snapshot.kalman.Q != null) this._kalman.Q = snapshot.kalman.Q; if (snapshot.kalman.R != null) this._kalman.R = snapshot.kalman.R; }
      if (snapshot.pca && snapshot.pca.left && snapshot.pca.left.fitted) {
        LEFT_PCA.mean = Float64Array.from(snapshot.pca.left.mean || []); LEFT_PCA.basis = snapshot.pca.left.basis || LEFT_PCA.basis; LEFT_PCA.fitted = true;
      }
      if (snapshot.pca && snapshot.pca.right && snapshot.pca.right.fitted) {
        RIGHT_PCA.mean = Float64Array.from(snapshot.pca.right.mean || []); RIGHT_PCA.basis = snapshot.pca.right.basis || RIGHT_PCA.basis; RIGHT_PCA.fitted = true;
      }
      if (snapshot.regressions) {
        this._regressions.polynomial.setData(snapshot.regressions.polynomial || []);
        this._regressions.rbf.setData(snapshot.regressions.rbf || []);
      }
      this._trainingRecords = [];
      this._calibrationRecords = [];
      return true;
    },
    saveCalibration(storeOptions) {
      if (storeOptions) this.configureStore(storeOptions);
      if (!this._store) this._store = new CalibrationStore();
      return this._store.save(this.getCalibrationSnapshot()).catch(() => false);
    },
    loadCalibration(storeOptions) {
      if (storeOptions) this.configureStore(storeOptions);
      if (!this._store) this._store = new CalibrationStore();
      return this._store.load().then(snap => {
        if (!snap || snap._incompatibleReason || snap._validationRequired) return snap;
        this.applyCalibrationSnapshot(snap);
        return snap;
      }).catch(() => null);
    },
    clearCalibration() { if (!this._store) this._store = new CalibrationStore(); return this._store.clear().catch(() => false); },
    isStorageAvailable() { if (!this._store) this._store = new CalibrationStore(); return this._store.available(); },

    attachVideoClock(video) {
      this.detachVideoClock();
      if (!video || typeof video.requestVideoFrameCallback !== 'function') return false;
      const state = { video, active: true, handle: null };
      const tick = (_now, metadata) => {
        if (!state.active) return;
        this._diagnostics.videoFrames++;
        this._diagnostics.lastVideoMetadata = metadata ? {
          mediaTime: metadata.mediaTime, presentedFrames: metadata.presentedFrames,
          expectedDisplayTime: metadata.expectedDisplayTime, processingDuration: metadata.processingDuration,
        } : null;
        state.handle = video.requestVideoFrameCallback(tick);
      };
      state.handle = video.requestVideoFrameCallback(tick);
      this._videoClock = state;
      return true;
    },
    detachVideoClock() {
      const s = this._videoClock;
      if (!s) return this;
      s.active = false;
      if (s.handle != null && s.video && typeof s.video.cancelVideoFrameCallback === 'function') {
        try { s.video.cancelVideoFrameCallback(s.handle); } catch (_) {}
      }
      this._videoClock = null;
      return this;
    },

    getDiagnostics() {
      this._initModels();
      const d = this._diagnostics;
      const elapsed = d.firstFrameAt && d.lastFrameAt > d.firstFrameAt ? (d.lastFrameAt - d.firstFrameAt) / 1000 : 0;
      return {
        version: LIBRARY_VERSION,
        libraryVersion: LIBRARY_VERSION,
        schemaVersion: SCHEMA_VERSION,
        featureVersion: FEATURE_VERSION,
        regressionMode: this._currentMode,
        pcaFitted: this.isPCAFitted(),
        trackingQuality: this._trackingQuality,
        gazeStability: this._gazeStability,
        driftRmse: this._watchdog.rmse || 0,
        driftState: this._watchdog.state || 'ok',
        gazeFrames: d.frames,
        effectiveGazeFps: elapsed > 0 ? d.frames / elapsed : 0,
        eyeFeatureCoverage: d.frames ? d.eyeFeatureFrames / d.frames : 0,
        fallbackFrames: d.fallbackFrames,
        blinkFrames: d.blinkFrames,
        saccadeFrames: d.saccadeFrames,
        explicitSamples: d.explicitSamples,
        adaptiveSamples: d.adaptiveSamples,
        calibrationSamples: d.calibrationSamples,
        videoFrames: d.videoFrames,
        lastVideoMetadata: d.lastVideoMetadata,
      };
    },

    PCABasis,
    PolynomialRegression,
    RBFRegression,
    EnsembleRegression,
    KalmanFilter,
    BlinkDetector,
    SaccadeDetector,
    DriftWatchdog,
    AdaptiveRecalibrator,
    GazeTargetResolver,
    AdaptiveDwellController,
    DwellTimer,
    CalibrationStore,
    _MemoryBackend,
  };

  api._initModels();
  return api;
});
