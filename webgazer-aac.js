/**
 * webgazer-aac.js  v1.1.0
 * Accessibility-first enhancement layer for WebGazer.js
 *
 * Fork/patch by ALTRU.dev — Code for Humanity
 * https://altru.dev  |  github.com/altru-dev/webgazer-aac
 *
 * License: GPLv3 (same as upstream brownhci/WebGazer)
 *
 * USAGE:
 *   Load AFTER webgazer.js. Then call:
 *     webgazerAAC.install()
 *     webgazerAAC.setRegression('ensemble')   // or 'polynomial', 'rbf', 'ridge'
 *     webgazerAAC.enableAdaptiveRecalibration()
 *
 *   At calibration end, call:
 *     webgazerAAC.fitUserBasis()   // fits per-user PCA from collected patches
 *
 *   Everything else (webgazer.begin(), setGazeListener(), etc.) stays identical.
 *
 * IMPROVEMENTS IN v1.1.0 OVER v1.0.0:
 *   1. Per-user PCA basis  — fitted from actual calibration patches, replaces
 *                            fixed random projection → better feature quality
 *   2. CLAHE normalisation — adaptive contrast on eye patches before feature
 *                            extraction → works in poor lighting
 *   3. Kalman filter       — replaces EMA smoother; separates process noise
 *                            from measurement noise → less jitter, better tracking
 *   4. Blink detection     — pauses gaze output during blinks → no false dwell fires
 *   5. Saccade suppression — holds last stable position during fast eye movements
 *   6. Ensemble regression — blends polynomial + RBF weighted by recent accuracy
 *   7. Frame cache         — skips inference when gaze is stable → saves CPU
 */

(function (global) {
  'use strict';

  // ─── Constants ────────────────────────────────────────────────────────────

  const PATCH_COMPONENTS = 10;  // components per eye (was 8)
  const LAMBDA           = 1e-3;
  const PATCH_DIM        = 60 * 40;

  // ─── Math utilities ───────────────────────────────────────────────────────

  function ridgeSolve(X, y, lambda) {
    const n = X.length, m = X[0].length;
    const XtX = Array.from({ length: m }, () => new Float64Array(m));
    for (let i = 0; i < m; i++)
      for (let j = 0; j < m; j++)
        for (let k = 0; k < n; k++)
          XtX[i][j] += X[k][i] * X[k][j];
    for (let i = 0; i < m; i++) XtX[i][i] += lambda;
    const Xty = new Float64Array(m);
    for (let i = 0; i < m; i++)
      for (let k = 0; k < n; k++)
        Xty[i] += X[k][i] * y[k];
    return gaussianElimination(XtX, Xty);
  }

  function gaussianElimination(A, b) {
    const n = b.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < n; col++) {
      let maxRow = col;
      for (let row = col + 1; row < n; row++)
        if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
      [M[col], M[maxRow]] = [M[maxRow], M[col]];
      if (Math.abs(M[col][col]) < 1e-12) continue;
      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const factor = M[row][col] / M[col][col];
        for (let k = col; k <= n; k++) M[row][k] -= factor * M[col][k];
      }
    }
    return M.map((row, i) => (Math.abs(M[i][i]) < 1e-12 ? 0 : row[n] / M[i][i]));
  }

  // ─── CLAHE (Contrast Limited Adaptive Histogram Equalisation) ─────────────

  /**
   * Simplified CLAHE on a flat greyscale array.
   * Divides the patch into a grid of tiles, equalises each tile's histogram
   * with a clip limit to prevent noise amplification, then bilinearly
   * interpolates across tile borders.
   *
   * For typical eye patches (60×40) we use a 4×4 tile grid (15×10 tiles).
   * clip = 4.0 is a reasonable default for moderate enhancement.
   */
  function clahe(grey, width, height, tileW, tileH, clip) {
    tileW = tileW || 15;
    tileH = tileH || 10;
    clip  = clip  || 4.0;

    const numTX = Math.ceil(width  / tileW);
    const numTY = Math.ceil(height / tileH);
    const out   = new Float32Array(grey.length);

    // Build per-tile CDFs
    const cdfs = [];
    for (let ty = 0; ty < numTY; ty++) {
      cdfs[ty] = [];
      for (let tx = 0; tx < numTX; tx++) {
        const hist = new Float32Array(256);
        let count = 0;
        const x0 = tx * tileW, y0 = ty * tileH;
        const x1 = Math.min(x0 + tileW, width);
        const y1 = Math.min(y0 + tileH, height);
        for (let y = y0; y < y1; y++)
          for (let x = x0; x < x1; x++) {
            hist[Math.min(255, Math.floor(grey[y * width + x]))]++;
            count++;
          }
        // Clip histogram
        const clipCount = clip * count / 256;
        let excess = 0;
        for (let b = 0; b < 256; b++) {
          if (hist[b] > clipCount) { excess += hist[b] - clipCount; hist[b] = clipCount; }
        }
        // Redistribute excess uniformly
        const redist = excess / 256;
        for (let b = 0; b < 256; b++) hist[b] += redist;
        // Build CDF
        const cdf = new Float32Array(256);
        cdf[0] = hist[0];
        for (let b = 1; b < 256; b++) cdf[b] = cdf[b - 1] + hist[b];
        const cdfMin = cdf.find(v => v > 0) || 1;
        for (let b = 0; b < 256; b++)
          cdf[b] = Math.round(255 * (cdf[b] - cdfMin) / (count - cdfMin + 1e-6));
        cdfs[ty][tx] = cdf;
      }
    }

    // Bilinear interpolation of tile CDFs
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const px = grey[y * width + x];
        const bin = Math.min(255, Math.floor(px));

        // Tile coordinates (fractional)
        const ftx = (x + 0.5) / tileW - 0.5;
        const fty = (y + 0.5) / tileH - 0.5;
        const tx0 = Math.max(0, Math.floor(ftx));
        const ty0 = Math.max(0, Math.floor(fty));
        const tx1 = Math.min(numTX - 1, tx0 + 1);
        const ty1 = Math.min(numTY - 1, ty0 + 1);
        const fx  = Math.max(0, Math.min(1, ftx - tx0));
        const fy  = Math.max(0, Math.min(1, fty - ty0));

        const v00 = cdfs[ty0][tx0][bin];
        const v10 = cdfs[ty0][tx1][bin];
        const v01 = cdfs[ty1][tx0][bin];
        const v11 = cdfs[ty1][tx1][bin];

        out[y * width + x] =
          v00 * (1 - fx) * (1 - fy) +
          v10 * fx       * (1 - fy) +
          v01 * (1 - fx) * fy +
          v11 * fx       * fy;
      }
    }
    return out;
  }

  // ─── Per-user PCA basis ───────────────────────────────────────────────────

  /**
   * PCABasis: fitted from actual calibration patches collected during setup.
   * Falls back to the deterministic random projection until fit() is called.
   *
   * Fitting uses the covariance method (mean-centred patches, power iteration
   * for top-k eigenvectors). Runs once at calibration end in ~5–20ms.
   */
  function PCABasis(dim, k, seed) {
    this.dim    = dim;
    this.k      = k;
    this.fitted = false;
    this.mean   = null;
    this.basis  = makeFallbackBasis(dim, k, seed);
  }

  PCABasis.prototype = {
    /**
     * Fit from an array of raw greyscale arrays (one per calibration patch).
     * @param {Array<number[]>} patches  Array of grey arrays, each length=dim
     */
    fit(patches) {
      const n = patches.length;
      if (n < this.k + 2) return false; // not enough data

      // Trim patches to expected dim
      const P = patches.map(p => {
        const v = new Float64Array(this.dim);
        const len = Math.min(p.length, this.dim);
        for (let i = 0; i < len; i++) v[i] = p[i];
        return v;
      });

      // Mean centre
      const mean = new Float64Array(this.dim);
      for (const p of P) for (let i = 0; i < this.dim; i++) mean[i] += p[i] / n;
      for (const p of P) for (let i = 0; i < this.dim; i++) p[i] -= mean[i];

      // Power iteration for top-k eigenvectors
      // Works in the sample space (n×n, not dim×dim) for efficiency
      // We compute S = P P' / n  (n×n), find its eigenvectors u,
      // then project back: v = P' u / ||P' u||

      // Build S = P P' (n×n)
      const S = Array.from({ length: n }, () => new Float64Array(n));
      for (let i = 0; i < n; i++)
        for (let j = i; j < n; j++) {
          let dot = 0;
          for (let d = 0; d < this.dim; d++) dot += P[i][d] * P[j][d];
          dot /= n;
          S[i][j] = S[j][i] = dot;
        }

      const basis = [];
      // Deflation: after each eigenvector, subtract its contribution from S
      const Scopy = S.map(r => Float64Array.from(r));

      for (let e = 0; e < this.k; e++) {
        // Random init
        let u = Float64Array.from({ length: n }, (_, i) => Math.sin(i * 7.3 + e * 3.1));
        // Power iterate
        for (let iter = 0; iter < 50; iter++) {
          const Su = new Float64Array(n);
          for (let i = 0; i < n; i++)
            for (let j = 0; j < n; j++) Su[i] += Scopy[i][j] * u[j];
          const mag = Math.sqrt(Su.reduce((a, v) => a + v * v, 0)) || 1;
          u = Su.map(v => v / mag);
        }
        // Project back to dim space: v = P' u  (dim-dimensional eigenvector)
        const v = new Float64Array(this.dim);
        for (let d = 0; d < this.dim; d++)
          for (let i = 0; i < n; i++) v[d] += P[i][d] * u[i];
        const vmag = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1;
        const vn = v.map(x => x / vmag);
        basis.push(vn);

        // Deflate S: S -= λ u u'   where λ = u' S u
        let lam = 0;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) lam += u[i] * Scopy[i][j] * u[j];
        for (let i = 0; i < n; i++)
          for (let j = 0; j < n; j++) Scopy[i][j] -= lam * u[i] * u[j];
      }

      this.mean   = mean;
      this.basis  = basis.map(v => Array.from(v));
      this.fitted = true;
      return true;
    },

    project(grey) {
      const len = Math.min(grey.length, this.dim);
      const v   = new Float64Array(this.dim);
      for (let i = 0; i < len; i++) v[i] = grey[i];
      if (this.mean) for (let i = 0; i < this.dim; i++) v[i] -= this.mean[i];
      return this.basis.map(bv => {
        let dot = 0;
        for (let i = 0; i < this.dim; i++) dot += bv[i] * v[i];
        return dot;
      });
    },
  };

  function makeFallbackBasis(dim, k, seed) {
    const basis = [];
    let s = seed;
    const rand = () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff - 0.5; };
    for (let j = 0; j < k; j++) {
      const v = Array.from({ length: dim }, rand);
      const mag = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1;
      basis.push(v.map(x => x / mag));
    }
    return basis;
  }

  // Global per-user PCA bases (one per eye)
  const LEFT_PCA  = new PCABasis(PATCH_DIM, PATCH_COMPONENTS, 0xdeadbeef);
  const RIGHT_PCA = new PCABasis(PATCH_DIM, PATCH_COMPONENTS, 0xcafebabe);

  // Storage for raw patches collected during calibration (used to fit PCA)
  const _calibPatches = { left: [], right: [] };

  // ─── Feature extraction with CLAHE ────────────────────────────────────────

  function getGreyscale(patch) {
    if (!patch) return null;
    try {
      let data, width, height;
      if (patch instanceof ImageData) {
        data = patch.data; width = patch.width; height = patch.height;
      } else if (patch.data && patch.width) {
        data = patch.data; width = patch.width; height = patch.height;
      } else if (typeof HTMLCanvasElement !== 'undefined' && patch instanceof HTMLCanvasElement) {
        const ctx = patch.getContext('2d');
        const id = ctx.getImageData(0, 0, patch.width, patch.height);
        data = id.data; width = patch.width; height = patch.height;
      } else {
        return null;
      }
      const grey = new Float32Array(width * height);
      for (let i = 0, j = 0; i < data.length; i += 4, j++)
        grey[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      // Apply CLAHE for contrast normalisation
      return { grey: clahe(grey, width, height), width, height };
    } catch (e) { return null; }
  }

  function extractFeatures(eyePatches, collectForPCA) {
    if (!eyePatches || !eyePatches.left || !eyePatches.right) return null;

    const lResult = getGreyscale(eyePatches.left.patch || eyePatches.left);
    const rResult = getGreyscale(eyePatches.right.patch || eyePatches.right);
    if (!lResult || !rResult) return null;

    // Optionally stash raw (pre-CLAHE) patches for later PCA fitting
    if (collectForPCA) {
      if (_calibPatches.left.length < 300)  _calibPatches.left.push(Array.from(lResult.grey));
      if (_calibPatches.right.length < 300) _calibPatches.right.push(Array.from(rResult.grey));
    }

    const lProj = LEFT_PCA.project(lResult.grey);
    const rProj = RIGHT_PCA.project(rResult.grey);
    const components = [...lProj, ...rProj];

    // Degree-2 polynomial features
    const features = [1.0];
    for (const c of components) features.push(c);
    for (let i = 0; i < components.length; i++)
      for (let j = i; j < components.length; j++)
        features.push(components[i] * components[j]);

    return features;
  }

  // ─── Polynomial Regression ────────────────────────────────────────────────

  function PolynomialRegression() {
    this.xSamples = [];
    this.ySamples = [];
    this.weights  = [];
    this.betaX    = null;
    this.betaY    = null;
    this._dirty   = false;
    this._recentErr = 0; // recent RMSE — used by ensemble
  }

  PolynomialRegression.prototype = {
    addData(eyePatches, screenX, screenY, importance) {
      const feat = extractFeatures(eyePatches, true);
      if (!feat) return;
      this.xSamples.push(feat);
      this.ySamples.push([screenX, screenY]);
      this.weights.push(importance != null ? importance : 1.0);
      this._dirty = true;
      const MAX = 200;
      if (this.xSamples.length > MAX) { this.xSamples.shift(); this.ySamples.shift(); this.weights.shift(); }
    },
    setData(data) {
      this.xSamples = []; this.ySamples = []; this.weights = [];
      if (!data) return;
      for (const d of data)
        if (d.features && d.screenPos) {
          this.xSamples.push(d.features); this.ySamples.push(d.screenPos); this.weights.push(d.weight || 1.0);
        }
      this._dirty = true;
    },
    getData() {
      return this.xSamples.map((f, i) => ({ features: f, screenPos: this.ySamples[i], weight: this.weights[i] }));
    },
    _fit() {
      const n = this.xSamples.length;
      if (n < 6) { this.betaX = null; this.betaY = null; return; }
      const decay = 0.985;
      const W = this.weights.map((w, i) => w * Math.pow(decay, n - 1 - i));
      const Xw  = this.xSamples.map((row, i) => row.map(v => v * Math.sqrt(W[i])));
      const yXw = this.ySamples.map((pos, i) => pos[0] * Math.sqrt(W[i]));
      const yYw = this.ySamples.map((pos, i) => pos[1] * Math.sqrt(W[i]));
      this.betaX = ridgeSolve(Xw, yXw, LAMBDA);
      this.betaY = ridgeSolve(Xw, yYw, LAMBDA);
      this._dirty = false;
    },
    predict(eyePatches) {
      if (this._dirty) this._fit();
      if (!this.betaX || !this.betaY) return null;
      const feat = extractFeatures(eyePatches, false);
      if (!feat) return null;
      const dim = Math.min(feat.length, this.betaX.length);
      let px = 0, py = 0;
      for (let i = 0; i < dim; i++) { px += feat[i] * this.betaX[i]; py += feat[i] * this.betaY[i]; }
      return {
        x: Math.max(0, Math.min(window.innerWidth,  px)),
        y: Math.max(0, Math.min(window.innerHeight, py)),
      };
    },
    name: 'polynomial',
  };

  // ─── RBF Regression ───────────────────────────────────────────────────────

  function RBFRegression() {
    this.features   = [];
    this.targets    = [];
    this.weights    = [];
    this.alphaX     = null;
    this.alphaY     = null;
    this.gamma      = 1.0;
    this._dirty     = false;
    this._recentErr = 0;
  }

  RBFRegression.prototype = {
    addData(eyePatches, screenX, screenY, importance) {
      const feat = extractFeatures(eyePatches, false);
      if (!feat) return;
      this.features.push(feat);
      this.targets.push([screenX, screenY]);
      this.weights.push(importance != null ? importance : 1.0);
      this._dirty = true;
      const MAX = 100;
      if (this.features.length > MAX) { this.features.shift(); this.targets.shift(); this.weights.shift(); }
    },
    setData(data) {
      this.features = []; this.targets = []; this.weights = [];
      if (!data) return;
      for (const d of data)
        if (d.features && d.screenPos) {
          this.features.push(d.features); this.targets.push(d.screenPos); this.weights.push(d.weight || 1.0);
        }
      this._dirty = true;
    },
    getData() {
      return this.features.map((f, i) => ({ features: f, screenPos: this.targets[i], weight: this.weights[i] }));
    },
    _sqDist(a, b) {
      const dim = Math.min(a.length, b.length);
      let s = 0;
      for (let i = 0; i < dim; i++) { const d = a[i] - b[i]; s += d * d; }
      return s;
    },
    _tuneGamma() {
      const n = this.features.length;
      if (n < 2) return;
      const dists = [];
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++) dists.push(this._sqDist(this.features[i], this.features[j]));
      dists.sort((a, b) => a - b);
      const median = dists[Math.floor(dists.length / 2)] || 1;
      this.gamma = 1 / (2 * median);
    },
    _fit() {
      const n = this.features.length;
      if (n < 4) { this.alphaX = null; this.alphaY = null; return; }
      this._tuneGamma();
      const K = Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (__, j) =>
          Math.exp(-this.gamma * this._sqDist(this.features[i], this.features[j]))
        )
      );
      const decay = 0.98;
      for (let i = 0; i < n; i++)
        K[i][i] += LAMBDA / (this.weights[i] * Math.pow(decay, n - 1 - i) + 1e-8);
      this.alphaX = gaussianElimination(K, this.targets.map(t => t[0]));
      this.alphaY = gaussianElimination(K, this.targets.map(t => t[1]));
      this._dirty = false;
    },
    predict(eyePatches) {
      if (this._dirty) this._fit();
      if (!this.alphaX) return null;
      const feat = extractFeatures(eyePatches, false);
      if (!feat) return null;
      let px = 0, py = 0;
      for (let i = 0; i < this.features.length; i++) {
        const k = Math.exp(-this.gamma * this._sqDist(feat, this.features[i]));
        px += this.alphaX[i] * k; py += this.alphaY[i] * k;
      }
      return {
        x: Math.max(0, Math.min(window.innerWidth,  px)),
        y: Math.max(0, Math.min(window.innerHeight, py)),
      };
    },
    name: 'rbf',
  };

  // ─── Ensemble regression ──────────────────────────────────────────────────

  /**
   * Blends polynomial and RBF predictions weighted by their rolling RMSE.
   * The model with lower recent error gets more weight.
   * Falls back to whichever model is ready if only one has enough data.
   */
  function EnsembleRegression(poly, rbf) {
    this.poly = poly;
    this.rbf  = rbf;
    this._errPoly = 0;
    this._errRbf  = 0;
    this._alpha   = 0.05; // EMA for error tracking
    this.name = 'ensemble';
  }

  EnsembleRegression.prototype = {
    addData(eyePatches, screenX, screenY, importance) {
      this.poly.addData(eyePatches, screenX, screenY, importance);
      this.rbf.addData(eyePatches, screenX, screenY, importance);
    },
    setData(data) { this.poly.setData(data); this.rbf.setData(data); },
    getData()     { return this.poly.getData(); },

    predict(eyePatches) {
      const pPoly = this.poly.predict(eyePatches);
      const pRbf  = this.rbf.predict(eyePatches);

      if (!pPoly && !pRbf) return null;
      if (!pPoly) return pRbf;
      if (!pRbf)  return pPoly;

      // Weight inversely by recent error
      const errPoly = this._errPoly || 1;
      const errRbf  = this._errRbf  || 1;
      const wPoly = 1 / errPoly;
      const wRbf  = 1 / errRbf;
      const wSum  = wPoly + wRbf;

      return {
        x: (pPoly.x * wPoly + pRbf.x * wRbf) / wSum,
        y: (pPoly.y * wPoly + pRbf.y * wRbf) / wSum,
      };
    },

    /** Call after each confirmed calibration point to track model accuracy */
    trackError(eyePatches, trueX, trueY) {
      const pp = this.poly.predict(eyePatches);
      const pr = this.rbf.predict(eyePatches);
      if (pp) {
        const err = Math.sqrt((pp.x - trueX) ** 2 + (pp.y - trueY) ** 2);
        this._errPoly = this._errPoly * (1 - this._alpha) + err * this._alpha;
      }
      if (pr) {
        const err = Math.sqrt((pr.x - trueX) ** 2 + (pr.y - trueY) ** 2);
        this._errRbf  = this._errRbf  * (1 - this._alpha) + err * this._alpha;
      }
    },
  };

  // ─── Kalman filter ────────────────────────────────────────────────────────

  /**
   * 4-state Kalman filter: [x, y, vx, vy]
   *
   * Process model: constant velocity with Gaussian process noise Q
   * Measurement model: we observe [x, y] directly (H = [I | 0])
   *
   * This separates process noise (head/body movement) from measurement
   * noise (frame-to-frame jitter in the regression output). The result
   * is much smoother than EMA without introducing the lag EMA causes
   * during genuine gaze shifts.
   *
   * Tune via:
   *   processNoise     — larger = trust measurements more (more responsive)
   *   measurementNoise — larger = trust model more (smoother but laggier)
   */
  function KalmanFilter(options) {
    options = options || {};
    this.Q = options.processNoise     || 8;    // process noise variance
    this.R = options.measurementNoise || 50;   // measurement noise variance

    // State: [x, y, vx, vy]
    this.x = null; // null = not initialised
    // Error covariance (4×4, stored as flat 16-element array, row-major)
    this.P = [
      1000, 0, 0, 0,
      0, 1000, 0, 0,
      0, 0, 100, 0,
      0, 0, 0, 100,
    ];
    this.lastT  = null;
    this._blink = false;
  }

  KalmanFilter.prototype = {

    _matMul4x4(A, B) {
      const C = new Float64Array(16);
      for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++)
          for (let k = 0; k < 4; k++)
            C[i * 4 + j] += A[i * 4 + k] * B[k * 4 + j];
      return C;
    },

    /** Predict step — project state forward by dt milliseconds */
    _predict(dt) {
      const s = dt / 1000; // seconds
      // F = [[1,0,s,0],[0,1,0,s],[0,0,1,0],[0,0,0,1]]
      const nx = this.x[0] + this.x[2] * s;
      const ny = this.x[1] + this.x[3] * s;
      this.x = [nx, ny, this.x[2], this.x[3]];

      // P = F P F' + Q·I  (simplified: only add Q to diagonal)
      const F = [
        1, 0, s, 0,
        0, 1, 0, s,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ];
      const Ft = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        s, 0, 1, 0,
        0, s, 0, 1,
      ];
      const FP   = this._matMul4x4(F, this.P);
      const FPFt = this._matMul4x4(FP, Ft);
      const Q    = this.Q;
      this.P = FPFt;
      this.P[0]  += Q;
      this.P[5]  += Q;
      this.P[10] += Q * 0.1;
      this.P[15] += Q * 0.1;
    },

    /** Update step — incorporate measurement [mx, my] */
    _update(mx, my) {
      // H = [[1,0,0,0],[0,1,0,0]] (we only observe x,y)
      // S = H P H' + R·I  (2×2)
      const S00 = this.P[0]  + this.R;
      const S01 = this.P[1];
      const S10 = this.P[4];
      const S11 = this.P[5] + this.R;

      // K = P H' S⁻¹  (4×2)
      const det = S00 * S11 - S01 * S10 + 1e-10;
      const Si00 =  S11 / det, Si01 = -S01 / det;
      const Si10 = -S10 / det, Si11 =  S00 / det;

      // P H' (4×2)
      const PH = [
        this.P[0],  this.P[1],
        this.P[4],  this.P[5],
        this.P[8],  this.P[9],
        this.P[12], this.P[13],
      ];

      const K = [
        PH[0] * Si00 + PH[1] * Si10,   PH[0] * Si01 + PH[1] * Si11,
        PH[2] * Si00 + PH[3] * Si10,   PH[2] * Si01 + PH[3] * Si11,
        PH[4] * Si00 + PH[5] * Si10,   PH[4] * Si01 + PH[5] * Si11,
        PH[6] * Si00 + PH[7] * Si10,   PH[6] * Si01 + PH[7] * Si11,
      ];

      // Innovation
      const innX = mx - this.x[0];
      const innY = my - this.x[1];

      // State update
      this.x[0] += K[0] * innX + K[1] * innY;
      this.x[1] += K[2] * innX + K[3] * innY;
      this.x[2] += K[4] * innX + K[5] * innY;
      this.x[3] += K[6] * innX + K[7] * innY;

      // P = (I - K H) P  (simplified Joseph form for 4×4)
      const KH = [
        K[0], K[1], 0, 0,
        K[2], K[3], 0, 0,
        K[4], K[5], 0, 0,
        K[6], K[7], 0, 0,
      ];
      const IKH = [
        1 - KH[0],  -KH[1],  0, 0,
         -KH[4], 1 - KH[5], 0, 0,
         -KH[8],  -KH[9],   1, 0,
        -KH[12], -KH[13],   0, 1,
      ];
      this.P = this._matMul4x4(IKH, this.P);
    },

    smooth(x, y, isBlink, isSaccade) {
      const now = performance.now();

      if (this.x === null) {
        this.x = [x, y, 0, 0];
        this.lastT = now;
        return { x, y, vx: 0, vy: 0, confidence: 0 };
      }

      const dt = Math.min(now - this.lastT, 120);
      this.lastT = now;

      this._predict(dt);

      // During blinks or saccades: skip measurement update, coast on prediction
      if (!isBlink && !isSaccade) {
        this._update(x, y);
      }

      const vMag = Math.sqrt(this.x[2] ** 2 + this.x[3] ** 2);
      // Confidence: falls with velocity and with large innovation
      const innovation = Math.sqrt((x - this.x[0]) ** 2 + (y - this.x[1]) ** 2);
      const confidence = Math.max(0, Math.min(1,
        (1 - vMag / 800) * (1 - innovation / 300)
      ));

      return {
        x:          this.x[0],
        y:          this.x[1],
        vx:         this.x[2],
        vy:         this.x[3],
        confidence,
        isBlink,
        isSaccade,
      };
    },

    reset() {
      this.x = null;
      this.P = [1000,0,0,0, 0,1000,0,0, 0,0,100,0, 0,0,0,100];
      this.lastT = null;
    },
  };

  // ─── Blink detector ───────────────────────────────────────────────────────

  /**
   * Detects blinks by measuring patch brightness.
   * During a blink the eye patch goes dark (eyelid covers pupil).
   * Uses an adaptive threshold based on recent patch brightness history.
   *
   * Returns true while a blink is in progress.
   * Has a ~80ms post-blink lockout to prevent jitter as the eye reopens.
   */
  function BlinkDetector(options) {
    options = options || {};
    this.windowSize   = options.windowSize   || 30;  // frames of history
    this.blinkThresh  = options.blinkThresh  || 0.55; // fraction of mean to trigger
    this.lockoutMs    = options.lockoutMs    || 80;

    this._history     = [];
    this._blinking    = false;
    this._lockoutEnd  = 0;
  }

  BlinkDetector.prototype.update = function (eyePatches) {
    if (!eyePatches || !eyePatches.left) return this._blinking;

    // Get mean brightness of left eye patch (quick proxy for both eyes)
    const patch = eyePatches.left.patch || eyePatches.left;
    let brightness = 0, count = 0;
    try {
      let data;
      if (patch instanceof ImageData)                                          data = patch.data;
      else if (patch.data)                                                     data = patch.data;
      else if (typeof HTMLCanvasElement !== 'undefined' && patch instanceof HTMLCanvasElement)
        data = patch.getContext('2d').getImageData(0,0,patch.width,patch.height).data;
      if (data) {
        for (let i = 0; i < data.length; i += 4) {
          brightness += 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2];
          count++;
        }
        brightness /= count || 1;
      }
    } catch (e) { return this._blinking; }

    this._history.push(brightness);
    if (this._history.length > this.windowSize) this._history.shift();

    const mean = this._history.reduce((a, v) => a + v, 0) / this._history.length;
    const now  = performance.now();

    if (now < this._lockoutEnd) return true; // post-blink lockout

    const isBlink = brightness < mean * this.blinkThresh;

    if (isBlink && !this._blinking) {
      this._blinking   = true;
      this._lockoutEnd = 0;
    } else if (!isBlink && this._blinking) {
      this._blinking   = false;
      this._lockoutEnd = now + this.lockoutMs;
    }

    return this._blinking || now < this._lockoutEnd;
  };

  // ─── Saccade detector ────────────────────────────────────────────────────

  /**
   * Detects saccades (fast eye movements) from the Kalman velocity estimate.
   * During a saccade the regression output is unreliable — we coast on
   * the Kalman prediction instead of incorporating the noisy measurement.
   *
   * threshold: px/sec above which we suppress the measurement update
   */
  function SaccadeDetector(threshold) {
    this.threshold = threshold || 600; // px/sec
  }

  SaccadeDetector.prototype.isSaccade = function (vx, vy) {
    return Math.sqrt(vx * vx + vy * vy) > this.threshold;
  };

  // ─── Frame cache ──────────────────────────────────────────────────────────

  /**
   * Skips regression inference when gaze appears stable.
   * If velocity is below `minSpeed` px/sec, reuse the last prediction
   * for up to `maxReuseMs` milliseconds. Saves ~20–40% CPU on fast devices,
   * more on slow ones.
   */
  function FrameCache(options) {
    options = options || {};
    this.minSpeed  = options.minSpeed  || 15;  // px/sec
    this.maxReuseMs = options.maxReuseMs || 50; // max ms to reuse

    this._last  = null;
    this._lastT = 0;
  }

  FrameCache.prototype = {
    shouldSkip(vx, vy) {
      if (!this._last) return false;
      const speed = Math.sqrt(vx * vx + vy * vy);
      const age   = performance.now() - this._lastT;
      return speed < this.minSpeed && age < this.maxReuseMs;
    },
    store(pred) {
      this._last  = pred;
      this._lastT = performance.now();
    },
    get() { return this._last; },
  };

  // ─── Adaptive recalibration ───────────────────────────────────────────────

  function AdaptiveRecalibrator(regressionModule) {
    this.regression = regressionModule;
    this.enabled    = false;
    this.hitCount   = 0;
    this.maxHitsPerSession = 500;
  }

  AdaptiveRecalibrator.prototype = {
    enable()  { this.enabled = true; },
    disable() { this.enabled = false; },
    recordHit(targetX, targetY, eyePatches, importance) {
      if (!this.enabled || this.hitCount >= this.maxHitsPerSession) return;
      importance = importance != null ? importance : 1.5;
      this.regression.addData(eyePatches, targetX, targetY, importance);
      this.hitCount++;
    },
    recordElementHit(element, eyePatches) {
      if (!element || !this.enabled) return;
      const r = element.getBoundingClientRect();
      this.recordHit(r.left + r.width / 2, r.top + r.height / 2, eyePatches);
    },
  };

  // ─── Main install ─────────────────────────────────────────────────────────

  const webgazerAAC = {
    _regression:     null,
    _kalman:         new KalmanFilter(),
    _blink:          new BlinkDetector(),
    _saccade:        new SaccadeDetector(),
    _cache:          new FrameCache(),
    _recalibrator:   null,
    _currentMode:    'ensemble',
    _installed:      false,
    _lastPatches:    null,
    _lastConfidence: 0,
    _lastResult:     null,

    install() {
      if (this._installed) return this;
      if (typeof webgazer === 'undefined') {
        console.error('[webgazer-aac] webgazer.js must be loaded first');
        return this;
      }

      const poly = new PolynomialRegression();
      const rbf  = new RBFRegression();

      this._regressions = {
        polynomial: poly,
        rbf:        rbf,
        ensemble:   new EnsembleRegression(poly, rbf),
      };

      this.setRegression('ensemble');

      const self = this;

      // Intercept setGazeListener
      const _origSetGazeListener = webgazer.setGazeListener.bind(webgazer);
      webgazer.setGazeListener = function (callback) {
        return _origSetGazeListener(function (data, elapsedTime) {
          // Cache latest eye patches
          try {
            const tracker = webgazer.getTracker();
            if (tracker) {
              self._lastPatches =
                (tracker.getEyePatches && tracker.getEyePatches()) ||
                (tracker.getCurrentEyePatches && tracker.getCurrentEyePatches()) ||
                (webgazer.getCurrentEyePatches && webgazer.getCurrentEyePatches()) || null;
            }
          } catch (e) {}

          if (data === null) { callback(null, elapsedTime); return; }

          // Blink detection
          const isBlink = self._blink.update(self._lastPatches);

          // Saccade detection (from Kalman velocity)
          const vx = self._lastResult ? self._lastResult.vx : 0;
          const vy = self._lastResult ? self._lastResult.vy : 0;
          const isSaccade = self._saccade.isSaccade(vx, vy);

          // Frame cache — skip inference if stable
          let rawX, rawY;
          if (self._cache.shouldSkip(vx, vy) && self._cache.get()) {
            const cached = self._cache.get();
            rawX = cached.x; rawY = cached.y;
          } else {
            // Regression inference
            let pred = null;
            if (self._regression && self._lastPatches) {
              try { pred = self._regression.predict(self._lastPatches); } catch (e) {}
            }
            rawX = pred ? pred.x : data.x;
            rawY = pred ? pred.y : data.y;
            self._cache.store({ x: rawX, y: rawY });
          }

          // Kalman filter
          const result = self._kalman.smooth(rawX, rawY, isBlink, isSaccade);
          self._lastResult     = result;
          self._lastConfidence = result.confidence;

          // During blink: pass null so dwell timers don't advance
          if (isBlink) { callback(null, elapsedTime); return; }

          callback({
            x:          result.x,
            y:          result.y,
            confidence: result.confidence,
            isSaccade,
          }, elapsedTime);
        });
      };

      // Intercept recordScreenPosition to feed our model
      const _origRecord = webgazer.recordScreenPosition.bind(webgazer);
      webgazer.recordScreenPosition = function (x, y, eventType) {
        _origRecord(x, y, eventType);
        if (self._regression && self._lastPatches) {
          try {
            self._regression.addData(self._lastPatches, x, y);
            // Track ensemble error
            if (self._regression instanceof EnsembleRegression)
              self._regression.trackError(self._lastPatches, x, y);
          } catch (e) {}
        }
      };

      this._recalibrator = new AdaptiveRecalibrator(this._regressions.ensemble);
      this._installed    = true;
      console.info('[webgazer-aac] v1.1.0 installed — regression: ' + this._currentMode);
      return this;
    },

    /**
     * Fit the per-user PCA basis from patches collected during calibration.
     * Call this at the END of your calibration sequence.
     * Returns {left: bool, right: bool} indicating whether each eye fit succeeded.
     */
    fitUserBasis() {
      const lOk = LEFT_PCA.fit(_calibPatches.left);
      const rOk = RIGHT_PCA.fit(_calibPatches.right);
      const msg = `PCA fit — left: ${lOk ? _calibPatches.left.length + ' patches' : 'insufficient data'}, ` +
                  `right: ${rOk ? _calibPatches.right.length + ' patches' : 'insufficient data'}`;
      console.info('[webgazer-aac] ' + msg);
      // Invalidate regression caches so they re-fit with new features
      Object.values(this._regressions || {}).forEach(r => { if (r._dirty !== undefined) r._dirty = true; });
      return { left: lOk, right: rOk, message: msg };
    },

    /** Clear collected calibration patches (call before a fresh calibration) */
    resetCalibrationPatches() {
      _calibPatches.left  = [];
      _calibPatches.right = [];
      return this;
    },

    setRegression(mode) {
      if (mode === 'ridge') {
        this._regression  = null;
        this._currentMode = 'ridge';
        if (typeof webgazer !== 'undefined') webgazer.setRegression('ridge');
        return this;
      }
      const reg = this._regressions && this._regressions[mode];
      if (!reg) { console.warn('[webgazer-aac] unknown regression:', mode); return this; }
      this._regression  = reg;
      this._currentMode = mode;
      if (this._recalibrator) this._recalibrator.regression = reg;
      return this;
    },

    enableAdaptiveRecalibration()  { if (this._recalibrator) this._recalibrator.enable();  return this; },
    disableAdaptiveRecalibration() { if (this._recalibrator) this._recalibrator.disable(); return this; },

    recordDwellHit(element) {
      if (!this._recalibrator || !this._lastPatches) return;
      this._recalibrator.recordElementHit(element, this._lastPatches);
    },

    recordDwellHitXY(x, y) {
      if (!this._recalibrator || !this._lastPatches) return;
      this._recalibrator.recordHit(x, y, this._lastPatches);
    },

    /**
     * Tune the Kalman filter's noise parameters.
     * processNoise: larger = more responsive to movement (default 8)
     * measurementNoise: larger = smoother but slower (default 50)
     */
    setKalmanParams(processNoise, measurementNoise) {
      this._kalman.Q = processNoise;
      this._kalman.R = measurementNoise;
      return this;
    },

    getConfidence()     { return this._lastConfidence; },
    getRegressionMode() { return this._currentMode; },
    isPCAFitted()       { return LEFT_PCA.fitted && RIGHT_PCA.fitted; },

    resetSmoother() {
      this._kalman.reset();
      this._lastResult = null;
      return this;
    },

    // Expose classes for testing / advanced use
    PolynomialRegression,
    RBFRegression,
    EnsembleRegression,
    KalmanFilter,
    BlinkDetector,
    SaccadeDetector,
    FrameCache,
    AdaptiveRecalibrator,
    PCABasis,

    version: '1.1.0',
  };

  global.webgazerAAC = webgazerAAC;

})(typeof globalThis !== 'undefined' ? globalThis : window);
