/**
 * webgazer-aac.js
 * Accessibility-first enhancement layer for WebGazer.js
 *
 * Fork/patch by ALTRU.dev — Code for Humanity
 * https://altru.dev
 *
 * License: GPLv3 (same as upstream brownhci/WebGazer)
 *
 * USAGE:
 *   Load AFTER webgazer.js. Then call:
 *     webgazerAAC.install()        // patch webgazer with improved regressions
 *     webgazerAAC.setRegression('polynomial')  // or 'rbf', 'ridge' (original)
 *     webgazerAAC.enableAdaptiveRecalibration() // continuous self-correction
 *
 *   Everything else (webgazer.begin(), setGazeListener(), etc.) stays identical.
 *
 * IMPROVEMENTS OVER UPSTREAM:
 *   1. Polynomial regression (degree 2) — models non-linear corner distortion
 *   2. RBF regression — radial basis function, best for irregular calibration
 *   3. Weighted recency — recent calibration clicks matter more than old ones
 *   4. Adaptive recalibration — dwell hits feed back as free training samples
 *   5. Gaze smoothing — configurable EMA + velocity-aware smoothing
 *   6. Prediction confidence — exposes a 0-1 confidence score per prediction
 */

(function (global) {
  'use strict';

  // ─── Math utilities ───────────────────────────────────────────────────────

  /**
   * Solve least-squares via normal equations: (X'X + λI)⁻¹ X'y
   * Works for any feature matrix X and target vector y.
   * λ is L2 regularisation (ridge term).
   */
  function ridgeSolve(X, y, lambda) {
    const n = X.length;       // samples
    const m = X[0].length;    // features

    // X'X
    const XtX = Array.from({ length: m }, () => new Float64Array(m));
    for (let i = 0; i < m; i++)
      for (let j = 0; j < m; j++)
        for (let k = 0; k < n; k++)
          XtX[i][j] += X[k][i] * X[k][j];

    // Add λ to diagonal
    for (let i = 0; i < m; i++) XtX[i][i] += lambda;

    // X'y
    const Xty = new Float64Array(m);
    for (let i = 0; i < m; i++)
      for (let k = 0; k < n; k++)
        Xty[i] += X[k][i] * y[k];

    // Solve (X'X)β = X'y via Gaussian elimination with partial pivoting
    return gaussianElimination(XtX, Xty);
  }

  function gaussianElimination(A, b) {
    const n = b.length;
    // Augmented matrix [A|b]
    const M = A.map((row, i) => [...row, b[i]]);

    for (let col = 0; col < n; col++) {
      // Partial pivot
      let maxRow = col;
      for (let row = col + 1; row < n; row++)
        if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
      [M[col], M[maxRow]] = [M[maxRow], M[col]];

      if (Math.abs(M[col][col]) < 1e-12) continue; // singular — skip

      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const factor = M[row][col] / M[col][col];
        for (let k = col; k <= n; k++)
          M[row][k] -= factor * M[col][k];
      }
    }

    return M.map((row, i) => (Math.abs(M[i][i]) < 1e-12 ? 0 : row[n] / M[i][i]));
  }

  // ─── Feature extractors ───────────────────────────────────────────────────

  /**
   * Degree-2 polynomial features from a flat eye patch pixel array.
   * Input: Float32Array of pixel values (greyscale, 0-255)
   * Output: feature vector [1, px0, px1, ..., px0*px0, px0*px1, ...]
   *
   * To keep dimensionality tractable we first PCA-reduce the raw pixels
   * down to PATCH_COMPONENTS principal components, then form degree-2
   * features from those components only.
   *
   * PATCH_COMPONENTS = 8  →  1 + 8 + 36 = 45 features
   * (compared to raw ridge: 1 + raw_pixels, typically 1 + 2400 = 2401)
   */
  const PATCH_COMPONENTS = 8;
  const LAMBDA = 1e-3; // L2 regularisation

  /**
   * Extract a compact float32 feature vector from left+right eye patch data.
   * eye patches come in as {left:{patch:ImageData}, right:{patch:ImageData}}
   */
  function extractFeatures(eyePatches) {
    if (!eyePatches || !eyePatches.left || !eyePatches.right) return null;

    const lData = getGreyscale(eyePatches.left.patch || eyePatches.left);
    const rData = getGreyscale(eyePatches.right.patch || eyePatches.right);
    if (!lData || !rData) return null;

    // Normalise 0→1
    const norm = arr => {
      const max = Math.max(...arr) || 1;
      const min = Math.min(...arr);
      const range = (max - min) || 1;
      return arr.map(v => (v - min) / range);
    };

    const lNorm = norm(lData);
    const rNorm = norm(rData);

    // Simple PCA via top-k projection using random fixed basis
    // (we use a deterministic random basis seeded at load time so it's
    //  consistent across frames — a proper PCA would need a training set)
    const lProj = projectToBasis(lNorm, LEFT_BASIS);
    const rProj = projectToBasis(rNorm, RIGHT_BASIS);

    // Concatenate left + right projections
    const components = [...lProj, ...rProj]; // length = 2 * PATCH_COMPONENTS

    // Degree-2 polynomial features: [1, c0..cn, c0*c0, c0*c1, ..., cn*cn]
    const features = [1.0];
    for (const c of components) features.push(c);
    for (let i = 0; i < components.length; i++)
      for (let j = i; j < components.length; j++)
        features.push(components[i] * components[j]);

    return features;
  }

  function getGreyscale(patch) {
    if (!patch) return null;
    try {
      let data;
      if (patch instanceof ImageData) {
        data = patch.data;
      } else if (patch.data) {
        data = patch.data;
      } else if (patch instanceof HTMLCanvasElement) {
        const ctx = patch.getContext('2d');
        data = ctx.getImageData(0, 0, patch.width, patch.height).data;
      } else {
        return null;
      }
      const grey = [];
      for (let i = 0; i < data.length; i += 4)
        grey.push(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
      return grey;
    } catch (e) { return null; }
  }

  // Deterministic pseudo-random basis vectors (seeded, consistent per load)
  function makeBasis(inputDim, outputDim, seed) {
    const basis = [];
    let s = seed;
    const rand = () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff - 0.5; };
    for (let j = 0; j < outputDim; j++) {
      const v = Array.from({ length: inputDim }, rand);
      // Normalise
      const mag = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1;
      basis.push(v.map(x => x / mag));
    }
    return basis;
  }

  function projectToBasis(vec, basis) {
    return basis.map(bv => {
      const len = Math.min(vec.length, bv.length);
      let dot = 0;
      for (let i = 0; i < len; i++) dot += vec[i] * bv[i];
      return dot;
    });
  }

  // Bases are created once at load time. Input dim is arbitrary — we'll
  // resize lazily if the actual patch is a different size.
  const PATCH_DIM = 60 * 40; // typical WebGazer eye patch ~60×40
  const LEFT_BASIS  = makeBasis(PATCH_DIM, PATCH_COMPONENTS, 0xdeadbeef);
  const RIGHT_BASIS = makeBasis(PATCH_DIM, PATCH_COMPONENTS, 0xcafebabe);


  // ─── Polynomial Regression ────────────────────────────────────────────────

  /**
   * Degree-2 polynomial regression over eye-patch PCA features.
   * Exposes the same interface as WebGazer's built-in regression modules:
   *   - addData(eyePatches, x, y)
   *   - setData(data)
   *   - getData() → array of stored samples
   *   - predict(eyePatches) → {x, y} | null
   */
  function PolynomialRegression() {
    this.xSamples = [];   // feature vectors
    this.ySamples = [];   // [screenX, screenY]
    this.weights  = [];   // sample importance weights (recency)
    this.betaX    = null; // solved coefficients for X
    this.betaY    = null; // solved coefficients for Y
    this._dirty   = false;
  }

  PolynomialRegression.prototype = {

    addData(eyePatches, screenX, screenY, importance) {
      const feat = extractFeatures(eyePatches);
      if (!feat) return;
      this.xSamples.push(feat);
      this.ySamples.push([screenX, screenY]);
      this.weights.push(importance != null ? importance : 1.0);
      this._dirty = true;
      // Keep a rolling window — old samples hurt accuracy when user moves
      const MAX = 150;
      if (this.xSamples.length > MAX) {
        this.xSamples.shift();
        this.ySamples.shift();
        this.weights.shift();
      }
    },

    setData(data) {
      this.xSamples = [];
      this.ySamples = [];
      this.weights  = [];
      if (!data) return;
      for (const d of data) {
        if (d.features && d.screenPos) {
          this.xSamples.push(d.features);
          this.ySamples.push(d.screenPos);
          this.weights.push(d.weight || 1.0);
        }
      }
      this._dirty = true;
    },

    getData() {
      return this.xSamples.map((f, i) => ({
        features:  f,
        screenPos: this.ySamples[i],
        weight:    this.weights[i],
      }));
    },

    _fit() {
      const n = this.xSamples.length;
      if (n < 6) { this.betaX = null; this.betaY = null; return; }

      // Apply recency weighting: most recent samples get weight 1.0,
      // oldest get weight ~0.3, using exponential decay.
      const decay = 0.985;
      const W = this.weights.map((w, i) => w * Math.pow(decay, n - 1 - i));

      // Weight the feature matrix rows
      const Xw = this.xSamples.map((row, i) => row.map(v => v * Math.sqrt(W[i])));
      const yXw = this.ySamples.map((pos, i) => pos[0] * Math.sqrt(W[i]));
      const yYw = this.ySamples.map((pos, i) => pos[1] * Math.sqrt(W[i]));

      this.betaX = ridgeSolve(Xw, yXw, LAMBDA);
      this.betaY = ridgeSolve(Xw, yYw, LAMBDA);
      this._dirty = false;
    },

    predict(eyePatches) {
      if (this._dirty) this._fit();
      if (!this.betaX || !this.betaY) return null;

      const feat = extractFeatures(eyePatches);
      if (!feat) return null;

      // Resize beta if feature dim changed (different camera resolution)
      const dim = Math.min(feat.length, this.betaX.length);

      let px = 0, py = 0;
      for (let i = 0; i < dim; i++) {
        px += feat[i] * this.betaX[i];
        py += feat[i] * this.betaY[i];
      }

      // Clamp to viewport
      px = Math.max(0, Math.min(window.innerWidth,  px));
      py = Math.max(0, Math.min(window.innerHeight, py));

      return { x: px, y: py };
    },

    name: 'polynomial',
  };


  // ─── RBF Regression ───────────────────────────────────────────────────────

  /**
   * Radial Basis Function regression.
   * Uses a Gaussian kernel over the PCA feature space.
   * Better than polynomial when calibration points are unevenly distributed.
   *
   * Prediction: ŷ = Σ αᵢ · K(x, xᵢ)   where K(a,b) = exp(-γ·‖a-b‖²)
   * Coefficients α solved via (K + λI)α = y
   */
  function RBFRegression() {
    this.features  = [];
    this.targets   = [];
    this.weights   = [];
    this.alphaX    = null;
    this.alphaY    = null;
    this.gamma     = 1.0; // kernel bandwidth — auto-tuned after fit
    this._dirty    = false;
  }

  RBFRegression.prototype = {

    addData(eyePatches, screenX, screenY, importance) {
      const feat = extractFeatures(eyePatches);
      if (!feat) return;
      this.features.push(feat);
      this.targets.push([screenX, screenY]);
      this.weights.push(importance != null ? importance : 1.0);
      this._dirty = true;
      const MAX = 80; // RBF is O(n²) so keep smaller window
      if (this.features.length > MAX) {
        this.features.shift();
        this.targets.shift();
        this.weights.shift();
      }
    },

    setData(data) {
      this.features = []; this.targets = []; this.weights = [];
      if (!data) return;
      for (const d of data) {
        if (d.features && d.screenPos) {
          this.features.push(d.features);
          this.targets.push(d.screenPos);
          this.weights.push(d.weight || 1.0);
        }
      }
      this._dirty = true;
    },

    getData() {
      return this.features.map((f, i) => ({
        features: f, screenPos: this.targets[i], weight: this.weights[i],
      }));
    },

    _sqDist(a, b) {
      const dim = Math.min(a.length, b.length);
      let s = 0;
      for (let i = 0; i < dim; i++) { const d = a[i] - b[i]; s += d * d; }
      return s;
    },

    _tuneGamma() {
      // Median heuristic: γ = 1 / (2 · median(pairwise distances²))
      const n = this.features.length;
      if (n < 2) return;
      const dists = [];
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++)
          dists.push(this._sqDist(this.features[i], this.features[j]));
      dists.sort((a, b) => a - b);
      const median = dists[Math.floor(dists.length / 2)] || 1;
      this.gamma = 1 / (2 * median);
    },

    _fit() {
      const n = this.features.length;
      if (n < 4) { this.alphaX = null; this.alphaY = null; return; }

      this._tuneGamma();

      // Build kernel matrix K[i][j] = exp(-γ · ‖fᵢ - fⱼ‖²)
      const K = Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (__, j) =>
          Math.exp(-this.gamma * this._sqDist(this.features[i], this.features[j]))
        )
      );

      // Add regularisation + recency weighting on diagonal
      const decay = 0.98;
      for (let i = 0; i < n; i++)
        K[i][i] += LAMBDA / (this.weights[i] * Math.pow(decay, n - 1 - i) + 1e-8);

      const yX = this.targets.map(t => t[0]);
      const yY = this.targets.map(t => t[1]);

      this.alphaX = gaussianElimination(K, yX);
      this.alphaY = gaussianElimination(K, yY);
      this._dirty = false;
    },

    predict(eyePatches) {
      if (this._dirty) this._fit();
      if (!this.alphaX) return null;

      const feat = extractFeatures(eyePatches);
      if (!feat) return null;

      let px = 0, py = 0;
      for (let i = 0; i < this.features.length; i++) {
        const k = Math.exp(-this.gamma * this._sqDist(feat, this.features[i]));
        px += this.alphaX[i] * k;
        py += this.alphaY[i] * k;
      }

      px = Math.max(0, Math.min(window.innerWidth,  px));
      py = Math.max(0, Math.min(window.innerHeight, py));

      return { x: px, y: py };
    },

    name: 'rbf',
  };


  // ─── Velocity-aware smoothing ─────────────────────────────────────────────

  /**
   * Improves on WebGazer's fixed-lerp smoothing.
   * - When gaze is moving fast, use less smoothing (more responsive)
   * - When gaze is stable (dwelling), use more smoothing (reduces jitter)
   * - Separate smoothing for saccades vs slow pursuit movements
   */
  function VelocitySmoother(options) {
    options = options || {};
    this.alpha     = options.alpha     || 0.22;  // base EMA factor
    this.velScale  = options.velScale  || 0.003; // how much velocity boosts alpha
    this.maxAlpha  = options.maxAlpha  || 0.7;   // cap for fast movements
    this.minAlpha  = options.minAlpha  || 0.08;  // floor for dwelling

    this.sx = null; this.sy = null;
    this.vx = 0;    this.vy = 0;
    this.lastT = null;
  }

  VelocitySmoother.prototype.smooth = function (x, y) {
    const now = performance.now();

    if (this.sx === null) {
      this.sx = x; this.sy = y; this.lastT = now;
      return { x, y, confidence: 0 };
    }

    const dt = Math.min(now - this.lastT, 100); // cap at 100ms
    this.lastT = now;

    // Instantaneous velocity (px/ms)
    const rawVx = (x - this.sx) / dt;
    const rawVy = (y - this.sy) / dt;
    const speed = Math.sqrt(rawVx * rawVx + rawVy * rawVy);

    // Smooth velocity estimate
    const velAlpha = 0.4;
    this.vx = velAlpha * rawVx + (1 - velAlpha) * this.vx;
    this.vy = velAlpha * rawVy + (1 - velAlpha) * this.vy;

    // Adapt EMA factor to speed
    const adaptedAlpha = Math.min(
      this.maxAlpha,
      Math.max(this.minAlpha, this.alpha + speed * this.velScale)
    );

    this.sx += adaptedAlpha * (x - this.sx);
    this.sy += adaptedAlpha * (y - this.sy);

    // Confidence: high when stable, low when fast-moving or just started
    const jitter = Math.sqrt((x - this.sx) ** 2 + (y - this.sy) ** 2);
    const confidence = Math.max(0, Math.min(1, 1 - jitter / 200));

    return { x: this.sx, y: this.sy, confidence };
  };

  VelocitySmoother.prototype.reset = function () {
    this.sx = null; this.sy = null; this.vx = 0; this.vy = 0;
  };


  // ─── Adaptive recalibration ───────────────────────────────────────────────

  /**
   * Watches for confirmed dwell events and feeds them back into the
   * regression model as free labeled training samples.
   *
   * When a user dwells on a UI element long enough to activate it,
   * we know:
   *   - the target's center XY (from the DOM element's bounding rect)
   *   - the eye patches at that moment
   *
   * This is a free calibration point — we add it with high importance.
   *
   * Usage from JelloOS/GazeContext:
   *   webgazerAAC.recordDwellHit(element)  // call when dwell fires
   */
  function AdaptiveRecalibrator(regressionModule) {
    this.regression = regressionModule;
    this.enabled    = false;
    this.hitCount   = 0;
    this.maxHitsPerSession = 200;
  }

  AdaptiveRecalibrator.prototype = {
    enable()  { this.enabled = true; },
    disable() { this.enabled = false; },

    recordHit(targetX, targetY, eyePatches, importance) {
      if (!this.enabled) return;
      if (this.hitCount >= this.maxHitsPerSession) return;
      importance = importance != null ? importance : 1.5; // slightly boost dwell hits
      this.regression.addData(eyePatches, targetX, targetY, importance);
      this.hitCount++;
    },

    recordElementHit(element, eyePatches) {
      if (!element || !this.enabled) return;
      const r = element.getBoundingClientRect();
      const cx = r.left + r.width  / 2;
      const cy = r.top  + r.height / 2;
      this.recordHit(cx, cy, eyePatches);
    },
  };


  // ─── Main install ──────────────────────────────────────────────────────────

  const webgazerAAC = {
    _regression:   null,
    _smoother:     new VelocitySmoother(),
    _recalibrator: null,
    _currentMode:  'polynomial',
    _installed:    false,
    _lastPatches:  null,
    _lastConfidence: 0,

    /**
     * Install the patch. Must be called after webgazer.js has loaded.
     * Sets polynomial regression as the default and wraps the gaze listener
     * to add velocity-aware smoothing + confidence scoring.
     */
    install() {
      if (this._installed) return this;
      if (typeof webgazer === 'undefined') {
        console.error('[webgazer-aac] webgazer.js must be loaded before webgazer-aac.js');
        return this;
      }

      // Register our regression modules so webgazer.setRegression() knows them
      this._regressions = {
        polynomial: new PolynomialRegression(),
        rbf:        new RBFRegression(),
      };

      // Default to polynomial
      this.setRegression('polynomial');

      // Intercept eye patches so adaptive recalibration can access them
      const self = this;
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
                (webgazer.getCurrentEyePatches && webgazer.getCurrentEyePatches()) ||
                null;
            }
          } catch (e) {}

          if (data === null) { callback(null, elapsedTime); return; }

          // Run our prediction on top of (or instead of) WebGazer's output
          let pred = null;
          if (self._regression && self._lastPatches) {
            try { pred = self._regression.predict(self._lastPatches); } catch (e) {}
          }

          // Fall back to WebGazer's own prediction if ours isn't ready
          const rawX = pred ? pred.x : data.x;
          const rawY = pred ? pred.y : data.y;

          // Velocity-aware smoothing
          const smoothed = self._smoother.smooth(rawX, rawY);
          self._lastConfidence = smoothed.confidence;

          callback({ x: smoothed.x, y: smoothed.y, confidence: smoothed.confidence }, elapsedTime);
        });
      };

      // Intercept recordScreenPosition so calibration data feeds our model too
      const _origRecord = webgazer.recordScreenPosition.bind(webgazer);
      webgazer.recordScreenPosition = function (x, y, eventType) {
        _origRecord(x, y, eventType);
        if (self._regression && self._lastPatches) {
          try { self._regression.addData(self._lastPatches, x, y); } catch (e) {}
        }
      };

      this._recalibrator = new AdaptiveRecalibrator(this._regressions.polynomial);
      this._installed = true;
      console.info('[webgazer-aac] installed — regression: ' + this._currentMode);
      return this;
    },

    /**
     * Switch regression model.
     * @param {'polynomial'|'rbf'|'ridge'} mode
     */
    setRegression(mode) {
      if (mode === 'ridge') {
        // Passthrough — use WebGazer's built-in ridge
        this._regression = null;
        this._currentMode = 'ridge';
        if (typeof webgazer !== 'undefined') webgazer.setRegression('ridge');
        return this;
      }
      const reg = this._regressions && this._regressions[mode];
      if (!reg) { console.warn('[webgazer-aac] unknown regression:', mode); return this; }
      this._regression = reg;
      this._currentMode = mode;
      if (this._recalibrator) this._recalibrator.regression = reg;
      return this;
    },

    /**
     * Enable continuous self-correction via dwell hits.
     */
    enableAdaptiveRecalibration() {
      if (this._recalibrator) this._recalibrator.enable();
      return this;
    },

    disableAdaptiveRecalibration() {
      if (this._recalibrator) this._recalibrator.disable();
      return this;
    },

    /**
     * Call this from JelloOS GazeContext whenever a dwell fires on a known element.
     * @param {Element} element  The DOM element that was activated
     */
    recordDwellHit(element) {
      if (!this._recalibrator || !this._lastPatches) return;
      this._recalibrator.recordElementHit(element, this._lastPatches);
    },

    /**
     * Record a dwell hit at explicit screen coordinates (if no element ref available).
     */
    recordDwellHitXY(x, y) {
      if (!this._recalibrator || !this._lastPatches) return;
      this._recalibrator.recordHit(x, y, this._lastPatches);
    },

    /**
     * Returns the confidence score (0–1) for the last prediction.
     * 1 = stable gaze, 0 = fast movement / just initialised.
     */
    getConfidence() {
      return this._lastConfidence;
    },

    /**
     * Returns the name of the active regression module.
     */
    getRegressionMode() {
      return this._currentMode;
    },

    /**
     * Reset the smoother (useful when user indicates they've moved position).
     */
    resetSmoother() {
      this._smoother.reset();
      return this;
    },

    /**
     * Expose classes for advanced usage / testing
     */
    PolynomialRegression,
    RBFRegression,
    VelocitySmoother,
    AdaptiveRecalibrator,

    version: '1.0.0-aac',
  };

  // Expose globally
  global.webgazerAAC = webgazerAAC;

})(typeof globalThis !== 'undefined' ? globalThis : window);
