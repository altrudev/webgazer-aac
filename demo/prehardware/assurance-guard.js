'use strict';

(function () {
  const aac = window.webgazerAAC;
  if (!aac || aac.__assuranceGuardInstalled) return;

  const state = {
    adaptiveRequested: false,
    driftRequested: false,
    adaptiveActivated: false,
    driftActivated: false,
    sessionAuthorizedFit: false,
    blockedDwellUpdates: 0,
    blockedForCalibration: 0,
    blockedForCaptureQuality: 0,
  };

  const originalEnableAdaptive = typeof aac.enableAdaptiveRecalibration === 'function'
    ? aac.enableAdaptiveRecalibration.bind(aac) : null;
  const originalEnableDrift = typeof aac.enableDriftWatchdog === 'function'
    ? aac.enableDriftWatchdog.bind(aac) : null;
  const originalFit = typeof aac.fitUserBasis === 'function'
    ? aac.fitUserBasis.bind(aac) : null;
  const originalCreateDwell = typeof aac.createDwellTimer === 'function'
    ? aac.createDwellTimer.bind(aac) : null;
  const originalClear = typeof aac.clearAllCalibration === 'function'
    ? aac.clearAllCalibration.bind(aac) : null;

  function fitted() {
    try { return !!(aac.isPCAFitted && aac.isPCAFitted()); }
    catch (_) { return false; }
  }

  function authorized() {
    return state.sessionAuthorizedFit && fitted();
  }

  function captureQuality() {
    try {
      const fusion = window.webgazerCaptureFusion;
      if (!fusion || typeof fusion.getLive !== 'function') return { available: false, reliability: 1 };
      const live = fusion.getLive() || {};
      return { available: !!live.available, reliability: Number(live.reliability || 0) };
    } catch (_) {
      return { available: false, reliability: 1 };
    }
  }

  function activatePostCalibration() {
    if (!authorized()) return false;
    if (state.adaptiveRequested && originalEnableAdaptive && !state.adaptiveActivated) {
      originalEnableAdaptive();
      state.adaptiveActivated = true;
    }
    if (state.driftRequested && originalEnableDrift && !state.driftActivated) {
      originalEnableDrift();
      state.driftActivated = true;
    }
    return true;
  }

  if (originalEnableAdaptive) {
    aac.enableAdaptiveRecalibration = function () {
      state.adaptiveRequested = true;
      activatePostCalibration();
      return aac;
    };
  }

  if (originalEnableDrift) {
    aac.enableDriftWatchdog = function () {
      state.driftRequested = true;
      activatePostCalibration();
      return aac;
    };
  }

  if (originalFit) {
    aac.fitUserBasis = function (...args) {
      const result = originalFit(...args);
      state.sessionAuthorizedFit = !!(result && result.rebuilt && fitted());
      if (state.sessionAuthorizedFit) activatePostCalibration();
      return result;
    };
  }

  if (originalClear) {
    aac.clearAllCalibration = async function (...args) {
      const result = await originalClear(...args);
      state.sessionAuthorizedFit = false;
      state.adaptiveActivated = false;
      state.driftActivated = false;
      return result;
    };
  }

  if (originalCreateDwell) {
    aac.createDwellTimer = function (...args) {
      const timer = originalCreateDwell(...args);
      if (!timer || typeof timer.updateFromGaze !== 'function') return timer;
      const originalUpdate = timer.updateFromGaze.bind(timer);
      timer.updateFromGaze = function (...updateArgs) {
        if (!authorized()) {
          state.blockedDwellUpdates++;
          state.blockedForCalibration++;
          return null;
        }
        const q = captureQuality();
        if (q.available && q.reliability < 0.35) {
          state.blockedDwellUpdates++;
          state.blockedForCaptureQuality++;
          return null;
        }
        activatePostCalibration();
        return originalUpdate(...updateArgs);
      };
      return timer;
    };
  }

  aac.__assuranceGuardInstalled = true;
  window.webgazerAACAssuranceGuard = {
    version: '0.2',
    isPCAFitted: fitted,
    isSessionAuthorized: authorized,
    getState: () => ({ ...state, pcaFitted: fitted(), sessionAuthorized: authorized(), captureQuality: captureQuality() }),
  };
})();
