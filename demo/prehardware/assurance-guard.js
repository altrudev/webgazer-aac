'use strict';

(function () {
  const aac = window.webgazerAAC;
  if (!aac || aac.__assuranceGuardInstalled) return;

  const state = {
    adaptiveRequested: false,
    driftRequested: false,
    adaptiveActivated: false,
    driftActivated: false,
    blockedDwellUpdates: 0,
  };

  const originalEnableAdaptive = typeof aac.enableAdaptiveRecalibration === 'function'
    ? aac.enableAdaptiveRecalibration.bind(aac) : null;
  const originalEnableDrift = typeof aac.enableDriftWatchdog === 'function'
    ? aac.enableDriftWatchdog.bind(aac) : null;
  const originalFit = typeof aac.fitUserBasis === 'function'
    ? aac.fitUserBasis.bind(aac) : null;
  const originalCreateDwell = typeof aac.createDwellTimer === 'function'
    ? aac.createDwellTimer.bind(aac) : null;

  function fitted() {
    try { return !!(aac.isPCAFitted && aac.isPCAFitted()); }
    catch (_) { return false; }
  }

  function activatePostCalibration() {
    if (!fitted()) return false;
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
      if (result && result.rebuilt && fitted()) activatePostCalibration();
      return result;
    };
  }

  if (originalCreateDwell) {
    aac.createDwellTimer = function (...args) {
      const timer = originalCreateDwell(...args);
      if (!timer || typeof timer.updateFromGaze !== 'function') return timer;
      const originalUpdate = timer.updateFromGaze.bind(timer);
      timer.updateFromGaze = function (...updateArgs) {
        if (!fitted()) {
          state.blockedDwellUpdates++;
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
    version: '0.1',
    isPCAFitted: fitted,
    getState: () => ({ ...state, pcaFitted: fitted() }),
  };
})();
