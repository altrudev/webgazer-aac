"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
global.window = {innerWidth:1920, innerHeight:1080};
global.performance = {now:Date.now.bind(Date)};
global.ImageData = class {constructor(data,width,height){Object.assign(this,{data,width,height});}};
require("../webgazer-aac.js");
test("AAC public API is available", () => {
  const aac=global.webgazerAAC;
  assert.equal(aac.version,"1.3.0");
  for(const name of ["DwellTimer","KalmanFilter","BlinkDetector","DriftWatchdog","CalibrationStore","PolynomialRegression","RBFRegression","EnsembleRegression"]) assert.equal(typeof aac[name],"function",name);
});
test("Dwell timer rejects insufficient confidence and ignores null gaze", () => {
  const aac=global.webgazerAAC;const timer=new aac.DwellTimer({dwellMs:100,minConfidence:0.5});
  let ticks=0;const target={dispatchEvent(e){if(e.type==="webgazer-aac:dwell-complete") ticks++;return true;}};
  timer.update(target,1,1,0.2,false,false); timer.update(target,1,1,0.2,false,false);
  timer.update(null,1,1,1,false,true);
  assert.equal(timer.progress,0);assert.equal(ticks,0);
});
