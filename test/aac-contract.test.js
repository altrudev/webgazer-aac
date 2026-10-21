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

test("Dwell completion never turns prediction into calibration evidence", () => {
 const aac=global.webgazerAAC;
 let recorded=0;
 const t=new aac.DwellTimer({dwellMs:50,holdAfterMs:0,aacRef:{recordDwellHitXY(){recorded++;}}});
 const target={dispatchEvent(){return true;}};
 const original=performance.now;
 let clock=1000;performance.now=()=>clock;
 try { t.update(target,10,20,0.95,false,false);clock+=90;t.update(target,10,20,0.95,false,false);assert.equal(recorded,0); }
 finally {performance.now=original;}
});
test("Legacy model-derived dwell API never retrains and confirmed positions require finite coordinates", () => {
 const aac=global.webgazerAAC;
 let trained=0,watched=0;
 const oldR=aac._recalibrator,oldW=aac._watchdog,oldP=aac._lastPatches;
 aac._recalibrator={recordHit(){trained++;},recordElementHit(){trained++;}};
 aac._watchdog={record(){watched++;}};
 aac._lastPatches={left:{},right:{}};
 try {
  aac.recordDwellHitXY(10,20);aac.recordDwellHit({getBoundingClientRect(){return {left:0,top:0,width:20,height:20};}});
  assert.equal(trained,0);assert.equal(watched,0);
  assert.equal(aac.recordConfirmedPosition(NaN,5),false);
  assert.equal(aac.recordConfirmedPosition(10,20),true);
  assert.equal(trained,1);assert.equal(watched,1);
 } finally {aac._recalibrator=oldR;aac._watchdog=oldW;aac._lastPatches=oldP;}
});
