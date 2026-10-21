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

test("Malformed gaze and backwards clock cannot activate dwell", () => {
 const t=new webgazerAAC.DwellTimer({dwellMs:100,maxPauseMs:1000});
 let fired=0;const target={dispatchEvent(e){if(e.type==="webgazer-aac:dwell-complete") fired++;return true;}};
 const original=performance.now;let clock=1000;performance.now=()=>clock;
 try {
  t.update(target,1,1,0.9,false,false);clock=900;t.update(target,1,1,0.9,false,false);
  assert.equal(t.progress,0);
  clock=950;t.update(target,NaN,1,0.9,false,false);
  clock=1000;t.update(target,1,1,NaN,false,false);
  assert.equal(t.progress,0);assert.equal(fired,0);
 }finally{performance.now=original;}
});
test("Extended gaze interruption discards partial dwell", () => {
 const t=new webgazerAAC.DwellTimer({dwellMs:500,maxPauseMs:150});
 const target={dispatchEvent(){return true;}};
 const original=performance.now;let clock=1000;performance.now=()=>clock;
 try {
  t.update(target,1,1,0.9,false,false);clock+=80;t.update(target,1,1,0.9,false,false);
  assert.ok(t.progress>0);
  clock+=10;t.update(null,1,1,0.9,false,true);
  clock+=160;t.update(null,1,1,0.9,false,true);
  assert.equal(t.progress,0);
  t.update(target,1,1,0.9,false,false);assert.equal(t.progress,0);
 }finally{performance.now=original;}
});

test("Calibration restoration rejects mismatched viewport before changing state", () => {
 const aac=global.webgazerAAC, before=aac._kalman.Q;
 const snap={aacVersion:"1.3.0",screenWidth:1000,screenHeight:700,kalman:{Q:29,R:60},pca:{left:{fitted:false},right:{fitted:false}}};
 assert.equal(aac.applyCalibrationSnapshot(snap),false);assert.equal(aac._kalman.Q,before);
});
test("Calibration restoration rejects invalid noise values without partial mutation", () => {
 const aac=global.webgazerAAC, before=aac._kalman.Q;
 const snap={aacVersion:"1.3.0",screenWidth:1920,screenHeight:1080,kalman:{Q:33,R:NaN},pca:{left:{fitted:false},right:{fitted:false}}};
 assert.equal(aac.applyCalibrationSnapshot(snap),false);assert.equal(aac._kalman.Q,before);
});
test("Rejected calibration load returns null, not a seemingly restored snapshot", async () => {
 const aac=global.webgazerAAC, old=aac._store, q=aac._kalman.Q;
 aac._store={load:()=>Promise.resolve({aacVersion:"1.3.0",screenWidth:800,screenHeight:600,kalman:{Q:99,R:10}})};
 try {assert.equal(await aac.loadCalibration(),null);assert.equal(aac._kalman.Q,q);}
 finally {aac._store=old;}
});
