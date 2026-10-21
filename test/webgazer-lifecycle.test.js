"use strict";
const {test}=require("node:test");
const assert=require("node:assert/strict");
global.window={innerWidth:1920,innerHeight:1080};
global.performance={now:Date.now.bind(Date)};
global.ImageData=class{constructor(data,width,height){Object.assign(this,{data,width,height});}};
require("../webgazer-aac.js");
test("WebGazer adapter suppresses interruption, rejects corrupt data, and recovers",()=>{
 const aac=global.webgazerAAC;
 let callback,output=[],up=true,throws=false;
 const patches={left:{},right:{}};
 const original=global.webgazer;
 global.webgazer={
  setRegression(){return this;},
  setGazeListener(cb){callback=cb;return this;},
  recordScreenPosition(){return this;},
  getTracker(){if(throws)throw Error("camera stopped");return up?{getEyePatches:()=>patches}:null;}
 };
 try{
  aac.install();
  global.webgazer.setGazeListener(v=>output.push(v));
  callback({x:120,y:90,eyeFeatures:patches},1);assert.ok(output.at(-1));assert.equal(Number.isFinite(output.at(-1).x),true);
  up=false;callback({x:200,y:100},2);assert.equal(output.at(-1),null);
  assert.equal(aac._lastPatches,null);assert.equal(aac._kalman.x,null);
  throws=true;callback({x:200,y:100},3);assert.equal(output.at(-1),null);
  throws=false;up=true;callback({x:130,y:95,eyeFeatures:patches},4);assert.ok(output.at(-1));
  callback({x:NaN,y:95},5);assert.equal(output.at(-1),null);
  callback(null,6);assert.equal(output.at(-1),null);
 } finally {global.webgazer=original;}
});

test("WebGazer 3.x eyeFeatures are consumed without invoking async getEyePatches",()=>{
 const aac=global.webgazerAAC;
 let callback, calls=0;const original=global.webgazer;
 global.webgazer={
  setRegression(){return this;},
  setGazeListener(cb){callback=cb;return this;},
  recordScreenPosition(){return this;},
  getTracker(){return {getEyePatches(){calls++;return Promise.resolve({left:{},right:{}});}};}
 };
 try{
  aac._installed=false;aac.install();
  const values=[];global.webgazer.setGazeListener(v=>values.push(v));
  callback({x:120,y:90,eyeFeatures:{left:{},right:{}}},1);
  assert.equal(calls,0);assert.ok(values.at(-1));
  callback({x:120,y:90},2);
  assert.equal(values.at(-1),null);assert.equal(calls,0);
 } finally{global.webgazer=original;}
});

test("Calibration gate rejects stale patches, freezes both models, and resets state",()=>{
 const aac=global.webgazerAAC;
 let callback, upstream=0;
 const original=global.webgazer;
 const w=8,h=8,rgba=new Uint8ClampedArray(w*h*4);
 for(let i=0;i<rgba.length;i+=4){rgba[i]=80;rgba[i+1]=50;rgba[i+2]=30;rgba[i+3]=255;}
 const patches={left:{patch:{width:w,height:h,data:rgba}},right:{patch:{width:w,height:h,data:rgba}}};
 global.webgazer={
  setRegression(){return this;},
  setGazeListener(cb){callback=cb;return this;},
  recordScreenPosition(){upstream++;return this;},
  getTracker(){return {getCurrentEyePatches:()=>patches};}
 };
 try {
  aac._installed=false;aac.install();
  aac.resetTraining();aac.lockTraining(false);
  assert.equal(global.webgazer.recordScreenPosition(100,80,"click"),false,"no eye frame yet");
  assert.equal(upstream,0);
  global.webgazer.setGazeListener(()=>{});
  callback({x:100,y:80,eyeFeatures:patches},0);
  assert.equal(global.webgazer.recordScreenPosition(100,80,"click"),undefined);
  assert.equal(upstream,1);
  assert.equal(aac.getTrainingEvidence().accepted,1);
  assert.equal(aac._regressions.polynomial.xSamples.length,1);
  aac.lockTraining(true);
  assert.equal(global.webgazer.recordScreenPosition(110,90,"click"),false);
  assert.equal(upstream,1);
  assert.equal(aac._regressions.polynomial.xSamples.length,1);
  aac.resetTraining();
  assert.equal(aac._regressions.polynomial.xSamples.length,0);
  assert.equal(aac._regressions.rbf.features.length,0);
  assert.equal(aac.getTrainingEvidence().accepted,0);
  assert.equal(aac.getTrainingEvidence().locked,true);
 } finally {aac.lockTraining(false);global.webgazer=original;}
});
