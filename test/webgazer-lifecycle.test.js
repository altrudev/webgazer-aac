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
