"use strict";
const {test}=require("node:test");
const assert=require("node:assert/strict");
global.window={innerWidth:1280,innerHeight:720};
global.performance={now:Date.now.bind(Date)};
require("../webgazer-aac.js");
const patch=(w,h)=>({patch:{width:w,height:h,data:Uint8ClampedArray.from({length:w*h*4},(_,i)=>i%4===3?255:(Math.floor(i/4)%w)*3%256)}});
test("different eye crop dimensions normalize to fixed-length finite features",()=>{
 const reg=global.webgazerAAC;
 const original=global.webgazer;
 global.webgazer={setRegression(){return this},setGazeListener(){return this},recordScreenPosition(){return this}};
 try{
  reg._installed=false;reg.install();
  const poly=reg._regressions.polynomial;
  poly.addData({left:patch(18,12),right:patch(90,60)},200,150);
  poly.addData({left:patch(60,40),right:patch(40,24)},250,170);
  assert.equal(poly.xSamples.length,2);
  assert.equal(poly.xSamples[0].length,poly.xSamples[1].length);
  assert.ok(poly.xSamples.every(features=>features.every(Number.isFinite)));
  poly.addData({left:patch(0,12),right:patch(60,40)},300,180);
  assert.equal(poly.xSamples.length,2,"malformed crop must not train regression");
 }finally{global.webgazer=original}
});
