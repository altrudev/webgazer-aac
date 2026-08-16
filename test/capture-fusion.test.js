'use strict';
const assert=require('node:assert/strict');
const C=require('../demo/prehardware/capture-fusion-core.js');

let passed=0;
function test(name,fn){fn();passed++;console.log(`ok ${passed} - ${name}`);}

test('clean eye scores above glare-contaminated eye',()=>{
  const clean=C.eyeReliability({visibility:1,glare:.01,clipped:0,apertureRatio:.22,jitter:.01,landmarkConfidence:1,occlusion:0});
  const glare=C.eyeReliability({visibility:1,glare:.15,clipped:.08,apertureRatio:.22,jitter:.01,landmarkConfidence:1,occlusion:0});
  assert.ok(clean>glare,`${clean} should exceed ${glare}`);
});

test('eyelid occlusion reduces reliability',()=>{
  const open=C.eyeReliability({visibility:1,glare:0,clipped:0,apertureRatio:.22,jitter:.01,occlusion:0});
  const closed=C.eyeReliability({visibility:1,glare:0,clipped:0,apertureRatio:.08,jitter:.01,occlusion:.8});
  assert.ok(open>closed);
});

test('bilateral fusion favors reliable eye',()=>{
  const fused=C.fuseEyes({u:.10,v:.02,reliability:.9},{u:-.20,v:.10,reliability:.1});
  assert.ok(Math.abs(fused.u-.10)<Math.abs(fused.u+.20));
  assert.ok(fused.reliability>0);
});

test('bilateral disagreement reduces combined reliability',()=>{
  const agree=C.fuseEyes({u:.1,v:.02,reliability:.9},{u:.11,v:.01,reliability:.9});
  const disagree=C.fuseEyes({u:.1,v:.02,reliability:.9},{u:-.1,v:-.1,reliability:.9});
  assert.ok(agree.reliability>disagree.reliability);
});

test('median suppresses one-frame iris spike',()=>{
  assert.equal(C.median([.1,.1,.1,.9,.11]),.1);
});

test('linear predictor fits deterministic calibration map',()=>{
  const samples=[];
  for(let i=0;i<12;i++){
    const u=(i%4)/3,v=Math.floor(i/4)/2;
    samples.push({features:[u,v,0,0,0,.2,.2],x:100+500*u,y:80+400*v,weight:1});
  }
  const model=C.fitLinear(samples);
  const p=C.predict(model,[.5,.5,0,0,0,.2,.2]);
  assert.ok(Math.abs(p.x-350)<2);
  assert.ok(Math.abs(p.y-280)<2);
});

test('leave-one-out reports finite geometric error',()=>{
  const samples=[];
  for(let i=0;i<12;i++){
    const u=(i%4)/3,v=Math.floor(i/4)/2;
    samples.push({features:[u,v,0,0,0,.2,.2],x:120+450*u,y:60+360*v,weight:.8});
  }
  const result=C.leaveOneOut(samples);
  assert.equal(result.pairs.length,12);
  assert.ok(Number.isFinite(result.rmse));
});

console.log(`capture fusion: ${passed}/${passed} passed`);
