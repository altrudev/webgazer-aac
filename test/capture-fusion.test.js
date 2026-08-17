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

test('blink and eyelid occlusion reduce reliability',()=>{
  const open=C.eyeReliability({visibility:1,glare:0,clipped:0,apertureRatio:.22,jitter:.01,occlusion:0});
  const closed=C.eyeReliability({visibility:1,glare:0,clipped:0,apertureRatio:.08,jitter:.01,occlusion:.8});
  const blink=C.eyeReliability({visibility:1,glare:0,clipped:0,apertureRatio:.04,jitter:.01,occlusion:.9,blink:true});
  assert.ok(open>closed);assert.equal(blink,0);
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

test('one-euro feature filter rejects a single large spike without freezing',()=>{
  const f=C.makeOneEuro({minCutoff:1,beta:.007});let t=0,v=0;
  for(let i=0;i<10;i++){t+=33;v=f.update(.1,t);}
  const spike=f.update(1,t+=33);const recovered=f.update(.11,t+=33);
  assert.ok(spike<1);assert.ok(recovered<spike);assert.ok(recovered>.09);
});

test('linear predictor fits deterministic calibration map',()=>{
  const samples=[];
  for(let i=0;i<18;i++){
    const u=(i%6)/5,v=Math.floor(i/6)/2;
    samples.push({features:[u,v,0,0,0,.2,.2],x:100+500*u,y:80+400*v,weight:1});
  }
  const model=C.fitLinear(samples);
  const p=C.predict(model,[.5,.5,0,0,0,.2,.2]);
  assert.ok(Math.abs(p.x-350)<2);assert.ok(Math.abs(p.y-280)<2);
});

test('RBF kernel ridge models a nonlinear calibration map',()=>{
  const samples=[];
  for(let iy=0;iy<7;iy++)for(let ix=0;ix<7;ix++){
    const u=-1+2*ix/6,v=-1+2*iy/6;
    const features=[u,v,u*.3,v*.2,0,0,0,0,0,.2,.2,.14,.95];
    samples.push({features,x:600+330*u+90*u*v+60*v*v,y:400+260*v+75*u*u-45*u*v,weight:.9});
  }
  const model=C.fitKernelRidge(samples,{maxSamples:80,lambda:.01});
  const features=[.27,-.34,.081,-.068,0,0,0,0,0,.2,.2,.14,.95];
  const p=C.predict(model,features),tx=600+330*.27+90*.27*-.34+60*.34*.34,ty=400+260*-.34+75*.27*.27-45*.27*-.34;
  assert.ok(Math.hypot(p.x-tx,p.y-ty)<55,`nonlinear error ${Math.hypot(p.x-tx,p.y-ty)}`);
});

test('kernel comparison returns finite held-out metrics',()=>{
  const samples=[];
  for(let i=0;i<36;i++){
    const u=-1+2*(i%6)/5,v=-1+2*Math.floor(i/6)/5;
    samples.push({features:[u,v,u,v,u,v,0,0,0,.2,.2,.14,.9],x:500+250*u+70*u*v,y:350+220*v+55*u*u,weight:.8});
  }
  for(const kernel of ['linear','poly2','rbf']){
    const result=C.kFold(samples,{kernel,folds:3,maxSamples:60});assert.ok(Number.isFinite(result.metrics.rmse),kernel);
  }
});

test('leave-region-out withholds spatial regions rather than neighboring samples',()=>{
  const samples=[];
  for(let y=0;y<6;y++)for(let x=0;x<6;x++){
    const nx=x/5,ny=y/5; samples.push({features:[nx,ny,nx,ny,nx,ny,0,0,0,.2,.2,.14,.95],x:nx*1200,y:ny*800,weight:1});
  }
  const result=C.leaveRegionOut(samples,{kernel:'linear',width:1200,height:800,cols:3,rows:3});
  assert.equal(result.regions,9);assert.ok(result.pairs.length>0);assert.ok(Number.isFinite(result.metrics.rmse));
});

test('smooth-pursuit Lissajous path stays inside safe margins and spans screen',()=>{
  const pts=Array.from({length:101},(_,i)=>C.pursuitPoint(i/100));
  assert.ok(pts.every(p=>p.nx>=.12&&p.nx<=.88&&p.ny>=.12&&p.ny<=.88));
  assert.ok(Math.max(...pts.map(p=>p.nx))-Math.min(...pts.map(p=>p.nx))>.7);
  assert.ok(Math.max(...pts.map(p=>p.ny))-Math.min(...pts.map(p=>p.ny))>.7);
});

test('positioning coach passes stable geometry and rejects poor eye quality',()=>{
  const good=C.coachAssessment({faceOffsetX:0,faceOffsetY:0,interocularNorm:.14,yaw:.05,pitch:.04,roll:.03,luminance:95,lightingAsymmetry:.05,leftReliability:.85,rightReliability:.8,fusedReliability:.82});
  const bad=C.coachAssessment({faceOffsetX:0,faceOffsetY:0,interocularNorm:.14,yaw:.05,pitch:.04,roll:.03,luminance:95,lightingAsymmetry:.05,leftReliability:.2,rightReliability:.18,fusedReliability:.19});
  assert.equal(good.pass,true);assert.equal(bad.pass,false);assert.ok(bad.reasons.includes('eye-quality'));
});

test('deterministic thinning preserves endpoints and maximum model budget',()=>{
  const a=Array.from({length:501},(_,i)=>({i})),b=C.deterministicThin(a,160);
  assert.equal(b.length,160);assert.equal(b[0].i,0);assert.equal(b[b.length-1].i,500);
});

console.log(`capture fusion: ${passed}/${passed} passed`);
