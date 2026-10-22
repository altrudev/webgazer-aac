const test=require('node:test'),assert=require('node:assert/strict');
const {createHybridObservationStream}=require('../hybrid/observation-bridge.js');
test('WebGazer predictions and iris geometry remain separately observable, never fused as validated gaze',()=>{
 let t=1000;const m=createHybridObservationStream({measureIris:()=>({available:true,left:{cx:3},right:{cx:5}}),now:()=>t});
 m.observeLandmarks([],1000);
 const x=m.comparePrediction({x:80,y:110});assert.equal(x.iris.left.cx,3);assert.deepEqual(x.webgazer,{x:80,y:110});assert.equal(x.qualifiedForDwell,false);
 t=1300;const expired=m.comparePrediction({x:80,y:110});assert.equal(expired.iris,null);assert.equal(expired.reason,'stale_or_missing_iris');
 m.reset();assert.equal(m.comparePrediction(null).frameSequence,null);
});
test('Invalid landmark timestamp and invalid prediction are rejected independently',()=>{
 const m=createHybridObservationStream({measureIris:()=>({available:false}),now:()=>1000});
 assert.equal(m.observeLandmarks([],2000),null);
 m.observeLandmarks([],1000);
 const v=m.comparePrediction({x:NaN,y:2});assert.equal(v.webgazer,null);assert.equal(v.reason,'iris_unavailable');
});
