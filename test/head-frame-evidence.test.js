const {test}=require('node:test'),assert=require('node:assert/strict');
const {measureHead,frameEvidence}=require('../hybrid/head-frame-evidence.js');
test('head proxy reports orientation and position without inventing degrees',()=>{
 let p=Array.from({length:478},()=>({x:.5,y:.5}));p[33]={x:.3,y:.4};p[263]={x:.7,y:.4};p[1]={x:.55,y:.55};p[10]={x:.5,y:.2};p[152]={x:.5,y:.8};
 const m=measureHead(p);assert.equal(m.available,true);assert.equal(m.rollProxyRad,0);assert.equal(m.units,'normalized_image_proxy_not_degrees');assert.ok(Math.abs(m.noseOffsetEyeWidths-.125)<1e-9);
 p[263]={x:.7,y:.45};assert.ok(measureHead(p).rollProxyRad>0);
 assert.equal(measureHead([]).available,false);
});
test('pair gate rejects old, future, skewed, missing and unpaired frames',()=>{
 assert.equal(frameEvidence({landmarkAt:900,gazeAt:920,now:950}).paired,true);
 assert.equal(frameEvidence({landmarkAt:700,gazeAt:920,now:950}).paired,false);
 assert.equal(frameEvidence({landmarkAt:901,gazeAt:950,now:900}).paired,false);
 assert.equal(frameEvidence({landmarkAt:800,gazeAt:950,now:960,maxAgeMs:200,maxPairSkewMs:80}).reason,'pair_skew');
 assert.equal(frameEvidence({landmarkAt:NaN,gazeAt:950,now:960}).paired,false);
 assert.equal(frameEvidence({landmarkAt:900,gazeAt:920,now:950}).sameSourceFrame,false);
});
