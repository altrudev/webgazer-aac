'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {measureFrame}=require('../hybrid/measurement-assurance');
function fixture() {
 const landmarks=Array.from({length:478},()=>({x:.5,y:.5,z:0}));
 for(const [corners,lids,rim,cx] of [
  [[33,133],[[160,144],[158,153]],[469,470,471,472],.3],
  [[362,263],[[385,380],[387,373]],[474,475,476,477],.7]
 ]) {
  corners.forEach((id,j)=>landmarks[id]={x:cx+(j? .1:-.1),y:.5});
  lids.forEach(([u,l],j)=>{landmarks[u]={x:cx+(j?.04:-.04),y:.48};landmarks[l]={x:cx+(j?.04:-.04),y:.52};});
  rim.forEach((id,j)=>landmarks[id]={x:cx+.02*Math.cos(j*Math.PI/2),y:.5+.04*Math.sin(j*Math.PI/2)});
 }
 return {landmarks,width:1000,height:500,frameTimeMs:100,now:150};
}
test('per-eye pixel geometry and measured quality never authorize gaze or dwell',()=>{
 const result=measureFrame(fixture());
 assert.equal(result.available,true);assert.equal(result.screenGaze,null);assert.equal(result.qualifiedForDwell,false);
 assert.equal(result.quality.kind,'measured_features');assert.equal('confidence' in result,false);
 for(const e of [result.quality.left,result.quality.right]) {
  assert.ok(Math.abs(e.eyelidOpenness-.1)<1e-12);
  assert.ok(Math.abs(e.iris.radiusPx-20)<1e-10);assert.ok(e.iris.residualPx<1e-10);
 }
});
test('independent lid closure remains measured and does not fabricate a blink probability',()=>{
 const f=fixture();for(const [u,l] of [[160,144],[158,153]])f.landmarks[l]={...f.landmarks[u]};
 const r=measureFrame(f);assert.equal(r.available,true);assert.equal(r.quality.left.eyelidOpenness,0);assert.ok(r.quality.right.eyelidOpenness>0);
});
test('openness and circle measurements are invariant under image-plane rotation',()=>{
 const f=fixture(),original=measureFrame(f);
 f.landmarks=f.landmarks.map(p=>({x:(-p.y*f.height+500)/f.width,y:p.x*f.width/f.height}));
 const rotated=measureFrame(f);assert.equal(rotated.available,true);
 assert.ok(Math.abs(rotated.quality.left.eyelidOpenness-original.quality.left.eyelidOpenness)<1e-12);
});
test('reject missing, malformed, stale and future frames without throwing',()=>{
 assert.equal(measureFrame().available,false);
 for(const change of [{landmarks:null},{landmarks:[]},{frameTimeMs:NaN},{now:99},{now:281},{width:Infinity},{height:0},{maxAgeMs:-1},{minIrisRadiusPx:NaN}])assert.equal(measureFrame({...fixture(),...change}).available,false);
 for(const value of [null,undefined,{x:NaN,y:0},{x:0,y:Infinity},{x:0,y:0,z:NaN}]) {
  const f=fixture();f.landmarks[470]=value;assert.equal(measureFrame(f).available,false);
 }
 const f=fixture();delete f.landmarks[20];assert.equal(measureFrame(f).available,false);
 assert.equal(measureFrame({...fixture(),now:280}).available,true);
});
test('reject collapsed eyes, collinear rims, small irises and poor circle fits',()=>{
 let f=fixture();f.landmarks[133]={...f.landmarks[33]};assert.equal(measureFrame(f).reason,'degenerate_eye_baseline');
 f=fixture();for(const i of [469,470,471,472])f.landmarks[i].y=.5;
 assert.equal(measureFrame(f).reason,'degenerate_iris_geometry');
 f=fixture();for(const i of [469,470,471,472]) {f.landmarks[i].x=.3+(f.landmarks[i].x-.3)*.01;f.landmarks[i].y=.5+(f.landmarks[i].y-.5)*.01;}
 assert.equal(measureFrame(f).reason,'iris_too_small');
 f=fixture();f.landmarks[472]={x:.3,y:.5};const r=measureFrame(f);
 assert.equal(r.reason,'iris_residual_too_large');assert.ok(r.quality.left.iris.relativeResidual>.2);
 for(const rejected of [r,measureFrame()]){assert.equal(rejected.qualifiedForDwell,false);assert.equal(rejected.screenGaze,null);}
});
test('reject finite input whose pixel geometry overflows and repeated rim points',()=>{
 const f=fixture();f.landmarks[1].x=Number.MAX_VALUE;
 assert.equal(measureFrame(f).reason,'nonfinite_geometry');
 const g=fixture();g.landmarks[470]={...g.landmarks[469]};
 assert.equal(measureFrame(g).reason,'degenerate_iris_geometry');
 const h=fixture();h.landmarks[133].x=h.landmarks[33].x+.001;
 assert.equal(measureFrame(h).reason,'eye_too_small');
});
