'use strict';
/* Independent camera-image measurements, not calibrated gaze accuracy or probability. */
(function(root) {
 const EYES = {
  left: {corners:[33,133], lids:[[160,144],[158,153]], rim:[469,470,471,472]},
  right: {corners:[362,263], lids:[[385,380],[387,373]], rim:[474,475,476,477]}
 };
 // Fit all four rim points in centered coordinates to avoid translation cancellation.
 function circle(points) {
  if(points.some((p,i)=>points.slice(i+1).some(q=>p.x===q.x&&p.y===q.y)))return null;
  const mx=points.reduce((s,p)=>s+p.x/4,0), my=points.reduce((s,p)=>s+p.y/4,0);
  const q=points.map(p=>({x:p.x-mx,y:p.y-my}));
  let xx=0,xy=0,yy=0,xb=0,yb=0;
  for(const p of q) {const b=p.x*p.x+p.y*p.y;xx+=p.x*p.x;xy+=p.x*p.y;yy+=p.y*p.y;xb+=p.x*b;yb+=p.y*b;}
  const det=xx*yy-xy*xy;
  if(!(xx>0&&yy>0&&det>1e-12*(xx+yy)**2))return null;
  const cx=(xb*yy-yb*xy)/(2*det),cy=(yb*xx-xb*xy)/(2*det);
  const distances=q.map(p=>Math.hypot(p.x-cx,p.y-cy));
  const radiusPx=distances.reduce((s,d)=>s+d/4,0);
  const residualPx=Math.sqrt(distances.reduce((s,d)=>s+(d-radiusPx)**2/4,0));
  const result={cx:mx+cx,cy:my+cy,radiusPx,residualPx,relativeResidual:residualPx/radiusPx};
  return Object.values(result).every(Number.isFinite)&&radiusPx>0?result:null;
 }
 function eye(points, spec, limits) {
  const [a,b]=spec.corners.map(i=>points[i]);
  const dx=b.x-a.x,dy=b.y-a.y,eyeWidthPx=Math.hypot(dx,dy);
  if(!Number.isFinite(eyeWidthPx)||eyeWidthPx<=0)return {available:false,reason:'degenerate_eye_baseline'};
  // Project lid separation onto the eye baseline normal, preserving roll invariance.
  const eyelidGapPx=spec.lids.reduce((sum,[u,l])=>sum+Math.abs((points[l].x-points[u].x)*(-dy/eyeWidthPx)+(points[l].y-points[u].y)*(dx/eyeWidthPx))/2,0);
  const iris=circle(spec.rim.map(i=>points[i]));
  const features={eyeWidthPx,eyelidGapPx,eyelidOpenness:eyelidGapPx/eyeWidthPx,iris};
  if(!Number.isFinite(features.eyelidOpenness)||!iris)return {available:false,reason:'degenerate_iris_geometry',...features};
  const reason=eyeWidthPx<limits.minEyeWidthPx?'eye_too_small':iris.radiusPx<limits.minIrisRadiusPx?'iris_too_small':iris.relativeResidual>limits.maxRelativeResidual?'iris_residual_too_large':null;
  return {available:reason===null,reason,...features};
 }
 /** Normalized MediaPipe x/y; dimensions in camera pixels; timestamps in one monotonic ms clock.
  * Openness = mean projected lid gap / corner width. Zero is a measurable closed lid.
  * Defaults are engineering gates, not clinically validated accuracy thresholds.
  */
 function measureFrame({landmarks,width,height,frameTimeMs,now,maxAgeMs=180,minEyeWidthPx=2,minIrisRadiusPx=1,maxRelativeResidual=.2}={}) {
  const base={schema:'aac-measurement-assurance/1',available:false,screenGaze:null,qualifiedForDwell:false};
  const reject=reason=>({...base,reason});
  if(![width,height,maxAgeMs,minEyeWidthPx,minIrisRadiusPx,maxRelativeResidual].every(Number.isFinite)||width<=0||height<=0||maxAgeMs<0||minEyeWidthPx<=0||minIrisRadiusPx<=0||maxRelativeResidual<0)return reject('invalid_options');
  if(![frameTimeMs,now].every(Number.isFinite))return reject('missing_timestamps');
  const ageMs=now-frameTimeMs;
  if(!Number.isFinite(ageMs)||ageMs<0||ageMs>maxAgeMs)return reject('stale_or_future_frame');
  if(!Array.isArray(landmarks)||landmarks.length!==478)return reject('landmarks_missing');
  const points=[];
  for(let i=0;i<478;i++) {
   const p=landmarks[i];
   if(!p||!Number.isFinite(p.x)||!Number.isFinite(p.y)||(p.z!==undefined&&!Number.isFinite(p.z)))return reject('nonfinite_landmarks');
   const x=p.x*width,y=p.y*height;
   if(!Number.isFinite(x)||!Number.isFinite(y))return reject('nonfinite_geometry');
   points.push({x,y});
  }
  const limits={minEyeWidthPx,minIrisRadiusPx,maxRelativeResidual};
  const left=eye(points,EYES.left,limits),right=eye(points,EYES.right,limits);
  return {...base,available:left.available&&right.available,reason:!left.available?left.reason:!right.available?right.reason:null,
   quality:{kind:'measured_features',units:'camera_pixels_and_dimensionless_ratios',ageMs,left,right}};
 }
 const api={measureFrame};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;
 root.AACMeasurementAssurance=api;
})(typeof window!=='undefined'?window:globalThis);
