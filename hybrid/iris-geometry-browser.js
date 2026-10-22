'use strict';
/* Browser iris contour measurement; camera px coordinates; not a gaze predictor. */
(function(root){
function fit(points){const [a,b,c,d]=points;if(!a||!b||!c||!d)return null;
const A=2*(b.x-a.x),B=2*(b.y-a.y),C=b.x*b.x+b.y*b.y-a.x*a.x-a.y*a.y;
const D=2*(c.x-a.x),E=2*(c.y-a.y),F=c.x*c.x+c.y*c.y-a.x*a.x-a.y*a.y;
const det=A*E-B*D;if(Math.abs(det)<1e-8)return null;
const x=(C*E-B*F)/det,y=(A*F-C*D)/det;
const r=points.reduce((s,p)=>s+Math.hypot(p.x-x,p.y-y),0)/4;
const residual=Math.sqrt(points.reduce((s,p)=>s+(Math.hypot(p.x-x,p.y-y)-r)**2,0)/4);
return Number.isFinite(r)&&r>0&&residual/r<.45?{x,y,r,residual}:null;}
function eye(points,corners,ids){const a=points[corners[0]],b=points[corners[1]],circle=fit(ids.map(i=>points[i]));
if(!a||!b||!circle)return null;
const dx=b.x-a.x,dy=b.y-a.y,w=Math.hypot(dx,dy);if(w<1e-6)return null;
return {cx:circle.x,cy:circle.y,radiusPx:circle.r,residualPx:circle.residual,
 horizontal:((circle.x-a.x)*dx+(circle.y-a.y)*dy)/(w*w),
 vertical:((circle.x-a.x)*(-dy)+(circle.y-a.y)*dx)/(w*w)};}
function measureIris(landmarks,width,height){if(!Array.isArray(landmarks)||landmarks.length<478||width<=0||height<=0)return {available:false,reason:'landmarks_missing'};
const p=landmarks.map(v=>({x:v.x*width,y:v.y*height}));
const left=eye(p,[33,133],[469,470,471,472]),right=eye(p,[362,263],[474,475,476,477]);
if(!left||!right)return {available:false,reason:'invalid_iris_contours'};
return {available:true,left,right,screenGaze:null};}
const api={measureIris};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.AACIrisGeometry=api;
})(typeof window!=='undefined'?window:globalThis);
