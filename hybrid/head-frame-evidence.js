'use strict';
// Face-landmark movement is a camera-image proxy, not a calibrated 3D angle.
(function(root){
function measureHead(landmarks){
 const ids=[33,263,1,152,10],p=ids.map(i=>landmarks?.[i]);
 if(p.some(v=>!v||!Number.isFinite(v.x)||!Number.isFinite(v.y)))return {available:false,reason:'head_landmarks_missing'};
 const [left,right,nose,chin,forehead]=p;
 const dx=right.x-left.x,dy=right.y-left.y,width=Math.hypot(dx,dy);
 if(width<0.01)return {available:false,reason:'degenerate_eye_baseline'};
 const midX=(left.x+right.x)/2,midY=(left.y+right.y)/2;
 return {available:true,rollProxyRad:Math.atan2(dy,dx),horizontalPosition:nose.x,verticalPosition:nose.y,
  noseOffsetEyeWidths:(nose.x-midX)/width,faceHeight:Math.hypot(chin.x-forehead.x,chin.y-forehead.y),
  eyeSpan:width,units:'normalized_image_proxy_not_degrees'};
}
function frameEvidence({landmarkAt,gazeAt,now,maxAgeMs=180,maxPairSkewMs=100}){
 if(![landmarkAt,gazeAt,now].every(Number.isFinite))return {paired:false,reason:'missing_timestamps'};
 if(landmarkAt>now||gazeAt>now||now-landmarkAt>maxAgeMs||now-gazeAt>maxAgeMs)return {paired:false,reason:'stale_or_future_frame'};
 if(Math.abs(landmarkAt-gazeAt)>maxPairSkewMs)return {paired:false,reason:'pair_skew'};
 return {paired:true,skewMs:Math.abs(landmarkAt-gazeAt),maxAgeMs:Math.max(now-landmarkAt,now-gazeAt),sameSourceFrame:false};
}
const api={measureHead,frameEvidence};if(typeof module!=='undefined')module.exports=api;root.AACHeadEvidence=api;
})(typeof window!=='undefined'?window:globalThis);
