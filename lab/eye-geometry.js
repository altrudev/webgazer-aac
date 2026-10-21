(function(root){'use strict';
const idx={left:[33,133,468],right:[362,263,473]};
function point(a,i){const p=a&&a[i];return p&&Number.isFinite(p[0])&&Number.isFinite(p[1])?p:null}
function eye(a,ids){const l=point(a,ids[0]),r=point(a,ids[1]),iris=point(a,ids[2]);if(!l||!r||!iris)return null;const w=Math.hypot(r[0]-l[0],r[1]-l[1]);if(w<6)return null;const t=((iris[0]-l[0])*(r[0]-l[0])+(iris[1]-l[1])*(r[1]-l[1]))/(w*w);return{horizontal:t,width:w}}
function measure(landmarks){const left=eye(landmarks,idx.left),right=eye(landmarks,idx.right);const nose=point(landmarks,1),center=point(landmarks,168);if(!left||!right)return{available:false,reason:'iris_or_eye_landmarks_unavailable'};return{available:true,left_horizontal:left.horizontal,right_horizontal:right.horizontal,eye_width_ratio:left.width/right.width,binocular_disagreement:Math.abs(left.horizontal-right.horizontal),head_proxy:nose&&center?{dx:nose[0]-center[0],dy:nose[1]-center[1]}:null};}
const api={measure};
if(typeof module!=='undefined'&&module.exports)module.exports=api;
root.aacEyeGeometry=api;
})(typeof window!=='undefined'?window:globalThis);
