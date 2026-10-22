'use strict';
// Side-channel combining camera iris geometry evidence with WebGazer screen estimates.
// Deliberately does not modify predictions or qualify them for AAC activation.
function createHybridObservationStream({measureIris, maxAgeMs=250, now=()=>performance.now()}){
 if(typeof measureIris!=='function')throw new TypeError('measureIris function required');
 let latest=null,sequence=0;
 return {
  observeLandmarks(points,frameTimeMs){
   if(!Number.isFinite(frameTimeMs)||frameTimeMs>now()+50){latest=null;return null;}
   const iris=measureIris(points);
   latest={sequence:++sequence,frameTimeMs,iris};
   return latest;
  },
  comparePrediction(prediction){
   const time=now();
   const irisFresh=!!latest&&time>=latest.frameTimeMs&&time-latest.frameTimeMs<=maxAgeMs;
   const gazeValid=!!prediction&&Number.isFinite(prediction.x)&&Number.isFinite(prediction.y);
   return {schema:'aac-hybrid-observation/1',frameSequence:irisFresh?latest.sequence:null,
    iris:irisFresh&&latest.iris.available?latest.iris:null,
    reason:!irisFresh?'stale_or_missing_iris':!latest.iris.available?'iris_unavailable':null,
    webgazer:gazeValid?{x:prediction.x,y:prediction.y}:null,
    qualifiedForDwell:false};
  },
  reset(){latest=null;sequence=0;}
 };
}
if(typeof module!=='undefined')module.exports={createHybridObservationStream};
