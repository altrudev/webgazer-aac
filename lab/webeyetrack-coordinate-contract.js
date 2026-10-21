"use strict";
/** Pure viewport mapper for WebEyeTrack centered normalized point-of-gaze. */
function mapWebEyeTrackGaze(normPog,width,height){
 if(!Array.isArray(normPog)||normPog.length!==2||!Number.isFinite(width)||!Number.isFinite(height)||width<=0||height<=0)return null;
 const [nx,ny]=normPog;if(!Number.isFinite(nx)||!Number.isFinite(ny)||nx<-.5||nx>.5||ny<-.5||ny>.5)return null;
 return {x:(nx+.5)*width,y:(ny+.5)*height};
}
module.exports={mapWebEyeTrackGaze};
