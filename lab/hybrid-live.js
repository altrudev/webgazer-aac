import {FaceLandmarker,FilesetResolver} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/+esm';
const $=id=>document.getElementById(id),video=$('camera'),canvas=$('eyes'),ctx=canvas.getContext('2d');
let landmarker=null,stream=null,active=false,raf=0,paired=0,rejected=0,lastTime=-1;
const bridge=window.AACHybridObservations.createHybridObservationStream({measureIris:pts=>window.AACIrisGeometry.measureIris(pts,video.videoWidth,video.videoHeight)});
function text(id,val){$(id).textContent=String(val)}
function stop(){active=false;cancelAnimationFrame(raf);if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}video.srcObject=null;landmarker?.close();landmarker=null;window.webgazer?.end?.();bridge.reset();ctx.clearRect(0,0,canvas.width,canvas.height);$('start').disabled=false;$('stop').disabled=true;text('status','Stopped');}
function draw(landmarks,m){const w=video.videoWidth,h=video.videoHeight,rc=video.getBoundingClientRect();const scale=Math.min(rc.width/w,rc.height/h),offX=(rc.width-w*scale)/2,offY=(rc.height-h*scale)/2;
canvas.width=rc.width*devicePixelRatio;canvas.height=rc.height*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);ctx.clearRect(0,0,rc.width,rc.height);
if(!m.available)return;for(const eye of [m.left,m.right]){ctx.strokeStyle='#ffcb6d';ctx.lineWidth=2;ctx.beginPath();ctx.arc(offX+(w-eye.cx)*scale,offY+eye.cy*scale,Math.max(3,eye.radiusPx*scale),0,Math.PI*2);ctx.stroke();}}
function loop(t){if(!active)return;raf=requestAnimationFrame(loop);if(!landmarker||video.readyState<2||t-lastTime<100)return;lastTime=t;
try{const results=landmarker.detectForVideo(video,t),p=results.faceLandmarks[0];if(!p){rejected++;text('iris','No face');return;}
const m=window.AACIrisGeometry.measureIris(p,video.videoWidth,video.videoHeight);
bridge.observeLandmarks(p,t);const report=bridge.comparePrediction(window.__lastWebGazer);
if(report.iris&&report.webgazer)paired++;else rejected++;
text('iris',m.available?'Both eyes measured':m.reason);text('left',m.available?m.left.horizontal.toFixed(3):'—');text('right',m.available?m.right.horizontal.toFixed(3):'—');text('paired',paired);text('rejected',rejected);draw(p,m);
}catch(err){rejected++;text('status','Landmark inference error: '+String(err).slice(0,100));}}
$('start').addEventListener('click',async()=>{$('start').disabled=true;text('status','Loading camera and landmark model');
try{const files=await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm');
landmarker=await FaceLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',delegate:'CPU'},runningMode:'VIDEO',numFaces:1});
stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});video.srcObject=stream;await video.play();
if(typeof window.webgazer==='undefined')throw Error('WebGazer unavailable');
window.webgazer.setRegression('ridge').setTracker('TFFaceMesh').saveDataAcrossSessions(false).setGazeListener(d=>{window.__lastWebGazer=d&&Number.isFinite(d.x)&&Number.isFinite(d.y)?{x:d.x,y:d.y}:null;text('gaze',window.__lastWebGazer?Math.round(d.x)+', '+Math.round(d.y):'—')});
await window.webgazer.begin();window.webgazer.removeMouseEventListeners?.();window.webgazer.showVideoPreview(false).showPredictionPoints(false);active=true;$('stop').disabled=false;text('status','Measuring. Screen gaze remains uncalibrated.');raf=requestAnimationFrame(loop);
}catch(err){stop();text('status','Failed: '+String(err).slice(0,150));}});
$('stop').addEventListener('click',stop);window.addEventListener('pagehide',stop);
