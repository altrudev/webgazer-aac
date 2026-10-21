(()=>{'use strict';
const $=id=>document.getElementById(id);
const points=[[.12,.14],[.50,.14],[.88,.14],[.12,.50],[.50,.50],[.88,.50],[.12,.86],[.50,.86],[.88,.86]];
let state='idle',idx=0,timer=null,lastValid=0,lastGaze=null,accepted=0,skipped=0,startedAt=0;
function status(s){$('guided-status').textContent=s}
function stop(){if(window.webgazerAAC?.lockTraining)window.webgazerAAC.lockTraining(true);if(timer){clearInterval(timer);timer=null;}state='idle';$('guided-target').hidden=true;$('guided-target').style.display='none';$('guided-start').disabled=false;}
function next(){
 if(idx>=points.length){const count=accepted;stop();status('Calibration complete: '+count+' explicitly paired eye samples; '+skipped+' samples skipped. Now run accuracy validation.');return}
 const [x,y]=points[idx],r=$('stage').getBoundingClientRect(),el=$('guided-target');
 el.hidden=false;el.style.display='grid';el.style.left=(100*x)+'%';el.style.top=(100*y)+'%';el.textContent=(idx+1)+'/9';
 state='settling';startedAt=performance.now();let perTarget=0;
 status('Look at target '+(idx+1)+' of 9. Keep your head steady; mouse movement is not needed.');
 timer=setInterval(()=>{
  if($('stop').disabled){stop();status('Calibration stopped when camera stopped.');return}
  const elapsed=performance.now()-startedAt;
  if(elapsed<850)return; // fixation settling
  if(elapsed>3100){clearInterval(timer);timer=null;idx++;next();return}
  const isFresh=lastGaze&&performance.now()-lastValid<180;
  if(!isFresh||perTarget>=7){if(!isFresh)skipped++;return}
  try{
   // Only this explicit guided target can submit training positions.
   // No mouse movement or click coordinates are used.
   const before=window.webgazerAAC.getTrainingEvidence().attempted;
   window.webgazer.recordScreenPosition(r.left+r.width*x,r.top+r.height*y,'click');
   const after=window.webgazerAAC.getTrainingEvidence().attempted;
   if(after!==before+1){skipped++;return}
   accepted++;perTarget++;lastValid=0;status('Target '+(idx+1)+'/9: '+perTarget+' fresh eye samples paired.');
  }catch(e){skipped++;status('Target '+(idx+1)+'/9: training sample rejected.')}
 },230);
}
$('guided-start').addEventListener('click',()=>{
 if($('stop').disabled){status('Start your camera first.');return}
 stop();accepted=0;skipped=0;idx=0;state='settling';
 if(window.webgazer&&typeof window.webgazer.clearData==='function'){
  // Clear any prior mouse-derived regression samples before controlled calibration.
  Promise.resolve(window.webgazer.clearData()).then(()=>{if(!window.webgazerAAC?.resetTraining?.())throw Error('AAC reset unavailable');window.webgazerAAC.lockTraining(false);next()}).catch(()=>{stop();status('Could not reset earlier training data; calibration not started.')});
 }else{stop();status('WebGazer model reset unavailable; refusing mixed calibration data.')}
});
$('guided-cancel').addEventListener('click',()=>{stop();status('Guided calibration cancelled.')});
window.addEventListener('aac-gaze',e=>{
 const d=e.detail;
 if(d&&Number.isFinite(d.x)&&Number.isFinite(d.y)){lastValid=performance.now();lastGaze=d}
});
window.addEventListener('aac-camera-stop',stop);
window.addEventListener('aac-validation-start',()=>{if(state!=='idle'){stop();status('Calibration ended before validation.')}});
window.aacGuidedCalibration=()=>({schema:'aac-guided-calibration/1',method:'nine-point-timed-fixation',explicit_samples:accepted,skipped_samples:skipped,finished:idx>=points.length&&state==='idle'});
})();
