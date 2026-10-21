(()=>{'use strict';
const $=id=>document.getElementById(id);let mode='calibrate',samples=[],targetIndex=0,started=0;
const targets=[{x:.15,y:.18},{x:.85,y:.18},{x:.50,y:.50},{x:.15,y:.82},{x:.85,y:.82}];
let completed=[],validTargets=0;
function med(a){a=[...a].sort((x,y)=>x-y);return a.length?(a[(a.length-1)>>1]+a[a.length>>1])/2:null}
function percentile(a,p){if(!a.length)return null;const b=[...a].sort((x,y)=>x-y);return b[Math.floor((b.length-1)*p)]}
function position(t){const rect=$('stage').getBoundingClientRect();return{x:rect.left+rect.width*t.x,y:rect.top+rect.height*t.y}}
function status(msg){$('validation-status').textContent=msg}
function guide(){const t=targets[targetIndex];const el=$('evaluation-target');if(!t){el.hidden=true;el.style.display='none';return}el.hidden=false;el.style.display='grid';el.style.left=(t.x*100)+'%';el.style.top=(t.y*100)+'%';el.textContent=String(targetIndex+1);status('Validation target '+(targetIndex+1)+'/5: look at the circle for 2 seconds. No clicks required.');}
function begin(){window.dispatchEvent(new Event('aac-validation-start'));if(!$('stop').disabled&&window.aacGuidedCalibration?.().finished){window.webgazerAAC?.lockTraining(true);mode='validate';samples=[];targetIndex=0;completed=[];validTargets=0;$('validation-start').disabled=true;started=performance.now();guide();}else status('Start camera and complete guided nine-point calibration before validation.')}
function record(d){if(mode!=='validate'||!d||!Number.isFinite(d.x)||!Number.isFinite(d.y))return;const elapsed=performance.now()-started;if(elapsed<350)return;const t=position(targets[targetIndex]);samples.push(Math.hypot(d.x-t.x,d.y-t.y));if(elapsed>=2200){const medErr=med(samples);completed.push({median_px:medErr,p90_px:percentile(samples,.9),count:samples.length});if(samples.length>=10)validTargets++;targetIndex++;samples=[];started=performance.now();if(targetIndex>=targets.length){mode='done';$('evaluation-target').hidden=true;$('evaluation-target').style.display='none';$('validation-start').disabled=false;const ok=completed.filter(x=>x.count>=10);status('Validation complete: '+ok.length+'/5 targets valid; median target error '+(med(ok.map(x=>x.median_px))?.toFixed(0)||'—')+' px.');}else guide();}}
function report(){const ok=completed.filter(x=>x.count>=10);return {protocol:'aac-holdout-v1',explicit_validation_click_training:false,target_count:completed.length,valid_targets:ok.length,samples_per_target:completed.map(x=>x.count),median_target_error_px:med(ok.map(x=>x.median_px)),median_target_p90_px:med(ok.map(x=>x.p90_px)),note:'Screen-coordinate estimate without validation clicks; model updates from other inputs have not been excluded.'}}
$('validation-start').onclick=begin;$('validation-cancel').onclick=()=>{mode='calibrate';$('evaluation-target').hidden=true;$('evaluation-target').style.display='none';$('validation-start').disabled=false;status('Validation canceled. You may calibrate with the numbered buttons.')};
window.addEventListener('aac-gaze',ev=>record(ev.detail));
window.addEventListener('aac-camera-stop',()=>{if(mode==='validate')$('validation-cancel').click()});
window.aacValidationReport=report;
})();
