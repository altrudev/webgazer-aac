'use strict';
(function(){
  const C=window.webgazerCaptureFusionCore;
  if(!C)return;
  const samples=[];
  const state={
    cohort:'unknown', faceMesh:null, ready:false, sending:false, timer:null,
    latestEyes:null, latestHead:null, latestWebGazer:null, latestEyePatch:{left:null,right:null},
    filtered:null, history:{left:[],right:[]}, model:null, evaluation:null, frames:0, missing:0,
  };
  const EYE={
    left:{outer:263,inner:362,upper:386,lower:374,iris:[468,469,470,471,472]},
    right:{outer:33,inner:133,upper:159,lower:145,iris:[473,474,475,476,477]},
  };
  function avg(a){return C.avg(a);} function stdev(a){return C.stdev(a);} function clamp(v){return C.clamp(v);}
  function point(lm,video){if(!lm)return null;const w=video.videoWidth||640,h=video.videoHeight||480;return [lm.x*w,lm.y*h];}
  function meanPoint(a){return a.length?[avg(a.map(p=>p[0])),avg(a.map(p=>p[1]))]:null;}
  function norm(v){const m=Math.hypot(v[0],v[1])||1;return [v[0]/m,v[1]/m];}
  function dot(a,b){return a[0]*b[0]+a[1]*b[1];}
  function subtract(a,b){return [a[0]-b[0],a[1]-b[1]];}
  function eyeLocal(lms,video,cfg){
    const outer=point(lms[cfg.outer],video),inner=point(lms[cfg.inner],video),upper=point(lms[cfg.upper],video),lower=point(lms[cfg.lower],video);
    const iris=meanPoint(cfg.iris.map(i=>point(lms[i],video)).filter(Boolean));
    if(!outer||!inner||!upper||!lower||!iris)return null;
    const center=[(outer[0]+inner[0])/2,(outer[1]+inner[1])/2],axis=subtract(inner,outer),width=Math.hypot(axis[0],axis[1]);
    if(width<4)return null;const ex=norm(axis),ey=[-ex[1],ex[0]],d=subtract(iris,center);
    const aperture=Math.hypot(upper[0]-lower[0],upper[1]-lower[1]);
    return {u:dot(d,ex)/width,v:dot(d,ey)/width,width,aperture,apertureRatio:aperture/width,center};
  }
  function headFeatures(lms,video,left,right){
    const nose=point(lms[1],video),forehead=point(lms[10],video),chin=point(lms[152],video);
    if(!nose||!forehead||!chin||!left||!right)return null;
    const eyeMid=[(left.center[0]+right.center[0])/2,(left.center[1]+right.center[1])/2];
    const inter=Math.hypot(left.center[0]-right.center[0],left.center[1]-right.center[1])||1;
    const faceH=Math.hypot(forehead[0]-chin[0],forehead[1]-chin[1])||1;
    return {
      yaw:(nose[0]-eyeMid[0])/inter,
      pitch:(nose[1]-eyeMid[1])/faceH,
      roll:Math.atan2(right.center[1]-left.center[1],right.center[0]-left.center[0]),
      interocularPx:inter,
    };
  }
  function patchQuality(eye){
    if(!eye)return {glare:0,clipped:0,contrast:0};const src=eye.patch||eye;if(!src.data||!src.width||!src.height)return {glare:0,clipped:0,contrast:0};
    const d=src.data,rgba=d.length>=src.width*src.height*4,vals=[];let bright=0,clip=0,n=0;const step=rgba?16:4;
    for(let i=0;i<d.length;i+=step){let y;if(rgba)y=.299*(d[i]||0)+.587*(d[i+1]||0)+.114*(d[i+2]||0);else y=Number(d[i])||0;vals.push(y);if(y>=235)bright++;if(y>=250)clip++;n++;}
    vals.sort((a,b)=>a-b);const p10=vals[Math.floor(vals.length*.1)]||0,p90=vals[Math.floor(vals.length*.9)]||0;
    return {glare:n?bright/n:0,clipped:n?clip/n:0,contrast:(p90-p10)/255};
  }
  function reliability(side,eye){
    const h=state.history[side],u=h.map(x=>x.u).slice(-7),v=h.map(x=>x.v).slice(-7);const jitter=Math.hypot(stdev(u),stdev(v));
    const q=state.latestEyePatch[side]||{glare:0,clipped:0};
    return C.eyeReliability({visibility:1,glare:q.glare,clipped:q.clipped,apertureRatio:eye.apertureRatio,jitter,landmarkConfidence:1,occlusion:eye.apertureRatio<.10?.7:eye.apertureRatio<.14?.35:0});
  }
  function filteredEye(side,eye){
    const h=state.history[side];h.push({u:eye.u,v:eye.v});if(h.length>7)h.shift();
    const mu=C.median(h.map(x=>x.u).slice(-5)),mv=C.median(h.map(x=>x.v).slice(-5));const prev=state.filtered&&state.filtered[side];
    return {...eye,u:C.ema(prev&&prev.u,mu,.45),v:C.ema(prev&&prev.v,mv,.45)};
  }
  function featureVector(fused,head,left,right){return [fused.u,fused.v,head?head.yaw:0,head?head.pitch:0,head?head.roll:0,left.apertureRatio,right.apertureRatio];}
  function onResults(results){
    const video=document.getElementById('webgazerVideoFeed'),lms=results&&results.multiFaceLandmarks&&results.multiFaceLandmarks[0];
    if(!video||!lms||lms.length<478){state.missing++;return;}
    let left=eyeLocal(lms,video,EYE.left),right=eyeLocal(lms,video,EYE.right);if(!left||!right){state.missing++;return;}
    left=filteredEye('left',left);right=filteredEye('right',right);state.filtered={left,right};
    left.reliability=reliability('left',left);right.reliability=reliability('right',right);
    const fused=C.fuseEyes(left,right),head=headFeatures(lms,video,left,right);state.latestEyes={left,right,fused};state.latestHead=head;state.frames++;
    if(state.model&&fused&&fused.reliability>=.25){const p=C.predict(state.model,featureVector(fused,head,left,right));state.latestIrisPrediction=p;}
    else state.latestIrisPrediction=null;
    renderLive();
  }
  async function loadScript(src){return new Promise((resolve,reject)=>{if(window.FaceMesh)return resolve();const s=document.createElement('script');s.src=src;s.onload=resolve;s.onerror=()=>reject(new Error('FaceMesh load failed'));document.head.appendChild(s);});}
  async function ensure(){if(state.ready)return;await loadScript('./mediapipe/face_mesh/face_mesh.js');state.faceMesh=new window.FaceMesh({locateFile:f=>`./mediapipe/face_mesh/${f}`});state.faceMesh.setOptions({maxNumFaces:1,refineLandmarks:true,minDetectionConfidence:.5,minTrackingConfidence:.5});state.faceMesh.onResults(onResults);state.ready=true;state.timer=setInterval(send,100);}
  async function send(){if(!state.ready||state.sending)return;const video=document.getElementById('webgazerVideoFeed');if(!video||video.readyState<2)return;state.sending=true;try{await state.faceMesh.send({image:video});}catch(_){state.missing++;}finally{state.sending=false;}}
  function patchGaze(){if(!window.webgazer||window.webgazer.__captureFusionPatched)return false;const original=window.webgazer.setGazeListener.bind(window.webgazer);window.webgazer.setGazeListener=function(listener){return original(function(gaze,elapsed){if(gaze){state.latestWebGazer={x:gaze.x,y:gaze.y};const f=gaze.eyeFeatures;if(f){state.latestEyePatch.left=patchQuality(f.left);state.latestEyePatch.right=patchQuality(f.right);}}return listener(gaze,elapsed);});};window.webgazer.__captureFusionPatched=true;return true;}
  function sampleAtTarget(target){
    const e=state.latestEyes,h=state.latestHead;if(!e||!e.fused||e.fused.reliability<.20)return;
    const features=featureVector(e.fused,h,e.left,e.right);samples.push({features,x:target.x,y:target.y,weight:e.fused.reliability,reliability:e.fused.reliability,webgazer:state.latestWebGazer?{...state.latestWebGazer}:null});
    if(samples.length>=5)state.model=C.fitLinear(samples);
    evaluate();renderLive();
  }
  function evaluate(){
    if(samples.length<6){state.evaluation=null;return;}
    const loo=C.leaveOneOut(samples),wgPairs=samples.filter(s=>s.webgazer).map(s=>({px:s.webgazer.x,py:s.webgazer.y,tx:s.x,ty:s.y}));
    const fusedPairs=[];for(let i=0;i<samples.length;i++){const train=samples.filter((_,j)=>j!==i),m=C.fitLinear(train),ip=C.predict(m,samples[i].features),wg=samples[i].webgazer;if(!ip||!wg)continue;const a=.35+.55*clamp(samples[i].reliability);fusedPairs.push({px:ip.x*a+wg.x*(1-a),py:ip.y*a+wg.y*(1-a),tx:samples[i].x,ty:samples[i].y});}
    state.evaluation={samples:samples.length,webgazerRmse:C.rmse(wgPairs),irisLooRmse:loo.rmse,fusedLooRmse:C.rmse(fusedPairs)};
  }
  function renderLive(){const q=document.getElementById('captureFusionLive'),e=document.getElementById('captureFusionEval');if(q){const x=state.latestEyes;if(!x)q.textContent='waiting for refined iris capture';else q.textContent=`L ${(x.left.reliability*100).toFixed(0)}% · R ${(x.right.reliability*100).toFixed(0)}% · bilateral ${(x.fused.agreement*100).toFixed(0)}% · capture ${(x.fused.reliability*100).toFixed(0)}%${x.fused.reliability<.35?' · PAUSE QUALITY':''}`;}if(e){const v=state.evaluation;e.textContent=!v?`${samples.length} calibration samples collected`:`${v.samples} samples · WebGazer ${v.webgazerRmse==null?'—':v.webgazerRmse.toFixed(1)+' px'} · iris LOO ${v.irisLooRmse==null?'—':v.irisLooRmse.toFixed(1)+' px'} · fused LOO ${v.fusedLooRmse==null?'—':v.fusedLooRmse.toFixed(1)+' px'}`;}}
  function mount(){
    const section=document.createElement('section');section.id='captureFusionPanel';section.className='panel';section.innerHTML=`<div class="panelHead"><div><p class="eyebrow">EXPERIMENTAL CAPTURE FUSION</p><h2>Iris + head pose + per-eye reliability</h2></div><label>Validation cohort <select id="eyewearCohort"><option value="unknown">Unknown</option><option value="none">No eyewear</option><option value="glasses">Glasses</option><option value="contacts">Contacts</option></select></label></div><p class="hint">Cohort is user-selected evidence only; eyewear is never inferred. Each eye is weighted independently for glare, clipping, aperture/occlusion, landmark stability and bilateral agreement.</p><div class="cameraComparisonGrid"><article><span>Live capture reliability</span><strong id="captureFusionLive">waiting</strong></article><article><span>Predictor comparison</span><strong id="captureFusionEval">0 calibration samples collected</strong></article></div><div class="controls"><button id="downloadCaptureFusion" type="button">Download fusion evidence</button></div>`;
    const evidence=document.querySelector('.evidencePanel');if(evidence&&evidence.parentNode)evidence.parentNode.insertBefore(section,evidence);else document.querySelector('main').appendChild(section);
    section.querySelector('#eyewearCohort').addEventListener('change',ev=>state.cohort=ev.target.value);
    section.querySelector('#downloadCaptureFusion').addEventListener('click',download);
    document.addEventListener('click',ev=>{const b=ev.target.closest&&ev.target.closest('.calibrationPoint');if(!b)return;const r=b.getBoundingClientRect();sampleAtTarget({x:r.left+r.width/2,y:r.top+r.height/2});});
  }
  function evidence(){return {schema:'webgazer-aac/capture-fusion-experiment/0.1',exportedAt:new Date().toISOString(),experimental:true,canonicalRuntimeModified:false,eyewearCohort:state.cohort,rawEyeMaterialPersisted:false,rawLandmarksPersisted:false,frames:state.frames,missing:state.missing,latest:{eyes:state.latestEyes,head:state.latestHead,eyePatchQuality:state.latestEyePatch},calibrationSamples:samples.length,evaluation:state.evaluation};}
  function download(){const blob=new Blob([JSON.stringify(evidence(),null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`webgazer-capture-fusion-${Date.now()}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),0);}
  mount();
  const poll=setInterval(()=>{if(patchGaze()){ensure().catch(()=>{});clearInterval(poll);}},100);
  window.webgazerCaptureFusion={getLive:()=>({available:!!state.latestEyes,reliability:state.latestEyes&&state.latestEyes.fused?state.latestEyes.fused.reliability:0,leftReliability:state.latestEyes?state.latestEyes.left.reliability:0,rightReliability:state.latestEyes?state.latestEyes.right.reliability:0,prediction:state.latestIrisPrediction}),getEvidence:evidence,getEvaluation:()=>state.evaluation};
})();