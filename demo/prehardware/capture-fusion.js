'use strict';
(function(){
  const C=window.webgazerCaptureFusionCore;
  if(!C)return;

  const PURSUIT_MS=18000;
  const MIN_CAPTURE_RELIABILITY=.40;
  const MAX_MODEL_SAMPLES=160;
  const sparseSamples=[];
  const pursuitSamples=[];
  const canvases={left:document.createElement('canvas'),right:document.createElement('canvas')};
  canvases.left.width=96;canvases.left.height=48;canvases.right.width=96;canvases.right.height=48;

  const state={
    cohort:'unknown',faceMesh:null,ready:false,starting:false,sending:false,
    latestEyes:null,latestHead:null,latestWebGazer:null,latestEyePatch:{left:null,right:null},
    filters:{left:{u:C.makeOneEuro(),v:C.makeOneEuro()},right:{u:C.makeOneEuro(),v:C.makeOneEuro()}},
    history:{left:[],right:[]},models:{},selectedKernel:null,selectedModel:null,evaluation:null,
    frames:0,missing:0,rejected:0,startupState:'waiting-for-webgazer',startupError:null,
    frameClock:{mode:null,callbackId:null,captured:0,lastCapture:null,latencies:[],frameIntervals:[],lastNow:null},
    pursuit:null,pursuitCompleted:false,coachPassSince:null,coachBypassed:false,
    latestIrisPrediction:null,latestFusedPrediction:null,
  };

  const EYE={
    left:{outer:263,inner:362,upper:386,lower:374,iris:[468,469,470,471,472],contour:[263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]},
    right:{outer:33,inner:133,upper:159,lower:145,iris:[473,474,475,476,477],contour:[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]},
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
    if(width<4)return null;
    const ex=norm(axis),ey=[-ex[1],ex[0]],d=subtract(iris,center),aperture=Math.hypot(upper[0]-lower[0],upper[1]-lower[1]);
    return {u:dot(d,ex)/width,v:dot(d,ey)/width,width,aperture,apertureRatio:aperture/width,center,outer,inner};
  }

  function headFeatures(lms,video,left,right){
    const nose=point(lms[1],video),forehead=point(lms[10],video),chin=point(lms[152],video);
    if(!nose||!forehead||!chin||!left||!right)return null;
    const eyeMid=[(left.center[0]+right.center[0])/2,(left.center[1]+right.center[1])/2];
    const inter=Math.hypot(left.center[0]-right.center[0],left.center[1]-right.center[1])||1;
    const faceH=Math.hypot(forehead[0]-chin[0],forehead[1]-chin[1])||1;
    const vw=video.videoWidth||640,vh=video.videoHeight||480;
    return {
      yaw:(nose[0]-eyeMid[0])/inter,
      pitch:(nose[1]-eyeMid[1])/faceH,
      roll:Math.atan2(right.center[1]-left.center[1],right.center[0]-left.center[0]),
      interocularPx:inter,interocularNorm:inter/vw,
      faceOffsetX:(eyeMid[0]-vw/2)/vw,faceOffsetY:(eyeMid[1]-vh*.42)/vh,
    };
  }

  function eyeCropQuality(lms,video,cfg,side){
    const pts=cfg.contour.map(i=>point(lms[i],video)).filter(Boolean);if(pts.length<8)return null;
    const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);
    let minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
    const w=maxX-minX,h=maxY-minY,padX=w*.45,padY=Math.max(h*.8,w*.18);
    minX=Math.max(0,minX-padX);maxX=Math.min(video.videoWidth,maxX+padX);minY=Math.max(0,minY-padY);maxY=Math.min(video.videoHeight,maxY+padY);
    const sw=maxX-minX,sh=maxY-minY;if(sw<4||sh<3)return null;
    const canvas=canvases[side],ctx=canvas.getContext('2d',{willReadFrequently:true});
    try{ctx.drawImage(video,minX,minY,sw,sh,0,0,canvas.width,canvas.height);}catch(_){return null;}
    const data=ctx.getImageData(0,0,canvas.width,canvas.height).data,vals=[];let bright=0,clip=0,dark=0;
    for(let i=0;i<data.length;i+=16){const y=.299*data[i]+.587*data[i+1]+.114*data[i+2];vals.push(y);if(y>=235)bright++;if(y>=250)clip++;if(y<=20)dark++;}
    vals.sort((a,b)=>a-b);const n=vals.length,p10=vals[Math.floor(n*.1)]||0,p90=vals[Math.floor(n*.9)]||0;
    return {glare:n?bright/n:0,clipped:n?clip/n:0,dark:n?dark/n:0,contrast:(p90-p10)/255,luminance:avg(vals),sourceWidthPx:sw,sourceHeightPx:sh,analysisWidth:canvas.width,analysisHeight:canvas.height};
  }

  function filterEye(side,eye,t){
    const f=state.filters[side];return {...eye,u:f.u.update(eye.u,t),v:f.v.update(eye.v,t)};
  }

  function reliability(side,eye){
    const h=state.history[side],u=h.map(x=>x.u).slice(-7),v=h.map(x=>x.v).slice(-7),jitter=Math.hypot(stdev(u),stdev(v));
    const q=state.latestEyePatch[side]||{glare:0,clipped:0};
    const blink=eye.apertureRatio<.075;
    return C.eyeReliability({visibility:1,glare:q.glare,clipped:q.clipped,apertureRatio:eye.apertureRatio,jitter,landmarkConfidence:1,occlusion:eye.apertureRatio<.10?.7:eye.apertureRatio<.14?.35:0,blink});
  }

  function featureVector(fused,head,left,right){
    return [
      fused.u,fused.v,left.u,left.v,right.u,right.v,
      head?head.yaw:0,head?head.pitch:0,head?head.roll:0,
      left.apertureRatio,right.apertureRatio,head?head.interocularNorm:0,fused.agreement||0
    ];
  }

  function currentWebGazer(){
    const dot=document.getElementById('gazeDot');if(!dot||dot.hidden)return null;
    const x=parseFloat(dot.style.left),y=parseFloat(dot.style.top);return Number.isFinite(x)&&Number.isFinite(y)?{x,y}:null;
  }

  function coachMetrics(left,right,fused,head){
    const lq=state.latestEyePatch.left,rq=state.latestEyePatch.right;
    const lum=lq&&rq?(lq.luminance+rq.luminance)/2:null;
    const asym=lq&&rq?Math.abs(lq.luminance-rq.luminance)/Math.max(20,(lq.luminance+rq.luminance)/2):null;
    return {
      faceOffsetX:head?head.faceOffsetX:1,faceOffsetY:head?head.faceOffsetY:1,interocularNorm:head?head.interocularNorm:0,
      yaw:head?head.yaw:1,pitch:head?head.pitch:1,roll:head?head.roll:1,luminance:lum,lightingAsymmetry:asym,
      leftReliability:left.reliability,rightReliability:right.reliability,fusedReliability:fused.reliability
    };
  }

  function updateCoach(left,right,fused,head){
    const assessment=C.coachAssessment(coachMetrics(left,right,fused,head));
    const now=performance.now();if(assessment.pass){if(state.coachPassSince==null)state.coachPassSince=now;}else state.coachPassSince=null;
    const stable=assessment.pass&&state.coachPassSince!=null&&now-state.coachPassSince>=1500;
    state.coach={...assessment,stable,stableMs:state.coachPassSince==null?0:now-state.coachPassSince};
    const button=document.getElementById('startPursuit');if(button&&!state.pursuit&&!state.pursuitCompleted)button.disabled=!stable;
  }

  function sampleFromFrame(target,source,frameCtx){
    const e=state.latestEyes,h=state.latestHead;if(!e||!e.fused||e.fused.reliability<MIN_CAPTURE_RELIABILITY||!target)return false;
    const features=featureVector(e.fused,h,e.left,e.right),wg=state.latestWebGazer?{...state.latestWebGazer}:null;
    const s={features,x:target.x,y:target.y,weight:e.fused.reliability,reliability:e.fused.reliability,webgazer:wg,source,captureTime:frameCtx&&frameCtx.captureTime||performance.now()};
    (source==='pursuit'?pursuitSamples:sparseSamples).push(s);return true;
  }

  function onResults(results){
    const emit=performance.now(),video=document.getElementById('webgazerVideoFeed'),lms=results&&results.multiFaceLandmarks&&results.multiFaceLandmarks[0],frameCtx=state.frameClock.lastCapture;
    if(!video||!lms||lms.length<478){state.missing++;return;}
    let left=eyeLocal(lms,video,EYE.left),right=eyeLocal(lms,video,EYE.right);if(!left||!right){state.missing++;return;}
    const t=frameCtx&&frameCtx.captureTime||emit;left=filterEye('left',left,t);right=filterEye('right',right,t);
    state.history.left.push({u:left.u,v:left.v});state.history.right.push({u:right.u,v:right.v});if(state.history.left.length>12)state.history.left.shift();if(state.history.right.length>12)state.history.right.shift();
    state.latestEyePatch.left=eyeCropQuality(lms,video,EYE.left,'left');state.latestEyePatch.right=eyeCropQuality(lms,video,EYE.right,'right');
    left.reliability=reliability('left',left);right.reliability=reliability('right',right);
    const fused=C.fuseEyes(left,right),head=headFeatures(lms,video,left,right);state.latestEyes={left,right,fused};state.latestHead=head;state.latestWebGazer=currentWebGazer();state.frames++;
    updateCoach(left,right,fused,head);
    if(frameCtx&&Number.isFinite(frameCtx.captureTime)){state.frameClock.latencies.push(emit-frameCtx.captureTime);if(state.frameClock.latencies.length>500)state.frameClock.latencies.shift();}
    if(state.pursuit&&frameCtx&&frameCtx.pursuitTarget){
      if(!sampleFromFrame(frameCtx.pursuitTarget,'pursuit',frameCtx))state.rejected++;
    }
    if(state.selectedModel&&fused&&fused.reliability>=.25){
      state.latestIrisPrediction=C.predict(state.selectedModel,featureVector(fused,head,left,right));
      const wg=state.latestWebGazer,ip=state.latestIrisPrediction;
      if(wg&&ip){const a=fusionAlpha(fused.reliability);state.latestFusedPrediction={x:ip.x*a+wg.x*(1-a),y:ip.y*a+wg.y*(1-a),irisWeight:a};}else state.latestFusedPrediction=ip;
    }else{state.latestIrisPrediction=null;state.latestFusedPrediction=null;}
    renderLive();
  }

  function fusionAlpha(reliability){
    const ev=state.evaluation;if(!ev||!Number.isFinite(ev.webgazer.rmse)||!Number.isFinite(ev.selected.rmse))return .25+.65*clamp(reliability);
    const iw=1/Math.max(1,ev.selected.rmse),ww=1/Math.max(1,ev.webgazer.rmse),base=iw/(iw+ww);return clamp(base*clamp(reliability),.10,.92);
  }

  async function loadScript(src){return new Promise((resolve,reject)=>{if(window.FaceMesh)return resolve();const s=document.createElement('script');s.src=src;s.onload=resolve;s.onerror=()=>reject(new Error('FaceMesh load failed'));document.head.appendChild(s);});}
  function sessionRunning(){const s=document.getElementById('sessionState');return !!(s&&s.textContent.trim().toLowerCase()==='running');}
  function webgazerStartupComplete(){try{const video=document.getElementById('webgazerVideoFeed');return !!(sessionRunning()&&window.webgazer&&video&&video.readyState>=2&&video.videoWidth>0&&video.videoHeight>0);}catch(_){return false;}}

  async function ensure(){
    if(state.ready||state.starting||!webgazerStartupComplete())return;
    state.starting=true;state.startupState='starting-refined-detector';state.startupError=null;renderLive();
    try{
      await new Promise(resolve=>setTimeout(resolve,1250));
      if(!webgazerStartupComplete())throw new Error('WebGazer session not running before refined detector startup');
      await loadScript('./mediapipe/face_mesh/face_mesh.js');
      state.faceMesh=new window.FaceMesh({locateFile:f=>`./mediapipe/face_mesh/${f}`});
      state.faceMesh.setOptions({maxNumFaces:1,refineLandmarks:true,minDetectionConfidence:.5,minTrackingConfidence:.5});
      state.faceMesh.onResults(onResults);state.ready=true;state.startupState='ready';startFrameClock();
    }catch(error){state.startupState='error';state.startupError=String(error&&error.message||error);}
    finally{state.starting=false;renderLive();}
  }

  function pursuitTargetAt(now){
    if(!state.pursuit)return null;const progress=clamp((now-state.pursuit.startedAt)/PURSUIT_MS);const p=C.pursuitPoint(progress),x=p.nx*window.innerWidth,y=p.ny*window.innerHeight;
    movePursuitMarker(x,y);return {x,y,nx:p.nx,ny:p.ny,progress};
  }

  async function processVideoFrame(now,metadata){
    if(state.frameClock.lastNow!=null){state.frameClock.frameIntervals.push(now-state.frameClock.lastNow);if(state.frameClock.frameIntervals.length>500)state.frameClock.frameIntervals.shift();}state.frameClock.lastNow=now;
    const target=pursuitTargetAt(now),ctx={captureTime:now,mediaTime:metadata&&metadata.mediaTime,expectedDisplayTime:metadata&&metadata.expectedDisplayTime,presentedFrames:metadata&&metadata.presentedFrames,pursuitTarget:target};
    state.frameClock.lastCapture=ctx;state.frameClock.captured++;
    if(state.pursuit&&now-state.pursuit.startedAt>=PURSUIT_MS)finishPursuit();
    if(!state.sending&&state.ready&&sessionRunning()){
      const video=document.getElementById('webgazerVideoFeed');if(video&&video.readyState>=2){state.sending=true;try{await state.faceMesh.send({image:video});}catch(_){state.missing++;}finally{state.sending=false;}}
    }
  }

  function startFrameClock(){
    const video=document.getElementById('webgazerVideoFeed');if(!video)return;
    if(typeof video.requestVideoFrameCallback==='function'){
      state.frameClock.mode='requestVideoFrameCallback';
      const tick=(now,metadata)=>{processVideoFrame(now,metadata).finally(()=>{if(state.ready&&sessionRunning())state.frameClock.callbackId=video.requestVideoFrameCallback(tick);});};
      state.frameClock.callbackId=video.requestVideoFrameCallback(tick);
    }else{
      state.frameClock.mode='interval-fallback';
      const timer=setInterval(()=>{if(!state.ready||!sessionRunning()){clearInterval(timer);return;}processVideoFrame(performance.now(),null);},50);state.frameClock.callbackId=timer;
    }
  }

  function stopFrameClock(){
    const video=document.getElementById('webgazerVideoFeed'),id=state.frameClock.callbackId;if(id==null)return;
    if(state.frameClock.mode==='requestVideoFrameCallback'&&video&&typeof video.cancelVideoFrameCallback==='function')video.cancelVideoFrameCallback(id);else if(state.frameClock.mode==='interval-fallback')clearInterval(id);
    state.frameClock.callbackId=null;
  }

  function sampleAtClick(target){sampleFromFrame(target,'sparse',state.frameClock.lastCapture);renderLive();}

  function buildEvaluation(samples){
    if(samples.length<18)return null;
    const kernels=['linear','poly2','rbf'],models={},scores={};
    for(const kernel of kernels){
      const cv=C.kFold(samples,{kernel,folds:4,maxSamples:MAX_MODEL_SAMPLES});
      scores[kernel]=cv.metrics;models[kernel]=C.fitModel(samples,{kernel,maxSamples:MAX_MODEL_SAMPLES});
    }
    const selected=kernels.filter(k=>Number.isFinite(scores[k].rmse)).sort((a,b)=>scores[a].rmse-scores[b].rmse)[0]||'linear';
    const wgPairs=samples.filter(s=>s.webgazer).map(s=>({px:s.webgazer.x,py:s.webgazer.y,tx:s.x,ty:s.y}));
    const region=C.leaveRegionOut(samples,{kernel:selected,width:window.innerWidth,height:window.innerHeight,maxSamples:MAX_MODEL_SAMPLES});
    const wg=C.evaluatePairs(wgPairs),chosen=scores[selected];
    return {models,scores,selected,webgazer:wg,selectedMetrics:chosen,regionOut:region.metrics,regionCount:region.regions};
  }

  function evaluateAndFit(){
    const training=pursuitSamples.length>=18?pursuitSamples:sparseSamples;
    const result=buildEvaluation(training);if(!result)return;
    state.models=result.models;state.selectedKernel=result.selected;state.selectedModel=result.models[result.selected];
    state.evaluation={
      source:pursuitSamples.length>=18?'smooth-pursuit':'sparse-click',samples:training.length,selectedKernel:result.selected,
      kernels:result.scores,webgazer:result.webgazer,selected:result.selectedMetrics,leaveRegionOut:result.regionOut,regionCount:result.regionCount,
      model:{type:state.selectedModel&&state.selectedModel.type,kernel:state.selectedModel&&state.selectedModel.kernel,trainingCount:state.selectedModel&&state.selectedModel.trainingCount||training.length,sourceCount:state.selectedModel&&state.selectedModel.sourceCount||training.length,gamma:state.selectedModel&&state.selectedModel.gamma||null,lambda:state.selectedModel&&state.selectedModel.lambda||null}
    };
    renderLive();
  }

  function movePursuitMarker(x,y){const m=document.getElementById('pursuitMarker');if(!m)return;m.hidden=false;m.style.left=`${x}px`;m.style.top=`${y}px`;}
  function startPursuit(bypass=false){
    if(state.pursuit||!state.ready)return;
    if(!(state.coach&&state.coach.stable)&&!bypass)return;
    pursuitSamples.length=0;state.pursuitCompleted=false;state.coachBypassed=!!bypass;state.pursuit={startedAt:performance.now(),durationMs:PURSUIT_MS};
    const btn=document.getElementById('startPursuit');if(btn){btn.disabled=true;btn.textContent='Pursuit calibration running…';}
    renderLive();
  }
  function finishPursuit(){
    if(!state.pursuit)return;state.pursuit=null;state.pursuitCompleted=true;const m=document.getElementById('pursuitMarker');if(m)m.hidden=true;
    const btn=document.getElementById('startPursuit');if(btn){btn.disabled=true;btn.textContent=`Pursuit captured ${pursuitSamples.length} samples`;}
    setTimeout(evaluateAndFit,0);renderLive();
  }

  function renderLive(){
    const q=document.getElementById('captureFusionLive'),e=document.getElementById('captureFusionEval'),coach=document.getElementById('captureCoach'),timing=document.getElementById('captureTiming');
    if(q){const x=state.latestEyes;if(!state.ready&&!x)q.textContent=state.startupState==='error'?`refined detector error: ${state.startupError}`:state.startupState==='starting-refined-detector'?'starting refined detector after WebGazer…':'waiting for running WebGazer session';else if(!x)q.textContent='waiting for refined iris capture';else q.textContent=`L ${(x.left.reliability*100).toFixed(0)}% · R ${(x.right.reliability*100).toFixed(0)}% · bilateral ${(x.fused.agreement*100).toFixed(0)}% · capture ${(x.fused.reliability*100).toFixed(0)}%${x.fused.reliability<.35?' · PAUSE QUALITY':''}`;}
    if(coach){const c=state.coach;coach.textContent=!c?'waiting for face geometry':c.stable?'READY · capture conditions stable':c.pass?`hold steady ${Math.min(1.5,c.stableMs/1000).toFixed(1)} / 1.5 s`:`adjust: ${c.reasons.join(', ')}`;}
    if(timing){const lat=state.frameClock.latencies,iv=state.frameClock.frameIntervals;timing.textContent=`${state.frameClock.mode||'waiting'} · inference p50 ${lat.length?C.quantile(lat,.5).toFixed(1):'—'} ms · p95 ${lat.length?C.quantile(lat,.95).toFixed(1):'—'} ms · frame ${iv.length?C.quantile(iv,.5).toFixed(1):'—'} ms`;}
    if(e){const v=state.evaluation;if(!v)e.textContent=`${pursuitSamples.length} pursuit · ${sparseSamples.length} click samples`;else e.textContent=`${v.source} ${v.samples} · selected ${v.selectedKernel} ${v.selected.rmse==null?'—':v.selected.rmse.toFixed(1)+' px CV'} · region ${v.leaveRegionOut.rmse==null?'—':v.leaveRegionOut.rmse.toFixed(1)+' px'} · WebGazer ${v.webgazer.rmse==null?'—':v.webgazer.rmse.toFixed(1)+' px'}`;}
  }

  function mount(){
    const section=document.createElement('section');section.id='captureFusionPanel';section.className='panel';
    section.innerHTML=`<div class="panelHead"><div><p class="eyebrow">EXPERIMENTAL CAPTURE FUSION</p><h2>Dense iris + head-pose gaze engine</h2></div><label>Validation cohort <select id="eyewearCohort"><option value="unknown">Unknown</option><option value="none">No eyewear</option><option value="glasses">Glasses</option><option value="contacts">Contacts</option></select></label></div>
      <p class="hint">Runs only after WebGazer is already running. Refined 478-landmark capture is frame-synchronized, eye-local, head-pose aware and reliability weighted. No raw eye images or landmarks are persisted.</p>
      <div class="cameraComparisonGrid"><article><span>Capture reliability</span><strong id="captureFusionLive">waiting for running WebGazer session</strong></article><article><span>Positioning coach</span><strong id="captureCoach">waiting</strong></article><article><span>Frame timing</span><strong id="captureTiming">waiting</strong></article><article><span>Predictor comparison</span><strong id="captureFusionEval">no dense calibration yet</strong></article></div>
      <div class="controls"><button id="startPursuit" type="button" disabled>Start 18 s smooth-pursuit calibration</button><button id="bypassPursuit" type="button">Start pursuit anyway</button><button id="downloadCaptureFusion" type="button">Download fusion evidence</button></div>
      <p class="hint">The experimental engine automatically compares linear, polynomial-2 and RBF kernel ridge models by held-out error, then selects the lowest-error model. WebGazer remains the untouched baseline.</p>`;
    const evidence=document.querySelector('.evidencePanel');if(evidence&&evidence.parentNode)evidence.parentNode.insertBefore(section,evidence);else document.querySelector('main').appendChild(section);
    section.querySelector('#eyewearCohort').addEventListener('change',ev=>state.cohort=ev.target.value);
    section.querySelector('#startPursuit').addEventListener('click',()=>startPursuit(false));section.querySelector('#bypassPursuit').addEventListener('click',()=>startPursuit(true));section.querySelector('#downloadCaptureFusion').addEventListener('click',download);
    document.addEventListener('click',ev=>{const b=ev.target.closest&&ev.target.closest('.calibrationPoint');if(!b)return;const r=b.getBoundingClientRect();sampleAtClick({x:r.left+r.width/2,y:r.top+r.height/2});});
    const marker=document.createElement('div');marker.id='pursuitMarker';marker.hidden=true;marker.setAttribute('aria-hidden','true');marker.style.cssText='position:fixed;width:24px;height:24px;border:4px solid #111827;border-radius:50%;transform:translate(-50%,-50%);z-index:7000;pointer-events:none;background:rgba(255,255,255,.86);box-shadow:0 0 0 5px rgba(255,255,255,.45)';document.body.appendChild(marker);
  }

  function timingEvidence(){const lat=state.frameClock.latencies,iv=state.frameClock.frameIntervals;return {mode:state.frameClock.mode,capturedFrames:state.frameClock.captured,inferenceLatencyMs:{p50:lat.length?C.quantile(lat,.5):null,p95:lat.length?C.quantile(lat,.95):null},frameIntervalMs:{p50:iv.length?C.quantile(iv,.5):null,p95:iv.length?C.quantile(iv,.95):null}};}
  function evidence(){
    const x=state.latestEyes;return {
      schema:'webgazer-aac/capture-fusion-experiment/0.3',exportedAt:new Date().toISOString(),experimental:true,canonicalRuntimeModified:false,
      eyewearCohort:state.cohort,rawEyeMaterialPersisted:false,rawLandmarksPersisted:false,
      startup:{state:state.startupState,error:state.startupError,deferredUntilWebGazerSessionRunning:true},
      calibration:{method:pursuitSamples.length?'smooth-pursuit-lissajous':'sparse-click',durationMs:pursuitSamples.length?PURSUIT_MS:null,pursuitSamples:pursuitSamples.length,sparseSamples:sparseSamples.length,rejectedFrames:state.rejected,coachBypassed:state.coachBypassed},
      frameTiming:timingEvidence(),coach:state.coach||null,frames:state.frames,missing:state.missing,
      latest:x?{leftReliability:x.left.reliability,rightReliability:x.right.reliability,bilateralAgreement:x.fused.agreement,captureReliability:x.fused.reliability,leftApertureRatio:x.left.apertureRatio,rightApertureRatio:x.right.apertureRatio,head:state.latestHead,eyePatchQuality:state.latestEyePatch}:null,
      evaluation:state.evaluation,
      livePrediction:{iris:state.latestIrisPrediction,fused:state.latestFusedPrediction,webgazer:state.latestWebGazer}
    };
  }
  function download(){const blob=new Blob([JSON.stringify(evidence(),null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`webgazer-capture-fusion-${Date.now()}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),0);}

  mount();renderLive();ensure();
  const ensurePoll=setInterval(()=>{if(state.ready||state.startupState==='error'){clearInterval(ensurePoll);return;}ensure();},250);
  window.addEventListener('beforeunload',stopFrameClock);
  window.webgazerCaptureFusion={
    getLive:()=>({available:!!state.latestEyes,reliability:state.latestEyes&&state.latestEyes.fused?state.latestEyes.fused.reliability:0,leftReliability:state.latestEyes?state.latestEyes.left.reliability:0,rightReliability:state.latestEyes?state.latestEyes.right.reliability:0,prediction:state.latestFusedPrediction||state.latestIrisPrediction,startupState:state.startupState,selectedKernel:state.selectedKernel}),
    getEvidence:evidence,getEvaluation:()=>state.evaluation,startPursuit:()=>startPursuit(false),stopFrameClock
  };
})();
