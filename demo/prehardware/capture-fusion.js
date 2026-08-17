'use strict';
(function(){
  const C=window.webgazerCaptureFusionCore;
  if(!C)return;

  const PURSUIT_MS=18000,MIN_CAPTURE_RELIABILITY=.40,MAX_MODEL_SAMPLES=160,ORIGIN=location.origin;
  const sparseSamples=[],pursuitSamples=[];
  const state={
    cohort:'unknown',ready:false,starting:false,startupState:'waiting-for-isolated-capture',startupError:null,iframe:null,
    latestEyes:null,latestHead:null,latestWebGazer:null,latestQuality:{left:null,right:null},latestVideo:null,
    filters:{left:{u:C.makeOneEuro(),v:C.makeOneEuro()},right:{u:C.makeOneEuro(),v:C.makeOneEuro()}},history:{left:[],right:[]},
    models:{},selectedKernel:null,selectedModel:null,evaluation:null,frames:0,missing:0,rejected:0,
    frameClock:{mode:null,callbackId:null,nextFrameId:1,inFlight:null,captured:0,dropped:0,latencies:[],frameIntervals:[],lastNow:null},
    pursuit:null,pursuitCompleted:false,coachPassSince:null,coachBypassed:false,coach:null,
    latestIrisPrediction:null,latestFusedPrediction:null,
  };

  function avg(a){return C.avg(a);} function stdev(a){return C.stdev(a);} function clamp(v){return C.clamp(v);}
  function sessionRunning(){const s=document.getElementById('sessionState');return !!(s&&s.textContent.trim().toLowerCase()==='running');}
  function currentWebGazer(){const dot=document.getElementById('gazeDot');if(!dot||dot.hidden)return null;const x=parseFloat(dot.style.left),y=parseFloat(dot.style.top);return Number.isFinite(x)&&Number.isFinite(y)?{x,y}:null;}

  function filterEye(side,eye,t){const f=state.filters[side];return {...eye,u:f.u.update(eye.u,t),v:f.v.update(eye.v,t)};}
  function reliability(side,eye){
    const h=state.history[side],u=h.map(x=>x.u).slice(-7),v=h.map(x=>x.v).slice(-7),jitter=Math.hypot(stdev(u),stdev(v)),q=state.latestQuality[side]||{glare:0,clipped:0},blink=eye.apertureRatio<.075;
    return C.eyeReliability({visibility:1,glare:q.glare,clipped:q.clipped,apertureRatio:eye.apertureRatio,jitter,landmarkConfidence:1,occlusion:eye.apertureRatio<.10?.7:eye.apertureRatio<.14?.35:0,blink});
  }
  function featureVector(fused,head,left,right){return [fused.u,fused.v,left.u,left.v,right.u,right.v,head?head.yaw:0,head?head.pitch:0,head?head.roll:0,left.apertureRatio,right.apertureRatio,head?head.interocularNorm:0,fused.agreement||0];}
  function coachMetrics(left,right,fused,head){
    const lq=state.latestQuality.left,rq=state.latestQuality.right,lum=lq&&rq?(lq.luminance+rq.luminance)/2:null,asym=lq&&rq?Math.abs(lq.luminance-rq.luminance)/Math.max(20,(lq.luminance+rq.luminance)/2):null;
    return {faceOffsetX:head?head.faceOffsetX:1,faceOffsetY:head?head.faceOffsetY:1,interocularNorm:head?head.interocularNorm:0,yaw:head?head.yaw:1,pitch:head?head.pitch:1,roll:head?head.roll:1,luminance:lum,lightingAsymmetry:asym,leftReliability:left.reliability,rightReliability:right.reliability,fusedReliability:fused.reliability};
  }
  function updateCoach(left,right,fused,head){
    const assessment=C.coachAssessment(coachMetrics(left,right,fused,head)),now=performance.now();if(assessment.pass){if(state.coachPassSince==null)state.coachPassSince=now;}else state.coachPassSince=null;
    const stable=assessment.pass&&state.coachPassSince!=null&&now-state.coachPassSince>=1500;state.coach={...assessment,stable,stableMs:state.coachPassSince==null?0:now-state.coachPassSince};
    const button=document.getElementById('startPursuit');if(button&&!state.pursuit&&!state.pursuitCompleted)button.disabled=!stable;
  }

  function sampleFromFrame(target,source,ctx){
    const e=state.latestEyes,h=state.latestHead;if(!e||!e.fused||e.fused.reliability<MIN_CAPTURE_RELIABILITY||!target)return false;
    const s={features:featureVector(e.fused,h,e.left,e.right),x:target.x,y:target.y,weight:e.fused.reliability,reliability:e.fused.reliability,webgazer:state.latestWebGazer?{...state.latestWebGazer}:null,source,captureTime:ctx&&ctx.captureTime||performance.now()};
    (source==='pursuit'?pursuitSamples:sparseSamples).push(s);return true;
  }

  function fusionAlpha(reliability){
    const ev=state.evaluation;if(!ev||!Number.isFinite(ev.webgazer.rmse)||!Number.isFinite(ev.selected.rmse))return .25+.65*clamp(reliability);
    const iw=1/Math.max(1,ev.selected.rmse),ww=1/Math.max(1,ev.webgazer.rmse),base=iw/(iw+ww);return clamp(base*clamp(reliability),.10,.92);
  }

  function handleCaptureResult(msg){
    const ctx=state.frameClock.inFlight;if(!ctx||msg.frameId!==ctx.frameId)return;
    state.frameClock.inFlight=null;const emit=performance.now();
    if(!msg.ok){state.missing++;renderLive();return;}
    let left=filterEye('left',msg.left,msg.captureTime||ctx.captureTime),right=filterEye('right',msg.right,msg.captureTime||ctx.captureTime);
    state.history.left.push({u:left.u,v:left.v});state.history.right.push({u:right.u,v:right.v});if(state.history.left.length>12)state.history.left.shift();if(state.history.right.length>12)state.history.right.shift();
    state.latestQuality=msg.quality||{left:null,right:null};left.reliability=reliability('left',left);right.reliability=reliability('right',right);
    const fused=C.fuseEyes(left,right);state.latestEyes={left,right,fused};state.latestHead=msg.head||null;state.latestVideo=msg.video||null;state.latestWebGazer=currentWebGazer();state.frames++;
    updateCoach(left,right,fused,state.latestHead);
    state.frameClock.latencies.push(emit-ctx.captureTime);if(state.frameClock.latencies.length>500)state.frameClock.latencies.shift();
    if(state.pursuit&&ctx.pursuitTarget){if(!sampleFromFrame(ctx.pursuitTarget,'pursuit',ctx))state.rejected++;}
    if(state.selectedModel&&fused&&fused.reliability>=.25){
      state.latestIrisPrediction=C.predict(state.selectedModel,featureVector(fused,state.latestHead,left,right));const wg=state.latestWebGazer,ip=state.latestIrisPrediction;
      if(wg&&ip){const a=fusionAlpha(fused.reliability);state.latestFusedPrediction={x:ip.x*a+wg.x*(1-a),y:ip.y*a+wg.y*(1-a),irisWeight:a};}else state.latestFusedPrediction=ip;
    }else{state.latestIrisPrediction=null;state.latestFusedPrediction=null;}
    renderLive();
  }

  function onMessage(event){
    if(event.origin!==ORIGIN||!state.iframe||event.source!==state.iframe.contentWindow)return;const msg=event.data||{};if(msg.source!=='webgazer-aac-refined-frame')return;
    if(msg.type==='ready'){state.ready=true;state.startupState='ready-isolated';state.startupError=null;startFrameClock();renderLive();return;}
    if(msg.type==='error'){state.startupState='error';state.startupError=msg.reason||'isolated detector error';renderLive();return;}
    if(msg.type==='result')handleCaptureResult(msg);
  }

  function ensureIsolatedCapture(){
    if(state.starting||state.ready||state.iframe)return;state.starting=true;state.startupState='starting-isolated-refined-capture';renderLive();
    const iframe=document.createElement('iframe');iframe.src=`./refined-capture-frame.html?t=${Date.now()}`;iframe.title='Isolated local refined iris capture';iframe.setAttribute('aria-hidden','true');iframe.tabIndex=-1;iframe.style.cssText='position:fixed;left:-10000px;top:-10000px;width:2px;height:2px;border:0;opacity:0;pointer-events:none';
    iframe.addEventListener('load',()=>{state.starting=false;state.startupState='isolated-frame-loaded';renderLive();});iframe.addEventListener('error',()=>{state.starting=false;state.startupState='error';state.startupError='isolated capture frame failed to load';renderLive();});state.iframe=iframe;document.body.appendChild(iframe);
  }

  function pursuitTargetAt(now){
    if(!state.pursuit)return null;const progress=clamp((now-state.pursuit.startedAt)/PURSUIT_MS),p=C.pursuitPoint(progress),x=p.nx*window.innerWidth,y=p.ny*window.innerHeight;movePursuitMarker(x,y);return {x,y,nx:p.nx,ny:p.ny,progress};
  }

  function requestInference(now,metadata){
    if(!state.ready||!state.iframe||!sessionRunning()||state.frameClock.inFlight)return;
    const target=pursuitTargetAt(now),ctx={frameId:state.frameClock.nextFrameId++,captureTime:now,mediaTime:metadata&&metadata.mediaTime,expectedDisplayTime:metadata&&metadata.expectedDisplayTime,presentedFrames:metadata&&metadata.presentedFrames,pursuitTarget:target};
    state.frameClock.inFlight=ctx;state.frameClock.captured++;state.iframe.contentWindow.postMessage({type:'process-frame',frameId:ctx.frameId,captureTime:ctx.captureTime,pursuitTarget:ctx.pursuitTarget},ORIGIN);
  }

  function startFrameClock(){
    const video=document.getElementById('webgazerVideoFeed');if(!video||state.frameClock.callbackId!=null)return;
    if(typeof video.requestVideoFrameCallback==='function'){
      state.frameClock.mode='requestVideoFrameCallback';
      const tick=(now,metadata)=>{
        state.frameClock.callbackId=null;if(state.frameClock.lastNow!=null){state.frameClock.frameIntervals.push(now-state.frameClock.lastNow);if(state.frameClock.frameIntervals.length>500)state.frameClock.frameIntervals.shift();}state.frameClock.lastNow=now;
        if(state.pursuit&&now-state.pursuit.startedAt>=PURSUIT_MS)finishPursuit();else pursuitTargetAt(now);
        if(state.frameClock.inFlight)state.frameClock.dropped++;else requestInference(now,metadata);
        if(state.ready&&sessionRunning())state.frameClock.callbackId=video.requestVideoFrameCallback(tick);
      };
      state.frameClock.callbackId=video.requestVideoFrameCallback(tick);
    }else{
      state.frameClock.mode='interval-fallback';state.frameClock.callbackId=setInterval(()=>{const now=performance.now();if(!sessionRunning())return;if(state.pursuit&&now-state.pursuit.startedAt>=PURSUIT_MS)finishPursuit();else pursuitTargetAt(now);if(state.frameClock.inFlight)state.frameClock.dropped++;else requestInference(now,null);},50);
    }
  }

  function stopFrameClock(){
    const video=document.getElementById('webgazerVideoFeed'),id=state.frameClock.callbackId;if(id==null)return;if(state.frameClock.mode==='requestVideoFrameCallback'&&video&&typeof video.cancelVideoFrameCallback==='function')video.cancelVideoFrameCallback(id);else clearInterval(id);state.frameClock.callbackId=null;
  }

  function buildEvaluation(samples){
    if(samples.length<18)return null;const kernels=['linear','poly2','rbf'],models={},scores={};
    for(const kernel of kernels){const cv=C.kFold(samples,{kernel,folds:4,maxSamples:MAX_MODEL_SAMPLES});scores[kernel]=cv.metrics;models[kernel]=C.fitModel(samples,{kernel,maxSamples:MAX_MODEL_SAMPLES});}
    const selected=kernels.filter(k=>Number.isFinite(scores[k].rmse)).sort((a,b)=>scores[a].rmse-scores[b].rmse)[0]||'linear';
    const wgPairs=samples.filter(s=>s.webgazer).map(s=>({px:s.webgazer.x,py:s.webgazer.y,tx:s.x,ty:s.y})),region=C.leaveRegionOut(samples,{kernel:selected,width:window.innerWidth,height:window.innerHeight,maxSamples:MAX_MODEL_SAMPLES});
    return {models,scores,selected,webgazer:C.evaluatePairs(wgPairs),selectedMetrics:scores[selected],regionOut:region.metrics,regionCount:region.regions};
  }
  function evaluateAndFit(){
    const training=pursuitSamples.length>=18?pursuitSamples:sparseSamples,result=buildEvaluation(training);if(!result)return;state.models=result.models;state.selectedKernel=result.selected;state.selectedModel=result.models[result.selected];
    state.evaluation={source:pursuitSamples.length>=18?'smooth-pursuit':'sparse-click',samples:training.length,selectedKernel:result.selected,kernels:result.scores,webgazer:result.webgazer,selected:result.selectedMetrics,leaveRegionOut:result.regionOut,regionCount:result.regionCount,model:{type:state.selectedModel&&state.selectedModel.type,kernel:state.selectedModel&&state.selectedModel.kernel,trainingCount:state.selectedModel&&state.selectedModel.trainingCount||training.length,sourceCount:state.selectedModel&&state.selectedModel.sourceCount||training.length,gamma:state.selectedModel&&state.selectedModel.gamma||null,lambda:state.selectedModel&&state.selectedModel.lambda||null}};renderLive();
  }

  function sampleAtClick(target){sampleFromFrame(target,'sparse',{captureTime:performance.now()});if(sparseSamples.length>=18&&!pursuitSamples.length)evaluateAndFit();renderLive();}
  function movePursuitMarker(x,y){const m=document.getElementById('pursuitMarker');if(!m)return;m.hidden=false;m.style.left=`${x}px`;m.style.top=`${y}px`;}
  function startPursuit(bypass=false){
    if(state.pursuit||!state.ready)return;if(!(state.coach&&state.coach.stable)&&!bypass)return;pursuitSamples.length=0;state.rejected=0;state.pursuitCompleted=false;state.coachBypassed=!!bypass;state.pursuit={startedAt:performance.now(),durationMs:PURSUIT_MS};const btn=document.getElementById('startPursuit');if(btn){btn.disabled=true;btn.textContent='Pursuit calibration running…';}renderLive();
  }
  function finishPursuit(){
    if(!state.pursuit)return;state.pursuit=null;state.pursuitCompleted=true;const m=document.getElementById('pursuitMarker');if(m)m.hidden=true;const btn=document.getElementById('startPursuit');if(btn){btn.disabled=true;btn.textContent=`Pursuit captured ${pursuitSamples.length} samples`;}setTimeout(evaluateAndFit,0);renderLive();
  }

  function timingEvidence(){const lat=state.frameClock.latencies,iv=state.frameClock.frameIntervals;return {mode:state.frameClock.mode,capturedFrames:state.frameClock.captured,droppedWhileBusy:state.frameClock.dropped,inferenceLatencyMs:{p50:lat.length?C.quantile(lat,.5):null,p95:lat.length?C.quantile(lat,.95):null},frameIntervalMs:{p50:iv.length?C.quantile(iv,.5):null,p95:iv.length?C.quantile(iv,.95):null}};}
  function featureSnapshot(){const x=state.latestEyes;if(!x)return null;return {capturedAt:performance.now(),left:{u:x.left.u,v:x.left.v,reliability:x.left.reliability,apertureRatio:x.left.apertureRatio,width:x.left.width},right:{u:x.right.u,v:x.right.v,reliability:x.right.reliability,apertureRatio:x.right.apertureRatio,width:x.right.width},fused:{u:x.fused.u,v:x.fused.v,reliability:x.fused.reliability,agreement:x.fused.agreement},head:state.latestHead,quality:state.latestQuality,video:state.latestVideo};}
  function evidence(){
    const x=state.latestEyes;return {schema:'webgazer-aac/capture-fusion-experiment/0.4',exportedAt:new Date().toISOString(),experimental:true,canonicalRuntimeModified:false,eyewearCohort:state.cohort,rawEyeMaterialPersisted:false,rawLandmarksPersisted:false,isolation:{sameOriginIframe:true,reason:'separate legacy MediaPipe Emscripten/WASM globals from WebGazer'},startup:{state:state.startupState,error:state.startupError,deferredUntilWebGazerSessionRunning:true},calibration:{method:pursuitSamples.length?'smooth-pursuit-lissajous':'sparse-click',durationMs:pursuitSamples.length?PURSUIT_MS:null,pursuitSamples:pursuitSamples.length,sparseSamples:sparseSamples.length,rejectedFrames:state.rejected,coachBypassed:state.coachBypassed},frameTiming:timingEvidence(),coach:state.coach||null,frames:state.frames,missing:state.missing,latest:x?{leftReliability:x.left.reliability,rightReliability:x.right.reliability,bilateralAgreement:x.fused.agreement,captureReliability:x.fused.reliability,leftApertureRatio:x.left.apertureRatio,rightApertureRatio:x.right.apertureRatio,head:state.latestHead,eyePatchQuality:state.latestQuality,video:state.latestVideo}:null,evaluation:state.evaluation,livePrediction:{iris:state.latestIrisPrediction,fused:state.latestFusedPrediction,webgazer:state.latestWebGazer}};
  }
  function download(){const blob=new Blob([JSON.stringify(evidence(),null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`webgazer-capture-fusion-${Date.now()}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),0);}

  function renderLive(){
    const q=document.getElementById('captureFusionLive'),e=document.getElementById('captureFusionEval'),coach=document.getElementById('captureCoach'),timing=document.getElementById('captureTiming');
    if(q){const x=state.latestEyes;if(!state.ready&&!x)q.textContent=state.startupState==='error'?`isolated detector error: ${state.startupError}`:`${state.startupState.replaceAll('-',' ')}`;else if(!x)q.textContent='waiting for refined iris capture';else q.textContent=`L ${(x.left.reliability*100).toFixed(0)}% · R ${(x.right.reliability*100).toFixed(0)}% · bilateral ${(x.fused.agreement*100).toFixed(0)}% · capture ${(x.fused.reliability*100).toFixed(0)}%${x.fused.reliability<.35?' · PAUSE QUALITY':''}`;}
    if(coach){const c=state.coach;coach.textContent=!c?'waiting for face geometry':c.stable?'READY · capture conditions stable':c.pass?`hold steady ${Math.min(1.5,c.stableMs/1000).toFixed(1)} / 1.5 s`:`adjust: ${c.reasons.join(', ')}`;}
    if(timing){const lat=state.frameClock.latencies,iv=state.frameClock.frameIntervals;timing.textContent=`${state.frameClock.mode||'waiting'} · inference p50 ${lat.length?C.quantile(lat,.5).toFixed(1):'—'} ms · p95 ${lat.length?C.quantile(lat,.95).toFixed(1):'—'} ms · frame ${iv.length?C.quantile(iv,.5).toFixed(1):'—'} ms · busy drops ${state.frameClock.dropped}`;}
    if(e){const v=state.evaluation;if(!v)e.textContent=`${pursuitSamples.length} pursuit · ${sparseSamples.length} click samples`;else e.textContent=`${v.source} ${v.samples} · selected ${v.selectedKernel} ${v.selected.rmse==null?'—':v.selected.rmse.toFixed(1)+' px CV'} · region ${v.leaveRegionOut.rmse==null?'—':v.leaveRegionOut.rmse.toFixed(1)+' px'} · WebGazer ${v.webgazer.rmse==null?'—':v.webgazer.rmse.toFixed(1)+' px'}`;}
  }

  function mount(){
    const section=document.createElement('section');section.id='captureFusionPanel';section.className='panel';section.innerHTML=`<div class="panelHead"><div><p class="eyebrow">EXPERIMENTAL CAPTURE FUSION</p><h2>Dense iris + head-pose gaze engine</h2></div><label>Validation cohort <select id="eyewearCohort"><option value="unknown">Unknown</option><option value="none">No eyewear</option><option value="glasses">Glasses</option><option value="contacts">Contacts</option></select></label></div><p class="hint">The refined 478-landmark detector runs inside an isolated same-origin frame so its legacy MediaPipe/WASM globals cannot collide with WebGazer. Capture is video-frame synchronized, per-eye quality weighted and local-only.</p><div class="cameraComparisonGrid"><article><span>Capture reliability</span><strong id="captureFusionLive">starting isolated capture</strong></article><article><span>Positioning coach</span><strong id="captureCoach">waiting</strong></article><article><span>Frame timing</span><strong id="captureTiming">waiting</strong></article><article><span>Predictor comparison</span><strong id="captureFusionEval">no dense calibration yet</strong></article></div><div class="controls"><button id="startPursuit" type="button" disabled>Start 18 s smooth-pursuit calibration</button><button id="bypassPursuit" type="button">Start pursuit anyway</button><button id="downloadCaptureFusion" type="button">Download fusion evidence</button></div><p class="hint">Linear, polynomial-2 and RBF kernel-ridge mappings are compared by held-out error. The lowest-error model is selected; WebGazer stays an untouched baseline.</p>`;
    const evidencePanel=document.querySelector('.evidencePanel');if(evidencePanel&&evidencePanel.parentNode)evidencePanel.parentNode.insertBefore(section,evidencePanel);else document.querySelector('main').appendChild(section);
    section.querySelector('#eyewearCohort').addEventListener('change',ev=>state.cohort=ev.target.value);section.querySelector('#startPursuit').addEventListener('click',()=>startPursuit(false));section.querySelector('#bypassPursuit').addEventListener('click',()=>startPursuit(true));section.querySelector('#downloadCaptureFusion').addEventListener('click',download);
    document.addEventListener('click',ev=>{const b=ev.target.closest&&ev.target.closest('.calibrationPoint');if(!b)return;const r=b.getBoundingClientRect();sampleAtClick({x:r.left+r.width/2,y:r.top+r.height/2});});
    const marker=document.createElement('div');marker.id='pursuitMarker';marker.hidden=true;marker.setAttribute('aria-hidden','true');marker.style.cssText='position:fixed;width:24px;height:24px;border:4px solid #111827;border-radius:50%;transform:translate(-50%,-50%);z-index:7000;pointer-events:none;background:rgba(255,255,255,.86);box-shadow:0 0 0 5px rgba(255,255,255,.45)';document.body.appendChild(marker);
  }

  addEventListener('message',onMessage);mount();renderLive();ensureIsolatedCapture();
  const session=document.getElementById('sessionState');if(session)new MutationObserver(()=>{if(sessionRunning()&&state.ready)startFrameClock();else if(!sessionRunning())stopFrameClock();}).observe(session,{childList:true,characterData:true,subtree:true});
  addEventListener('beforeunload',()=>{stopFrameClock();removeEventListener('message',onMessage);if(state.iframe)state.iframe.remove();});
  window.webgazerCaptureFusion={getLive:()=>({available:!!state.latestEyes,reliability:state.latestEyes&&state.latestEyes.fused?state.latestEyes.fused.reliability:0,leftReliability:state.latestEyes?state.latestEyes.left.reliability:0,rightReliability:state.latestEyes?state.latestEyes.right.reliability:0,prediction:state.latestFusedPrediction||state.latestIrisPrediction,startupState:state.startupState,selectedKernel:state.selectedKernel}),getFeatureSnapshot:featureSnapshot,getEvidence:evidence,getEvaluation:()=>state.evaluation,startPursuit:()=>startPursuit(false),stopFrameClock};
})();
