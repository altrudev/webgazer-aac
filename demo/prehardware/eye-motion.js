'use strict';

(function () {
  const SAMPLE_MS = 3000;
  const SEND_INTERVAL_MS = 100;
  const stages = ['CENTER','LEFT','RIGHT','UP','DOWN'];
  const measurements = {};
  let active = null, faceMesh = null, faceMeshReady = false, sending = false, timer = null, marker = null;

  const EYES = {
    left: { contour:[466,388,387,386,385,384,398,263,249,390,373,374,380,381,382,362], corners:[362,263] },
    right:{ contour:[246,161,160,159,158,157,173,33,7,163,144,145,153,154,155,133], corners:[33,133] },
  };
  const IRIS_GROUPS = [[468,469,470,471,472],[473,474,475,476,477]];
  const TARGETS = { CENTER:[0.50,0.50], LEFT:[0.18,0.50], RIGHT:[0.82,0.50], UP:[0.50,0.18], DOWN:[0.50,0.82] };

  const avg=v=>v.length?v.reduce((a,b)=>a+b,0)/v.length:0;
  const stdev=v=>{if(v.length<2)return 0;const m=avg(v);return Math.sqrt(v.reduce((a,x)=>a+(x-m)*(x-m),0)/v.length);};
  const dist=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
  const dot=(a,b)=>a[0]*b[0]+a[1]*b[1];

  function loadScript(src){return new Promise((resolve,reject)=>{const existing=[...document.scripts].find(s=>s.src&&s.src.includes(src));if(existing&&window.FaceMesh)return resolve();const s=document.createElement('script');s.src=src;s.onload=resolve;s.onerror=()=>reject(new Error(`Could not load ${src}`));document.head.appendChild(s);});}
  async function ensureFaceMesh(){if(faceMeshReady&&faceMesh)return faceMesh;await loadScript('./mediapipe/face_mesh/face_mesh.js');if(typeof window.FaceMesh!=='function')throw new Error('Local MediaPipe FaceMesh did not expose window.FaceMesh');faceMesh=new window.FaceMesh({locateFile:file=>`./mediapipe/face_mesh/${file}`});faceMesh.setOptions({maxNumFaces:1,refineLandmarks:true,minDetectionConfidence:0.5,minTrackingConfidence:0.5});faceMesh.onResults(onResults);faceMeshReady=true;return faceMesh;}
  function pointPx(lm,video){if(!lm)return null;const w=video.videoWidth||video.clientWidth||640,h=video.videoHeight||video.clientHeight||480;return [Number(lm.x)*w,Number(lm.y)*h];}
  function meanPoint(points){const valid=points.filter(Boolean);return valid.length?[avg(valid.map(p=>p[0])),avg(valid.map(p=>p[1]))]:null;}

  function eyeGeometry(pos, cfg, video){
    const contour=cfg.contour.map(i=>pointPx(pos[i],video)).filter(Boolean);if(contour.length<8)return null;
    let a=pointPx(pos[cfg.corners[0]],video),b=pointPx(pos[cfg.corners[1]],video);if(!a||!b)return null;
    if(a[0]>b[0]){const t=a;a=b;b=t;}
    const mid=[(a[0]+b[0])/2,(a[1]+b[1])/2],width=dist(a,b);if(width<4)return null;
    const ux=(b[0]-a[0])/width,uy=(b[1]-a[1])/width;
    const axis=[ux,uy],perp=[-uy,ux];
    const vertical=contour.map(p=>dot([p[0]-mid[0],p[1]-mid[1]],perp));
    return {mid,width,axis,perp,aperture:Math.max(...vertical)-Math.min(...vertical)};
  }

  function extract(pos,video){
    if(!Array.isArray(pos)||pos.length<478)return null;
    const geometry={left:eyeGeometry(pos,EYES.left,video),right:eyeGeometry(pos,EYES.right,video)};if(!geometry.left||!geometry.right)return null;
    const irisCenters=IRIS_GROUPS.map(g=>meanPoint(g.map(i=>pointPx(pos[i],video))));if(!irisCenters[0]||!irisCenters[1])return null;
    const used=new Set(),eyes={};
    for(const side of ['left','right']){
      const g=geometry[side];let idx=dist(irisCenters[0],g.mid)<=dist(irisCenters[1],g.mid)?0:1;if(used.has(idx))idx=idx===0?1:0;used.add(idx);
      const iris=irisCenters[idx],rel=[iris[0]-g.mid[0],iris[1]-g.mid[1]];
      eyes[side]={u:dot(rel,g.axis)/g.width,v:dot(rel,g.perp)/g.width,eyeWidth:g.width,eyeAperture:g.aperture,irisX:iris[0],irisY:iris[1]};
    }
    return {landmarkCount:pos.length,eyes};
  }

  function gazePoint(){const dotEl=document.getElementById('gazeDot');if(!dotEl||dotEl.hidden)return null;const x=parseFloat(dotEl.style.left),y=parseFloat(dotEl.style.top);return Number.isFinite(x)&&Number.isFinite(y)?[x,y]:null;}
  function onResults(results){if(!active)return;const video=document.getElementById('webgazerVideoFeed'),pos=results&&results.multiFaceLandmarks&&results.multiFaceLandmarks[0],s=video&&pos?extract(pos,video):null;if(!s){active.missing++;return;}active.samples++;active.landmarkCount=s.landmarkCount;for(const side of ['left','right']){active[side].u.push(s.eyes[side].u);active[side].v.push(s.eyes[side].v);active[side].w.push(s.eyes[side].eyeWidth);active[side].a.push(s.eyes[side].eyeAperture);}const g=gazePoint();if(g){active.gx.push(g[0]);active.gy.push(g[1]);}}
  async function sendFrame(){if(!active||sending||!faceMesh)return;const video=document.getElementById('webgazerVideoFeed');if(!video||video.readyState<2)return;sending=true;try{await faceMesh.send({image:video});}catch(_){if(active)active.missing++;}finally{sending=false;}}
  function summarize(run){const result={stage:run.stage,samples:run.samples,missing:run.missing,landmarkCount:run.landmarkCount,coordinateSystem:'eye-corner-local-width-normalized'};for(const side of ['left','right'])result[side]={irisU:avg(run[side].u),irisV:avg(run[side].v),irisJitter:Math.hypot(stdev(run[side].u),stdev(run[side].v)),eyeWidthPx:avg(run[side].w),eyeAperturePx:avg(run[side].a)};result.gazeJitterPx=run.gx.length?Math.hypot(stdev(run.gx),stdev(run.gy)):null;return result;}
  function updateSummary(){const out=document.getElementById('eyeMotionSummary');if(!out)return;const c=measurements.CENTER,l=measurements.LEFT,r=measurements.RIGHT,u=measurements.UP,d=measurements.DOWN;if(!c){out.textContent='Measure CENTER first.';return;}if(c.samples===0){out.textContent='No refined iris landmarks were captured.';return;}if(!(l&&r&&u&&d)){out.textContent=`CENTER captured ${c.samples} frames · 478-landmark iris detector active · center jitter L ${c.left.irisJitter.toFixed(3)} / R ${c.right.irisJitter.toFixed(3)}.`;return;}const horiz=((r.left.irisU-l.left.irisU)+(r.right.irisU-l.right.irisU))/2;const vert=((d.left.irisV-u.left.irisV)+(d.right.irisV-u.right.irisV))/2;const centerJitter=(c.left.irisJitter+c.right.irisJitter)/2;const hAgree=Math.sign(r.left.irisU-l.left.irisU)===Math.sign(r.right.irisU-l.right.irisU),vAgree=Math.sign(d.left.irisV-u.left.irisV)===Math.sign(d.right.irisV-u.right.irisV);const measurable=Math.abs(horiz)>=0.05&&Math.abs(vert)>=0.025&&centerJitter<=0.05&&hAgree&&vAgree;out.textContent=`${measurable?'MEASURABLE DIRECT IRIS MOVEMENT':'PARTIAL / UNSTABLE DIRECT IRIS MOVEMENT'} · H ${horiz.toFixed(3)} eye-width · V ${vert.toFixed(3)} eye-width · center jitter ${centerJitter.toFixed(3)} · bilateral H ${hAgree?'yes':'no'} / V ${vAgree?'yes':'no'} · WebGazer center jitter ${c.gazeJitterPx==null?'—':c.gazeJitterPx.toFixed(1)+' px'}`;}
  function moveMarker(stage){if(!marker)return;const p=TARGETS[stage]||TARGETS.CENTER;marker.style.left=`${p[0]*100}%`;marker.style.top=`${p[1]*100}%`;marker.hidden=false;}
  async function begin(stage){if(active)return;const out=document.getElementById('eyeMotionSummary');try{const video=document.getElementById('webgazerVideoFeed');if(!video){if(out)out.textContent='Start the verified camera session first.';return;}if(out)out.textContent='Loading local refined FaceMesh…';await ensureFaceMesh();}catch(error){if(out)out.textContent=`Direct iris detector unavailable: ${error.message||error}`;return;}active={stage,samples:0,missing:0,landmarkCount:0,left:{u:[],v:[],w:[],a:[]},right:{u:[],v:[],w:[],a:[]},gx:[],gy:[]};moveMarker(stage);if(out)out.textContent=`Measuring ${stage} for 3 seconds. Keep your head still and look only at the marker.`;document.querySelectorAll('[data-eye-stage]').forEach(b=>b.disabled=true);timer=setInterval(sendFrame,SEND_INTERVAL_MS);sendFrame();setTimeout(()=>{clearInterval(timer);timer=null;const run=active;active=null;if(marker)marker.hidden=true;measurements[stage]=summarize(run);document.querySelectorAll('[data-eye-stage]').forEach(b=>b.disabled=false);updateSummary();},SAMPLE_MS);}
  function mount(){if(document.getElementById('eyeMotionPanel'))return;const section=document.createElement('section');section.id='eyeMotionPanel';section.className='panel';section.innerHTML=`<div class="panelHead"><div><p class="eyebrow">DIRECT EYE MOVEMENT</p><h2>Eye-local iris diagnostic</h2></div></div><p class="hint">Independent of WebGazer gaze regression. Iris position is projected into an eye-corner coordinate frame, compensating for roll and avoiding unstable eye-height normalization. No eye images or raw landmarks are persisted.</p><div class="controls">${stages.map(s=>`<button type="button" data-eye-stage="${s}">${s}</button>`).join('')}<button id="downloadEyeMotion" type="button">Download eye-motion evidence</button></div><p id="eyeMotionSummary" class="hint">Measure CENTER first.</p>`;const evidence=document.querySelector('.evidencePanel');if(evidence&&evidence.parentNode)evidence.parentNode.insertBefore(section,evidence);else document.querySelector('main').appendChild(section);section.querySelectorAll('[data-eye-stage]').forEach(b=>b.addEventListener('click',()=>begin(b.dataset.eyeStage)));document.getElementById('downloadEyeMotion').addEventListener('click',()=>{const evidence={schema:'webgazer-aac/iris-motion-diagnostic/0.3',exportedAt:new Date().toISOString(),detector:'local-mediapipe-face-mesh-refine-landmarks',coordinateSystem:'eye-corner-local-width-normalized',rawEyeMaterialPersisted:false,rawLandmarksPersisted:false,measurements};const blob=new Blob([JSON.stringify(evidence,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`webgazer-iris-motion-${Date.now()}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),0);});marker=document.createElement('div');marker.id='eyeMotionMarker';marker.setAttribute('aria-hidden','true');marker.style.cssText='position:fixed;width:28px;height:28px;border:5px solid #111827;border-radius:50%;transform:translate(-50%,-50%);z-index:6000;pointer-events:none;background:rgba(255,255,255,.78)';marker.hidden=true;document.body.appendChild(marker);}
  mount();window.webgazerEyeMotion={getMeasurements:()=>JSON.parse(JSON.stringify(measurements))};
})();
