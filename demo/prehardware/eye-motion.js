'use strict';

(function () {
  const SAMPLE_MS = 3000;
  const SEND_INTERVAL_MS = 100;
  const stages = ['CENTER','LEFT','RIGHT','UP','DOWN'];
  const measurements = {};
  let active = null;
  let faceMesh = null;
  let faceMeshReady = false;
  let sending = false;
  let timer = null;
  let marker = null;

  const EYES = {
    left: [466,388,387,386,385,384,398,263,249,390,373,374,380,381,382,362],
    right: [246,161,160,159,158,157,173,33,7,163,144,145,153,154,155,133],
  };
  const IRIS_GROUPS = [[468,469,470,471,472],[473,474,475,476,477]];
  const TARGETS = {
    CENTER: [0.50,0.50],
    LEFT: [0.18,0.50],
    RIGHT: [0.82,0.50],
    UP: [0.50,0.18],
    DOWN: [0.50,0.82],
  };

  function avg(values){ return values.length ? values.reduce((a,b)=>a+b,0)/values.length : 0; }
  function stdev(values){ if(values.length<2)return 0; const m=avg(values); return Math.sqrt(values.reduce((a,v)=>a+(v-m)*(v-m),0)/values.length); }
  function dist(a,b){ return Math.hypot(a[0]-b[0],a[1]-b[1]); }

  function loadScript(src) {
    return new Promise((resolve,reject)=>{
      const existing=[...document.scripts].find(s=>s.src&&s.src.includes(src));
      if(existing && window.FaceMesh) return resolve();
      const s=document.createElement('script');
      s.src=src;
      s.onload=resolve;
      s.onerror=()=>reject(new Error(`Could not load ${src}`));
      document.head.appendChild(s);
    });
  }

  async function ensureFaceMesh() {
    if(faceMeshReady && faceMesh) return faceMesh;
    await loadScript('./mediapipe/face_mesh/face_mesh.js');
    if(typeof window.FaceMesh!=='function') throw new Error('Local MediaPipe FaceMesh did not expose window.FaceMesh');
    faceMesh=new window.FaceMesh({
      locateFile: file => `./mediapipe/face_mesh/${file}`,
    });
    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    faceMesh.onResults(onResults);
    faceMeshReady=true;
    return faceMesh;
  }

  function pointPx(lm, video) {
    if(!lm) return null;
    const w=video.videoWidth||video.clientWidth||640;
    const h=video.videoHeight||video.clientHeight||480;
    return [Number(lm.x)*w, Number(lm.y)*h];
  }

  function meanPoint(points) {
    const valid=points.filter(Boolean);
    if(!valid.length) return null;
    return [avg(valid.map(p=>p[0])),avg(valid.map(p=>p[1]))];
  }

  function eyeBox(pos, indices, video) {
    const pts=indices.map(i=>pointPx(pos[i],video)).filter(Boolean);
    if(pts.length<8) return null;
    const xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]);
    const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
    return {minX,maxX,minY,maxY,width:maxX-minX,height:maxY-minY,center:[(minX+maxX)/2,(minY+maxY)/2]};
  }

  function extract(pos,video) {
    if(!Array.isArray(pos) || pos.length<478) return null;
    const boxes={left:eyeBox(pos,EYES.left,video),right:eyeBox(pos,EYES.right,video)};
    if(!boxes.left||!boxes.right) return null;
    const irisCenters=IRIS_GROUPS.map(group=>meanPoint(group.map(i=>pointPx(pos[i],video))));
    if(!irisCenters[0]||!irisCenters[1]) return null;
    const used=new Set();
    const eyes={};
    for(const side of ['left','right']) {
      const b=boxes[side];
      let idx=dist(irisCenters[0],b.center)<=dist(irisCenters[1],b.center)?0:1;
      if(used.has(idx)) idx=idx===0?1:0;
      used.add(idx);
      const iris=irisCenters[idx];
      eyes[side]={
        nx:b.width>0?(iris[0]-b.minX)/b.width:0.5,
        ny:b.height>0?(iris[1]-b.minY)/b.height:0.5,
        eyeWidth:b.width,
        eyeHeight:b.height,
      };
    }
    return {landmarkCount:pos.length,eyes};
  }

  function gazePoint() {
    const dot=document.getElementById('gazeDot');
    if(!dot||dot.hidden)return null;
    const x=parseFloat(dot.style.left),y=parseFloat(dot.style.top);
    return Number.isFinite(x)&&Number.isFinite(y)?[x,y]:null;
  }

  function onResults(results) {
    if(!active) return;
    const video=document.getElementById('webgazerVideoFeed');
    const pos=results&&results.multiFaceLandmarks&&results.multiFaceLandmarks[0];
    const s=video&&pos?extract(pos,video):null;
    if(!s){active.missing++;return;}
    active.samples++;
    active.landmarkCount=s.landmarkCount;
    for(const side of ['left','right']) {
      active[side].x.push(s.eyes[side].nx);
      active[side].y.push(s.eyes[side].ny);
      active[side].w.push(s.eyes[side].eyeWidth);
      active[side].h.push(s.eyes[side].eyeHeight);
    }
    const g=gazePoint(); if(g){active.gx.push(g[0]);active.gy.push(g[1]);}
  }

  async function sendFrame() {
    if(!active||sending||!faceMesh)return;
    const video=document.getElementById('webgazerVideoFeed');
    if(!video||video.readyState<2)return;
    sending=true;
    try{await faceMesh.send({image:video});}
    catch(_){if(active)active.missing++;}
    finally{sending=false;}
  }

  function summarize(run) {
    const result={stage:run.stage,samples:run.samples,missing:run.missing,landmarkCount:run.landmarkCount};
    for(const side of ['left','right']) {
      result[side]={
        irisNx:avg(run[side].x),irisNy:avg(run[side].y),
        irisJitter:Math.hypot(stdev(run[side].x),stdev(run[side].y)),
        eyeWidthPx:avg(run[side].w),eyeHeightPx:avg(run[side].h),
      };
    }
    result.gazeJitterPx=run.gx.length?Math.hypot(stdev(run.gx),stdev(run.gy)):null;
    return result;
  }

  function updateSummary() {
    const out=document.getElementById('eyeMotionSummary');if(!out)return;
    const c=measurements.CENTER,l=measurements.LEFT,r=measurements.RIGHT,u=measurements.UP,d=measurements.DOWN;
    if(!c){out.textContent='Measure CENTER first.';return;}
    if(c.samples===0){out.textContent='No refined iris landmarks were captured. Check the local FaceMesh assets and camera feed.';return;}
    if(!(l&&r&&u&&d)){
      out.textContent=`CENTER captured ${c.samples} iris frames · landmark count ${c.landmarkCount} · center jitter L ${c.left.irisJitter.toFixed(3)} / R ${c.right.irisJitter.toFixed(3)}. Continue LEFT / RIGHT / UP / DOWN.`;
      return;
    }
    const horiz=((r.left.irisNx-l.left.irisNx)+(r.right.irisNx-l.right.irisNx))/2;
    const vert=((d.left.irisNy-u.left.irisNy)+(d.right.irisNy-u.right.irisNy))/2;
    const centerJitter=(c.left.irisJitter+c.right.irisJitter)/2;
    const leftRightAgreement=Math.sign(r.left.irisNx-l.left.irisNx)===Math.sign(r.right.irisNx-l.right.irisNx);
    const upDownAgreement=Math.sign(d.left.irisNy-u.left.irisNy)===Math.sign(d.right.irisNy-u.right.irisNy);
    const measurable=Math.abs(horiz)>=0.06&&Math.abs(vert)>=0.06&&centerJitter<=0.06&&leftRightAgreement&&upDownAgreement;
    out.textContent=`${measurable?'MEASURABLE DIRECT IRIS MOVEMENT':'WEAK / UNSTABLE DIRECT IRIS MOVEMENT'} · horizontal ${horiz.toFixed(3)} eye-width · vertical ${vert.toFixed(3)} eye-height · center jitter ${centerJitter.toFixed(3)} · bilateral agreement H ${leftRightAgreement?'yes':'no'} / V ${upDownAgreement?'yes':'no'} · gaze jitter ${c.gazeJitterPx==null?'—':c.gazeJitterPx.toFixed(1)+' px'}`;
  }

  function moveMarker(stage) {
    if(!marker)return;
    const p=TARGETS[stage]||TARGETS.CENTER;
    marker.style.left=`${p[0]*100}%`;
    marker.style.top=`${p[1]*100}%`;
    marker.hidden=false;
  }

  async function begin(stage) {
    if(active)return;
    const out=document.getElementById('eyeMotionSummary');
    try{
      const video=document.getElementById('webgazerVideoFeed');
      if(!video){if(out)out.textContent='Start the verified camera session first.';return;}
      if(out)out.textContent='Loading local refined FaceMesh…';
      await ensureFaceMesh();
    }catch(error){if(out)out.textContent=`Direct iris detector unavailable: ${error.message||error}`;return;}
    active={stage,samples:0,missing:0,landmarkCount:0,left:{x:[],y:[],w:[],h:[]},right:{x:[],y:[],w:[],h:[]},gx:[],gy:[]};
    moveMarker(stage);
    if(out)out.textContent=`Measuring ${stage} for 3 seconds. Keep your head still and look only at the marker.`;
    document.querySelectorAll('[data-eye-stage]').forEach(b=>b.disabled=true);
    timer=setInterval(sendFrame,SEND_INTERVAL_MS);
    sendFrame();
    setTimeout(()=>{
      clearInterval(timer);timer=null;
      const run=active;
      active=null;
      if(marker)marker.hidden=true;
      measurements[stage]=summarize(run);
      document.querySelectorAll('[data-eye-stage]').forEach(b=>b.disabled=false);
      updateSummary();
    },SAMPLE_MS);
  }

  function mount() {
    if(document.getElementById('eyeMotionPanel'))return;
    const section=document.createElement('section');section.id='eyeMotionPanel';section.className='panel';
    section.innerHTML=`<div class="panelHead"><div><p class="eyebrow">DIRECT EYE MOVEMENT</p><h2>Refined iris landmark diagnostic</h2></div></div>
      <p class="hint">Independent of WebGazer gaze regression. Runs a separate local MediaPipe FaceMesh pass with refined eye/iris landmarks. No eye images or raw landmarks are persisted.</p>
      <div class="controls">${stages.map(s=>`<button type="button" data-eye-stage="${s}">${s}</button>`).join('')}<button id="downloadEyeMotion" type="button">Download eye-motion evidence</button></div>
      <p id="eyeMotionSummary" class="hint">Measure CENTER first.</p>`;
    const evidence=document.querySelector('.evidencePanel');
    if(evidence&&evidence.parentNode)evidence.parentNode.insertBefore(section,evidence);else document.querySelector('main').appendChild(section);
    section.querySelectorAll('[data-eye-stage]').forEach(b=>b.addEventListener('click',()=>begin(b.dataset.eyeStage)));
    document.getElementById('downloadEyeMotion').addEventListener('click',()=>{
      const evidence={
        schema:'webgazer-aac/iris-motion-diagnostic/0.2',
        exportedAt:new Date().toISOString(),
        detector:'local-mediapipe-face-mesh-refine-landmarks',
        rawEyeMaterialPersisted:false,
        rawLandmarksPersisted:false,
        measurements,
      };
      const blob=new Blob([JSON.stringify(evidence,null,2)],{type:'application/json'});
      const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`webgazer-iris-motion-${Date.now()}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),0);
    });

    marker=document.createElement('div');
    marker.id='eyeMotionMarker';
    marker.setAttribute('aria-hidden','true');
    marker.style.cssText='position:fixed;width:28px;height:28px;border:5px solid #111827;border-radius:50%;transform:translate(-50%,-50%);z-index:6000;pointer-events:none;background:rgba(255,255,255,.78)';
    marker.hidden=true;
    document.body.appendChild(marker);
  }

  mount();
  window.webgazerEyeMotion={getMeasurements:()=>JSON.parse(JSON.stringify(measurements))};
})();
