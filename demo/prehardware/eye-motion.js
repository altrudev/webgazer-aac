'use strict';
(function(){
  const SAMPLE_MS=3000,POLL_MS=40,stages=['CENTER','LEFT','RIGHT','UP','DOWN'],measurements={};
  const TARGETS={CENTER:[.50,.50],LEFT:[.18,.50],RIGHT:[.82,.50],UP:[.50,.18],DOWN:[.50,.82]};
  let active=null,timer=null,marker=null,lastFrame=-1;
  const avg=v=>v.length?v.reduce((a,b)=>a+b,0)/v.length:0;
  const stdev=v=>{if(v.length<2)return 0;const m=avg(v);return Math.sqrt(v.reduce((a,x)=>a+(x-m)*(x-m),0)/v.length);};
  function gazePoint(){const d=document.getElementById('gazeDot');if(!d||d.hidden)return null;const x=parseFloat(d.style.left),y=parseFloat(d.style.top);return Number.isFinite(x)&&Number.isFinite(y)?[x,y]:null;}
  function fusion(){return window.webgazerCaptureFusion||null;}
  function collect(){
    if(!active)return;const f=fusion();if(!f||typeof f.getFeatureSnapshot!=='function'){active.missing++;return;}
    const ev=typeof f.getEvidence==='function'?f.getEvidence():null,frame=ev&&Number(ev.frames);if(Number.isFinite(frame)&&frame===lastFrame)return;if(Number.isFinite(frame))lastFrame=frame;
    const s=f.getFeatureSnapshot();if(!s||!s.left||!s.right){active.missing++;return;}active.samples++;active.landmarkCount=478;
    for(const side of ['left','right']){const e=s[side];active[side].u.push(e.u);active[side].v.push(e.v);active[side].w.push(e.width||0);active[side].a.push((e.apertureRatio||0)*(e.width||0));active[side].r.push(e.reliability||0);}
    const g=gazePoint();if(g){active.gx.push(g[0]);active.gy.push(g[1]);}
  }
  function summarize(run){
    const out={stage:run.stage,samples:run.samples,missing:run.missing,landmarkCount:run.landmarkCount,coordinateSystem:'eye-corner-local-width-normalized',source:'shared-isolated-refined-capture'};
    for(const side of ['left','right'])out[side]={irisU:avg(run[side].u),irisV:avg(run[side].v),irisJitter:Math.hypot(stdev(run[side].u),stdev(run[side].v)),eyeWidthPx:avg(run[side].w),eyeAperturePx:avg(run[side].a),reliability:avg(run[side].r)};
    out.gazeJitterPx=run.gx.length?Math.hypot(stdev(run.gx),stdev(run.gy)):null;return out;
  }
  function updateSummary(){
    const out=document.getElementById('eyeMotionSummary'),c=measurements.CENTER,l=measurements.LEFT,r=measurements.RIGHT,u=measurements.UP,d=measurements.DOWN;if(!out)return;
    if(!c){out.textContent='Measure CENTER first.';return;}if(c.samples===0){out.textContent='No shared refined iris samples were captured. Wait for Experimental Capture Fusion to show live reliability.';return;}
    if(!(l&&r&&u&&d)){out.textContent=`CENTER captured ${c.samples} isolated refined frames · center jitter L ${c.left.irisJitter.toFixed(3)} / R ${c.right.irisJitter.toFixed(3)}.`;return;}
    const horiz=((r.left.irisU-l.left.irisU)+(r.right.irisU-l.right.irisU))/2,vert=((d.left.irisV-u.left.irisV)+(d.right.irisV-u.right.irisV))/2,centerJitter=(c.left.irisJitter+c.right.irisJitter)/2,hAgree=Math.sign(r.left.irisU-l.left.irisU)===Math.sign(r.right.irisU-l.right.irisU),vAgree=Math.sign(d.left.irisV-u.left.irisV)===Math.sign(d.right.irisV-u.right.irisV),measurable=Math.abs(horiz)>=.05&&Math.abs(vert)>=.025&&centerJitter<=.05&&hAgree&&vAgree;
    out.textContent=`${measurable?'MEASURABLE DIRECT IRIS MOVEMENT':'PARTIAL / UNSTABLE DIRECT IRIS MOVEMENT'} · H ${horiz.toFixed(3)} eye-width · V ${vert.toFixed(3)} eye-width · center jitter ${centerJitter.toFixed(3)} · bilateral H ${hAgree?'yes':'no'} / V ${vAgree?'yes':'no'} · WebGazer center jitter ${c.gazeJitterPx==null?'—':c.gazeJitterPx.toFixed(1)+' px'}`;
  }
  function moveMarker(stage){if(!marker)return;const p=TARGETS[stage]||TARGETS.CENTER;marker.style.left=`${p[0]*100}%`;marker.style.top=`${p[1]*100}%`;marker.hidden=false;}
  function begin(stage){
    if(active)return;const out=document.getElementById('eyeMotionSummary'),f=fusion(),live=f&&typeof f.getLive==='function'?f.getLive():null;if(!live||!live.available){if(out)out.textContent='Wait until Experimental Capture Fusion shows live per-eye reliability; this diagnostic now reuses that isolated detector to avoid a second MediaPipe/WASM runtime.';return;}
    active={stage,samples:0,missing:0,landmarkCount:478,left:{u:[],v:[],w:[],a:[],r:[]},right:{u:[],v:[],w:[],a:[],r:[]},gx:[],gy:[]};lastFrame=-1;moveMarker(stage);if(out)out.textContent=`Measuring ${stage} for 3 seconds from the shared isolated iris detector. Keep your head still and look only at the marker.`;document.querySelectorAll('[data-eye-stage]').forEach(b=>b.disabled=true);timer=setInterval(collect,POLL_MS);collect();setTimeout(()=>{clearInterval(timer);timer=null;const run=active;active=null;if(marker)marker.hidden=true;measurements[stage]=summarize(run);document.querySelectorAll('[data-eye-stage]').forEach(b=>b.disabled=false);updateSummary();},SAMPLE_MS);
  }
  function mount(){
    if(document.getElementById('eyeMotionPanel'))return;const section=document.createElement('section');section.id='eyeMotionPanel';section.className='panel';section.innerHTML=`<div class="panelHead"><div><p class="eyebrow">DIRECT EYE MOVEMENT</p><h2>Shared isolated iris diagnostic</h2></div></div><p class="hint">Reuses the Experimental Capture Fusion detector instead of loading another FaceMesh/WASM runtime. Iris position is eye-local and roll compensated. No eye images or raw landmarks are persisted.</p><div class="controls">${stages.map(s=>`<button type="button" data-eye-stage="${s}">${s}</button>`).join('')}<button id="downloadEyeMotion" type="button">Download eye-motion evidence</button></div><p id="eyeMotionSummary" class="hint">Measure CENTER after the capture-fusion panel is live.</p>`;
    const evidence=document.querySelector('.evidencePanel');if(evidence&&evidence.parentNode)evidence.parentNode.insertBefore(section,evidence);else document.querySelector('main').appendChild(section);section.querySelectorAll('[data-eye-stage]').forEach(b=>b.addEventListener('click',()=>begin(b.dataset.eyeStage)));
    document.getElementById('downloadEyeMotion').addEventListener('click',()=>{const evidence={schema:'webgazer-aac/iris-motion-diagnostic/0.4',exportedAt:new Date().toISOString(),detector:'shared-isolated-refined-capture',coordinateSystem:'eye-corner-local-width-normalized',rawEyeMaterialPersisted:false,rawLandmarksPersisted:false,measurements};const blob=new Blob([JSON.stringify(evidence,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`webgazer-iris-motion-${Date.now()}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),0);});
    marker=document.createElement('div');marker.id='eyeMotionMarker';marker.setAttribute('aria-hidden','true');marker.style.cssText='position:fixed;width:28px;height:28px;border:5px solid #111827;border-radius:50%;transform:translate(-50%,-50%);z-index:6000;pointer-events:none;background:rgba(255,255,255,.78)';marker.hidden=true;document.body.appendChild(marker);
  }
  mount();window.webgazerEyeMotion={getMeasurements:()=>JSON.parse(JSON.stringify(measurements))};
})();
