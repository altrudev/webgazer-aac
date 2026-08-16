'use strict';

(function () {
  const SAMPLE_MS = 3000;
  const stages = ['CENTER','LEFT','RIGHT','UP','DOWN'];
  const measurements = {};
  let active = null;
  let poll = null;

  const EYES = {
    left: [466,388,387,386,385,384,398,263,249,390,373,374,380,381,382,362],
    right: [246,161,160,159,158,157,173,33,7,163,144,145,153,154,155,133],
  };
  const IRIS_GROUPS = [[468,469,470,471,472],[473,474,475,476,477]];

  function meanPoint(points) {
    const valid = points.filter(Boolean);
    if (!valid.length) return null;
    return [valid.reduce((a,p)=>a+p[0],0)/valid.length, valid.reduce((a,p)=>a+p[1],0)/valid.length];
  }
  function dist(a,b){ return Math.hypot(a[0]-b[0], a[1]-b[1]); }
  function stdev(values){ if(values.length<2)return 0; const m=values.reduce((a,b)=>a+b,0)/values.length; return Math.sqrt(values.reduce((a,v)=>a+(v-m)*(v-m),0)/values.length); }
  function avg(values){ return values.length ? values.reduce((a,b)=>a+b,0)/values.length : 0; }

  function eyeBox(pos, indices) {
    const pts = indices.map(i => pos[i]).filter(Boolean);
    if (pts.length < 8) return null;
    const xs = pts.map(p=>p[0]), ys=pts.map(p=>p[1]);
    const minX=Math.min(...xs), maxX=Math.max(...xs), minY=Math.min(...ys), maxY=Math.max(...ys);
    return { minX,maxX,minY,maxY,width:maxX-minX,height:maxY-minY,center:[(minX+maxX)/2,(minY+maxY)/2] };
  }

  function sampleLandmarks() {
    try {
      if (!window.webgazer || typeof window.webgazer.getTracker !== 'function') return null;
      const tracker = window.webgazer.getTracker();
      const pos = tracker && typeof tracker.getPositions === 'function' ? tracker.getPositions() : null;
      if (!Array.isArray(pos) || pos.length < 478) return { available:false, landmarkCount:Array.isArray(pos)?pos.length:0 };

      const boxes = { left: eyeBox(pos,EYES.left), right: eyeBox(pos,EYES.right) };
      if (!boxes.left || !boxes.right) return { available:false, landmarkCount:pos.length };
      const irisCenters = IRIS_GROUPS.map(g => meanPoint(g.map(i=>pos[i])));
      if (!irisCenters[0] || !irisCenters[1]) return { available:false, landmarkCount:pos.length };

      const assigned = {};
      for (const side of ['left','right']) {
        const b = boxes[side];
        const idx = dist(irisCenters[0], b.center) <= dist(irisCenters[1], b.center) ? 0 : 1;
        const iris = irisCenters[idx];
        assigned[side] = {
          nx: b.width > 0 ? (iris[0]-b.minX)/b.width : 0.5,
          ny: b.height > 0 ? (iris[1]-b.minY)/b.height : 0.5,
          irisX: iris[0], irisY: iris[1], eyeWidth:b.width, eyeHeight:b.height,
        };
      }
      return { available:true, landmarkCount:pos.length, eyes:assigned };
    } catch (_) { return null; }
  }

  function gazePoint() {
    const dot = document.getElementById('gazeDot');
    if (!dot || dot.hidden) return null;
    const x = parseFloat(dot.style.left), y=parseFloat(dot.style.top);
    return Number.isFinite(x)&&Number.isFinite(y) ? [x,y] : null;
  }

  function collect() {
    if (!active) return;
    const s = sampleLandmarks();
    if (!s || !s.available) { active.missing++; return; }
    active.samples++;
    active.landmarkCount = s.landmarkCount;
    for (const side of ['left','right']) {
      active[side].x.push(s.eyes[side].nx);
      active[side].y.push(s.eyes[side].ny);
      active[side].w.push(s.eyes[side].eyeWidth);
      active[side].h.push(s.eyes[side].eyeHeight);
    }
    const g=gazePoint(); if(g){ active.gx.push(g[0]); active.gy.push(g[1]); }
  }

  function summarize(run) {
    const result = { stage:run.stage, samples:run.samples, missing:run.missing, landmarkCount:run.landmarkCount };
    for (const side of ['left','right']) {
      result[side] = {
        irisNx: avg(run[side].x), irisNy: avg(run[side].y),
        irisJitter: Math.hypot(stdev(run[side].x),stdev(run[side].y)),
        eyeWidthPx: avg(run[side].w), eyeHeightPx: avg(run[side].h),
      };
    }
    result.gazeJitterPx = run.gx.length ? Math.hypot(stdev(run.gx),stdev(run.gy)) : null;
    return result;
  }

  function updateSummary() {
    const out=document.getElementById('eyeMotionSummary'); if(!out)return;
    const c=measurements.CENTER, l=measurements.LEFT, r=measurements.RIGHT, u=measurements.UP, d=measurements.DOWN;
    if(!c){ out.textContent='Measure CENTER first.'; return; }
    if(!(l&&r&&u&&d)){ out.textContent=`CENTER measured. Iris jitter L ${c.left.irisJitter.toFixed(3)} · R ${c.right.irisJitter.toFixed(3)}. Continue LEFT / RIGHT / UP / DOWN.`; return; }
    const horiz = ((r.left.irisNx-l.left.irisNx)+(r.right.irisNx-l.right.irisNx))/2;
    const vert = ((d.left.irisNy-u.left.irisNy)+(d.right.irisNy-u.right.irisNy))/2;
    const centerJitter=(c.left.irisJitter+c.right.irisJitter)/2;
    const measurable = Math.abs(horiz)>=0.08 && Math.abs(vert)>=0.08 && centerJitter<=0.08;
    out.textContent = `${measurable?'MEASURABLE EYE MOVEMENT':'WEAK / UNSTABLE EYE MOVEMENT'} · horizontal span ${horiz.toFixed(3)} eye-width · vertical span ${vert.toFixed(3)} eye-height · center iris jitter ${centerJitter.toFixed(3)} · gaze jitter ${c.gazeJitterPx==null?'—':c.gazeJitterPx.toFixed(1)+' px'}`;
  }

  function begin(stage) {
    if(active)return;
    active={stage,samples:0,missing:0,landmarkCount:0,left:{x:[],y:[],w:[],h:[]},right:{x:[],y:[],w:[],h:[]},gx:[],gy:[]};
    const out=document.getElementById('eyeMotionSummary'); if(out)out.textContent=`Measuring ${stage} for 3 seconds. Move only your eyes toward ${stage.toLowerCase()}; keep your head still.`;
    document.querySelectorAll('[data-eye-stage]').forEach(b=>b.disabled=true);
    poll=setInterval(collect,50);
    setTimeout(()=>{
      clearInterval(poll); poll=null;
      measurements[stage]=summarize(active); active=null;
      document.querySelectorAll('[data-eye-stage]').forEach(b=>b.disabled=false);
      updateSummary();
    },SAMPLE_MS);
  }

  function mount() {
    if(document.getElementById('eyeMotionPanel'))return;
    const section=document.createElement('section'); section.id='eyeMotionPanel'; section.className='panel';
    section.innerHTML=`<div class="panelHead"><div><p class="eyebrow">DIRECT EYE MOVEMENT</p><h2>Iris landmark diagnostic</h2></div></div>
      <p class="hint">Independent of gaze regression. Uses MediaPipe iris landmarks normalized inside each eye. No eye images are saved.</p>
      <div class="controls">${stages.map(s=>`<button type="button" data-eye-stage="${s}">${s}</button>`).join('')}<button id="downloadEyeMotion" type="button">Download eye-motion evidence</button></div>
      <p id="eyeMotionSummary" class="hint">Measure CENTER first.</p>`;
    const evidence=document.querySelector('.evidencePanel');
    if(evidence&&evidence.parentNode)evidence.parentNode.insertBefore(section,evidence); else document.querySelector('main').appendChild(section);
    section.querySelectorAll('[data-eye-stage]').forEach(b=>b.addEventListener('click',()=>begin(b.dataset.eyeStage)));
    document.getElementById('downloadEyeMotion').addEventListener('click',()=>{
      const evidence={schema:'webgazer-aac/iris-motion-diagnostic/0.1',exportedAt:new Date().toISOString(),rawEyeMaterialPersisted:false,measurements};
      const blob=new Blob([JSON.stringify(evidence,null,2)],{type:'application/json'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download=`webgazer-iris-motion-${Date.now()}.json`; a.click(); setTimeout(()=>URL.revokeObjectURL(url),0);
    });
  }

  mount();
  window.webgazerEyeMotion={getMeasurements:()=>JSON.parse(JSON.stringify(measurements))};
})();
