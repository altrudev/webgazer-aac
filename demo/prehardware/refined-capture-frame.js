'use strict';
(function(){
  const ORIGIN=location.origin;
  const frameCanvas=document.createElement('canvas');
  const canvases={left:document.createElement('canvas'),right:document.createElement('canvas')};
  canvases.left.width=96;canvases.left.height=48;canvases.right.width=96;canvases.right.height=48;
  const EYE={
    left:{outer:263,inner:362,upper:386,lower:374,iris:[468,469,470,471,472],contour:[263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]},
    right:{outer:33,inner:133,upper:159,lower:145,iris:[473,474,475,476,477],contour:[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]},
  };
  let faceMesh=null,pending=null,ready=false,busy=false;
  function avg(a){return a.length?a.reduce((x,y)=>x+y,0)/a.length:0;}
  function dimensions(source){return {w:source.videoWidth||source.width||640,h:source.videoHeight||source.height||480};}
  function point(lm,source){if(!lm)return null;const {w,h}=dimensions(source);return [lm.x*w,lm.y*h];}
  function meanPoint(a){return a.length?[avg(a.map(p=>p[0])),avg(a.map(p=>p[1]))]:null;}
  function norm(v){const m=Math.hypot(v[0],v[1])||1;return [v[0]/m,v[1]/m];}
  function dot(a,b){return a[0]*b[0]+a[1]*b[1];}
  function subtract(a,b){return [a[0]-b[0],a[1]-b[1]];}
  function eyeLocal(lms,source,cfg){
    const outer=point(lms[cfg.outer],source),inner=point(lms[cfg.inner],source),upper=point(lms[cfg.upper],source),lower=point(lms[cfg.lower],source),iris=meanPoint(cfg.iris.map(i=>point(lms[i],source)).filter(Boolean));
    if(!outer||!inner||!upper||!lower||!iris)return null;const center=[(outer[0]+inner[0])/2,(outer[1]+inner[1])/2],axis=subtract(inner,outer),width=Math.hypot(axis[0],axis[1]);if(width<4)return null;
    const ex=norm(axis),ey=[-ex[1],ex[0]],d=subtract(iris,center),aperture=Math.hypot(upper[0]-lower[0],upper[1]-lower[1]);return {u:dot(d,ex)/width,v:dot(d,ey)/width,width,aperture,apertureRatio:aperture/width,center};
  }
  function headFeatures(lms,source,left,right){
    const nose=point(lms[1],source),forehead=point(lms[10],source),chin=point(lms[152],source);if(!nose||!forehead||!chin)return null;const eyeMid=[(left.center[0]+right.center[0])/2,(left.center[1]+right.center[1])/2],inter=Math.hypot(left.center[0]-right.center[0],left.center[1]-right.center[1])||1,faceH=Math.hypot(forehead[0]-chin[0],forehead[1]-chin[1])||1,{w:vw,h:vh}=dimensions(source);
    return {yaw:(nose[0]-eyeMid[0])/inter,pitch:(nose[1]-eyeMid[1])/faceH,roll:Math.atan2(right.center[1]-left.center[1],right.center[0]-left.center[0]),interocularPx:inter,interocularNorm:inter/vw,faceOffsetX:(eyeMid[0]-vw/2)/vw,faceOffsetY:(eyeMid[1]-vh*.42)/vh};
  }
  function eyeCropQuality(lms,source,cfg,side){
    const pts=cfg.contour.map(i=>point(lms[i],source)).filter(Boolean);if(pts.length<8)return null;const {w:sourceW,h:sourceH}=dimensions(source),xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);let minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);const w=maxX-minX,h=maxY-minY,padX=w*.45,padY=Math.max(h*.8,w*.18);
    minX=Math.max(0,minX-padX);maxX=Math.min(sourceW,maxX+padX);minY=Math.max(0,minY-padY);maxY=Math.min(sourceH,maxY+padY);const sw=maxX-minX,sh=maxY-minY;if(sw<4||sh<3)return null;const canvas=canvases[side],ctx=canvas.getContext('2d',{willReadFrequently:true});try{ctx.drawImage(source,minX,minY,sw,sh,0,0,canvas.width,canvas.height);}catch(_){return null;}
    const data=ctx.getImageData(0,0,canvas.width,canvas.height).data,vals=[];let bright=0,clip=0,dark=0;for(let i=0;i<data.length;i+=16){const y=.299*data[i]+.587*data[i+1]+.114*data[i+2];vals.push(y);if(y>=235)bright++;if(y>=250)clip++;if(y<=20)dark++;}vals.sort((a,b)=>a-b);const n=vals.length,p10=vals[Math.floor(n*.1)]||0,p90=vals[Math.floor(n*.9)]||0;return {glare:n?bright/n:0,clipped:n?clip/n:0,dark:n?dark/n:0,contrast:(p90-p10)/255,luminance:avg(vals),sourceWidthPx:sw,sourceHeightPx:sh,analysisWidth:canvas.width,analysisHeight:canvas.height};
  }
  function post(message){parent.postMessage({source:'webgazer-aac-refined-frame',...message},ORIGIN);}
  function onResults(results){
    const ctx=pending;pending=null;busy=false;try{const lms=results&&results.multiFaceLandmarks&&results.multiFaceLandmarks[0];if(!ctx||!lms||lms.length<478){post({type:'result',frameId:ctx&&ctx.frameId||null,ok:false,reason:'missing-478-landmarks'});return;}const left=eyeLocal(lms,frameCanvas,EYE.left),right=eyeLocal(lms,frameCanvas,EYE.right);if(!left||!right){post({type:'result',frameId:ctx.frameId,ok:false,reason:'eye-geometry-unavailable'});return;}post({type:'result',frameId:ctx.frameId,ok:true,captureTime:ctx.captureTime,pursuitTarget:ctx.pursuitTarget||null,landmarkCount:lms.length,left,right,head:headFeatures(lms,frameCanvas,left,right),quality:{left:eyeCropQuality(lms,frameCanvas,EYE.left,'left'),right:eyeCropQuality(lms,frameCanvas,EYE.right,'right')},video:{width:frameCanvas.width,height:frameCanvas.height}});}catch(error){post({type:'result',frameId:ctx&&ctx.frameId||null,ok:false,reason:String(error&&error.message||error)});}
  }
  async function init(){try{if(typeof FaceMesh!=='function')throw new Error('FaceMesh global unavailable inside isolated frame');faceMesh=new FaceMesh({locateFile:file=>`./mediapipe/face_mesh/${file}`});faceMesh.setOptions({maxNumFaces:1,refineLandmarks:true,minDetectionConfidence:.5,minTrackingConfidence:.5});faceMesh.onResults(onResults);ready=true;post({type:'ready',isolatedGlobal:true,refineLandmarks:true,sameRealmInputCanvas:true});}catch(error){post({type:'error',reason:String(error&&error.message||error)});}}
  addEventListener('message',async event=>{
    if(event.origin!==ORIGIN||event.source!==parent)return;const msg=event.data||{};if(msg.type!=='process-frame')return;if(!ready||!faceMesh){post({type:'result',frameId:msg.frameId,ok:false,reason:'detector-not-ready'});return;}if(busy){post({type:'result',frameId:msg.frameId,ok:false,reason:'detector-busy'});return;}
    const video=parent.document.getElementById('webgazerVideoFeed');if(!video||video.readyState<2||!video.videoWidth||!video.videoHeight){post({type:'result',frameId:msg.frameId,ok:false,reason:'video-not-ready'});return;}if(frameCanvas.width!==video.videoWidth||frameCanvas.height!==video.videoHeight){frameCanvas.width=video.videoWidth;frameCanvas.height=video.videoHeight;}try{frameCanvas.getContext('2d',{alpha:false}).drawImage(video,0,0,frameCanvas.width,frameCanvas.height);}catch(error){post({type:'result',frameId:msg.frameId,ok:false,reason:`frame-copy-failed: ${error&&error.message||error}`});return;}
    busy=true;pending={frameId:msg.frameId,captureTime:msg.captureTime,pursuitTarget:msg.pursuitTarget||null};try{await faceMesh.send({image:frameCanvas});}catch(error){const ctx=pending;pending=null;busy=false;post({type:'result',frameId:ctx&&ctx.frameId||msg.frameId,ok:false,reason:String(error&&error.message||error)});}
  });
  init();
})();
