(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  root.webgazerCaptureFusionCore=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){'use strict';
  function clamp(v,lo=0,hi=1){return Math.max(lo,Math.min(hi,Number.isFinite(v)?v:lo));}
  function avg(a){return a.length?a.reduce((x,y)=>x+y,0)/a.length:0;}
  function median(a){if(!a.length)return 0;const b=a.slice().sort((x,y)=>x-y);const m=Math.floor(b.length/2);return b.length%2?b[m]:(b[m-1]+b[m])/2;}
  function quantile(a,q){if(!a.length)return null;const b=a.slice().sort((x,y)=>x-y),p=clamp(q,0,1)*(b.length-1),lo=Math.floor(p),hi=Math.ceil(p);return lo===hi?b[lo]:b[lo]+(b[hi]-b[lo])*(p-lo);}
  function stdev(a){if(a.length<2)return 0;const m=avg(a);return Math.sqrt(avg(a.map(v=>(v-m)*(v-m))));}

  function eyeReliability(q={}){
    const visibility=clamp(q.visibility==null?1:q.visibility);
    const glare=clamp(q.glare||0);
    const clipped=clamp(q.clipped||0);
    const apertureRatio=clamp((q.apertureRatio==null?0.20:q.apertureRatio)/0.22);
    const stability=1-clamp((q.jitter||0)/0.08);
    const landmark=clamp(q.landmarkConfidence==null?1:q.landmarkConfidence);
    const occlusion=clamp(q.occlusion||0);
    const reflectionPenalty=clamp(Math.max(glare/0.12,clipped/0.05));
    const blinkPenalty=q.blink?1:0;
    const score=visibility*0.20+apertureRatio*0.20+stability*0.22+landmark*0.18+(1-occlusion)*0.10+(1-reflectionPenalty)*0.10;
    return clamp(score*(1-blinkPenalty));
  }

  function bilateralAgreement(left,right){
    if(!left||!right)return 0;
    const du=Math.abs((left.u||0)-(right.u||0));
    const dv=Math.abs((left.v||0)-(right.v||0));
    return clamp(1-(du/0.18+dv/0.14)/2);
  }

  function fuseEyes(left,right){
    if(!left&&!right)return null;
    if(!left)return {...right,source:'right-only',agreement:0.5};
    if(!right)return {...left,source:'left-only',agreement:0.5};
    const wl=Math.max(0,left.reliability||0),wr=Math.max(0,right.reliability||0),sum=wl+wr;
    if(sum<1e-6)return {u:0,v:0,reliability:0,agreement:0,source:'none'};
    const agreement=bilateralAgreement(left,right);
    return {
      u:(left.u*wl+right.u*wr)/sum,
      v:(left.v*wl+right.v*wr)/sum,
      reliability:clamp(((wl+wr)/2)*(0.55+0.45*agreement)),
      agreement,
      source:'bilateral-weighted'
    };
  }

  function ema(prev,next,alpha){if(prev==null)return next;return prev+(next-prev)*clamp(alpha,0,1);}

  function smoothingFactor(cutoff,dt){const r=2*Math.PI*Math.max(1e-6,cutoff)*Math.max(1e-6,dt);return r/(r+1);}
  function makeOneEuro({minCutoff=1,beta=0.007,dCutoff=1}={}){
    let xPrev=null,dxPrev=0,tPrev=null;
    return {
      update(x,tMs){
        const t=Number.isFinite(tMs)?tMs:Date.now();
        if(xPrev==null){xPrev=x;tPrev=t;return x;}
        const dt=Math.max(1,(t-tPrev))/1000;
        const dx=(x-xPrev)/dt;
        const ad=smoothingFactor(dCutoff,dt);
        const dxHat=ema(dxPrev,dx,ad);
        const cutoff=minCutoff+beta*Math.abs(dxHat);
        const a=smoothingFactor(cutoff,dt);
        const xHat=ema(xPrev,x,a);
        xPrev=xHat;dxPrev=dxHat;tPrev=t;return xHat;
      },
      reset(){xPrev=null;dxPrev=0;tPrev=null;}
    };
  }

  function solve(A,b){
    const n=b.length,M=A.map((r,i)=>r.slice().concat([b[i]]));
    for(let c=0;c<n;c++){
      let p=c;for(let r=c+1;r<n;r++)if(Math.abs(M[r][c])>Math.abs(M[p][c]))p=r;
      [M[c],M[p]]=[M[p],M[c]];
      let d=M[c][c];
      if(Math.abs(d)<1e-12){M[c][c]+=1e-8;d=M[c][c];}
      if(Math.abs(d)<1e-12)continue;
      for(let k=c;k<=n;k++)M[c][k]/=d;
      for(let r=0;r<n;r++){
        if(r===c)continue;
        const f=M[r][c];if(Math.abs(f)<1e-16)continue;
        for(let k=c;k<=n;k++)M[r][k]-=f*M[c][k];
      }
    }
    return M.map(r=>Number.isFinite(r[n])?r[n]:0);
  }

  function standardizer(samples){
    if(!samples||!samples.length)return null;
    const n=samples[0].features.length,mean=Array(n).fill(0),sd=Array(n).fill(1);
    for(let j=0;j<n;j++)mean[j]=avg(samples.map(s=>Number(s.features[j])||0));
    for(let j=0;j<n;j++){const v=stdev(samples.map(s=>Number(s.features[j])||0));sd[j]=v>1e-6?v:1;}
    return {mean,sd};
  }
  function normalizeFeatures(features,norm){return features.map((v,i)=>((Number(v)||0)-norm.mean[i])/norm.sd[i]);}

  function expandPoly2(f){const out=f.slice();for(let i=0;i<f.length;i++)for(let j=i;j<f.length;j++)out.push(f[i]*f[j]);return out;}

  function fitLinear(samples,lambda=1e-3,options={}){
    if(!samples||samples.length<5)return null;
    const norm=standardizer(samples),kind=options.kernel==='poly2'?'poly2':'linear';
    const expand=f=>kind==='poly2'?expandPoly2(normalizeFeatures(f,norm)):normalizeFeatures(f,norm);
    const first=expand(samples[0].features),m=first.length+1;
    const XtX=Array.from({length:m},()=>Array(m).fill(0)),XtyX=Array(m).fill(0),XtyY=Array(m).fill(0);
    for(const s of samples){
      const row=[1].concat(expand(s.features)),w=Math.max(1e-4,s.weight==null?1:s.weight);
      for(let i=0;i<m;i++){
        XtyX[i]+=w*row[i]*s.x;XtyY[i]+=w*row[i]*s.y;
        for(let j=0;j<m;j++)XtX[i][j]+=w*row[i]*row[j];
      }
    }
    for(let i=1;i<m;i++)XtX[i][i]+=lambda;
    return {type:'ridge',kernel:kind,bx:solve(XtX,XtyX),by:solve(XtX,XtyY),featureCount:samples[0].features.length,norm,lambda};
  }

  function deterministicThin(samples,maxSamples=160){
    if(samples.length<=maxSamples)return samples.slice();
    const out=[];
    for(let i=0;i<maxSamples;i++)out.push(samples[Math.round(i*(samples.length-1)/(maxSamples-1))]);
    return out;
  }
  function squaredDistance(a,b){let s=0;for(let i=0;i<a.length;i++){const d=a[i]-b[i];s+=d*d;}return s;}
  function estimateGamma(z){
    if(z.length<2)return 1;
    const ds=[];const step=Math.max(1,Math.floor(z.length/24));
    for(let i=0;i<z.length;i+=step)for(let j=i+step;j<z.length;j+=step)ds.push(squaredDistance(z[i],z[j]));
    const med=median(ds.filter(v=>v>1e-9));
    return med>1e-9?1/(2*med):1/z[0].length;
  }
  function fitKernelRidge(samples,{lambda=0.02,gamma=null,maxSamples=160}={}){
    if(!samples||samples.length<8)return null;
    const training=deterministicThin(samples,maxSamples),norm=standardizer(training),z=training.map(s=>normalizeFeatures(s.features,norm));
    const g=Number.isFinite(gamma)&&gamma>0?gamma:estimateGamma(z),n=training.length;
    const K=Array.from({length:n},()=>Array(n).fill(0));
    for(let i=0;i<n;i++)for(let j=i;j<n;j++){
      const k=Math.exp(-g*squaredDistance(z[i],z[j]));K[i][j]=k;K[j][i]=k;
    }
    for(let i=0;i<n;i++){
      const w=Math.max(0.08,training[i].weight==null?1:training[i].weight);
      K[i][i]+=lambda/w;
    }
    return {
      type:'krr',kernel:'rbf',featureCount:training[0].features.length,norm,gamma:g,lambda,
      centers:z,alphaX:solve(K,training.map(s=>s.x)),alphaY:solve(K,training.map(s=>s.y)),trainingCount:n,sourceCount:samples.length
    };
  }

  function fitModel(samples,{kernel='rbf',lambda,gamma,maxSamples}={}){
    if(kernel==='rbf')return fitKernelRidge(samples,{lambda:lambda==null?0.02:lambda,gamma,maxSamples:maxSamples||160});
    return fitLinear(samples,lambda==null?1e-3:lambda,{kernel:kernel==='poly2'?'poly2':'linear'});
  }

  function predict(model,features){
    if(!model||!features||features.length!==model.featureCount)return null;
    if(model.type==='krr'){
      const z=normalizeFeatures(features,model.norm);let x=0,y=0;
      for(let i=0;i<model.centers.length;i++){
        const k=Math.exp(-model.gamma*squaredDistance(z,model.centers[i]));x+=k*model.alphaX[i];y+=k*model.alphaY[i];
      }
      return Number.isFinite(x)&&Number.isFinite(y)?{x,y}:null;
    }
    let f=normalizeFeatures(features,model.norm);if(model.kernel==='poly2')f=expandPoly2(f);
    const row=[1].concat(f),x=row.reduce((s,v,i)=>s+v*model.bx[i],0),y=row.reduce((s,v,i)=>s+v*model.by[i],0);
    return Number.isFinite(x)&&Number.isFinite(y)?{x,y}:null;
  }

  function rmse(pairs){if(!pairs.length)return null;return Math.sqrt(avg(pairs.map(p=>(p.px-p.tx)**2+(p.py-p.ty)**2)));}
  function evaluatePairs(pairs){
    if(!pairs||!pairs.length)return {count:0,rmse:null,medianError:null,p95Error:null,meanError:null};
    const errors=pairs.map(p=>Math.hypot(p.px-p.tx,p.py-p.ty));
    return {count:pairs.length,rmse:rmse(pairs),meanError:avg(errors),medianError:median(errors),p95Error:quantile(errors,.95)};
  }

  function leaveOneOut(samples){
    if(!samples||samples.length<6)return {rmse:null,pairs:[]};const pairs=[];
    for(let i=0;i<samples.length;i++){
      const train=samples.filter((_,j)=>j!==i),m=fitLinear(train);const p=predict(m,samples[i].features);
      if(p)pairs.push({px:p.x,py:p.y,tx:samples[i].x,ty:samples[i].y});
    }
    return {rmse:rmse(pairs),pairs};
  }

  function kFold(samples,{kernel='rbf',folds=5,maxSamples=160}={}){
    if(!samples||samples.length<10)return {pairs:[],metrics:evaluatePairs([])};
    const k=Math.max(2,Math.min(folds,Math.floor(samples.length/2))),pairs=[];
    for(let fold=0;fold<k;fold++){
      const train=[],test=[];
      samples.forEach((s,i)=>(i%k===fold?test:train).push(s));
      const model=fitModel(train,{kernel,maxSamples});if(!model)continue;
      for(const s of test){const p=predict(model,s.features);if(p)pairs.push({px:p.x,py:p.y,tx:s.x,ty:s.y});}
    }
    return {pairs,metrics:evaluatePairs(pairs)};
  }

  function regionKey(sample,cols=3,rows=3,width=1,height=1){
    const x=clamp(sample.x/Math.max(1,width),0,.999999),y=clamp(sample.y/Math.max(1,height),0,.999999);
    return `${Math.floor(x*cols)}:${Math.floor(y*rows)}`;
  }
  function leaveRegionOut(samples,{kernel='rbf',cols=3,rows=3,width=1,height=1,maxSamples=160}={}){
    if(!samples||samples.length<18)return {pairs:[],metrics:evaluatePairs([]),regions:0};
    const groups=new Map();for(const s of samples){const k=regionKey(s,cols,rows,width,height);if(!groups.has(k))groups.set(k,[]);groups.get(k).push(s);}
    const pairs=[];
    for(const [key,test] of groups){const train=samples.filter(s=>regionKey(s,cols,rows,width,height)!==key);if(train.length<8)continue;const model=fitModel(train,{kernel,maxSamples});if(!model)continue;for(const s of test){const p=predict(model,s.features);if(p)pairs.push({px:p.x,py:p.y,tx:s.x,ty:s.y,region:key});}}
    return {pairs,metrics:evaluatePairs(pairs),regions:groups.size};
  }

  function pursuitPoint(progress,{margin=0.12,phase=Math.PI/2}={}){
    const t=clamp(progress,0,1)*Math.PI*2;
    const x=.5+.5*(1-2*margin)*Math.sin(3*t+phase);
    const y=.5+.5*(1-2*margin)*Math.sin(2*t);
    return {nx:clamp(x,margin,1-margin),ny:clamp(y,margin,1-margin)};
  }

  function coachAssessment(m={}){
    const reasons=[];
    const faceCentered=Math.abs(m.faceOffsetX||0)<=.16&&Math.abs(m.faceOffsetY||0)<=.18;
    const distanceOk=(m.interocularNorm||0)>=.08&&(m.interocularNorm||0)<=.28;
    const poseOk=Math.abs(m.yaw||0)<=.28&&Math.abs(m.pitch||0)<=.22&&Math.abs(m.roll||0)<=.22;
    const lightingOk=(m.luminance==null||((m.luminance>=45)&&(m.luminance<=205)))&&(m.lightingAsymmetry==null||m.lightingAsymmetry<=.28);
    const eyesOk=(m.leftReliability||0)>=.35&&(m.rightReliability||0)>=.35&&(m.fusedReliability||0)>=.40;
    if(!faceCentered)reasons.push('face-centering');if(!distanceOk)reasons.push('distance');if(!poseOk)reasons.push('head-pose');if(!lightingOk)reasons.push('lighting');if(!eyesOk)reasons.push('eye-quality');
    return {pass:reasons.length===0,reasons,criteria:{faceCentered,distanceOk,poseOk,lightingOk,eyesOk}};
  }

  return {
    clamp,avg,median,quantile,stdev,eyeReliability,bilateralAgreement,fuseEyes,ema,makeOneEuro,
    standardizer,normalizeFeatures,fitLinear,fitKernelRidge,fitModel,predict,rmse,evaluatePairs,leaveOneOut,kFold,leaveRegionOut,
    deterministicThin,pursuitPoint,coachAssessment
  };
});
