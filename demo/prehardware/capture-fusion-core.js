(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  root.webgazerCaptureFusionCore=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){'use strict';
  function clamp(v,lo=0,hi=1){return Math.max(lo,Math.min(hi,Number.isFinite(v)?v:lo));}
  function avg(a){return a.length?a.reduce((x,y)=>x+y,0)/a.length:0;}
  function median(a){if(!a.length)return 0;const b=a.slice().sort((x,y)=>x-y);const m=Math.floor(b.length/2);return b.length%2?b[m]:(b[m-1]+b[m])/2;}
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
    return clamp(visibility*0.22 + apertureRatio*0.20 + stability*0.22 + landmark*0.18 + (1-occlusion)*0.10 + (1-reflectionPenalty)*0.08);
  }
  function bilateralAgreement(left,right){
    if(!left||!right)return 0;
    const du=Math.abs((left.u||0)-(right.u||0));
    const dv=Math.abs((left.v||0)-(right.v||0));
    return clamp(1-(du/0.18+dv/0.14)/2);
  }
  function fuseEyes(left,right){
    if(!left&&!right)return null;
    if(!left)return {...right,source:'right-only'};
    if(!right)return {...left,source:'left-only'};
    const wl=Math.max(0,left.reliability||0),wr=Math.max(0,right.reliability||0),sum=wl+wr;
    if(sum<1e-6)return {u:0,v:0,reliability:0,source:'none'};
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
  function solve(A,b){
    const n=b.length,M=A.map((r,i)=>r.slice().concat([b[i]]));
    for(let c=0;c<n;c++){
      let p=c;for(let r=c+1;r<n;r++)if(Math.abs(M[r][c])>Math.abs(M[p][c]))p=r;
      [M[c],M[p]]=[M[p],M[c]];const d=M[c][c];if(Math.abs(d)<1e-10)continue;
      for(let r=0;r<n;r++){if(r===c)continue;const f=M[r][c]/d;for(let k=c;k<=n;k++)M[r][k]-=f*M[c][k];}
    }
    return M.map((r,i)=>Math.abs(r[i])<1e-10?0:r[n]/r[i]);
  }
  function fitLinear(samples,lambda=1e-3){
    if(!samples||samples.length<5)return null;
    const m=samples[0].features.length+1,XtX=Array.from({length:m},()=>Array(m).fill(0)),XtyX=Array(m).fill(0),XtyY=Array(m).fill(0);
    for(const s of samples){const row=[1].concat(s.features),w=Math.max(1e-4,s.weight==null?1:s.weight);for(let i=0;i<m;i++){XtyX[i]+=w*row[i]*s.x;XtyY[i]+=w*row[i]*s.y;for(let j=0;j<m;j++)XtX[i][j]+=w*row[i]*row[j];}}
    for(let i=0;i<m;i++)XtX[i][i]+=lambda;
    return {bx:solve(XtX,XtyX),by:solve(XtX,XtyY),featureCount:m-1};
  }
  function predict(model,features){if(!model||!features||features.length!==model.featureCount)return null;const row=[1].concat(features);return {x:row.reduce((s,v,i)=>s+v*model.bx[i],0),y:row.reduce((s,v,i)=>s+v*model.by[i],0)};}
  function rmse(pairs){if(!pairs.length)return null;return Math.sqrt(avg(pairs.map(p=>(p.px-p.tx)**2+(p.py-p.ty)**2)));}
  function leaveOneOut(samples){
    if(!samples||samples.length<6)return {rmse:null,pairs:[]};const pairs=[];
    for(let i=0;i<samples.length;i++){const train=samples.filter((_,j)=>j!==i),m=fitLinear(train);const p=predict(m,samples[i].features);if(p)pairs.push({px:p.x,py:p.y,tx:samples[i].x,ty:samples[i].y});}
    return {rmse:rmse(pairs),pairs};
  }
  return {clamp,avg,median,stdev,eyeReliability,bilateralAgreement,fuseEyes,ema,fitLinear,predict,rmse,leaveOneOut};
});