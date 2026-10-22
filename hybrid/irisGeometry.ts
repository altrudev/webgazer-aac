// Independent 2D eye geometry. Does not infer screen gaze or degrees.
export type Landmark = { x:number;y:number;z?:number } | [number,number,number?];
export type EyeMeasurement = { cx:number;cy:number;horizontal:number;vertical:number;radiusPx:number;residualPx:number };
function xy(p:Landmark|undefined):[number,number]|null {
 if(!p)return null;
 const x=Array.isArray(p)?p[0]:p.x,y=Array.isArray(p)?p[1]:p.y;
 return Number.isFinite(x)&&Number.isFinite(y)?[x,y]:null;
}
function fitCircle(points:[number,number][]):{x:number;y:number;r:number;residual:number}|null {
 if(points.length<3)return null;
 const p0=points[0],a=points[1],b=points[2];
 const A=2*(a[0]-p0[0]),B=2*(a[1]-p0[1]),C=a[0]*a[0]+a[1]*a[1]-p0[0]*p0[0]-p0[1]*p0[1];
 const D=2*(b[0]-p0[0]),E=2*(b[1]-p0[1]),F=b[0]*b[0]+b[1]*b[1]-p0[0]*p0[0]-p0[1]*p0[1];
 const det=A*E-B*D;if(Math.abs(det)<1e-8)return null;
 const x=(C*E-B*F)/det,y=(A*F-C*D)/det;
 const radius=points.reduce((sum,p)=>sum+Math.hypot(p[0]-x,p[1]-y),0)/points.length;
 const residual=Math.sqrt(points.reduce((sum,p)=>sum+(Math.hypot(p[0]-x,p[1]-y)-radius)**2,0)/points.length);
 return Number.isFinite(radius)&&radius>0?{x,y,r:radius,residual}:null;
}
function eye(points:Landmark[],corners:[number,number],iris:number[]):EyeMeasurement|null {
 const a=xy(points[corners[0]]),b=xy(points[corners[1]]),rim=iris.map(i=>xy(points[i]));
 if(!a||!b||rim.some(v=>!v))return null;
 const width=Math.hypot(b[0]-a[0],b[1]-a[1]);if(width<1e-6)return null;
 const circle=fitCircle(rim as [number,number][]);if(!circle||circle.residual>circle.r*.45)return null;
 const ux=(b[0]-a[0])/width,uy=(b[1]-a[1])/width;
 return {cx:circle.x,cy:circle.y,horizontal:((circle.x-a[0])*ux+(circle.y-a[1])*uy)/width,vertical:((circle.x-a[0])*(-uy)+(circle.y-a[1])*ux)/width,radiusPx:circle.r,residualPx:circle.residual};
}
export function measureIris(points:Landmark[]|null|undefined){
 if(!points||points.length<478)return {available:false as const,reason:'iris_landmarks_missing'};
 const left=eye(points,[33,133],[469,470,471,472]),right=eye(points,[362,263],[474,475,476,477]);
 if(!left||!right)return {available:false as const,reason:'invalid_iris_contours'};
 const nose=xy(points[1]),bridge=xy(points[168]);
 return {available:true as const,left,right,headMotionProxy:nose&&bridge?{dx:nose[0]-bridge[0],dy:nose[1]-bridge[1]}:null,screenGaze:null};
}
