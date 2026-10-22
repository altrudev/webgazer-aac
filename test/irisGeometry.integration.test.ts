import {test} from 'node:test';
import assert from 'node:assert/strict';
import {measureIris} from '../hybrid/irisGeometry.ts';
test('center of circle and direction change without predicting screen gaze',()=>{
 const p=Array.from({length:478},()=>({x:0,y:0}));
 p[33]={x:10,y:20};p[133]={x:50,y:20};p[362]={x:80,y:20};p[263]={x:120,y:20};
 const put=(ids:number[],cx:number)=>{const offsets=[[5,0],[0,5],[-5,0],[0,-5]];ids.forEach((id,i)=>p[id]={x:cx+offsets[i][0],y:20+offsets[i][1]})};
 put([469,470,471,472],30);put([474,475,476,477],100);
 const a=measureIris(p);assert.equal(a.available,true);if(!a.available)return;
 assert.ok(Math.abs(a.left.cx-30)<1e-6);assert.equal(a.screenGaze,null);
 put([469,470,471,472],36);const b=measureIris(p);assert.equal(b.available,true);if(!b.available)return;
 assert.ok(b.left.horizontal>a.left.horizontal);
});
test('missing or degenerate iris contours reject measurement',()=>{
 assert.equal(measureIris(Array.from({length:468},()=>({x:0,y:0}))).available,false);
 assert.equal(measureIris(Array.from({length:478},()=>({x:0,y:0}))).available,false);
});
