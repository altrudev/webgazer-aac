const {test}=require('node:test'),assert=require('node:assert/strict');const {measureIris}=require('../hybrid/iris-geometry-browser.js');
test('measured iris direction is independent of screen gaze',()=>{const p=Array.from({length:478},()=>({x:0,y:0}));p[33]={x:.1,y:.2};p[133]={x:.5,y:.2};p[362]={x:.6,y:.2};p[263]={x:1,y:.2};
function put(ids,c){const d=[[.04,0],[0,.04],[-.04,0],[0,-.04]];ids.forEach((i,j)=>p[i]={x:c+d[j][0],y:.2+d[j][1]});}
put([469,470,471,472],.3);put([474,475,476,477],.8);
const a=measureIris(p,1000,700);assert.equal(a.available,true);assert.equal(a.screenGaze,null);
put([469,470,471,472],.35);const b=measureIris(p,1000,700);assert.ok(b.left.horizontal>a.left.horizontal);
assert.equal(measureIris(p.slice(0,468),1000,700).available,false);
});