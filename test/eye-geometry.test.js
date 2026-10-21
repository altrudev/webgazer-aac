'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const geom=require('../lab/eye-geometry.js');
test('independent iris offset requires both eyes and iris landmarks',()=>{
 const a=Array.from({length:478},()=>[0,0,0]);
 a[33]=[10,20];a[133]=[50,20];a[468]=[30,20];
 a[362]=[80,20];a[263]=[120,20];a[473]=[100,20];
 a[1]=[65,30];a[168]=[65,26];
 const m=geom.measure(a);assert.equal(m.available,true);assert.equal(m.left_horizontal,.5);assert.equal(m.right_horizontal,.5);assert.equal(m.binocular_disagreement,0);
 a[473]=[110,20];assert.ok(geom.measure(a).binocular_disagreement>.2);
 a.length=468;assert.equal(geom.measure(a).available,false);
});
