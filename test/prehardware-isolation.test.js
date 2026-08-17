'use strict';
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..');
const read=p=>fs.readFileSync(path.join(root,p),'utf8');
let passed=0;
function test(name,fn){fn();passed++;console.log(`ok ${passed} - ${name}`);}
const main=read('demo/prehardware/capture-fusion.js');
const frame=read('demo/prehardware/refined-capture-frame.js');
const html=read('demo/prehardware/refined-capture-frame.html');
const eye=read('demo/prehardware/eye-motion.js');
const index=read('demo/prehardware/index.html');

test('main page fusion layer does not instantiate legacy FaceMesh runtime',()=>{
  assert.equal(/new\s+(?:window\.)?FaceMesh\s*\(/.test(main),false);
  assert.equal(/face_mesh\/face_mesh\.js/.test(main),false);
});

test('refined FaceMesh runtime is confined to same-origin isolated frame',()=>{
  assert.ok(/new\s+FaceMesh\s*\(/.test(frame));
  assert.ok(html.includes('./mediapipe/face_mesh/face_mesh.js'));
  assert.ok(main.includes('refined-capture-frame.html'));
  assert.ok(main.includes("event.origin!==ORIGIN"));
});

test('direct eye-motion diagnostic reuses shared isolated capture',()=>{
  assert.ok(eye.includes('getFeatureSnapshot'));
  assert.equal(/new\s+(?:window\.)?FaceMesh\s*\(/.test(eye),false);
  assert.equal(/face_mesh\/face_mesh\.js/.test(eye),false);
});

test('fusion script itself remains deferred until lab session is running',()=>{
  assert.ok(index.includes("session.textContent.trim().toLowerCase() !== 'running'"));
  assert.equal(/<script\s+src=["']\.\/capture-fusion\.js/.test(index),false);
});

test('isolated refined capture is local-only with no external network URL',()=>{
  assert.equal(/https?:\/\//.test(frame),false);
  assert.equal(/https?:\/\//.test(html),false);
});

test('frame-synchronized capture uses requestVideoFrameCallback with fallback',()=>{
  assert.ok(main.includes('requestVideoFrameCallback'));
  assert.ok(main.includes("frameClock.mode='interval-fallback'"));
  assert.ok(main.includes('frameClock.inFlight'));
});

console.log(`prehardware isolation: ${passed}/${passed} passed`);
