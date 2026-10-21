"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict");
const {mapWebEyeTrackGaze:m}=require("../lab/webeyetrack-coordinate-contract");
test("WebEyeTrack screen coordinates preserve directional movement",()=>{
 const w=1600,h=900;
 assert.deepEqual(m([0,0],w,h),{x:800,y:450});
 assert.deepEqual(m([-.5,-.5],w,h),{x:0,y:0});
 assert.deepEqual(m([.5,.5],w,h),{x:1600,y:900});
 assert.ok(m([.2,0],w,h).x>m([-.2,0],w,h).x);
 assert.ok(m([0,.2],w,h).y>m([0,-.2],w,h).y);
});
test("Reject invalid predictions instead of clamping and claiming valid",()=>{
 for(const p of [[.6,0],[-.6,0],[0,NaN],null,[0]])assert.equal(m(p,1600,900),null);
});
