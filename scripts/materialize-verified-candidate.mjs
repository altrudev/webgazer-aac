import fs from 'node:fs';
import crypto from 'node:crypto';
import zlib from 'node:zlib';

const source = new URL('../ddc/candidate/webgazer-aac.compact.js.gz.b64', import.meta.url);
const target = new URL('../webgazer-aac.js', import.meta.url);
const expectedSha256 = 'e2049bad33eb37a6bfeb375490455d78a33d8cb31bb16339fa2cabf02715f5e1';
const expectedGitBlob = '2b98d877627af794f507cf1d7911a456135f0d82';
const expectedBytes = 51666;

const encoded = fs.readFileSync(source, 'utf8').trim();
const candidate = zlib.gunzipSync(Buffer.from(encoded, 'base64'));
const sha256 = crypto.createHash('sha256').update(candidate).digest('hex');
const gitBlob = crypto.createHash('sha1').update(Buffer.concat([
  Buffer.from(`blob ${candidate.length}\0`), candidate,
])).digest('hex');

if (candidate.length !== expectedBytes || sha256 !== expectedSha256 || gitBlob !== expectedGitBlob) {
  console.error('Verified candidate identity mismatch. Promotion refused.');
  console.error({ bytes: candidate.length, sha256, gitBlob });
  process.exit(1);
}

fs.writeFileSync(target, candidate);
console.log('Verified candidate materialized to webgazer-aac.js');
console.log({ bytes: candidate.length, sha256, gitBlob });
