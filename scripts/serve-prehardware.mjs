import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const port = Number(process.env.PORT || 4818);
const codespaceWebGazer = process.env.WEBGAZER_353_PATH || '/workspaces/WebGazer/dist/webgazer.js';
const codespaceWebGazerPackage = process.env.WEBGAZER_353_PACKAGE || '/workspaces/WebGazer/package.json';
const codespaceMediaPipeRoot = process.env.WEBGAZER_353_MEDIAPIPE || path.join(path.dirname(codespaceWebGazer), 'mediapipe');
const expectedWebGazerVersion = '3.5.3';
const types = {
  '.html':'text/html; charset=utf-8',
  '.js':'text/javascript; charset=utf-8',
  '.mjs':'text/javascript; charset=utf-8',
  '.css':'text/css; charset=utf-8',
  '.json':'application/json; charset=utf-8',
  '.wasm':'application/wasm',
  '.binarypb':'application/octet-stream',
  '.data':'application/octet-stream',
};

function commonHeaders(extra = {}) {
  return {
    'Cache-Control': 'no-store',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Permissions-Policy': 'camera=(self)',
    'X-Content-Type-Options': 'nosniff',
    ...extra,
  };
}

function hashMediaPipeTree(rootDir) {
  const entries = [];
  const walk = dir => {
    for (const name of fs.readdirSync(dir).sort()) {
      const absolute = path.join(dir, name);
      const stat = fs.statSync(absolute);
      if (stat.isDirectory()) {
        walk(absolute);
      } else if (stat.isFile()) {
        const rel = path.relative(rootDir, absolute).split(path.sep).join('/');
        const data = fs.readFileSync(absolute);
        entries.push({ rel, bytes: data.length, sha256: crypto.createHash('sha256').update(data).digest('hex') });
      }
    }
  };
  walk(rootDir);
  const canonical = entries.map(e => `${e.rel}\0${e.bytes}\0${e.sha256}\n`).join('');
  return {
    files: entries.length,
    bytes: entries.reduce((sum, e) => sum + e.bytes, 0),
    sha256: crypto.createHash('sha256').update(canonical).digest('hex'),
  };
}

function inspectCodespaceWebGazer() {
  const result = {
    available: false,
    verified: false,
    expectedVersion: expectedWebGazerVersion,
    version: null,
    packageName: null,
    bundleBytes: null,
    bundleSha256: null,
    mediaPipeAvailable: false,
    mediaPipeFiles: null,
    mediaPipeBytes: null,
    mediaPipeSha256: null,
    source: 'codespace-build',
    reason: null,
  };
  try {
    if (!fs.existsSync(codespaceWebGazerPackage)) throw new Error(`package metadata missing: ${codespaceWebGazerPackage}`);
    if (!fs.existsSync(codespaceWebGazer) || !fs.statSync(codespaceWebGazer).isFile()) throw new Error(`bundle missing: ${codespaceWebGazer}`);
    if (!fs.existsSync(codespaceMediaPipeRoot) || !fs.statSync(codespaceMediaPipeRoot).isDirectory()) throw new Error(`MediaPipe assets missing: ${codespaceMediaPipeRoot}`);
    const faceMeshRoot = path.join(codespaceMediaPipeRoot, 'face_mesh');
    if (!fs.existsSync(faceMeshRoot) || !fs.statSync(faceMeshRoot).isDirectory()) throw new Error(`MediaPipe face_mesh assets missing: ${faceMeshRoot}`);

    const pkg = JSON.parse(fs.readFileSync(codespaceWebGazerPackage, 'utf8'));
    const bundle = fs.readFileSync(codespaceWebGazer);
    const mediaPipe = hashMediaPipeTree(codespaceMediaPipeRoot);
    result.available = true;
    result.version = typeof pkg.version === 'string' ? pkg.version : null;
    result.packageName = typeof pkg.name === 'string' ? pkg.name : null;
    result.bundleBytes = bundle.length;
    result.bundleSha256 = crypto.createHash('sha256').update(bundle).digest('hex');
    result.mediaPipeAvailable = true;
    result.mediaPipeFiles = mediaPipe.files;
    result.mediaPipeBytes = mediaPipe.bytes;
    result.mediaPipeSha256 = mediaPipe.sha256;
    result.verified = result.version === expectedWebGazerVersion && result.mediaPipeAvailable;
    if (!result.verified) result.reason = `package version ${result.version || 'unknown'} does not match ${expectedWebGazerVersion}`;
  } catch (error) {
    result.reason = String(error && error.message || error);
  }
  return result;
}

function serveVerifiedMediaPipe(raw, res) {
  const identity = inspectCodespaceWebGazer();
  if (!identity.verified) {
    res.writeHead(409, commonHeaders({ 'Content-Type':'application/json; charset=utf-8' }));
    res.end(JSON.stringify(identity));
    return;
  }

  const prefix = '/demo/prehardware/mediapipe/';
  const rel = raw.slice(prefix.length);
  const target = path.resolve(codespaceMediaPipeRoot, rel);
  if (target !== codespaceMediaPipeRoot && !target.startsWith(codespaceMediaPipeRoot + path.sep)) {
    res.writeHead(403, commonHeaders({ 'Content-Type':'text/plain; charset=utf-8' }));
    res.end('Forbidden');
    return;
  }

  try {
    const stat = fs.statSync(target);
    if (!stat.isFile()) throw new Error('not a file');
    const data = fs.readFileSync(target);
    res.writeHead(200, commonHeaders({
      'Content-Type': types[path.extname(target)] || 'application/octet-stream',
      'X-WebGazer-Version': identity.version,
      'X-WebGazer-MediaPipe-SHA256': identity.mediaPipeSha256,
    }));
    res.end(data);
  } catch (_) {
    res.writeHead(404, commonHeaders({ 'Content-Type':'text/plain; charset=utf-8' }));
    res.end('MediaPipe asset not found');
  }
}

const server = http.createServer((req, res) => {
  const raw = decodeURIComponent((req.url || '/').split('?')[0]);

  if (raw === '/') {
    res.writeHead(302, commonHeaders({ 'Location': '/demo/prehardware/' }));
    res.end();
    return;
  }

  if (raw === '/__webgazer__/status') {
    const identity = inspectCodespaceWebGazer();
    res.writeHead(identity.verified ? 200 : 409, commonHeaders({ 'Content-Type':'application/json; charset=utf-8' }));
    res.end(JSON.stringify(identity));
    return;
  }

  if (raw === '/__webgazer__/webgazer.js') {
    const identity = inspectCodespaceWebGazer();
    if (!identity.verified) {
      res.writeHead(409, commonHeaders({ 'Content-Type':'application/json; charset=utf-8' }));
      res.end(JSON.stringify(identity));
      return;
    }
    const data = fs.readFileSync(codespaceWebGazer);
    res.writeHead(200, commonHeaders({
      'Content-Type':'text/javascript; charset=utf-8',
      'X-WebGazer-Version': identity.version,
      'X-WebGazer-SHA256': identity.bundleSha256,
      'X-WebGazer-MediaPipe-SHA256': identity.mediaPipeSha256,
    }));
    res.end(data);
    return;
  }

  if (raw.startsWith('/demo/prehardware/mediapipe/')) {
    serveVerifiedMediaPipe(raw, res);
    return;
  }

  const rel = raw.replace(/^\/+/, '');
  const file = path.resolve(root, rel);
  if (file !== root && !file.startsWith(root + path.sep)) {
    res.writeHead(403, commonHeaders({ 'Content-Type':'text/plain; charset=utf-8' })).end('Forbidden'); return;
  }
  let target = file;
  try {
    if (fs.statSync(target).isDirectory()) target = path.join(target, 'index.html');
    const data = fs.readFileSync(target);
    res.writeHead(200, commonHeaders({ 'Content-Type': types[path.extname(target)] || 'application/octet-stream' }));
    res.end(data);
  } catch (_) {
    res.writeHead(404, commonHeaders({ 'Content-Type':'text/plain; charset=utf-8' })).end('Not found');
  }
});

server.listen(port, '127.0.0.1', () => {
  const identity = inspectCodespaceWebGazer();
  console.log(`webgazer-aac pre-hardware lab: http://127.0.0.1:${port}/demo/prehardware/`);
  if (identity.verified) {
    console.log(`Codespace WebGazer ${identity.version} verified: ${identity.bundleBytes} bytes sha256=${identity.bundleSha256}`);
    console.log(`MediaPipe assets verified: ${identity.mediaPipeFiles} files ${identity.mediaPipeBytes} bytes sha256=${identity.mediaPipeSha256}`);
  } else {
    console.log(`Codespace WebGazer unavailable or unverified: ${identity.reason || 'unknown reason'}`);
  }
});
