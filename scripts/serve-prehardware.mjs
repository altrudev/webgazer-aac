import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const port = Number(process.env.PORT || 4818);
const codespaceWebGazer = process.env.WEBGAZER_353_PATH || '/workspaces/WebGazer/dist/webgazer.js';
const codespaceWebGazerPackage = process.env.WEBGAZER_353_PACKAGE || '/workspaces/WebGazer/package.json';
const expectedWebGazerVersion = '3.5.3';
const types = { '.html':'text/html; charset=utf-8', '.js':'text/javascript; charset=utf-8', '.css':'text/css; charset=utf-8', '.json':'application/json; charset=utf-8' };

function commonHeaders(extra = {}) {
  return {
    'Cache-Control': 'no-store',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Permissions-Policy': 'camera=(self)',
    ...extra,
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
    source: 'codespace-build',
    reason: null,
  };
  try {
    if (!fs.existsSync(codespaceWebGazerPackage)) throw new Error(`package metadata missing: ${codespaceWebGazerPackage}`);
    if (!fs.existsSync(codespaceWebGazer) || !fs.statSync(codespaceWebGazer).isFile()) throw new Error(`bundle missing: ${codespaceWebGazer}`);
    const pkg = JSON.parse(fs.readFileSync(codespaceWebGazerPackage, 'utf8'));
    const bundle = fs.readFileSync(codespaceWebGazer);
    result.available = true;
    result.version = typeof pkg.version === 'string' ? pkg.version : null;
    result.packageName = typeof pkg.name === 'string' ? pkg.name : null;
    result.bundleBytes = bundle.length;
    result.bundleSha256 = crypto.createHash('sha256').update(bundle).digest('hex');
    result.verified = result.version === expectedWebGazerVersion;
    if (!result.verified) result.reason = `package version ${result.version || 'unknown'} does not match ${expectedWebGazerVersion}`;
  } catch (error) {
    result.reason = String(error && error.message || error);
  }
  return result;
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
    }));
    res.end(data);
    return;
  }

  const rel = raw.replace(/^\/+/, '');
  const file = path.resolve(root, rel);
  if (file !== root && !file.startsWith(root + path.sep)) {
    res.writeHead(403).end('Forbidden'); return;
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
  } else {
    console.log(`Codespace WebGazer unavailable or unverified: ${identity.reason || 'unknown reason'}`);
  }
});
