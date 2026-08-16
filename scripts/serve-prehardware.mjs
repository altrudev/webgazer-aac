import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const port = Number(process.env.PORT || 4818);
const expectedWebGazerVersion = '3.5.3';
const codespaceWebGazer = process.env.WEBGAZER_353_PATH || '/workspaces/WebGazer/dist/webgazer.js';
const codespaceWebGazerPackage = process.env.WEBGAZER_353_PACKAGE_PATH || path.resolve(path.dirname(codespaceWebGazer), '..', 'package.json');
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
    identityVerified: false,
    expectedVersion: expectedWebGazerVersion,
    version: null,
    source: 'codespace-build',
    bundleSha256: null,
    bundleBytes: null,
    reason: null,
  };
  try {
    if (!fs.existsSync(codespaceWebGazer) || !fs.statSync(codespaceWebGazer).isFile()) {
      result.reason = 'bundle-missing';
      return result;
    }
    if (!fs.existsSync(codespaceWebGazerPackage) || !fs.statSync(codespaceWebGazerPackage).isFile()) {
      result.reason = 'package-metadata-missing';
      return result;
    }
    const pkg = JSON.parse(fs.readFileSync(codespaceWebGazerPackage, 'utf8'));
    const data = fs.readFileSync(codespaceWebGazer);
    result.available = true;
    result.version = typeof pkg.version === 'string' ? pkg.version : null;
    result.bundleBytes = data.length;
    result.bundleSha256 = crypto.createHash('sha256').update(data).digest('hex');
    result.identityVerified = result.version === expectedWebGazerVersion;
    if (!result.identityVerified) result.reason = 'version-mismatch';
    return result;
  } catch (error) {
    result.reason = `inspection-error:${String(error.message || error)}`;
    return result;
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
    const status = inspectCodespaceWebGazer();
    res.writeHead(status.identityVerified ? 200 : 409, commonHeaders({ 'Content-Type':'application/json; charset=utf-8' }));
    res.end(JSON.stringify(status));
    return;
  }

  if (raw === '/__webgazer__/webgazer.js') {
    const status = inspectCodespaceWebGazer();
    if (!status.identityVerified) {
      res.writeHead(409, commonHeaders({ 'Content-Type':'text/plain; charset=utf-8' }));
      res.end(`WebGazer identity not verified: expected ${expectedWebGazerVersion}, observed ${status.version || 'unknown'} (${status.reason || 'unverified'})`);
      return;
    }
    try {
      const data = fs.readFileSync(codespaceWebGazer);
      res.writeHead(200, commonHeaders({
        'Content-Type':'text/javascript; charset=utf-8',
        'X-WebGazer-Version': status.version,
        'X-WebGazer-SHA256': status.bundleSha256,
      }));
      res.end(data);
    } catch (_) {
      res.writeHead(404, commonHeaders({ 'Content-Type':'text/plain; charset=utf-8' }));
      res.end('Verified WebGazer bundle became unavailable.');
    }
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
  const status = inspectCodespaceWebGazer();
  console.log(`webgazer-aac pre-hardware lab: http://127.0.0.1:${port}/demo/prehardware/`);
  if (status.identityVerified) {
    console.log(`Codespace WebGazer verified: v${status.version}, ${status.bundleBytes} bytes, sha256 ${status.bundleSha256}`);
  } else {
    console.log(`Codespace WebGazer NOT verified: ${status.reason || 'unknown'}; expected v${expectedWebGazerVersion}, observed ${status.version || 'unknown'}`);
  }
});
