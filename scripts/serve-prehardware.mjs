import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const port = Number(process.env.PORT || 4818);
const codespaceWebGazer = process.env.WEBGAZER_353_PATH || '/workspaces/WebGazer/dist/webgazer.js';
const types = { '.html':'text/html; charset=utf-8', '.js':'text/javascript; charset=utf-8', '.css':'text/css; charset=utf-8', '.json':'application/json; charset=utf-8' };

function commonHeaders(extra = {}) {
  return {
    'Cache-Control': 'no-store',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Permissions-Policy': 'camera=(self)',
    ...extra,
  };
}

const server = http.createServer((req, res) => {
  const raw = decodeURIComponent((req.url || '/').split('?')[0]);

  if (raw === '/') {
    res.writeHead(302, commonHeaders({ 'Location': '/demo/prehardware/' }));
    res.end();
    return;
  }

  if (raw === '/__webgazer__/status') {
    const exists = fs.existsSync(codespaceWebGazer) && fs.statSync(codespaceWebGazer).isFile();
    const body = JSON.stringify({ available: exists, expectedVersion: '3.5.3', source: exists ? 'codespace-build' : 'missing' });
    res.writeHead(exists ? 200 : 404, commonHeaders({ 'Content-Type':'application/json; charset=utf-8' }));
    res.end(body);
    return;
  }

  if (raw === '/__webgazer__/webgazer.js') {
    try {
      const data = fs.readFileSync(codespaceWebGazer);
      res.writeHead(200, commonHeaders({ 'Content-Type':'text/javascript; charset=utf-8' }));
      res.end(data);
    } catch (_) {
      res.writeHead(404, commonHeaders({ 'Content-Type':'text/plain; charset=utf-8' }));
      res.end('WebGazer 3.5.3 build not found. Expected /workspaces/WebGazer/dist/webgazer.js');
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
  const available = fs.existsSync(codespaceWebGazer);
  console.log(`webgazer-aac pre-hardware lab: http://127.0.0.1:${port}/demo/prehardware/`);
  console.log(available
    ? `Codespace WebGazer build available: ${codespaceWebGazer}`
    : `Codespace WebGazer build not found at ${codespaceWebGazer}`);
});
