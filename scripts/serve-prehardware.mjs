import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const port = Number(process.env.PORT || 4818);
const types = { '.html':'text/html; charset=utf-8', '.js':'text/javascript; charset=utf-8', '.css':'text/css; charset=utf-8', '.json':'application/json; charset=utf-8' };

const server = http.createServer((req, res) => {
  const raw = decodeURIComponent((req.url || '/').split('?')[0]);

  // Codespaces commonly opens a forwarded port at `/`. Redirect to the
  // canonical lab path so relative CSS/JS URLs resolve correctly.
  if (raw === '/') {
    res.writeHead(302, {
      'Location': '/demo/prehardware/',
      'Cache-Control': 'no-store'
    });
    res.end();
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
    res.writeHead(200, {
      'Content-Type': types[path.extname(target)] || 'application/octet-stream',
      'Cache-Control': 'no-store',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Permissions-Policy': 'camera=(self)'
    });
    res.end(data);
  } catch (_) {
    res.writeHead(404, { 'Content-Type':'text/plain; charset=utf-8' }).end('Not found');
  }
});

server.listen(port, '127.0.0.1', () => {
  console.log(`webgazer-aac pre-hardware lab: http://127.0.0.1:${port}/demo/prehardware/`);
  console.log('Load a trusted local WebGazer 3.5.3 JavaScript file in the lab before starting the camera.');
});
