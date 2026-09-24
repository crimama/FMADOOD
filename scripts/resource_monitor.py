#!/usr/bin/env python3
"""A small local dashboard for GPU, CPU, and memory across compute hosts."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
ASKPASS = ROOT / "scripts" / "ssh_askpass_env.sh"
SAMPLE_COMMAND = r"""bash -lc '
read -r _ u n s i w h si st _ < /proc/stat
t1=$((u+n+s+i+w+h+si+st)); b1=$((u+n+s+w+h+si+st))
sleep 0.2
read -r _ u n s i w h si st _ < /proc/stat
t2=$((u+n+s+i+w+h+si+st)); b2=$((u+n+s+w+h+si+st))
cpu=$(( (b2-b1)*10000/(t2-t1) ))
awk -v cpu="$cpu" "/MemTotal/ {total=\$2} /MemAvailable/ {available=\$2} END {printf \"CPU\\t%.1f\\nMEM\\t%.0f\\t%.0f\\n\", cpu/100, total-available, total}" /proc/meminfo
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | sed "s/^/GPU\\t/" || true
'"""


@dataclass(frozen=True)
class Host:
    name: str
    ssh_target: Optional[str] = None
    ssh_port: Optional[str] = None
    password: Optional[str] = None


def load_remote(name: str) -> Host:
    """Load only the values required to connect from the ignored local preset."""
    preset = ROOT / "configs" / "remote" / f"{name}.env"
    values: Dict[str, str] = {}
    if preset.exists():
        for line in preset.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export ") and "=" in line:
                key, value = line[7:].split("=", 1)
                values[key] = value.strip().strip("'\"")
    return Host(name, values.get("REMOTE_HOST"), values.get("REMOTE_PORT", "22"), values.get("SSH_PASSWORD"))


def parse_sample(output: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"cpu": None, "memory": None, "gpus": []}
    for row in csv.reader(io.StringIO(output), delimiter="\t"):
        if not row:
            continue
        if row[0] == "CPU" and len(row) == 2:
            result["cpu"] = float(row[1])
        elif row[0] == "MEM" and len(row) == 3:
            used, total = (float(value) for value in row[1:])
            result["memory"] = {"used_gib": round(used / 1048576, 1), "total_gib": round(total / 1048576, 1)}
        elif row[0] == "GPU" and len(row) == 2:
            gpu = next(csv.reader([row[1],], skipinitialspace=True))
            if len(gpu) != 6:
                continue
            result["gpus"].append(
                {"index": gpu[0].strip(), "name": gpu[1].strip(), "utilization": float(gpu[2]),
                 "used_mib": float(gpu[3]), "total_mib": float(gpu[4]), "temperature": float(gpu[5])}
            )
    return result


def collect(host: Host) -> Dict[str, Any]:
    started = time.monotonic()
    if host.ssh_target:
        if not host.password:
            return {"name": host.name, "error": "SSH password is missing from the local preset."}
        env = os.environ.copy()
        env.update({"SSH_PASSWORD": host.password, "SSH_ASKPASS": str(ASKPASS), "SSH_ASKPASS_REQUIRE": "force", "DISPLAY": "monitor:0"})
        command = ["ssh", "-p", host.ssh_port or "22", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=accept-new", host.ssh_target, SAMPLE_COMMAND]
    else:
        env = None
        command = ["bash", "-lc", SAMPLE_COMMAND]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, env=env, check=True)
        data = parse_sample(completed.stdout)
        if data["cpu"] is None or data["memory"] is None:
            raise ValueError("received an incomplete system sample")
        data.update({"name": host.name, "latency_ms": round((time.monotonic() - started) * 1000)})
        return data
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"name": host.name, "error": str(exc), "latency_ms": round((time.monotonic() - started) * 1000)}


PAGE = r"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Compute Monitor</title><style>
*{box-sizing:border-box}body{margin:0;background:#f6f7fb;color:#182230;font:16px system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:32px 20px}header{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}h1{font-size:24px;margin:0}small{color:#697386}.hosts{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:16px}.card{background:white;border-radius:14px;padding:20px;box-shadow:0 2px 12px #18223012}.title{display:flex;justify-content:space-between;align-items:center;margin-bottom:17px;font-size:20px;font-weight:700}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:12px}.metric{background:#f8faff;border-radius:10px;padding:12px}.metric b{font-size:21px;display:block;margin-top:3px}.bar{height:7px;background:#e8edf5;border-radius:9px;overflow:hidden;margin-top:10px}.bar i{display:block;height:100%;background:#276ef1;border-radius:9px}.gpu{border-top:1px solid #edf0f5;margin-top:17px;padding-top:14px}.gpuhead{display:flex;justify-content:space-between;font-weight:600}.gpu small{display:block;margin-top:3px}.error{color:#c23934;background:#fff2f0;padding:12px;border-radius:8px}@media(max-width:500px){main{padding:20px 12px}}
</style><main><header><div><h1>Compute Monitor</h1><small>GPU · CPU · RAM &nbsp;|&nbsp; 5초마다 갱신</small></div><small id="updated">연결 중…</small></header><section class="hosts" id="hosts"></section></main><script>
const pct=x=>Math.max(0,Math.min(100,x)); const bar=x=>`<div class="bar"><i style="width:${pct(x)}%"></i></div>`;
function host(h){if(h.error)return `<article class="card"><div class="title">${h.name}<small>오류</small></div><div class="error">${h.error}</div></article>`;let m=h.memory,p=m.used_gib/m.total_gib*100;return `<article class="card"><div class="title">${h.name}<small>${h.latency_ms} ms</small></div><div class="metrics"><div class="metric"><small>CPU</small><b>${h.cpu.toFixed(1)}%</b>${bar(h.cpu)}</div><div class="metric"><small>RAM</small><b>${m.used_gib} / ${m.total_gib} GiB</b>${bar(p)}</div></div><div class="gpu">${h.gpus.length?h.gpus.map(g=>{let p=g.used_mib/g.total_mib*100;return `<div class="gpuhead"><span>GPU ${g.index} · ${g.utilization.toFixed(0)}%</span><span>${(g.used_mib/1024).toFixed(1)} / ${(g.total_mib/1024).toFixed(1)} GiB</span></div><small>${g.name} · ${g.temperature.toFixed(0)}°C</small>${bar(p)}`}).join('<div style="height:14px"></div>'):'<small>GPU를 감지하지 못했습니다.</small>'}</div></article>`}
const apiPath=(location.pathname==='/' ? '' : location.pathname.replace(/\/$/,''))+'/api/status';
async function refresh(){try{let d=await fetch(apiPath,{cache:'no-store'}).then(r=>r.json());document.querySelector('#hosts').innerHTML=d.hosts.map(host).join('');document.querySelector('#updated').textContent='갱신 '+new Date().toLocaleTimeString('ko-KR')}catch(e){document.querySelector('#updated').textContent='갱신 실패'}}refresh();setInterval(refresh,5000);
</script>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/status":
            with ThreadPoolExecutor(max_workers=3) as pool:
                hosts = list(pool.map(collect, self.server.hosts))  # type: ignore[attr-defined]
            body = json.dumps({"hosts": hosts}).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        elif self.path == "/":
            body = PAGE.encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        else:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.hosts = [Host("local"), load_remote("dsba3"), load_remote("dsba5")]  # type: ignore[attr-defined]
    print(f"Compute monitor: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
