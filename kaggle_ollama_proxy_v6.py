# ============================================================
# COMPLETE v13: Ollama + OpenAI-compatible proxy + tunnel
#  - NEW MODEL: Qwen3-Coder-30B-A3B-Instruct UNCENSORED (abliterated)
#    Q4_K_M GGUF from HuggingFace (auto-falls back to Q4_K_S/IQ4_XS)
#  - HEARTBEAT: [HEARTBEAT #N] line every 60s + /heartbeat endpoint
#  - THINK SPLIT: Ollama `thinking` -> `reasoning_content`, and
#    `...` blocks stripped -> Freebuff shows thinking separately
#  - CONTEXT FIX: 32k context (Freebuff's prompt is ~13.5k tokens;
#    the old 8192 cap caused 'request exceeds the available context')
#  - norm_msgs(): converts OpenAI array-of-parts content to plain
#    strings (fixes Ollama 'cannot unmarshal array' 400)
#  - STREAMING FIX: httpx client lives inside the generator (520 fix)
#  - RE-RUN SAFE: kills old leftovers; if an old proxy still holds
#    port 8000, auto-switches to a free port (no restart needed)
#  - runs CONTINUOUSLY (never stops)
# ============================================================
import os, re, json, time, uuid, socket, subprocess, threading, urllib.request

PORT       = 11434   # Ollama
PROXY_PORT = 8000    # OpenAI-compatible proxy (tunnel THIS)
OLLAMA_BIN = "/usr/local/bin/ollama"
HF_REPO    = "mradermacher/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-GGUF"
QUANT      = ["Q4_K_M", "Q4_K_S", "IQ4_XS", "Q4_0"]   # Q4 first; auto-falls to smaller
MODEL      = "qwen3-coder-30b-uncensored:q4_k_m"
CONTEXT    = 32768   # was 8192 -> Freebuff's prompt (~13.5k tokens) needs more
WORKDIR    = "/kaggle/working/ocrs"

os.makedirs(WORKDIR, exist_ok=True)
sh0 = lambda c: subprocess.run(c, shell=True, executable="/bin/bash",
                               stdin=subprocess.DEVNULL).returncode

if not os.path.exists(OLLAMA_BIN):
    sh0("apt-get update -qq && apt-get install -y -qq zstd pciutils curl wget")
    sh0("curl -fsSL https://ollama.com/install.sh | sh")
sh0("pip install -q fastapi uvicorn httpx huggingface_hub")

env = os.environ.copy()
env["PATH"] = f"/usr/local/bin:/usr/bin:/bin:{env.get('PATH','')}"
env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
env.update({"CUDA_VISIBLE_DEVICES": "0,1", "OLLAMA_SCHED_SPREAD": "1",
            "OLLAMA_NUM_PARALLEL": "2", "OLLAMA_KEEP_ALIVE": "24h",
            "OLLAMA_NUM_GPU": "999", "OLLAMA_CONTEXT_LENGTH": str(CONTEXT),
            "OLLAMA_KV_CACHE_TYPE": "q8_0",
            "OLLAMA_HOST": f"127.0.0.1:{PORT}"})

def sh(cmd, timeout=7200):
    try:
        p = subprocess.run(cmd, shell=True, executable="/bin/bash",
                           capture_output=True, text=True, timeout=timeout,
                           env=env, stdin=subprocess.DEVNULL)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# ---- FIXED: parameter is `timeout` in both functions ----
def api_get(path, timeout=5):
    with opener.open(urllib.request.Request(f"http://127.0.0.1:{PORT}{path}"), timeout=timeout) as r:
        return json.loads(r.read().decode())

def api_post(path, payload, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())
# ---------------------------------------------------------

# === 1) Start Ollama ===
print("=== Starting Ollama (PROXY BUILD v13) ===", flush=True)
# kill leftovers from previous cell runs so the NEW code actually serves
sh("pkill -9 -f 'ollama serve'; pkill -9 -f cloudflared; pkill -9 -f uvicorn", timeout=10)
time.sleep(2)
# If an OLD proxy from a previous run of this cell still holds the port
# (it lives inside the kernel process - a cell re-run cannot kill it),
# just pick the next free port instead of dying. The old proxy becomes
# irrelevant: we killed the old tunnels above and tunnel to OUR port.
_picked = None
for _p in range(PROXY_PORT, PROXY_PORT + 10):
    for _ in range(3):
        s = socket.socket()
        try:
            s.bind(("0.0.0.0", _p)); s.close(); _picked = _p; break
        except OSError:
            s.close(); time.sleep(1)
    if _picked is not None:
        break
if _picked is None:
    raise RuntimeError(f"no free port in {PROXY_PORT}-{PROXY_PORT+9}; restart the kernel and rerun")
if _picked != PROXY_PORT:
    print(f"NOTE: port {PROXY_PORT} busy (old proxy from a previous run) -> using port {_picked} instead", flush=True)
PROXY_PORT = _picked
subprocess.Popen(f"{OLLAMA_BIN} serve > {WORKDIR}/ollama.log 2>&1",
                 shell=True, env=env)
for _ in range(30):
    if sh0(f"curl -sf http://127.0.0.1:{PORT}/api/version") == 0:
        break
    time.sleep(1)
else:
    raise RuntimeError("ollama did not start")
print("ollama: UP", flush=True)

# === 2) Model check / download ===
def model_present():
    tags = {t["name"] for t in api_get("/api/tags").get("models", [])}
    return MODEL in tags or any(t.split(":")[0] == MODEL for t in tags)

if not model_present():
    print("model missing -> downloading from HuggingFace...", flush=True)
    from huggingface_hub import hf_hub_download
    with opener.open(f"https://huggingface.co/api/models/{HF_REPO}/tree/main", timeout=60) as r:
        tree = json.loads(r.read().decode())
    files = [f["path"] for f in tree if f["path"].lower().endswith(".gguf")]
    cand = None
    for q in QUANT:                       # respect QUANT order -> Q4_K_M first
        hit = [p for p in files if q.lower() in p.lower()]
        if hit:
            cand = hit[0]
            break
    if cand is None:
        raise RuntimeError(f"no {'/'.join(QUANT)} gguf found in {HF_REPO}")
    print(f"downloading: {cand}", flush=True)
    gguf = hf_hub_download(repo_id=HF_REPO, filename=cand)
    with open(f"{WORKDIR}/Modelfile", "w") as f:
        f.write(f"FROM {gguf}\nPARAMETER num_ctx {CONTEXT}\n")
    rc, out = sh(f"{OLLAMA_BIN} create {MODEL} -f {WORKDIR}/Modelfile", timeout=3600)
    if rc != 0:
        raise RuntimeError(f"ollama create failed:\n{out}")
print("model present: True", flush=True)

# === 3) Warm-up (FIXED call) ===
print("=== Warm-up ===", flush=True)
t0 = time.time()
resp = api_post("/api/chat", {"model": MODEL,
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    "stream": False, "keep_alive": "24h",
    "options": {"num_ctx": CONTEXT}}, timeout=900)
print(f"warm-up OK in {time.time()-t0:.1f}s -> {resp.get('message',{}).get('content','')!r}", flush=True)

# === 4) OpenAI-compatible proxy ===
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx, uvicorn

app = FastAPI()
OLLAMA_URL = f"http://127.0.0.1:{PORT}"

# ---- heartbeat state ----
START_TS  = time.time()
REQ_COUNT = 0
LAST_REQ  = 0.0
BEATS     = 0

def norm_msgs(msgs):
    """Freebuff/OpenAI send content as an ARRAY OF PARTS; Ollama's Go API needs a
    plain string (+ optional images[]). Convert, or Ollama 400s with
    'cannot unmarshal array into Go struct field .ChatRequest.messages.content'."""
    out = []
    for m in msgs:
        c = m.get("content", "")
        images = list(m.get("images") or [])
        if isinstance(c, list):
            texts = []
            for part in c:
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict):
                    t = part.get("type", "text")
                    if t == "text":
                        texts.append(str(part.get("text", "")))
                    elif t == "image_url":
                        url = (part.get("image_url") or {}).get("url", "") or ""
                        if url.startswith("data:") and "," in url:
                            images.append(url.split(",", 1)[-1])
            c = "".join(texts)
        elif c is None:
            c = ""
        elif not isinstance(c, str):
            c = json.dumps(c)
        nm = {"role": m.get("role", "user"), "content": c}
        if images:
            nm["images"] = images
        out.append(nm)
    return out

def err_msg(raw):
    try:
        o = json.loads(raw)
        if isinstance(o, dict):
            return o.get("error", raw) if isinstance(o.get("error"), str) else raw
    except Exception:
        pass
    return raw

_THINK_OPEN  = chr(60) + 'think' + chr(62)
_THINK_CLOSE = chr(60) + chr(47) + 'think' + chr(62)

def split_think(text):
    """Extract think blocks -> (reasoning, clean_content)."""
    reasoning, content, rest = [], [], text or ''
    while True:
        a = rest.find(_THINK_OPEN)
        if a == -1:
            content.append(rest)
            break
        b = rest.find(_THINK_CLOSE, a)
        content.append(rest[:a])
        if b == -1:
            reasoning.append(rest[a + len(_THINK_OPEN):])
            break
        reasoning.append(rest[a + len(_THINK_OPEN):b])
        rest = rest[b + len(_THINK_CLOSE):]
    return ''.join(reasoning), ''.join(content)

class ThinkSplitter:
    """Streaming splitter: routes `...` text to reasoning and
    everything else to content, SAFE across chunk boundaries (a tag split
    between two chunks is buffered until it can be classified)."""
    OPEN, CLOSE = _THINK_OPEN, _THINK_CLOSE

    def __init__(self):
        self.buf, self.in_think = "", False

    def feed(self, text):
        self.buf += text or ""
        reasoning = content = ""
        while True:
            tag = self.CLOSE if self.in_think else self.OPEN
            i = self.buf.find(tag)
            if i == -1:
                # hold back a trailing partial tag split across chunks
                hold = 0
                for k in range(1, min(len(tag) - 1, len(self.buf)) + 1):
                    if tag.startswith(self.buf[-k:]):
                        hold = k
                out, self.buf = self.buf[:len(self.buf) - hold], self.buf[len(self.buf) - hold:]
                if self.in_think:
                    reasoning += out
                else:
                    content += out
                break
            head, self.buf = self.buf[:i], self.buf[i + len(tag):]
            if self.in_think:
                reasoning += head
                self.in_think = False
            else:
                content += head
                self.in_think = True
        return reasoning, content

    def flush(self):
        out, self.buf = self.buf, ""
        return (out, "") if self.in_think else ("", out)

@app.get("/")
@app.get("/health")
async def health():
    return {"status": "ok", "build": "v13", "context": CONTEXT, "provider": "ollama", "model": MODEL}

@app.get("/heartbeat")
async def heartbeat():
    return {"status": "ok", "build": "v13", "beat": BEATS, "ts": int(time.time()),
            "uptime_s": int(time.time() - START_TS), "model": MODEL,
            "requests": REQ_COUNT,
            "last_request_ago_s": (int(time.time() - LAST_REQ) if LAST_REQ else None)}

@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model",
            "created": int(time.time()), "owned_by": "ollama"}]}

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat(request: Request):
    global REQ_COUNT, LAST_REQ
    body = await request.json()
    REQ_COUNT += 1
    LAST_REQ  = time.time()
    model = body.get("model", MODEL)
    stream = bool(body.get("stream", False))
    opts = {}
    if "temperature" in body:     opts["temperature"] = body["temperature"]
    if "top_p" in body:           opts["top_p"] = body["top_p"]
    if "max_tokens" in body:      opts["num_predict"] = body["max_tokens"]
    opts["num_ctx"] = int(body.get("num_ctx", CONTEXT))  # fix: request needs > 13.5k ctx
    payload = {"model": model, "messages": norm_msgs(body.get("messages", [])),
               "stream": stream, "keep_alive": "24h", "options": opts}

    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def mk_client():
        return httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=15.0),
                                 trust_env=False)

    if not stream:
        async with mk_client() as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat",
                                  json={**payload, "stream": False})
            if r.status_code != 200:
                return JSONResponse({"error": {"message": err_msg(r.text), "type": "ollama_error"}},
                                    status_code=r.status_code)
            data = r.json()
        msg = data.get("message", {})
        rsn, clean = split_think(msg.get("content", ""))
        rsn = (msg.get("thinking") or "") + rsn   # Ollama reasoning field + any ...` blocks
        message = {"role": "assistant", "content": clean}
        if rsn:
            message["reasoning_content"] = rsn   # Freebuff shows this in the thinking panel
        return {"id": cmpl_id, "object": "chat.completion", "created": created,
                "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": message}],
                "usage": {"prompt_tokens": data.get("prompt_eval_count", 0),
                          "completion_tokens": data.get("eval_count", 0),
                          "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)}}

    async def gen():
        # FIX: the httpx client must live INSIDE the generator. The old code opened it
        # in an `async with` in the endpoint, which CLOSED it before streaming began,
        # killing the response mid-flight -> Cloudflare HTTP 520.
        async with mk_client() as client:
            # instant first byte (OpenAI-style empty role chunk) so Cloudflare's
            # ~100s origin timeout never trips even on a cold model
            yield "data: " + json.dumps({"id": cmpl_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                             "finish_reason": None}]}) + "\n\n"
            def chunk(delta):
                return "data: " + json.dumps({"id": cmpl_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}) + "\n\n"

            splitter = ThinkSplitter()
            try:
                async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as r:
                    if r.status_code != 200:
                        err = (await r.aread()).decode()
                        yield f"data: {json.dumps({'error': {'message': err_msg(err)}})}\n\n"
                    else:
                        async for line in r.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            m = obj.get("message", {})
                            if m.get("thinking"):
                                # Ollama's reasoning field -> Freebuff thinking panel
                                yield chunk({"reasoning_content": m["thinking"]})
                            if m.get("content"):
                                rsn, cnt = splitter.feed(m["content"])
                                if rsn:
                                    yield chunk({"reasoning_content": rsn})
                                if cnt:
                                    yield chunk({"content": cnt})
                            if obj.get("done"):
                                rsn, cnt = splitter.flush()
                                if rsn:
                                    yield chunk({"reasoning_content": rsn})
                                if cnt:
                                    yield chunk({"content": cnt})
                                yield "data: " + json.dumps({"id": cmpl_id, "object": "chat.completion.chunk",
                                    "created": created, "model": model,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': {'message': f'proxy stream error: {e}'}})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

def proxy_wd():
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="warning")

# === 5) Tunnel (cloudflared quick tunnel) ===
CF_BIN = f"{WORKDIR}/cloudflared"
if not os.path.exists(CF_BIN):
    sh0(f"curl -fsSL -o {CF_BIN} https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x {CF_BIN}")

TUNNEL_LOG = f"{WORKDIR}/tunnel.log"
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

def tunnel_wd():
    while True:
        try:
            with open(TUNNEL_LOG, "w") as lf:
                subprocess.Popen([CF_BIN, "tunnel", "--url", f"http://127.0.0.1:{PROXY_PORT}",
                                  "--no-autoupdate"], stdout=lf, stderr=subprocess.STDOUT, env=env)
            for _ in range(120):
                time.sleep(2)
                try:
                    txt = open(TUNNEL_LOG).read()
                except Exception:
                    txt = ""
                m = URL_RE.search(txt)
                if m:
                    print(f"\n>>> TUNNEL URL: {m.group(0)}/v1\n", flush=True)
                    print("COPY THIS TO FREEBUFF (Base URL) - REPLACE the old one in Freebuff:", flush=True)
                    print(f"{m.group(0)}/v1", flush=True)
                    print(f'SELF-CHECK: open {m.group(0)}/health in a browser -> must show "build": "v13"', flush=True)
                    try:
                        open(f"{WORKDIR}/freebuff_url.txt", "w").write(f"{m.group(0)}/v1\n")
                    except Exception:
                        pass
                    break
            # keep watching; restart if process dies
            while True:
                time.sleep(30)
                if sh0("pgrep -f 'cloudflared tunnel' >/dev/null") != 0:
                    print("cloudflared died -> restarting", flush=True)
                    break
        except Exception as e:
            print("tunnel_wd error:", e, flush=True)
            time.sleep(5)

def url_mon():
    last = ""
    while True:
        time.sleep(15)
        try:
            txt = open(TUNNEL_LOG).read()
            m = URL_RE.search(txt)
            if m and m.group(0) != last:
                last = m.group(0)
                print(f"\n[TUNNEL] Base URL for Freebuff: {last}/v1\n", flush=True)
        except Exception:
            pass

def ollama_wd():
    while True:
        time.sleep(30)
        if sh0(f"curl -sf http://127.0.0.1:{PORT}/api/version >/dev/null") != 0:
            print("ollama died -> restarting", flush=True)
            subprocess.Popen(f"{OLLAMA_BIN} serve > {WORKDIR}/ollama.log 2>&1", shell=True, env=env)
            time.sleep(10)

def heartbeat_wd():
    """Print a [HEARTBEAT] line every 60s so you can SEE the proxy is alive."""
    global BEATS
    while True:
        time.sleep(60)
        BEATS += 1
        up = int(time.time() - START_TS)
        ol = "UP" if sh0(f"curl -sf http://127.0.0.1:{PORT}/api/version >/dev/null") == 0 else "DOWN"
        last = f"{int(time.time()-LAST_REQ)}s ago" if LAST_REQ else "never"
        print(f"[HEARTBEAT #{BEATS}] up {up//3600}h{up%3600//60:02d}m | ollama={ol} | "
              f"reqs={REQ_COUNT} (last: {last}) | model={MODEL}", flush=True)

for fn in (ollama_wd, proxy_wd, tunnel_wd, url_mon, heartbeat_wd):
    threading.Thread(target=fn, daemon=True).start()

print("\nALL DONE - PROXY v13 running (32k ctx + think/reasoning split + heartbeat). Cell stays alive.", flush=True)
print("Heartbeat: a [HEARTBEAT #N] line prints every 60s; also check <url>/heartbeat", flush=True)
while True:
    time.sleep(30)
