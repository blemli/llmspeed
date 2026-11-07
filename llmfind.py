#!/usr/bin/env python3
"""llmfind - which llm? accuracy (y) vs cost (x), dot size = speed. red = api, green = open weights, priced by graphics card write-off."""
import email.message, flask, flask_caching, flask_limiter, flask_limiter.util as util, json, logging, logli, math, os, re, requests, secrets, smtplib, sqlite3, threading, time, yaml

CFG = yaml.safe_load(open("llmfind.yaml"))
DB = os.environ.get("DB", "/data/llmfind.db" if os.path.isdir("/data") else "llmfind.db")
SMTP = {k: os.environ.get("SMTP_" + k) for k in ("HOST", "USER", "PASS")}
URL = os.environ.get("URL", "https://llmfind.problem.li")
AA = os.environ.get("AA_KEY", "").strip() or None  # optional artificialanalysis.ai key: speed (dot size) + missing intelligence indices
W, H, L, R, T, B = 960, 540, 44, 16, 14, 34  # chart box

app = flask.Flask(__name__, template_folder=".")
logli.setup("llmfind")
cache = flask_caching.Cache(app, config={"CACHE_TYPE": "SimpleCache"})
limiter = flask_limiter.Limiter(key_func=util.get_remote_address, app=app, storage_uri="memory://")
app.template_filter("date")(lambda t: time.strftime("%Y-%m", time.gmtime(t)))
logging.info("boot")


def db():
    c = sqlite3.connect(DB, isolation_level=None)
    c.executescript("create table if not exists kv(k text primary key, v text, ts real);"
                    "create table if not exists subs(email text primary key, token text, ok int, ts real);"
                    "create table if not exists seen(id text primary key, ts real)")
    return c


def kv(k, v=None):
    """tiny json store: kv(k) -> (value, age in seconds), kv(k, v) stores and returns v"""
    if v is not None:
        db().execute("replace into kv values(?,?,?)", (k, json.dumps(v), time.time()))
        return v
    r = db().execute("select v, ts from kv where k=?", (k,)).fetchone()
    return (json.loads(r[0]), time.time() - r[1]) if r else (None, 1e9)


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def pick(c, *keys):
    """first of keys anywhere in a nested config"""
    for k in keys:
        if k in c:
            return c[k]
    for v in c.values():
        if isinstance(v, dict) and (r := pick(v, *keys)):
            return r


def params(hf):
    """total & active parameters of a hugging face repo: safetensors + moe config, the name knows best (Qwen3.5-397B-A17B). lossy."""
    p, _ = kv("p:" + hf)
    if p:
        return p
    s, total, active = hf.lower(), None, None
    try:
        j = requests.get(f"https://huggingface.co/api/models/{hf}", params={"expand[]": "safetensors"}, timeout=30).json()
        total = (j.get("safetensors") or {}).get("total")
        c = requests.get(f"https://huggingface.co/{hf}/raw/main/config.json", timeout=30).json()  # 401 on gated repos -> dense, lossy
        k, n = pick(c, "num_experts_per_tok", "num_experts_per_token", "moe_top_k", "moe_topk"), pick(c, "n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts")
        active = total * k / n * 1.5 if total and k and n else total  # x1.5: attention + shared experts always fire
    except Exception as e:
        logging.info(f"hf {hf}: {e}")
    if m := re.search(r"(\d+(?:\.\d+)?)([bt])-?a(\d+(?:\.\d+)?)b", s):  # 397B-A17B
        total, active = float(m[1]) * (1e12 if m[2] == "t" else 1e9), float(m[3]) * 1e9
    elif m := re.search(r"(\d+(?:\.\d+)?)b-\d+e", s):  # Llama-4-Maverick-17B-128E
        active = float(m[1]) * 1e9
    elif not total and (m := re.search(r"(\d+(?:\.\d+)?)b", s)):
        total = float(m[1]) * 1e9
    p = dict(total=total, active=active or total)
    return kv("p:" + hf, p) if total else p


def hour(g):
    """usd per card-hour: write-off + electricity, times box overhead, divided by utilization"""
    return (g["price"] / (CFG["writeoff_years"] * 8760) + g["watts"] / 1000 * CFG["kwh_usd"]) * CFG["overhead"] / CFG["utilization"]


def gpu_cost(p):
    """usd per 1M input/output tokens on the cheapest card that holds the fp8 weights. a floor, not a quote."""
    best = None
    for g in CFG["gpus"]:
        cards = math.ceil(p["total"] * CFG["bytes_per_param"] / (g["vram_gb"] * 1e9))
        if cards > CFG["max_cards"]:
            continue
        out = hour(g) / (g["tflops"] * 1e12 * CFG["mfu_decode"] / (2 * p["active"]) * 3600) * 1e6
        if not best or out < best["out"]:
            best = dict(inp=out * CFG["mfu_decode"] / CFG["mfu_prefill"], out=out, gpu=f'{cards}× {g["name"]}')
    return best


def arena():
    """arena.ai model slugs, scraped from the direct-chat page's react payload. on failure keep the last list."""
    try:
        t = requests.get("https://arena.ai/text/direct", headers={"RSC": "1", "User-Agent": "Mozilla/5.0 llmfind"}, timeout=30).text
        names = sorted(set(re.findall(r'"publicName":"([^"]+)"', t)))
        return kv("arena", names) if names else kv("arena")[0] or []
    except Exception as e:
        logging.warning(f"arena: {e}")
        return kv("arena")[0] or []


def speeds():
    """artificial analysis free api, by normalized name: median output tokens/s and intelligence index. needs AA_KEY."""
    out, page = {}, 1
    try:
        while AA:
            j = requests.get("https://artificialanalysis.ai/api/v2/language/models/free", params={"page": page}, headers={"x-api-key": AA}, timeout=60).json()
            for m in j.get("data", []):
                out.setdefault(norm(re.sub(r"\(.*?\)", "", m["name"])), dict(tps=(m.get("performance") or {}).get("median_output_tokens_per_second"),
                                                                           ii=(m.get("evaluations") or {}).get("artificial_analysis_intelligence_index")))
            if not j.get("pagination", {}).get("has_more"):
                break
            page += 1
    except Exception as e:
        logging.warning(f"aa: {e}")
    return out


LOCK = threading.Lock()


def refresh(force=False):
    """openrouter: every model with an artificial analysis index. open weights get their cost from the card, api models from the price list."""
    with LOCK:
        if not force and kv("models")[1] < 3600:
            return
        _refresh()


def _refresh():
    d = requests.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]
    aa, slugs, models = speeds(), arena(), {}
    for m in sorted(d, key=lambda m: ":" in m["id"]):  # plain ids before :free / :batch variants
        base, b = m["id"].split(":")[0], (m.get("benchmarks") or {}).get("artificial_analysis") or {}
        if base in models or not any(b.get(k) is not None for k in ("intelligence_index", "coding_index")):
            continue
        r = dict(id=base, name=re.sub(r" \((free|batch)\)$", "", m["name"].split(": ", 1)[-1]), creator=base.split("/")[0], created=m["created"], hf=m.get("hugging_face_id") or "",
                 idx=b, inp=float(m["pricing"]["prompt"]) * 1e6, out=float(m["pricing"]["completion"]) * 1e6, gpu=None, tps=None)
        if r["hf"] and (p := params(r["hf"]))["total"] and p["active"]:
            r.update(gpu_cost(p) or {}, **p)
        if a := aa.get(norm(r["name"])):
            r["tps"] = a["tps"]
            r["idx"]["intelligence_index"] = r["idx"].get("intelligence_index") or a["ii"]
        pat = re.compile(re.escape(slug(re.sub(r"-\d{4}$", "", base.split("/")[1]))) + r"(-[a-z].*)?")  # gpt-5.5 ~ gpt-5-5-high
        r["arena"] = min((n for n in slugs if pat.fullmatch(slug(n))), key=len, default=None)
        r["url"] = f"https://arena.ai/text/direct?model_a={r['arena']}" if r["arena"] else f"https://openrouter.ai/chat?models={base}"
        models[base] = r
    c, now = db(), time.time()
    old = {i for (i,) in c.execute("select id from seen")}
    new = [r for r in models.values() if r["id"] not in old]
    c.executemany("insert into seen values(?,?)", [(r["id"], now) for r in new])
    if old and new:
        notify(new)
    kv("models", list(models.values()))
    cache.clear()
    logging.info(f"refresh: {len(models)} models, {sum(1 for r in models.values() if r['gpu'])} on cards, {len(new)} new, {len(slugs)} arena slugs, {len(aa)} aa")


def mail(to, subject, body):
    if not all(SMTP.values()):
        return logging.warning(f"no smtp, dropped mail to {to}: {subject}")
    m = email.message.EmailMessage()
    m["From"], m["To"], m["Subject"] = SMTP["USER"], to, subject
    m.set_content(body)
    with smtplib.SMTP(SMTP["HOST"], 587, timeout=30) as s:
        s.starttls()
        s.login(SMTP["USER"], SMTP["PASS"])
        s.send_message(m)


def notify(new):
    lines = "\n".join(f"- {r['name']} ({r['creator']}{', open weights' if r['hf'] else ''})" for r in new)
    for e, t in db().execute("select email, token from subs where ok=1"):
        try:
            mail(e, f"llmfind: {len(new)} new model{'s' if len(new) > 1 else ''}", f"new on {URL}:\n\n{lines}\n\nunsubscribe: {URL}/bye/{t}")
        except Exception as x:
            logging.error(f"mail {e}: {x}")


def loop():
    while True:
        if kv("models")[1] > 86400:
            try:
                refresh()
            except Exception as e:
                logging.error(f"refresh: {e}")
        time.sleep(600)


threading.Thread(target=loop, daemon=True).start()


@cache.memoize(600)
def rows(s):
    """models that have the scenario's index, cost blended at the scenario's in:out token mix"""
    sc = CFG["scenarios"][s]
    ti, to = sc["tokens"]
    out = []
    for r in kv("models")[0] or []:
        acc, cost = r["idx"].get(sc["index"]), (r["inp"] * ti + r["out"] * to) / (ti + to)
        if acc is not None and cost > 0:
            out.append(dict(r, acc=acc, cost=cost, new=time.time() - r["created"] < 14 * 86400))
    return sorted(out, key=lambda r: -r["acc"])


def chart(rs, dots):
    """pixel positions: x = log10 cost, y = index. labels only on the pareto front - best accuracy for the money."""
    pts = rs + dots
    xs = [math.log10(p["cost"]) for p in pts] or [-1, 1]
    x0, x1 = math.floor(min(xs)), math.ceil(max(xs) + 0.3)
    y1 = min(100, 10 * math.ceil((max([p["acc"] for p in pts] + [50]) + 8) / 10))
    mx = max([p["tps"] or 0 for p in pts] + [1])
    for p in pts:
        p["cx"] = round(L + (math.log10(p["cost"]) - x0) / (x1 - x0) * (W - L - R), 1)
        p["cy"] = round(H - B - p["acc"] / y1 * (H - T - B), 1)
        p["r"] = round(3 + 11 * math.sqrt(p["tps"] / mx), 1) if p.get("tps") else 6
    best = -1
    for p in sorted(rs, key=lambda p: p["cost"]):
        p["label"], best = p["acc"] > best, max(best, p["acc"])
    boxes = []  # label slots: right of the dot, else below, else above, else no label. your own dots always get one.
    for p in dots + sorted((p for p in rs if p["label"]), key=lambda p: -p["acc"]):
        w, p["label"] = 5.5 * len(p["name"]) + 4, p in dots
        for dy in (4, 15, -8):
            x, y = p["cx"] + p["r"] + 2, p["cy"] + dy
            x = x if x + w < W - R else p["cx"] - p["r"] - 2 - w
            if all(x > b[2] or x + w < b[0] or y - 9 > b[3] or y + 2 < b[1] for b in boxes):
                p["lx"], p["ly"], p["label"] = round(x, 1), round(y, 1), True
                boxes.append((x, y - 9, x + w, y + 2))
                break
        p.setdefault("lx", round(p["cx"] + p["r"] + 2, 1)), p.setdefault("ly", round(p["cy"] + 4, 1))
    return dict(x0=x0, x1=x1, y1=y1, xt=[(10 ** e, round(L + (e - x0) / (x1 - x0) * (W - L - R), 1)) for e in range(x0, x1 + 1)],
                yt=[(v, round(H - B - v / y1 * (H - T - B), 1)) for v in range(0, y1 + 1, 10 if y1 <= 60 else 20)])


def cookie_dots():
    try:
        return [d for d in json.loads(flask.request.cookies.get("dots", "[]")) if "acc" in d and "cost" in d][:20]
    except ValueError:
        return []


@app.route("/")
@app.route("/<s>")
@limiter.limit("70/minute, 1000/hour")
def index(s="chat"):
    if s not in CFG["scenarios"]:
        flask.abort(404)
    rs, dots = [dict(r) for r in rows(s)], [d for d in cookie_dots() if d.get("s") == s]
    return flask.render_template("llmfind.html", s=s, sc=CFG["scenarios"][s], cfg=CFG, rs=rs, dots=dots, g=chart(rs, dots), hour=hour,
                                 msg=flask.request.args.get("m"), age=kv("models")[1], W=W, H=H, L=L, R=R, T=T, B=B)


@app.post("/dot")
@limiter.limit("30/minute")
def dot():
    f = flask.request.form
    s = f.get("s") if f.get("s") in CFG["scenarios"] else "chat"
    resp = flask.redirect("/" + s)
    if f.get("clear"):
        resp.delete_cookie("dots")
        return resp
    try:
        dots = cookie_dots() + [dict(s=s, name=f["name"][:30], acc=min(100, max(0, float(f["acc"]))), cost=max(1e-4, float(f["cost"])), tps=float(f["tps"]) if f.get("tps") else None)]
    except (KeyError, ValueError):
        flask.abort(400)
    resp.set_cookie("dots", json.dumps(dots[-20:]), max_age=365 * 86400, samesite="Lax")
    return resp


@app.post("/subscribe")
@limiter.limit("5/hour")
def subscribe():
    e, s = flask.request.form.get("email", "").strip().lower(), flask.request.form.get("s", "chat")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", e):
        return flask.redirect(f"/{s}?m=that is not a mail address")
    c = db()
    c.execute("insert or ignore into subs values(?,?,0,?)", (e, secrets.token_urlsafe(16), time.time()))
    t, = c.execute("select token from subs where email=?", (e,)).fetchone()
    mail(e, "llmfind: confirm", f"click to get a mail whenever a new model shows up on {URL}:\n\n{URL}/ok/{t}\n\nnot you? just ignore this.")
    logging.info(f"subscribe {e}")
    return flask.redirect(f"/{s}?m=check your inbox")


@app.route("/ok/<t>")
def ok(t):
    db().execute("update subs set ok=1 where token=?", (t,))
    return flask.redirect("/chat?m=subscribed - you get a mail when a new model appears")


@app.route("/bye/<t>")
def bye(t):
    db().execute("delete from subs where token=?", (t,))
    return flask.redirect("/chat?m=unsubscribed")


@app.route("/favicon.svg")
@app.route("/llmfind.svg")
def favicon():
    return flask.send_from_directory(app.root_path, "llmfind.svg")


@app.route("/up")
def up():
    return "OK", 200


if __name__ == "__main__":
    app.run(port=8080, debug=True, use_reloader=False)
