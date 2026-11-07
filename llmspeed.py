#!/usr/bin/env python3
"""llmspeed - intelligence (y) against output tokens/s (x) for every model artificial analysis measures. red = api, green = open weights."""
import email.message, flask, flask_caching, flask_limiter, flask_limiter.util as util, json, logging, logli, math, os, re, requests, secrets, smtplib, sqlite3, threading, time, urllib.parse, werkzeug.middleware.proxy_fix, yaml

CFG = yaml.safe_load(open("llmspeed.yaml"))
DB = os.environ.get("DB", "/data/llmspeed.db" if os.path.isdir("/data") else "llmspeed.db")
SMTP = {k: os.environ.get("SMTP_" + k) for k in ("HOST", "USER", "PASS")}
URL = os.environ.get("URL", "https://llmspeed.problem.li")
AA = os.environ.get("AA_KEY", "").strip() or None  # artificialanalysis.ai free data api key - without it there is no data
W, H, L, R, T, B = 960, 540, 44, 16, 14, 34  # chart box
MSG = dict(inbox="check your inbox", bad="that is not a mail address", ok="watching - you get a mail when a new model lands inside", bye="stopped watching")

app = flask.Flask(__name__, template_folder=".")
app.wsgi_app = werkzeug.middleware.proxy_fix.ProxyFix(app.wsgi_app, x_for=1, x_proto=1)  # kamal-proxy in front: rate limit per client, not per proxy
logli.setup("llmspeed")
cache = flask_caching.Cache(app, config={"CACHE_TYPE": "SimpleCache"})
limiter = flask_limiter.Limiter(key_func=util.get_remote_address, app=app, storage_uri="memory://")
logging.info("boot")


def db():
    c = sqlite3.connect(DB, isolation_level=None)
    c.executescript("create table if not exists kv(k text primary key, v text, ts real);"
                    "create table if not exists watch(token text primary key, email text, ok int, name text, tps0 real, tps1 real, ii0 real, ii1 real, ts real);"
                    "create table if not exists seen(id text primary key, ts real); drop table if exists subs")
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


def family(name):
    """'GPT-6.1 Sol (High)' -> 'GPT-6.1 Sol': the variants of a family differ in reasoning effort"""
    return re.sub(r"\s*\(.*?\)\s*", " ", name).strip()


def openrouter():
    """open weights flag + id by name, from openrouter's model list (no key)"""
    out = {}
    try:
        for m in requests.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]:
            out.setdefault(norm(re.sub(r" \((free|batch)\)$", "", m["name"].split(": ", 1)[-1])), dict(hf=bool(m.get("hugging_face_id")), id=m["id"].split(":")[0]))
    except Exception as e:
        logging.warning(f"openrouter: {e}")
    return out


def hf_open(name, creator):
    """not on openrouter: is there a hugging face repo of that name under that creator? a lossy open-weights guess, cached."""
    v, _ = kv("hf:" + name)
    if v is not None:
        return v
    try:
        j = requests.get("https://huggingface.co/api/models", params=dict(search=name, limit=10), timeout=30).json()
        v = any(norm(name) in norm(m["id"]) and norm(creator)[:5] in norm(m["id"].split("/")[0]) for m in j)
    except Exception as e:
        logging.info(f"hf {name}: {e}")
        v = False
    return kv("hf:" + name, v)


def arena():
    """arena.ai model slugs, scraped from the direct-chat page's react payload. on failure keep the last list."""
    try:
        t = requests.get("https://arena.ai/text/direct", headers={"RSC": "1", "User-Agent": "Mozilla/5.0 llmspeed"}, timeout=30).text
        names = sorted(set(re.findall(r'"publicName":"([^"]+)"', t)))
        return kv("arena", names) if names else kv("arena")[0] or []
    except Exception as e:
        logging.warning(f"arena: {e}")
        return kv("arena")[0] or []


def aa():
    """artificial analysis free data api: every language model, all pages"""
    out, page = [], 1
    while AA:
        j = requests.get("https://artificialanalysis.ai/api/v2/language/models/free", params={"page": page}, headers={"x-api-key": AA}, timeout=60).json()
        out += j.get("data", [])
        if not j.get("pagination", {}).get("has_more"):
            break
        page += 1
    return out


LOCK = threading.Lock()


def refresh(force=False):
    with LOCK:
        if not force and kv("models")[1] < 3600:
            return
        _refresh()


def _refresh():
    """every artificial analysis model with a speed and an intelligence index (or pinned), open weights looked up on openrouter / hugging face"""
    rows, orm, slugs, models, pins = aa(), openrouter(), arena(), {}, set(CFG.get("pin") or [])
    for m in rows:
        tps, ii = (m.get("performance") or {}).get("median_output_tokens_per_second"), (m.get("evaluations") or {}).get("artificial_analysis_intelligence_index")
        if not tps or (ii is None and m["slug"] not in pins):
            continue
        fam, creator, p = family(m["name"]), (m.get("model_creator") or {}).get("name") or "", m.get("pricing") or {}
        o = orm.get(norm(m["name"])) or orm.get(norm(fam))
        r = dict(slug=m["slug"], name=m["name"].replace(", Default Fallback", ""), family=fam, creator=creator, created=m.get("release_date") or "", ii=ii, tps=tps,
                 inp=p.get("price_1m_input_tokens"), out=p.get("price_1m_output_tokens"), hf=o["hf"] if o else hf_open(fam, creator), extra=False)
        pat = re.compile(re.escape(slug(fam)) + r"(-[a-z].*)?")  # gpt-6-luna ~ gpt-6-luna-max
        a = min((n for n in slugs if pat.fullmatch(slug(n))), key=len, default=None)
        r["url"] = f"https://arena.ai/text/direct?model_a={a}" if a else f"https://openrouter.ai/chat?models={o['id']}" if o else f"https://artificialanalysis.ai/models/{m['slug']}"
        models[m["slug"]] = r
    for x in CFG.get("extra") or []:
        if x.get("tps"):
            models["x:" + x["name"]] = dict(slug="x:" + x["name"], name=x["name"], family=x["name"], creator=x.get("creator") or "", created="", ii=x.get("ii"), tps=x["tps"],
                                             inp=None, out=None, hf=bool(x.get("open")), url=x.get("url") or "https://arena.ai/text/direct", extra=True)
    if not models:
        return logging.warning("refresh: no models - is AA_KEY set?")
    c, now = db(), time.time()
    old = {i for (i,) in c.execute("select id from seen")}
    new = [r for r in models.values() if r["slug"] not in old]
    c.executemany("insert into seen values(?,?)", [(r["slug"], now) for r in new])
    if old and new:
        notify(new)
    kv("models", list(models.values()))
    cache.clear()
    logging.info(f"refresh: {len(rows)} aa models, {len(models)} shown, {sum(1 for r in models.values() if r['hf'])} open, {len(new)} new, {len(orm)} openrouter, {len(slugs)} arena slugs")


def mail(to, subject, body):
    if not all(SMTP.values()):
        return logging.warning(f"no smtp, dropped mail to @{to.split('@')[-1]}: {subject}")
    m = email.message.EmailMessage()
    m["From"], m["To"], m["Subject"] = SMTP["USER"], to, subject
    m.set_content(body)
    with smtplib.SMTP(SMTP["HOST"], 587, timeout=30) as s:
        s.starttls()
        s.login(SMTP["USER"], SMTP["PASS"])
        s.send_message(m)


def inside(w, r):
    return w["tps0"] <= r["tps"] <= w["tps1"] and w["ii0"] <= (r["ii"] or 0) <= w["ii1"]


def line(r):
    return f"- {r['name']} ({r['creator']}{', open weights' if r['hf'] else ''}): {r['tps']:.0f} tokens/s, intelligence {r['ii'] if r['ii'] is not None else '?'}"


def notify(new):
    """every confirmed watcher whose region a new model landed in gets one mail"""
    for t, e, name, tps0, tps1, ii0, ii1 in db().execute("select token, email, name, tps0, tps1, ii0, ii1 from watch where ok=1").fetchall():
        hits = [r for r in new if inside(dict(tps0=tps0, tps1=tps1, ii0=ii0, ii1=ii1), r)]
        if hits:
            try:
                mail(e, f"llmspeed: {len(hits)} new in {name}", f"new inside '{name}' on {URL}:\n\n" + "\n".join(map(line, hits)) + f"\n\nstop watching: {URL}/bye/{t}")
            except Exception as x:
                logging.error(f"mail @{e.split('@')[-1]}: {x}")


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
def rows():
    ms, fortnight = kv("models")[0] or [], time.strftime("%Y-%m-%d", time.gmtime(time.time() - 14 * 86400))
    for r in ms:
        r["new"] = r["created"] > fortnight  # iso dates compare as strings
    return sorted(ms, key=lambda r: -(r["ii"] if r["ii"] is not None else -1))


def chart(rs, regions):
    """x = log10 tokens/s, y = intelligence index, unrated models sit on the x axis. labels on the pareto front: fast and smart."""
    xs = [math.log10(p["tps"]) for p in rs] + [math.log10(v) for w in regions for v in (w["tps0"], w["tps1"])] or [1, 3]
    x0, x1 = math.floor(min(xs)), round(max(xs) + 0.08, 2)  # left edge a decade, right edge just past the fastest model
    y1 = min(100, 10 * math.ceil((max([p["ii"] or 0 for p in rs] + [w["ii1"] for w in regions] + [50]) + 8) / 10))
    X, Y = lambda v: round(L + (math.log10(v) - x0) / (x1 - x0) * (W - L - R), 1), lambda v: round(H - B - v / y1 * (H - T - B), 1)
    for p in rs:
        p["cx"], p["cy"], p["r"] = X(p["tps"]), Y(p["ii"] or 0), 6
    for w in regions:
        w["x"], w["y"], w["w"], w["h"] = X(w["tps0"]), Y(w["ii1"]), round(X(w["tps1"]) - X(w["tps0"]), 1), round(Y(w["ii0"]) - Y(w["ii1"]), 1)
    best = -1
    for p in sorted(rs, key=lambda p: -p["tps"]):  # fast first
        p["label"] = p["ii"] is None or p["ii"] > best or p.get("extra")
        best = max(best, p["ii"] or -1)
    boxes = []  # label slots: right of the dot, else below, else above, else no label
    for p in sorted((p for p in rs if p["label"]), key=lambda p: -(p["ii"] or 0)):
        w, p["label"] = 5.5 * len(p["name"]) + 4, False
        for dy in (4, 15, -8):
            x, y = p["cx"] + p["r"] + 2, p["cy"] + dy
            x = x if x + w < W - R else p["cx"] - p["r"] - 2 - w
            if all(x > b[2] or x + w < b[0] or y - 9 > b[3] or y + 2 < b[1] for b in boxes):
                p["lx"], p["ly"], p["label"] = round(x, 1), round(y, 1), True
                boxes.append((x, y - 9, x + w, y + 2))
                break
    fams = {}
    for p in rs:
        if p["ii"] is not None:
            fams.setdefault(p["family"], []).append(p)
    lines = [("#2a9d3f" if g[0]["hf"] else "#e01b24", " ".join(f"{p['cx']},{p['cy']}" for p in sorted(g, key=lambda p: p["cx"]))) for g in fams.values() if len(g) > 1]
    return dict(x0=x0, x1=x1, y1=y1, lines=lines, xt=[(f"{10 ** e:,}", X(10 ** e)) for e in range(x0, math.floor(x1) + 1)],
                yt=[(v, Y(v)) for v in range(0, y1 + 1, 10 if y1 <= 60 else 20)])


def cookie_regions():
    try:
        return [dict(t=str(w["t"])[:40], name=str(w["name"])[:30], tps0=max(0.1, float(w["tps0"])), tps1=max(0.1, float(w["tps1"])), ii0=max(0.0, float(w["ii0"])), ii1=min(100.0, float(w["ii1"])))
                for w in json.loads(flask.request.cookies.get("regions", "[]"))][:10]
    except (ValueError, TypeError, KeyError):
        return []


@app.before_request
def same_origin():
    o = flask.request.origin
    if flask.request.method == "POST" and o and urllib.parse.urlparse(o).netloc != flask.request.host:
        flask.abort(403)


@app.route("/")
@limiter.limit("70/minute, 1000/hour")
def index():
    rs, regions = [dict(r) for r in rows()], cookie_regions()
    return flask.render_template("llmspeed.html", cfg=CFG, rs=rs, regions=regions, g=chart(rs, regions), msg=MSG.get(flask.request.args.get("m")), age=kv("models")[1],
                                 W=W, H=H, L=L, R=R, T=T, B=B)


@app.post("/watch")
@limiter.limit("5/hour")
def watch():
    """a region of the chart + a mail address: double opt-in, then a mail whenever a new model lands inside"""
    f, resp = flask.request.form, flask.redirect("/?m=inbox")
    if f.get("clear"):
        db().executemany("delete from watch where token=?", [(w["t"],) for w in cookie_regions()])
        resp = flask.redirect("/")
        resp.delete_cookie("regions")
        return resp
    e = f.get("email", "").strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", e):
        return flask.redirect("/?m=bad")
    try:
        a, b, c, d = (float(f[k]) for k in ("tps0", "tps1", "ii0", "ii1"))
    except (KeyError, ValueError):
        flask.abort(400)
    w = dict(t=secrets.token_urlsafe(16), name=f.get("name", "").strip()[:30] or "my region", tps0=max(0.1, min(a, b)), tps1=max(0.1, a, b), ii0=max(0.0, min(c, d)), ii1=min(100.0, max(c, d)))
    db().execute("insert into watch values(?,?,0,?,?,?,?,?,?)", (w["t"], e, w["name"], w["tps0"], w["tps1"], w["ii0"], w["ii1"], time.time()))
    now = [r for r in rows() if inside(w, r)]
    mail(e, "llmspeed: confirm", f"click to get a mail whenever a new model lands inside '{w['name']}' ({w['tps0']:g}-{w['tps1']:g} tokens/s, intelligence {w['ii0']:g}-{w['ii1']:g}) on {URL}:\n\n"
         f"{URL}/ok/{w['t']}\n\ninside right now:\n" + ("\n".join(map(line, now)) or "- nothing yet") + "\n\nnot you? just ignore this.")
    logging.info(f"watch @{e.split('@')[1]}: {w['name']}, {len(now)} inside")
    resp.set_cookie("regions", json.dumps((cookie_regions() + [w])[-10:]), max_age=365 * 86400, samesite="Lax")
    return resp


@app.route("/ok/<t>")
def ok(t):
    db().execute("update watch set ok=1 where token=?", (t,))
    return flask.redirect("/?m=ok")


@app.route("/bye/<t>")
def bye(t):
    db().execute("delete from watch where token=?", (t,))
    return flask.redirect("/?m=bye")


@app.route("/favicon.svg")
@app.route("/llmspeed.svg")
def favicon():
    return flask.send_from_directory(app.root_path, "llmspeed.svg")


@app.route("/up")
def up():
    return "OK", 200


if __name__ == "__main__":
    app.run(port=8080, debug=True, use_reloader=False)
