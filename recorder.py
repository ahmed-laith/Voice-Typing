"""Voice sample recorder for Voice Typing.

Shows one sentence at a time in the browser; you read it the way you normally talk, and each take is
saved as samples/<id>.wav with the exact text you said in samples/manifest.json. benchmark.py then
replays these recordings through different models and settings to find what understands you best.

Recording uses the same microphone, sample rate and code path as the app. Everything stays local:
the page is served on 127.0.0.1 only and the files never leave this folder.
"""
from __future__ import annotations

import json
import secrets
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import sounddevice as sd

import voice_typing as vt

SAMPLES = vt.APP_DIR / "samples"
SENTENCES = vt.APP_DIR / "training_sentences.txt"
CUSTOM = SAMPLES / "custom_sentences.txt"
MANIFEST = SAMPLES / "manifest.json"
TOKEN = secrets.token_urlsafe(12)  # only this page may drive the recorder


def load_items() -> list[dict]:
    items, category = [], ""
    for line in SENTENCES.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("#"):
            category = line.lstrip("# ").strip()
        elif line:
            items.append({"id": f"s{len(items) + 1:03d}", "text": line, "category": category})
    if CUSTOM.exists():
        for i, line in enumerate(l.strip() for l in CUSTOM.read_text(encoding="utf-8-sig").splitlines() if l.strip()):
            items.append({"id": f"c{i + 1:03d}", "text": line, "category": "جملك أنت"})
    return items


class Recorder:
    def __init__(self):
        SAMPLES.mkdir(exist_ok=True)
        self.cfg = vt.load_config()
        self.items = load_items()
        self.manifest: dict = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
        self.lock = threading.Lock()
        self.stream = None
        self.frames: list = []
        self.sr = vt.SAMPLE_RATE
        self.level = 0.0
        self.t0 = 0.0
        self.current: str | None = None

    def _save_manifest(self):
        tmp = MANIFEST.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(MANIFEST)

    def state(self):
        return {
            "items": [{**it, "done": it["id"] in self.manifest,
                       "said": self.manifest.get(it["id"], {}).get("text", it["text"])} for it in self.items],
            "recording": self.current,
            "level": self.level,
            "elapsed": time.time() - self.t0 if self.current else 0,
        }

    def _cb(self, indata, frames, t, status):
        chunk = indata[:, 0].copy()
        self.frames.append(chunk)
        self.level = float(np.sqrt(np.mean(chunk ** 2)))

    def start(self, item_id: str):
        with self.lock:
            if self.current:
                return
            dev = self.cfg["input_device"]
            self.frames = []
            try:
                self.sr = vt.SAMPLE_RATE
                self.stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32", device=dev, callback=self._cb)
            except Exception:
                self.sr = int(sd.query_devices(dev, "input")["default_samplerate"])
                self.stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32", device=dev, callback=self._cb)
            self.stream.start()
            self.t0, self.current = time.time(), item_id

    def stop(self, said: str):
        with self.lock:
            if not self.current:
                return {"error": "not recording"}
            self.stream.stop()
            self.stream.close()
            item_id, self.current, self.stream, self.level = self.current, None, None, 0.0
            audio = vt.to_16k(np.concatenate(self.frames) if self.frames else np.zeros(0, np.float32), self.sr)
            secs = len(audio) / vt.SAMPLE_RATE
            if secs < 0.5:
                return {"error": "short", "secs": secs}
            vt.save_wav(SAMPLES / f"{item_id}.wav", audio)
            item = next(it for it in self.items if it["id"] == item_id)
            loud = vt.loudest_seconds_rms(audio)
            peak = float(np.max(np.abs(audio)))
            self.manifest[item_id] = {"file": f"{item_id}.wav", "text": said.strip() or item["text"],
                                      "original": item["text"], "category": item["category"], "lang": "ar",
                                      "secs": round(secs, 2), "loud_rms": round(loud, 4), "peak": round(peak, 3),
                                      "recorded": time.strftime("%Y-%m-%d %H:%M:%S")}
            self._save_manifest()
            quality = "quiet" if loud < vt.MIN_SPEECH_RMS * 2.5 else "clipped" if peak > 0.98 else "ok"
            return {"ok": True, "secs": round(secs, 1), "quality": quality}

    def set_said(self, item_id: str, said: str):
        with self.lock:
            if item_id in self.manifest and said.strip():
                self.manifest[item_id]["text"] = said.strip()
                self._save_manifest()

    def add_custom(self, text: str) -> str:
        with self.lock:
            text = " ".join(text.split())
            with CUSTOM.open("a", encoding="utf-8") as f:
                f.write(text + "\n")
            self.items = load_items()
            return self.items[-1]["id"]


REC: Recorder
_auto_lock = threading.Lock()


def _auto_manifest() -> Path:
    return vt.AUTO_DIR / "manifest.json"


def load_auto() -> dict:
    p = _auto_manifest()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_auto(data: dict):
    p = _auto_manifest()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def suggest_rules(hyp: str, text: str) -> list[dict]:
    """Word-level differences between what the app wrote and what you corrected it to, as
    'wrong => right' rules for corrections.txt (one or two words each side)."""
    import difflib
    import re

    def words(s):
        return re.sub(r"[،,.!?؟:;\"()«»]", " ", s).split()

    def sim(x, y):
        return difflib.SequenceMatcher(None, x, y).ratio()

    def align(xs, ys):
        """Pairs words of two changed blocks (1:1, 2:1, 1:2) so that similar words end up together."""
        n, m = len(xs), len(ys)
        best = {(0, 0): (0.0, [])}
        for i in range(n + 1):
            for j in range(m + 1):
                if (i, j) not in best:
                    continue
                cost, segs = best[(i, j)]
                moves = []
                if i < n and j < m:
                    moves.append((1, 1, 1 - sim(xs[i], ys[j])))
                if i + 1 < n and j < m:
                    moves.append((2, 1, 1 - sim(xs[i] + xs[i + 1], ys[j])))
                if i < n and j + 1 < m:
                    moves.append((1, 2, 1 - sim(xs[i], ys[j] + ys[j + 1])))
                if i < n:
                    moves.append((1, 0, 1.0))
                if j < m:
                    moves.append((0, 1, 1.0))
                for di, dj, c in moves:
                    key = (i + di, j + dj)
                    if key not in best or cost + c < best[key][0]:
                        best[key] = (cost + c, segs + [(i, i + di, j, j + dj, c)])
        return best[(n, m)][1]

    a, b = words(hyp), words(text)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != "replace" or max(i2 - i1, j2 - j1) > 8:
            continue
        for s1, s2, t1, t2, cost in align(a[i1:i2], b[j1:j2]):
            if not (s2 > s1 and t2 > t1) or cost > 0.6:  # skip insertions, deletions, unrelated words
                continue
            wrong, right = " ".join(a[i1:i2][s1:s2]), " ".join(b[j1:j2][t1:t2])
            rule = {"wrong": wrong, "right": right}
            if wrong != right and wrong.strip() and right.strip() and rule not in out:
                out.append(rule)
    return out


def add_rule(wrong: str, right: str) -> bool:
    """Appends 'wrong => right' to corrections.txt unless the same rule is already there."""
    line = f"{wrong.strip()} => {right.strip()}"
    existing = vt.CORR_PATH.read_text(encoding="utf-8-sig") if vt.CORR_PATH.exists() else ""
    if line in (l.strip() for l in existing.splitlines()):
        return False
    with vt.CORR_PATH.open("a", encoding="utf-8") as f:
        f.write(("" if existing.endswith("\n") or not existing else "\n") + line + "\n")
    return True


def review_items() -> list[dict]:
    with _auto_lock:
        data = load_auto()
    items = [{"id": k, "hyp": v.get("hyp", ""), "text": v.get("text"), "lang": v.get("lang"), "secs": v.get("secs"),
              "recorded": v.get("recorded", ""), "note": v.get("note", "")} for k, v in data.items()]
    return sorted(items, key=lambda x: x["recorded"], reverse=True)


def review_save(item_id: str, text: str) -> dict:
    with _auto_lock:
        data = load_auto()
        if item_id not in data:
            return {"error": "unknown id"}
        data[item_id]["text"] = text.strip()
        data[item_id]["text_source"] = "reviewed by you"
        save_auto(data)
        hyp = data[item_id].get("hyp", "")
    return {"ok": True, "suggestions": suggest_rules(hyp, text) if text.strip() != hyp.strip() else []}


def review_add_vocab(word: str) -> dict:
    w = word.strip()
    if not w:
        return {"error": "empty word"}
    vocab_file = vt.APP_DIR / "vocabulary.txt"
    existing = set()
    if vocab_file.exists():
        existing = {l.strip().lower() for l in vocab_file.read_text(encoding="utf-8-sig").splitlines() if l.strip()}
    if w.lower() in existing:
        return {"ok": True, "added": False, "msg": "موجود مسبقاً في القاموس"}
    with vocab_file.open("a", encoding="utf-8") as f:
        f.write(("\n" if vocab_file.exists() and not vocab_file.read_text(encoding="utf-8-sig").endswith("\n") else "") + w + "\n")
    return {"ok": True, "added": True, "msg": "أُضيف للقاموس بنجاح"}


def review_ai_action(text: str, action: str) -> dict:
    t = text.strip()
    if not t:
        return {"error": "النص فارغ"}

    key_file = vt.APP_DIR / "gemini_key.txt"
    if not key_file.exists():
        return {"error": "لم يتم العثور على مفتاح Gemini في gemini_key.txt"}
    key = key_file.read_text(encoding="utf-8").strip()
    if not key:
        return {"error": "مفتاح Gemini فارغ"}

    prompts = {
        "summary": "أنت مساعد شخصي ذكي. لخص هذا النص الصوتي باللغة العربية في 2 إلى 3 نقاط رئيسية واضحة ومباشرة بدون أي مقدمات أو استطراد:\n\n",
        "prompt": "أنت خبير هندسة برومبتات ذكاء اصطناعي (Prompt Engineer). خذ هذا الكلام الشفوي العفوي وحوله إلى برومبت تنفيذي هندسي منظم ومفصل وموجه لوكلاء الذكاء الاصطناعي (Claude Code / Antigravity) لتنفيذ المطلوب بأعلى دقة:\n\n",
        "formal": "أنت كاتب محترف. أعد صياغة هذا الكلام الشفوي العربي إلى رسالة أو بريد إلكتروني رسمي، رصين، وأنيق وخالٍ من الأخطاء وجاهز للإرسال مباشرة:\n\n",
        "polish": "أنت مصفي نصوص ذكي. أزل التأتأة والتردد وتكرار الكلمات غير المفيد من هذا النص الصوتي، واجعله نصاً طبيعياً وسلساً مع الحفاظ على لهجة المتكلم وكلماته الأصلية:\n\n"
    }
    system_prompt = prompts.get(action, prompts["summary"])

    import urllib.request
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": system_prompt + t}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024}
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            out = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return {"ok": True, "result": out}
    except Exception as e:
        return {"error": str(e)}


def review_delete(item_id: str) -> dict:
    with _auto_lock:
        data = load_auto()
        entry = data.pop(item_id, None)
        if entry:
            (vt.AUTO_DIR / entry["file"]).unlink(missing_ok=True)
            save_auto(data)
    return {"ok": entry is not None}


_rules_lock = threading.Lock()


def _rule_lines() -> list[str]:
    return vt.CORR_PATH.read_text(encoding="utf-8-sig").splitlines() if vt.CORR_PATH.exists() else []


def _write_rule_lines(lines: list[str]):
    tmp = vt.CORR_PATH.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(vt.CORR_PATH)  # the app re-reads the file when it changes


def parse_rules(lines: list[str]) -> list[dict]:
    """Rules of corrections.txt with their line numbers and the section (first comment after a blank line)."""
    out, section, candidate, prev_blank = [], "عام", None, True
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            prev_blank = True
            continue
        if s.startswith("#"):
            body = s.lstrip("#").strip()
            if prev_blank and body and "=>" not in body:
                candidate = body
            prev_blank = False
            continue
        if "=>" in s:
            if candidate:
                section, candidate = candidate, None
            wrong, right = (p.strip() for p in s.split("=>", 1))
            out.append({"line": i, "raw": raw, "wrong": wrong, "right": right, "section": section})
        prev_blank = False
    return out


def check_rule(wrong: str, right: str) -> tuple[str | None, str | None]:
    """(error, warning) for a proposed rule."""
    wrong, right = wrong.strip(), right.strip()
    if not wrong or not right:
        return "اكتب الكلمة الخاطئة والصحيحة معاً", None
    if "=>" in wrong or "=>" in right or "\n" in wrong + right:
        return "لا تستعمل => ولا سطراً جديداً داخل القاعدة", None
    if wrong == right:
        return "الكلمتان متطابقتان", None
    if len(wrong.replace(" ", "").rstrip("*")) <= 2:
        return None, "قاعدة قصيرة جداً: قد تغيّر كلمات سليمة. جرّبها في مربع التجربة أولاً"
    return None, None


def rules_state() -> dict:
    with _rules_lock:
        rules = parse_rules(_rule_lines())
    return {"rules": rules, "sections": list(dict.fromkeys(r["section"] for r in rules))}


def rules_add(wrong: str, right: str, section: str = "") -> dict:
    error, warning = check_rule(wrong, right)
    if error:
        return {"error": error}
    line = f"{wrong.strip()} => {right.strip()}"
    with _rules_lock:
        lines = _rule_lines()
        rules = parse_rules(lines)
        if any(r["raw"].strip() == line for r in rules):
            return {"added": False, "warning": "هذه القاعدة موجودة مسبقاً"}
        same = [r for r in rules if r["section"] == section]
        if same:  # keep it with its section
            lines.insert(same[-1]["line"] + 1, line)
        else:
            lines += ([""] if lines and lines[-1].strip() else []) + [line]
        _write_rule_lines(lines)
    return {"added": True, "warning": warning}


def rules_delete(line: int, raw: str) -> dict:
    with _rules_lock:
        lines = _rule_lines()
        if not (0 <= line < len(lines)) or lines[line].strip() != raw.strip():
            return {"error": "changed"}  # the file changed since the page loaded: it reloads
        del lines[line]
        _write_rule_lines(lines)
    return {"ok": True}


def rules_update(line: int, raw: str, wrong: str, right: str) -> dict:
    error, warning = check_rule(wrong, right)
    if error:
        return {"error": error}
    with _rules_lock:
        lines = _rule_lines()
        if not (0 <= line < len(lines)) or lines[line].strip() != raw.strip():
            return {"error": "changed"}
        lines[line] = f"{wrong.strip()} => {right.strip()}"
        _write_rule_lines(lines)
    return {"ok": True, "warning": warning}


def rules_test(text: str) -> dict:
    """Applies the rules one after another, like the app does, and reports which ones changed the text."""
    with _rules_lock:
        rules = parse_rules(_rule_lines())
    cur, fired = text, []
    for r in rules:
        nxt = vt.apply_corrections(cur, [r["raw"].strip()])
        if nxt != cur:
            fired.append({"rule": r["raw"].strip(), "before": cur, "after": nxt})
            cur = nxt
    return {"result": cur, "fired": fired}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE.replace("__TOKEN__", TOKEN).encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/review":
            self._send(200, REVIEW_PAGE.replace("__TOKEN__", TOKEN).encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/rules":
            self._send(200, RULES_PAGE.replace("__TOKEN__", TOKEN).encode("utf-8"), "text/html; charset=utf-8")
        elif self.path.startswith("/api/rules"):
            self._json(rules_state())
        elif self.path.startswith("/api/review"):
            self._json(review_items())
        elif self.path.startswith("/auto/"):
            f = vt.AUTO_DIR / Path(self.path.split("?")[0]).name
            if f.suffix == ".wav" and f.exists():
                self._send(200, f.read_bytes(), "audio/wav")
            else:
                self._send(404, b"", "text/plain")
        elif self.path.startswith("/api/state"):
            self._json(REC.state())
        elif self.path.startswith("/samples/"):
            name = Path(self.path.split("?")[0]).name
            f = SAMPLES / name
            if f.suffix == ".wav" and f.exists():
                self._send(200, f.read_bytes(), "audio/wav")
            else:
                self._send(404, b"", "text/plain")
        else:
            self._send(404, b"", "text/plain")

    def do_POST(self):
        if self.headers.get("X-Token") != TOKEN:
            return self._json({"error": "forbidden"}, 403)
        n = int(self.headers.get("Content-Length") or 0)
        data = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/api/start":
                REC.start(data["id"])
                self._json({"ok": True})
            elif self.path == "/api/stop":
                self._json(REC.stop(data.get("said", "")))
            elif self.path == "/api/said":
                REC.set_said(data["id"], data.get("said", ""))
                self._json({"ok": True})
            elif self.path == "/api/custom":
                self._json({"id": REC.add_custom(data["text"])})
            elif self.path == "/api/review/save":
                self._json(review_save(data["id"], data.get("text", "")))
            elif self.path == "/api/review/rule":
                self._json({"added": add_rule(data["wrong"], data["right"])})
            elif self.path == "/api/review/delete":
                self._json(review_delete(data["id"]))
            elif self.path == "/api/review/ai_action":
                self._json(review_ai_action(data.get("text", ""), data.get("action", "summary")))
            elif self.path == "/api/review/vocab":
                self._json(review_add_vocab(data.get("word", "")))
            elif self.path == "/api/rules/add":
                self._json(rules_add(data.get("wrong", ""), data.get("right", ""), data.get("section", "")))
            elif self.path == "/api/rules/delete":
                self._json(rules_delete(int(data["line"]), data.get("raw", "")))
            elif self.path == "/api/rules/update":
                self._json(rules_update(int(data["line"]), data.get("raw", ""), data.get("wrong", ""), data.get("right", "")))
            elif self.path == "/api/rules/test":
                self._json(rules_test(data.get("text", "")))
            else:
                self._json({"error": "unknown"}, 404)
        except Exception as e:
            vt.log.exception("recorder %s", self.path)
            self._json({"error": str(e)}, 500)


PAGE = r"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>تسجيل عينات صوتك</title>
<style>
  :root { --bg:#141418; --card:#1f1f26; --line:#34343f; --text:#f1f1f4; --muted:#9a9aa8;
          --yellow:#facc15; --red:#e5484d; --green:#3ecf8e; --blue:#4f9dff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font-family:"Segoe UI", Tahoma, sans-serif; }
  main { max-width: 860px; margin: 0 auto; padding: 28px 16px 60px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 18px; line-height: 1.7; }
  .progress { height: 8px; background: var(--card); border-radius: 99px; overflow: hidden; }
  .progress > div { height: 100%; background: var(--green); width: 0; transition: width .3s; }
  .count { color: var(--muted); margin: 8px 0 18px; font-size: 14px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 26px; }
  .cat { color: var(--blue); font-size: 14px; margin-bottom: 10px; }
  .sentence { font-size: 30px; line-height: 1.8; font-weight: 600; min-height: 110px; }
  .controls { display: flex; align-items: center; gap: 18px; margin-top: 18px; flex-wrap: wrap; }
  #rec { width: 84px; height: 84px; border-radius: 50%; border: 3px solid #5a5a68; background: #26262e;
         cursor: pointer; display: grid; place-items: center; flex: none; }
  #rec span { width: 34px; height: 34px; border-radius: 50%; background: var(--red); display: block; transition: background .2s; }
  #rec.on span { background: var(--yellow); animation: pulse 1s infinite; }
  @keyframes pulse { 50% { opacity: .65; } }
  .meter { flex: 1; min-width: 160px; height: 12px; background: #26262e; border-radius: 99px; overflow: hidden; }
  .meter > div { height: 100%; width: 0; background: var(--green); transition: width .08s; }
  #timer { font-variant-numeric: tabular-nums; color: var(--muted); min-width: 48px; }
  .status { margin-top: 14px; min-height: 26px; line-height: 1.7; }
  .status.ok { color: var(--green); } .status.warn { color: var(--yellow); } .status.err { color: var(--red); }
  label { display: block; color: var(--muted); font-size: 14px; margin: 18px 0 6px; }
  textarea, input { width: 100%; background: #16161b; color: var(--text); border: 1px solid var(--line);
         border-radius: 12px; padding: 12px; font: inherit; font-size: 18px; line-height: 1.7; resize: vertical; }
  .row { display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }
  button.b { background: #2b2b35; color: var(--text); border: 1px solid var(--line); border-radius: 12px;
             padding: 10px 18px; font: inherit; cursor: pointer; }
  button.b.primary { background: var(--blue); border-color: var(--blue); color: #0b1220; font-weight: 600; }
  audio { width: 100%; margin-top: 12px; }
  .dots { display: flex; flex-wrap: wrap; gap: 6px; margin: 22px 0 0; }
  .dot { width: 30px; height: 30px; border-radius: 8px; background: var(--card); border: 1px solid var(--line);
         color: var(--muted); font-size: 12px; display: grid; place-items: center; cursor: pointer; }
  .dot.done { background: #173a2c; border-color: var(--green); color: var(--green); }
  .dot.cur { outline: 2px solid var(--blue); }
  .help { color: var(--muted); font-size: 14px; line-height: 1.9; margin-top: 26px; }
  kbd { background: #2b2b35; border: 1px solid var(--line); border-radius: 6px; padding: 1px 7px; font-size: 13px; }
  .finished { display: none; background: #173a2c; border: 1px solid var(--green); border-radius: 14px;
              padding: 16px; margin-bottom: 18px; line-height: 1.8; }
</style>
</head>
<body>
<main>
  <p style="margin:0 0 10px"><a href="/review" style="color:#4f9dff">← مراجعة وتصحيح إملاءاتك الأخيرة</a> &nbsp;&nbsp; <a href="/rules" style="color:#4f9dff">← قواعد التصحيح</a></p>
  <h1>تسجيل عينات صوتك</h1>
  <p class="sub">اقرأ كل جملة <b>بطريقتك الطبيعية</b> كأنك تسولف. إذا قلتها بشكل مختلف عن المكتوب، عدّل النص تحتها ليطابق كلامك بالضبط.</p>
  <div class="finished" id="finished">خلصت كل الجمل 👏 ارجع إلى Claude واكتب له «خلصت». تقدر تضيف جمل من عندك بالأسفل إذا تحب.</div>
  <div class="progress"><div id="bar"></div></div>
  <div class="count" id="count"></div>

  <div class="card">
    <div class="cat" id="cat"></div>
    <div class="sentence" id="sentence"></div>
    <div class="controls">
      <button id="rec" title="تسجيل (Space)"><span></span></button>
      <div class="meter"><div id="lvl"></div></div>
      <div id="timer">0.0</div>
    </div>
    <div class="status" id="status">اضغط الزر أو <kbd>Space</kbd> وابدأ الكلام.</div>
    <audio id="player" controls hidden></audio>
    <label for="said">النص كما قلته فعلاً</label>
    <textarea id="said" rows="2"></textarea>
    <div class="row">
      <button class="b" id="prev">→ السابقة</button>
      <button class="b primary" id="next">التالية ← (Enter)</button>
    </div>
  </div>

  <div class="dots" id="dots"></div>

  <label for="custom">أضف جملة تقولها كثيراً (ثم سجّلها)</label>
  <div class="row" style="margin-top:0">
    <input id="custom" placeholder="مثلاً: حبيبي شوفلي هذا الملف وخبرني رأيك">
    <button class="b" id="addCustom">أضف</button>
  </div>

  <p class="help"><kbd>Space</kbd> يبدأ ويوقف التسجيل · <kbd>Enter</kbd> الجملة التالية · تكلم من نفس المسافة اللي تستعملها عادة ·
  إعادة التسجيل تستبدل القديم · كل الملفات تبقى على جهازك في مجلد <b>samples</b>.</p>
</main>
<script>
const TOKEN = "__TOKEN__";
let items = [], idx = 0, recording = false, poll = null;
const $ = id => document.getElementById(id);

async function api(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json", "X-Token": TOKEN },
                                body: JSON.stringify(body || {}) });
  return r.json();
}
async function load() {
  const s = await (await fetch("/api/state")).json();
  items = s.items;
  const first = items.findIndex(i => !i.done);
  idx = first === -1 ? 0 : first;
  render();
}
function render() {
  const it = items[idx], done = items.filter(i => i.done).length;
  $("bar").style.width = (100 * done / items.length) + "%";
  $("count").textContent = `الجملة ${idx + 1} من ${items.length} · مسجّل: ${done}`;
  $("finished").style.display = done === items.length ? "block" : "none";
  $("cat").textContent = it.category;
  $("sentence").textContent = it.text;
  $("said").value = it.said;
  const p = $("player");
  if (it.done) { p.src = `/samples/${it.id}.wav?t=${Date.now()}`; p.hidden = false; } else { p.hidden = true; p.removeAttribute("src"); }
  if (!recording) setStatus(it.done ? "مسجّلة ✓ تقدر تسمعها، أو تعيد التسجيل." : "اضغط الزر أو Space وابدأ الكلام.", it.done ? "ok" : "");
  $("dots").innerHTML = items.map((x, i) =>
    `<div class="dot ${x.done ? "done" : ""} ${i === idx ? "cur" : ""}" data-i="${i}">${i + 1}</div>`).join("");
}
function setStatus(t, cls) { const s = $("status"); s.textContent = t; s.className = "status " + (cls || ""); }

let busy = false;
async function toggle() {
  if (busy) return;
  busy = true;
  try { await toggleInner(); } finally { busy = false; }
}
async function toggleInner() {
  if (!recording) {
    $("player").pause();
    await api("/api/start", { id: items[idx].id });
    recording = true; $("rec").classList.add("on");
    setStatus("يسجّل... تكلم الآن، واضغط مرة ثانية لما تخلص.", "warn");
    poll = setInterval(async () => {
      const s = await (await fetch("/api/state")).json();
      $("lvl").style.width = Math.min(100, s.level * 1400) + "%";
      $("timer").textContent = s.elapsed.toFixed(1);
    }, 100);
  } else {
    clearInterval(poll); recording = false; $("rec").classList.remove("on"); $("lvl").style.width = "0";
    const r = await api("/api/stop", { said: $("said").value });
    if (r.error === "short") { setStatus("التسجيل قصير جداً، أعده.", "err"); return; }
    if (r.error) { setStatus("خطأ: " + r.error, "err"); return; }
    items[idx].done = true; items[idx].said = $("said").value;
    render();
    if (r.quality === "quiet") setStatus("انحفظت، بس الصوت ضعيف. قرّب من المايك أو ارفع صوتك شوية وأعدها.", "warn");
    else if (r.quality === "clipped") setStatus("انحفظت، بس الصوت عالي لدرجة يتشوش. ابتعد شوية وأعدها.", "warn");
    else setStatus(`انحفظت ✓ (${r.secs} ثانية). اسمعها إذا تحب، ثم Enter للتالية.`, "ok");
  }
}
async function saveSaid() {
  const it = items[idx];
  if (it.done && $("said").value.trim() && $("said").value !== it.said) {
    await api("/api/said", { id: it.id, said: $("said").value }); it.said = $("said").value;
  }
}
async function go(d) {
  if (recording) return;
  await saveSaid();
  idx = Math.max(0, Math.min(items.length - 1, idx + d)); render();
}
// clicking must not leave a button focused, or Space/Enter would press it a second time
document.querySelectorAll("button").forEach(b => b.addEventListener("mousedown", e => e.preventDefault()));
$("rec").onclick = toggle;
$("next").onclick = () => go(1);
$("prev").onclick = () => go(-1);
$("said").addEventListener("blur", saveSaid);
$("dots").onclick = async e => { const i = e.target.dataset.i; if (i !== undefined && !recording) { await saveSaid(); idx = +i; render(); } };
$("addCustom").onclick = async () => {
  const t = $("custom").value.trim(); if (!t) return;
  await api("/api/custom", { text: t }); $("custom").value = "";
  const s = await (await fetch("/api/state")).json(); items = s.items; idx = items.length - 1; render();
};
document.addEventListener("keydown", e => {
  const typing = ["TEXTAREA", "INPUT"].includes(document.activeElement.tagName);
  if (e.code === "Space" && !typing) { e.preventDefault(); toggle(); }
  else if (e.key === "Enter" && !typing) { e.preventDefault(); go(1); }
});
load();
</script>
</body>
</html>
"""


REVIEW_PAGE = r"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>مكتبة Voice Typing الذكية</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-dark: #0B0B0B;
    --surface: #141417;
    --surface-card: #18181D;
    --surface-hover: #1E1E24;
    --border: rgba(247, 247, 245, 0.08);
    --border-focus: rgba(255, 106, 0, 0.5);
    --text-primary: #F7F7F5;
    --text-muted: #9E9EA8;
    --accent-orange: #FF6A00;
    --accent-orange-dim: rgba(255, 106, 0, 0.15);
    --green: #34C759;
    --green-dim: rgba(52, 199, 89, 0.15);
    --blue: #0A84FF;
    --blue-dim: rgba(10, 132, 255, 0.15);
    --purple: #BF5AF2;
    --purple-dim: rgba(191, 90, 242, 0.15);
    --red: #FF453A;
    --red-dim: rgba(255, 69, 58, 0.15);
    --yellow: #FFD60A;
    --font-arabic: 'Tajawal', -apple-system, BlinkMacSystemFont, "Segoe UI", Tahoma, sans-serif;
    --font-mono: 'JetBrains Mono', Consolas, monospace;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background-color: var(--bg-dark);
    color: var(--text-primary);
    font-family: var(--font-arabic);
    line-height: 1.6;
    min-height: 100vh;
    padding-bottom: 60px;
  }

  /* Sticky Top Bar */
  .top-header {
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(11, 11, 11, 0.92);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--border);
    padding: 14px 20px;
  }

  .header-inner {
    max-width: 1080px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .brand .logo-icon {
    width: 34px;
    height: 34px;
    background: var(--accent-orange);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 4px 12px rgba(255, 106, 0, 0.35);
  }

  .brand h1 {
    font-size: 1.25rem;
    font-weight: 900;
    letter-spacing: -0.3px;
  }

  .brand h1 span {
    color: var(--accent-orange);
  }

  .nav-links {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .nav-btn {
    font-family: var(--font-arabic);
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text-muted);
    text-decoration: none;
    padding: 6px 14px;
    border-radius: 8px;
    background: var(--surface);
    border: 1px solid var(--border);
    transition: all 0.2s ease;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .nav-btn:hover {
    color: var(--text-primary);
    border-color: rgba(255, 255, 255, 0.2);
    background: var(--surface-hover);
  }

  .nav-btn.primary {
    background: var(--accent-orange-dim);
    color: var(--accent-orange);
    border-color: rgba(255, 106, 0, 0.35);
  }

  .nav-btn.primary:hover {
    background: var(--accent-orange);
    color: #fff;
  }

  /* Main Container */
  main {
    max-width: 1080px;
    margin: 0 auto;
    padding: 24px 16px;
  }

  /* KPI Metrics Grid */
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 14px;
    margin-bottom: 24px;
  }

  .metric-card {
    background: var(--surface-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 18px;
    display: flex;
    align-items: center;
    gap: 14px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    transition: transform 0.2s ease, border-color 0.2s ease;
  }

  .metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.15);
  }

  .metric-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
  }

  .metric-icon.orange { background: var(--accent-orange-dim); color: var(--accent-orange); }
  .metric-icon.green { background: var(--green-dim); color: var(--green); }
  .metric-icon.blue { background: var(--blue-dim); color: var(--blue); }
  .metric-icon.purple { background: var(--purple-dim); color: var(--purple); }

  .metric-info h4 {
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--text-muted);
    margin-bottom: 2px;
  }

  .metric-info .val {
    font-family: var(--font-mono);
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--text-primary);
  }

  /* Search & Filter Toolbar */
  .toolbar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 24px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .search-wrapper {
    position: relative;
    width: 100%;
  }

  .search-icon {
    position: absolute;
    right: 14px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 1.1rem;
    color: var(--text-muted);
    pointer-events: none;
  }

  .search-input {
    width: 100%;
    background: #0E0E11;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 44px 12px 38px;
    font-family: var(--font-arabic);
    font-size: 1rem;
    color: var(--text-primary);
    transition: all 0.2s ease;
  }

  .search-input:focus {
    outline: none;
    border-color: var(--accent-orange);
    box-shadow: 0 0 0 3px rgba(255, 106, 0, 0.15);
  }

  .search-clear {
    position: absolute;
    left: 12px;
    top: 50%;
    transform: translateY(-50%);
    background: transparent;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 1rem;
    display: none;
  }

  .filter-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
  }

  .chips-group {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }

  .filter-label {
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--text-muted);
    margin-left: 4px;
  }

  .filter-chip {
    font-family: var(--font-arabic);
    font-size: 0.78rem;
    font-weight: 700;
    padding: 5px 12px;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: var(--surface-card);
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .filter-chip:hover {
    color: var(--text-primary);
    background: var(--surface-hover);
  }

  .filter-chip.active {
    background: var(--accent-orange);
    color: #fff;
    border-color: var(--accent-orange);
  }

  .results-meta {
    font-size: 0.82rem;
    color: var(--text-muted);
    font-weight: 600;
  }

  .results-meta b {
    color: var(--accent-orange);
    font-family: var(--font-mono);
  }

  /* Dictation Cards List */
  .list-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .dictation-card {
    background: var(--surface-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.35);
  }

  .dictation-card:hover {
    border-color: rgba(255, 255, 255, 0.14);
    box-shadow: 0 6px 25px rgba(0, 0, 0, 0.5);
  }

  .dictation-card.reviewed {
    border-left: 4px solid var(--green);
  }

  .card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 12px;
  }

  .card-badges {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .pill {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 12px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    border: 1px solid transparent;
  }

  .pill.date {
    background: var(--surface);
    color: var(--text-muted);
    font-family: var(--font-mono);
    border-color: var(--border);
  }

  .pill.dur {
    background: var(--accent-orange-dim);
    color: var(--accent-orange);
    font-family: var(--font-mono);
  }

  .pill.lang {
    background: var(--blue-dim);
    color: var(--blue);
  }

  .pill.done {
    background: var(--green-dim);
    color: var(--green);
  }

  .pill.words {
    background: var(--surface);
    color: #bbb;
    border-color: var(--border);
  }

  /* Audio Bar */
  .audio-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
    background: #0E0E11;
    padding: 8px 12px;
    border-radius: 12px;
    border: 1px solid var(--border);
  }

  audio {
    flex: 1;
    height: 32px;
  }

  .speed-selector {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .speed-btn {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
    background: var(--surface);
    color: var(--text-muted);
    border: 1px solid var(--border);
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .speed-btn:hover {
    color: var(--text-primary);
    background: var(--surface-hover);
  }

  .speed-btn.active {
    background: var(--accent-orange);
    color: #fff;
    border-color: var(--accent-orange);
  }

  /* Card Content */
  .transcript-box {
    position: relative;
    margin-bottom: 14px;
  }

  .transcript-editor {
    width: 100%;
    background: #0E0E11;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 16px;
    font-family: var(--font-arabic);
    font-size: 1.05rem;
    line-height: 1.8;
    color: var(--text-primary);
    direction: rtl;
    resize: vertical;
    transition: border-color 0.2s ease;
  }

  .transcript-editor:focus {
    outline: none;
    border-color: var(--border-focus);
    background: #111115;
  }

  .raw-hyp-preview {
    font-size: 0.85rem;
    color: #888;
    background: rgba(255, 255, 255, 0.03);
    padding: 8px 12px;
    border-radius: 8px;
    margin-bottom: 12px;
    line-height: 1.6;
    border-right: 3px solid #444;
  }

  .raw-hyp-preview b {
    color: var(--text-muted);
  }

  /* Card Actions Toolbar */
  .card-actions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
  }

  .main-btns {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .btn {
    font-family: var(--font-arabic);
    font-size: 0.82rem;
    font-weight: 700;
    padding: 6px 14px;
    border-radius: 8px;
    border: 1px solid transparent;
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .btn.btn-copy {
    background: var(--surface);
    color: var(--text-primary);
    border-color: var(--border);
  }
  .btn.btn-copy:hover {
    background: var(--surface-hover);
    border-color: rgba(255, 255, 255, 0.2);
  }

  .btn.btn-correct {
    background: var(--green-dim);
    color: var(--green);
    border-color: rgba(52, 199, 89, 0.3);
  }
  .btn.btn-correct:hover {
    background: var(--green);
    color: #0B0B0B;
  }

  .btn.btn-save {
    background: var(--accent-orange-dim);
    color: var(--accent-orange);
    border-color: rgba(255, 106, 0, 0.35);
  }
  .btn.btn-save:hover {
    background: var(--accent-orange);
    color: #fff;
  }

  .btn.btn-del {
    background: transparent;
    color: var(--red);
    border-color: transparent;
  }
  .btn.btn-del:hover {
    background: var(--red-dim);
    border-color: rgba(255, 69, 58, 0.3);
  }

  /* AI Action Buttons (Phase 2) */
  .ai-toolbar {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }

  .btn-ai {
    font-family: var(--font-arabic);
    font-size: 0.78rem;
    font-weight: 700;
    padding: 5px 12px;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.05);
    color: #D1D1D6;
    border: 1px solid var(--border);
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }

  .btn-ai:hover {
    background: var(--purple-dim);
    color: var(--purple);
    border-color: rgba(191, 90, 242, 0.4);
  }

  .btn-ai.active {
    background: var(--purple);
    color: #fff;
  }

  /* AI Output Box */
  .ai-result-box {
    margin-top: 12px;
    background: #111116;
    border: 1px solid rgba(191, 90, 242, 0.3);
    border-radius: 12px;
    padding: 14px 16px;
    position: relative;
    animation: fadeIn 0.25s ease;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(-4px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .ai-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    font-size: 0.8rem;
    font-weight: 800;
    color: var(--purple);
  }

  .ai-text {
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--text-primary);
    white-space: pre-wrap;
    margin-bottom: 10px;
  }

  .ai-actions {
    display: flex;
    gap: 8px;
  }

  /* Suggestion Chips */
  .sugg-box {
    margin-top: 12px;
    padding: 10px 14px;
    background: #151A24;
    border: 1px solid rgba(10, 132, 255, 0.25);
    border-radius: 12px;
    font-size: 0.82rem;
  }

  .sugg-box h5 {
    font-size: 0.8rem;
    color: var(--blue);
    margin-bottom: 6px;
  }

  .rule-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #0E0E11;
    border: 1px solid var(--border);
    padding: 3px 10px;
    border-radius: 8px;
    margin: 3px 4px;
    font-family: var(--font-mono);
    font-size: 0.78rem;
  }

  .rule-chip button {
    background: var(--blue-dim);
    color: var(--blue);
    border: 1px solid rgba(10, 132, 255, 0.3);
    border-radius: 6px;
    padding: 2px 8px;
    font-family: var(--font-arabic);
    font-weight: 700;
    cursor: pointer;
  }

  /* Jargon Extractor Chips */
  .jargon-chips {
    margin-top: 8px;
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    align-items: center;
  }

  .jargon-chip {
    background: rgba(255, 204, 0, 0.1);
    color: var(--yellow);
    border: 1px solid rgba(255, 204, 0, 0.25);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-family: var(--font-mono);
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .jargon-chip:hover {
    background: var(--yellow);
    color: #000;
  }

  /* Load More Button */
  .load-more-wrap {
    display: flex;
    justify-content: center;
    margin: 30px 0;
  }

  .btn-load-more {
    font-family: var(--font-arabic);
    font-size: 0.95rem;
    font-weight: 700;
    padding: 12px 32px;
    border-radius: 30px;
    background: var(--surface);
    color: var(--text-primary);
    border: 1px solid var(--border);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .btn-load-more:hover {
    background: var(--surface-hover);
    border-color: rgba(255, 255, 255, 0.2);
    transform: translateY(-2px);
  }

  /* Toast Notification */
  .toast {
    position: fixed;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%) translateY(80px);
    background: #1C1C22;
    color: #fff;
    border: 1px solid var(--border);
    padding: 10px 22px;
    border-radius: 30px;
    font-weight: 700;
    font-size: 0.9rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    gap: 8px;
    z-index: 1000;
    opacity: 0;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    pointer-events: none;
  }

  .toast.show {
    transform: translateX(-50%) translateY(0);
    opacity: 1;
  }

  mark {
    background: rgba(255, 106, 0, 0.3);
    color: #fff;
    border-radius: 2px;
    padding: 0 2px;
  }

  .empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-muted);
  }

  .empty-state .icon {
    font-size: 3rem;
    margin-bottom: 12px;
  }

  @media (max-width: 768px) {
    .header-inner { flex-direction: column; align-items: stretch; }
    .nav-links { justify-content: flex-end; }
    .card-actions { flex-direction: column; align-items: stretch; }
  }
</style>
</head>
<body>

  <!-- Sticky Top Header -->
  <header class="top-header">
    <div class="header-inner">
      <div class="brand">
        <div class="logo-icon">🎙️</div>
        <h1>مكتبة <span>Voice Typing</span></h1>
      </div>
      <nav class="nav-links">
        <a href="/" class="nav-btn">← جمل القياس</a>
        <a href="/rules" class="nav-btn">← قواعد التصحيح</a>
        <button id="btnExportMd" class="nav-btn primary">📥 تصدير Markdown</button>
        <button id="btnCopyAll" class="nav-btn">📋 نسخ المعروض</button>
      </nav>
    </div>
  </header>

  <main>
    <!-- Productivity Metrics Grid -->
    <section class="metrics-grid">
      <div class="metric-card">
        <div class="metric-icon orange">🎙️</div>
        <div class="metric-info">
          <h4>إجمالي الإملاءات</h4>
          <div class="val" id="statTotalCount">0</div>
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-icon green">⏱️</div>
        <div class="metric-info">
          <h4>إجمالي الساعات الصوتية</h4>
          <div class="val" id="statTotalHours">0h</div>
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-icon blue">📝</div>
        <div class="metric-info">
          <h4>إجمالي الكلمات</h4>
          <div class="val" id="statTotalWords">0</div>
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-icon purple">⚡</div>
        <div class="metric-info">
          <h4>الوقت الموفر المقدر</h4>
          <div class="val" id="statTimeSaved">0h</div>
        </div>
      </div>
    </section>

    <!-- Toolbar: Search & Filters -->
    <section class="toolbar">
      <div class="search-wrapper">
        <span class="search-icon">🔍</span>
        <input type="text" id="searchInput" class="search-input" placeholder="ابحث في 1,600+ تسجيل صوتي، الكلمات، والملاحظات فوراً...">
        <button id="searchClear" class="search-clear">✕</button>
      </div>

      <div class="filter-row">
        <!-- Status Filter -->
        <div class="chips-group">
          <span class="filter-label">الحالة:</span>
          <button class="filter-chip active" data-filter="status" data-val="all">الكل</button>
          <button class="filter-chip" data-filter="status" data-val="unreviewed">غير مراجَع</button>
          <button class="filter-chip" data-filter="status" data-val="reviewed">مراجَع ✓</button>
        </div>

        <!-- Duration Filter -->
        <div class="chips-group">
          <span class="filter-label">المدة:</span>
          <button class="filter-chip active" data-filter="dur" data-val="all">الكل</button>
          <button class="filter-chip" data-filter="dur" data-val="short">&lt; 5ث</button>
          <button class="filter-chip" data-filter="dur" data-val="med">5-20ث</button>
          <button class="filter-chip" data-filter="dur" data-val="long">&gt; 20ث</button>
        </div>

        <!-- Date Filter -->
        <div class="chips-group">
          <span class="filter-label">الفترة:</span>
          <button class="filter-chip active" data-filter="date" data-val="all">الكل</button>
          <button class="filter-chip" data-filter="date" data-val="today">اليوم</button>
          <button class="filter-chip" data-filter="date" data-val="week">هذا الأسبوع</button>
          <button class="filter-chip" data-filter="date" data-val="month">هذا الشهر</button>
        </div>

        <!-- Results Counter -->
        <div class="results-meta">
          عرض <b id="filteredCount">0</b> من أصل <b id="totalCount">0</b>
        </div>
      </div>
    </section>

    <!-- Cards Container -->
    <section id="dictationList" class="list-container"></section>

    <!-- Load More -->
    <div id="loadMoreWrap" class="load-more-wrap" style="display:none;">
      <button id="btnLoadMore" class="btn-load-more">عرض المزيد من الإملاءات ↓</button>
    </div>
  </main>

  <!-- Toast Notification -->
  <div id="toast" class="toast">
    <span class="toast-icon">✓</span>
    <span id="toastMsg" class="toast-msg">تمت العملية بنجاح</span>
  </div>

<script>
const TOKEN = "__TOKEN__";
const $ = id => document.getElementById(id);

let rawItems = [];
let filteredItems = [];
let renderedCount = 0;
const BATCH_SIZE = 30;

// Current Active Filters
const filters = {
  search: "",
  status: "all",
  dur: "all",
  date: "all"
};

async function api(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Token": TOKEN },
    body: JSON.stringify(body)
  });
  return r.json();
}

function showToast(msg) {
  const toast = $("toast");
  $("toastMsg").textContent = msg;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2400);
}

function esc(s) {
  return (s || "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function highlightText(text, q) {
  if (!q || !text) return esc(text);
  const escapedQ = q.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const regex = new RegExp(`(${escapedQ})`, "gi");
  const parts = text.split(regex);
  return parts.map((p, i) => regex.test(p) ? `<mark>${esc(p)}</mark>` : esc(p)).join("");
}

function extractTechWords(text) {
  if (!text) return [];
  const words = text.match(/[A-Za-z][A-Za-z0-9_.-]{2,}/g) || [];
  const unique = Array.from(new Set(words.map(w => w.trim())));
  return unique.slice(0, 4);
}

function updateMetrics() {
  const total = rawItems.length;
  let totalSecs = 0;
  let totalWords = 0;

  rawItems.forEach(i => {
    totalSecs += (i.secs || 0);
    const content = (i.text ?? i.hyp ?? "");
    const words = content.trim().split(/\s+/).filter(Boolean).length;
    totalWords += words;
  });

  const hours = (totalSecs / 3600).toFixed(1);
  // Typing speed comparison: average typing is 40 wpm, speaking is ~140 wpm
  const typingTimeHours = (totalWords / 40) / 60;
  const speakingTimeHours = totalSecs / 3600;
  const timeSaved = Math.max(0, typingTimeHours - speakingTimeHours).toFixed(1);

  $("statTotalCount").textContent = total.toLocaleString();
  $("statTotalHours").textContent = `${hours}h`;
  $("statTotalWords").textContent = totalWords.toLocaleString();
  $("statTimeSaved").textContent = `~${timeSaved}h`;
}

function applyFilters() {
  const q = filters.search.trim().toLowerCase();
  const now = new Date();

  filteredItems = rawItems.filter(item => {
    // Search
    if (q) {
      const hyp = (item.hyp || "").toLowerCase();
      const text = (item.text || "").toLowerCase();
      const rec = (item.recorded || "").toLowerCase();
      if (!hyp.includes(q) && !text.includes(q) && !rec.includes(q)) return false;
    }

    // Status
    if (filters.status === "unreviewed" && item.text !== null) return false;
    if (filters.status === "reviewed" && item.text === null) return false;

    // Duration
    const s = item.secs || 0;
    if (filters.dur === "short" && s >= 5) return false;
    if (filters.dur === "med" && (s < 5 || s > 20)) return false;
    if (filters.dur === "long" && s <= 20) return false;

    // Date
    if (filters.date !== "all" && item.recorded) {
      const itemDate = new Date(item.recorded.replace(" ", "T"));
      const diffMs = now - itemDate;
      const diffDays = diffMs / (1000 * 60 * 60 * 24);
      if (filters.date === "today" && diffDays > 1) return false;
      if (filters.date === "week" && diffDays > 7) return false;
      if (filters.date === "month" && diffDays > 30) return false;
    }

    return true;
  });

  $("filteredCount").textContent = filteredItems.length.toLocaleString();
  $("totalCount").textContent = rawItems.length.toLocaleString();

  renderedCount = 0;
  $("dictationList").innerHTML = "";
  renderNextBatch();
}

function renderNextBatch() {
  const nextItems = filteredItems.slice(renderedCount, renderedCount + BATCH_SIZE);
  if (!nextItems.length && renderedCount === 0) {
    $("dictationList").innerHTML = `
      <div class="empty-state">
        <div class="icon">🔍</div>
        <h3>لم يتم العثور على أي نتائج مطابقة</h3>
        <p>جرب تغيير كلمة البحث أو إعادة تعيين الفلاتر.</p>
      </div>`;
    $("loadMoreWrap").style.display = "none";
    return;
  }

  const html = nextItems.map(item => {
    const isReviewed = item.text !== null;
    const currentText = item.text ?? item.hyp ?? "";
    const wordsCount = currentText.trim().split(/\s+/).filter(Boolean).length;
    const techWords = extractTechWords(currentText);

    return `
      <div class="dictation-card ${isReviewed ? 'reviewed' : ''}" id="c-${item.id}">
        <div class="card-top">
          <div class="card-badges">
            <span class="pill date">📅 ${esc(item.recorded || 'غير محدد')}</span>
            <span class="pill dur">⏱️ ${item.secs || 0}ث</span>
            <span class="pill lang">${item.lang === 'ar' ? 'عربي' : item.lang === 'en' ? 'English' : 'تلقائي'}</span>
            <span class="pill words">${wordsCount} كلمة</span>
            ${isReviewed ? '<span class="pill done">✓ مراجَع</span>' : ''}
          </div>
        </div>

        <div class="audio-bar">
          <audio controls preload="none" src="/auto/${item.id}.wav" id="aud-${item.id}"></audio>
          <div class="speed-selector">
            <button class="speed-btn active" onclick="setSpeed('${item.id}', 1.0, this)">1.0x</button>
            <button class="speed-btn" onclick="setSpeed('${item.id}', 1.25, this)">1.25x</button>
            <button class="speed-btn" onclick="setSpeed('${item.id}', 1.5, this)">1.5x</button>
            <button class="speed-btn" onclick="setSpeed('${item.id}', 2.0, this)">2.0x</button>
          </div>
        </div>

        ${isReviewed && item.hyp && item.hyp !== item.text ? `
          <div class="raw-hyp-preview">
            <b>النص المفرغ أصلاً:</b> ${highlightText(item.hyp, filters.search)}
          </div>` : ''}

        <div class="transcript-box">
          <textarea class="transcript-editor" id="t-${item.id}" rows="${Math.min(8, Math.max(2, Math.ceil(currentText.length / 65)))}">${esc(currentText)}</textarea>
        </div>

        ${techWords.length ? `
          <div class="jargon-chips">
            <span style="font-size:0.75rem; color:#888;">مصطلحات مكتشفة:</span>
            ${techWords.map(w => `<span class="jargon-chip" onclick="addVocab('${esc(w)}')">+ أضف "${esc(w)}" للقاموس</span>`).join('')}
          </div>` : ''}

        <div class="card-actions">
          <div class="main-btns">
            <button class="btn btn-copy" onclick="copyText('${item.id}')">📋 نسخ النص</button>
            <button class="btn btn-correct" onclick="same('${item.id}')">✓ صحيح</button>
            <button class="btn btn-save" onclick="save('${item.id}')">💾 حفظ التعديل</button>
            <button class="btn btn-del" onclick="del('${item.id}')">🗑️ حذف</button>
          </div>

          <div class="ai-toolbar">
            <span style="font-size:0.75rem; color:#888;">أوامر AI:</span>
            <button class="btn-ai" onclick="runAi('${item.id}', 'summary')">✨ تلخيص</button>
            <button class="btn-ai" onclick="runAi('${item.id}', 'prompt')">💡 برومبت</button>
            <button class="btn-ai" onclick="runAi('${item.id}', 'formal')">📧 رسمي</button>
            <button class="btn-ai" onclick="runAi('${item.id}', 'polish')">🧹 تصفية</button>
          </div>
        </div>

        <div id="ai-box-${item.id}"></div>
        <div id="s-${item.id}"></div>
      </div>
    `;
  }).join("");

  $("dictationList").insertAdjacentHTML("beforeend", html);
  renderedCount += nextItems.length;

  if (renderedCount < filteredItems.length) {
    $("loadMoreWrap").style.display = "flex";
  } else {
    $("loadMoreWrap").style.display = "none";
  }
}

function setSpeed(id, rate, btn) {
  const aud = $("aud-" + id);
  if (aud) aud.playbackRate = rate;
  const parent = btn.parentElement;
  parent.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

function copyText(id) {
  const text = $("t-" + id).value;
  navigator.clipboard.writeText(text).then(() => {
    showToast("تم نسخ النص إلى الحافظة ✓");
  });
}

async function addVocab(word) {
  const r = await api("/api/review/vocab", { word });
  if (r.ok) {
    showToast(r.added ? `أُضيف "${word}" للقاموس بنجاح ✓` : `"${word}" موجود مسبقاً في القاموس`);
  } else {
    showToast("تعذر الإضافة: " + (r.error || "خطأ غير معروف"));
  }
}

async function same(id) {
  const item = rawItems.find(i => i.id === id);
  if (!item) return;
  $("t-" + id).value = item.hyp;
  await save(id);
}

async function save(id) {
  const text = $("t-" + id).value;
  const r = await api("/api/review/save", { id, text });
  const item = rawItems.find(i => i.id === id);
  if (item) item.text = text;

  const card = $("c-" + id);
  if (card) card.classList.add("reviewed");

  const box = $("s-" + id);
  let html = `<div style="color:var(--green); font-size:0.85rem; margin-top:8px;">انحفظ بنجاح ✓</div>`;
  if (r.suggestions && r.suggestions.length) {
    html += `
      <div class="sugg-box">
        <h5>قواعد تصحيح مقترحة من تعديلك (تُطبَّق تلقائياً في الإملاءات القادمة):</h5>
        ${r.suggestions.map((s, k) => `
          <span class="rule-chip" id="r-${id}-${k}">
            ${esc(s.wrong)} ← ${esc(s.right)}
            <button onclick='applyRule("${id}", ${k}, ${JSON.stringify(s.wrong)}, ${JSON.stringify(s.right)})'>أضف القاعدة</button>
          </span>
        `).join("")}
      </div>`;
  }
  box.innerHTML = html;
  showToast("تم الحفظ بنجاح ✓");
  updateMetrics();
}

async function applyRule(id, k, wrong, right) {
  const r = await api("/api/review/rule", { wrong, right });
  const chip = $(`r-${id}-${k}`);
  if (chip) {
    chip.innerHTML = `${esc(wrong)} ← ${esc(right)} ${r.added ? "✓ أُضيفت" : "(موجودة)"}`;
  }
  showToast(r.added ? "تمت إضافة القاعدة إلى corrections.txt ✓" : "القاعدة موجودة مسبقاً");
}

async function del(id) {
  if (!confirm("هل أنت متأكد من حذف هذا التسجيل الصوتي ونصه نهائياً؟")) return;
  const r = await api("/api/review/delete", { id });
  if (r.ok) {
    rawItems = rawItems.filter(i => i.id !== id);
    filteredItems = filteredItems.filter(i => i.id !== id);
    const card = $("c-" + id);
    if (card) card.remove();
    showToast("تم حذف التسجيل بنجاح");
    updateMetrics();
  }
}

// Phase 2: AI Action Runner
async function runAi(id, action) {
  const text = $("t-" + id).value;
  if (!text.trim()) {
    showToast("النص فارغ!");
    return;
  }

  const box = $("ai-box-" + id);
  box.innerHTML = `
    <div class="ai-result-box">
      <div class="ai-header">
        <span>جاري المعالجة عبر Gemini Flash Lite... ⚡</span>
      </div>
      <div class="ai-text" style="color:#888;">يرجى الانتظار ثانية واحدة...</div>
    </div>`;

  const titles = {
    summary: "ملخص في نقاط عمل (Summary)",
    prompt: "برومبت هندسي منظم (Promptify)",
    formal: "صياغة رسمية / بريد إلكتروني",
    polish: "تصفية التأتأة والتردد"
  };

  try {
    const r = await api("/api/review/ai_action", { text, action });
    if (r.ok && r.result) {
      box.innerHTML = `
        <div class="ai-result-box">
          <div class="ai-header">
            <span>✨ نتيجة الذكاء الاصطناعي: ${titles[action] || action}</span>
            <button onclick="this.closest('.ai-result-box').remove()" style="background:none; border:none; color:#888; cursor:pointer;">✕ إغلاق</button>
          </div>
          <div class="ai-text" id="ai-res-text-${id}">${esc(r.result)}</div>
          <div class="ai-actions">
            <button class="btn btn-copy" onclick="copyAiResult('${id}')">📋 نسخ النتيجة</button>
            <button class="btn btn-save" onclick="applyAiToEditor('${id}')">استبدال النص في المحرر ↵</button>
          </div>
        </div>`;
      showToast("تمت معالجة النص بالذكاء الاصطناعي ✓");
    } else {
      box.innerHTML = `<div style="color:var(--red); font-size:0.85rem; margin-top:8px;">تعذر المعالجة: ${esc(r.error || 'خطأ غير معروف')}</div>`;
    }
  } catch(e) {
    box.innerHTML = `<div style="color:var(--red); font-size:0.85rem; margin-top:8px;">حدث خطأ في الاتصال: ${esc(e.message)}</div>`;
  }
}

function copyAiResult(id) {
  const resEl = $("ai-res-text-" + id);
  if (resEl) {
    navigator.clipboard.writeText(resEl.textContent).then(() => {
      showToast("تم نسخ نتيجة الذكاء الاصطناعي ✓");
    });
  }
}

function applyAiToEditor(id) {
  const resEl = $("ai-res-text-" + id);
  if (resEl) {
    $("t-" + id).value = resEl.textContent;
    $("ai-box-" + id).innerHTML = "";
    showToast("تم نقل النتيجة إلى محرر النص ✓");
  }
}

// Markdown Exporter
function exportMarkdown() {
  if (!filteredItems.length) {
    showToast("لا توجد نصوص لتصديرها!");
    return;
  }

  let md = `# أرشيف إملاءات Voice Typing\n\n`;
  md += `تم التصدير بتاريخ: ${new Date().toLocaleString('ar-EG')}\n`;
  md += `إجمالي الإملاءات المصدرة: ${filteredItems.length}\n\n---\n\n`;

  filteredItems.forEach((i, idx) => {
    const text = i.text ?? i.hyp ?? "";
    md += `### ${idx + 1}. تسجيل [${i.recorded || 'بدون تاريخ'}] (المدة: ${i.secs || 0}ث)\n\n`;
    md += `${text}\n\n`;
    if (i.hyp && i.text && i.hyp !== i.text) {
      md += `> **المفرغ أولاً:** ${i.hyp}\n\n`;
    }
    md += `---\n\n`;
  });

  const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `VoiceTyping_Export_${new Date().toISOString().slice(0,10)}.md`;
  a.click();
  showToast("تم تنزيل ملف Markdown بنجاح 📥");
}

function copyAllFiltered() {
  if (!filteredItems.length) return;
  const allText = filteredItems.map(i => (i.text ?? i.hyp ?? "").trim()).filter(Boolean).join("\n\n---\n\n");
  navigator.clipboard.writeText(allText).then(() => {
    showToast(`تم نسخ ${filteredItems.length} إملاء إلى الحافظة ✓`);
  });
}

// Infinite Scroll / Load More
window.addEventListener("scroll", () => {
  if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 500) {
    if (renderedCount < filteredItems.length) {
      renderNextBatch();
    }
  }
});

// Event Listeners
$("btnLoadMore").addEventListener("click", renderNextBatch);
$("btnExportMd").addEventListener("click", exportMarkdown);
$("btnCopyAll").addEventListener("click", copyAllFiltered);

// Search
$("searchInput").addEventListener("input", e => {
  filters.search = e.target.value;
  $("searchClear").style.display = filters.search ? "block" : "none";
  applyFilters();
});

$("searchClear").addEventListener("click", () => {
  $("searchInput").value = "";
  filters.search = "";
  $("searchClear").style.display = "none";
  applyFilters();
});

// Filter Chips
document.querySelectorAll(".filter-chip").forEach(chip => {
  chip.addEventListener("click", () => {
    const filterType = chip.getAttribute("data-filter");
    const val = chip.getAttribute("data-val");
    chip.parentElement.querySelectorAll(".filter-chip").forEach(c => c.classList.remove("active"));
    chip.classList.add("active");
    filters[filterType] = val;
    applyFilters();
  });
});

// Init Load
async function load() {
  try {
    rawItems = await (await fetch("/api/review")).json();
    updateMetrics();
    applyFilters();
  } catch(e) {
    console.error("Load error:", e);
    showToast("تعذر تحميل البيانات من الخادم");
  }
}

load();
</script>
</body>
</html>
"""


RULES_PAGE = r"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>قواعد التصحيح</title>
<style>
  :root { --bg:#141418; --card:#1f1f26; --line:#34343f; --text:#f1f1f4; --muted:#9a9aa8;
          --green:#3ecf8e; --blue:#4f9dff; --red:#e5484d; --yellow:#facc15; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font-family:"Segoe UI", Tahoma, sans-serif; }
  main { max-width: 900px; margin: 0 auto; padding: 26px 16px 70px; }
  h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 0 0 10px; }
  a { color: var(--blue); }
  .nav { display:flex; gap:18px; margin-bottom:10px; font-size:14px; }
  .sub { color: var(--muted); line-height: 1.9; margin: 0 0 16px; font-size: 14.5px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 16px; margin-bottom: 14px; }
  input, textarea, select { background:#16161b; color:var(--text); border:1px solid var(--line); border-radius:10px;
          padding:9px 12px; font: inherit; font-size: 17px; }
  textarea { width:100%; resize: vertical; line-height: 1.8; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  .row input { flex:1; min-width: 130px; }
  .arrow { color: var(--muted); }
  button { background:#2b2b35; color:var(--text); border:1px solid var(--line); border-radius:10px; padding:8px 14px;
           font: inherit; cursor:pointer; }
  button.primary { background: var(--blue); border-color: var(--blue); color:#0b1220; font-weight:600; }
  button.del { color: var(--red); }
  button.save { background:#173a2c; border-color: var(--green); color: var(--green); display:none; }
  .rule { display:flex; gap:8px; align-items:center; padding:6px 0; border-top:1px solid #292932; }
  .rule input { flex:1; min-width: 100px; padding: 6px 10px; font-size: 16px; }
  .sec { margin-top: 6px; }
  .sec > summary { cursor:pointer; padding:8px 0; font-weight:600; list-style:none; }
  .sec > summary::before { content:"▾ "; color: var(--muted); }
  .count { color: var(--muted); font-weight:400; font-size: 13px; }
  .result { margin-top:10px; font-size: 19px; line-height:1.9; padding:10px 12px; background:#16161b; border-radius:10px;
            border:1px solid var(--line); min-height: 44px; white-space: pre-wrap; }
  .fired { margin-top:10px; font-size:14px; color: var(--muted); line-height: 2; }
  .fired code { background:#26262e; padding:1px 8px; border-radius:6px; color:var(--text); }
  .msg { margin-top:8px; min-height:22px; font-size:14px; } .ok { color:var(--green); } .warn { color:var(--yellow); } .err { color:var(--red); }
  .search { width:100%; margin: 4px 0 12px; }
  .hint { color: var(--muted); font-size:13px; margin-top:8px; line-height:1.8; }
</style>
</head>
<body>
<main>
  <div class="nav"><a href="/review">← مراجعة الإملاءات</a><a href="/">← تسجيل جمل القياس</a></div>
  <h1>قواعد التصحيح</h1>
  <p class="sub">كل قاعدة: <b>الخطأ ← الصحيح</b>. تُطبَّق على كل إملاء بعد التفريغ، والتعديل يسري <b>فوراً</b> بدون إعادة تشغيل.
  القاعدة تطابق <b>الكلمة كاملة</b> (فقاعدة «شاي ← چاي» لا تلمس «شايف»)، وتعمل مع السوابق و ف ب ل ك ال (فتصحّح «والشاي» أيضاً).
  النجمة <b>*</b> في آخر الطرفين تُبقي نهاية الكلمة: «أحكي* ← أحچي*» تحوّل «أحكيلك» إلى «أحچيلك».</p>

  <div class="card">
    <h2>جرّب جملة</h2>
    <textarea id="try" rows="2" placeholder="اكتب جملة كما يكتبها الموديل وشوف شنو تسوي بيها القواعد"></textarea>
    <div class="result" id="result"></div>
    <div class="fired" id="fired"></div>
  </div>

  <div class="card">
    <h2>أضف قاعدة</h2>
    <div class="row">
      <input id="nw" placeholder="الخطأ (كما يكتبه الموديل)">
      <span class="arrow">←</span>
      <input id="nr" placeholder="الصحيح">
      <select id="ns"></select>
      <button class="primary" id="add">أضف</button>
    </div>
    <div class="msg" id="addmsg"></div>
  </div>

  <input class="search" id="q" placeholder="ابحث في القواعد...">
  <div id="list"></div>
</main>
<script>
const TOKEN = "__TOKEN__";
const $ = id => document.getElementById(id);
let rules = [], sections = [];
async function api(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json", "X-Token": TOKEN }, body: JSON.stringify(body || {}) });
  return r.json();
}
const esc = s => (s || "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
async function load() {
  const s = await (await fetch("/api/rules")).json();
  rules = s.rules; sections = s.sections;
  $("ns").innerHTML = sections.map(x => `<option>${esc(x)}</option>`).join("") || "<option>عام</option>";
  render(); runTest();
}
function render() {
  const q = $("q").value.trim().toLowerCase();
  const shown = rules.filter(r => !q || r.wrong.toLowerCase().includes(q) || r.right.toLowerCase().includes(q));
  $("list").innerHTML = sections.map(sec => {
    const rs = shown.filter(r => r.section === sec);
    if (!rs.length) return "";
    return `<details class="card sec" open><summary>${esc(sec)} <span class="count">(${rs.length})</span></summary>` + rs.map(r => `
      <div class="rule" data-line="${r.line}">
        <input value="${esc(r.wrong)}" data-k="wrong" oninput="dirty(${r.line})">
        <span class="arrow">←</span>
        <input value="${esc(r.right)}" data-k="right" oninput="dirty(${r.line})">
        <button class="save" id="s-${r.line}" onclick="save(${r.line})">حفظ</button>
        <button class="del" onclick="del(${r.line})">حذف</button>
      </div>`).join("") + `</details>`;
  }).join("") || `<div class="card">لا توجد قواعد${q ? " تطابق البحث" : ""}.</div>`;
}
const find = line => rules.find(r => r.line === line);
const rowOf = line => document.querySelector(`.rule[data-line="${line}"]`);
function dirty(line) {
  const row = rowOf(line), r = find(line);
  const changed = row.querySelector('[data-k="wrong"]').value !== r.wrong || row.querySelector('[data-k="right"]').value !== r.right;
  $("s-" + line).style.display = changed ? "inline-block" : "none";
}
async function save(line) {
  const row = rowOf(line), r = find(line);
  const res = await api("/api/rules/update", { line, raw: r.raw, wrong: row.querySelector('[data-k="wrong"]').value, right: row.querySelector('[data-k="right"]').value });
  if (res.error === "changed") { alert("الملف تغيّر، سأعيد تحميل القواعد"); return load(); }
  if (res.error) return alert(res.error);
  if (res.warning) alert(res.warning);
  await load();
}
async function del(line) {
  const r = find(line);
  if (!confirm(`حذف القاعدة؟\n${r.wrong} ← ${r.right}`)) return;
  const res = await api("/api/rules/delete", { line, raw: r.raw });
  if (res.error === "changed") alert("الملف تغيّر، سأعيد تحميل القواعد");
  await load();
}
$("add").onclick = async () => {
  const res = await api("/api/rules/add", { wrong: $("nw").value, right: $("nr").value, section: $("ns").value });
  const m = $("addmsg");
  if (res.error) { m.className = "msg err"; m.textContent = res.error; return; }
  m.className = "msg " + (res.added ? "ok" : "warn");
  m.textContent = (res.added ? "أُضيفت ✓ " : "") + (res.warning || "");
  if (res.added) { $("nw").value = ""; $("nr").value = ""; await load(); }
};
let timer;
function runTest() {
  clearTimeout(timer);
  timer = setTimeout(async () => {
    const text = $("try").value;
    if (!text.trim()) { $("result").textContent = ""; $("fired").innerHTML = ""; return; }
    const r = await api("/api/rules/test", { text });
    $("result").textContent = r.result;
    $("fired").innerHTML = r.fired.length
      ? "القواعد التي غيّرت الجملة:<br>" + r.fired.map(f => `<code>${esc(f.rule)}</code>`).join(" ")
      : "لا توجد قاعدة غيّرت شيئاً.";
  }, 250);
}
$("try").oninput = runTest;
$("q").oninput = render;
load();
</script>
</body>
</html>
"""


def main():
    global REC
    vt.setup_logging()
    REC = Recorder()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Voice sample recorder is running at {url}")
    print("Keep this window open while recording. Close it when you are done.")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url + ("review" if "--review" in sys.argv else "rules" if "--rules" in sys.argv else ""))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
