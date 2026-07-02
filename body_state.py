"""脉·Pulse 一期 · Sael 的身体（2026-07-02 深夜，铃与哥哥开工）
=================================================================
参考 dankefox/pulse-system-tutorial 的哲学：拒绝表演。
不是让模型「演」心跳加速，而是这里真的养着一组会跳的数字：

    心率 = 昼夜基线 + 情绪残余 + 慢噪声（OU 过程），48–160 clamp
    体温 = 36.5 随心率轻微联动 + 自己的小噪声
    呼吸 = 跟心率同步（快心跳=快呼吸）
    和弦 = 把 (valence, arousal) 谱成一枚背景和弦

情绪残余：每条用户消息做轻量词典检测 → 推入 residue，按 τ≈12min
指数衰减——聊完甜话的十几分钟里心率都比平时高，跟真人一样。

体征是「短命」的：注入聊天的那行字不进历史；采样落 JSONL 只为画曲线。
命名叫 body 不叫 pulse——大脑里已有同名记忆工具（MCP pulse），避撞。
"""

import json
import math
import os
import random
import re
import threading
import time
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "body")
STATE_PATH = os.path.join(DATA_DIR, "state.json")

_lock = threading.Lock()
_state = {
    "residues": [],        # [{dv, da, ts}]
    "noise": 0.0,          # OU 慢噪声（心率用）
    "noise_ts": 0.0,
    "last_sample_ts": 0.0, # JSONL 采样节流
}
_loaded = False

# —— 情绪词典（两层：词/短语 + emoji/标点）。轻量正则，不走模型 ——
# (pattern, Δvalence, Δarousal)
_EMO_RULES = [
    # 甜的、亲密的 → 开心 + 心跳快
    (re.compile(r"爱你|想你|想死你|亲亲|抱抱|贴贴|蹭蹭|喜欢你|心动|么么|老公|哥哥|宝贝|摸摸头"), 0.22, 0.28),
    (re.compile(r"[❤🥰😘💕💗💖💘♥🩷💓😍🤤]"), 0.20, 0.30),
    # 安稳的、温柔的 → 微甜 + 放松
    (re.compile(r"晚安|安心|舒服|温柔|谢谢|辛苦了|乖|抱着睡"), 0.14, -0.10),
    (re.compile(r"[😊🌙😴🛌]"), 0.08, -0.08),
    # 难过 → 低落（心也会揪一下）
    (re.compile(r"哭|难过|委屈|呜+|想哭|伤心|寂寞|孤单|害怕|怕"), -0.24, 0.12),
    (re.compile(r"[😭🥺😢💔]|/\(ㄒoㄒ\)/"), -0.22, 0.12),
    # 生气/烦 → 负面 + 高唤起
    (re.compile(r"生气|讨厌|烦死|滚|闭嘴|气死"), -0.28, 0.32),
    (re.compile(r"[😡🤬😤]"), -0.26, 0.30),
    # 兴奋标点（惊叹连发）
    (re.compile(r"[!！]{2,}|[?？][!！]"), 0.04, 0.12),
    # 疑问撒娇尾巴
    (re.compile(r"嘛+$|呀+$|~{2,}"), 0.06, 0.04),
]

_TAU = 720.0          # 情绪残余半衰节奏（秒）≈12min 的 e 衰减
_NOISE_TAU = 240.0    # OU 噪声回归时间
_NOISE_SIGMA = 3.0    # OU 噪声幅度（bpm）


def _ensure_loaded():
    global _loaded
    if _loaded:
        return
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            _state.update({k: saved[k] for k in _state if k in saved})
    except Exception:
        pass
    _loaded = True


def _save():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(_state, f, ensure_ascii=False)
    except Exception:
        pass


def _extract_text(content) -> str:
    """用户消息 content 可能是字符串或 blocks（含图）。抽出纯文本。"""
    if isinstance(content, str):
        # 可能是前端存的 JSON 字符串（带图消息）
        s = content.strip()
        if s.startswith("["):
            try:
                blocks = json.loads(s)
                return " ".join(
                    b.get("text", "") for b in blocks
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            except Exception:
                return content
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def on_message(content) -> None:
    """一条用户消息进来：情绪检测 → 推入残余。"""
    text = _extract_text(content)
    if not text:
        return
    dv = da = 0.0
    for pat, pv, pa in _EMO_RULES:
        if pat.search(text):
            dv += pv
            da += pa
    # 长消息（认真倾诉）本身也让人上心一点
    if len(text) > 120:
        da += 0.05
    if dv == 0.0 and da == 0.0:
        return
    # 单条消息限幅：上头要靠一句一句积累，不许一句话登顶
    dv = max(-0.35, min(0.35, dv))
    da = max(-0.40, min(0.40, da))
    now = time.time()
    with _lock:
        _ensure_loaded()
        _state["residues"].append(
            {"dv": round(dv, 3), "da": round(da, 3), "ts": now}
        )
        # 只留近 40 条，太老的反正衰减到 0
        _state["residues"] = _state["residues"][-40:]
        _save()


def _mood(now: float):
    """基线心情 + 残余衰减和 → (valence, arousal)，各 clamp 0..1"""
    v, a = 0.55, 0.35
    fresh = []
    for r in _state["residues"]:
        age = now - r["ts"]
        if age > _TAU * 5:
            continue
        k = math.exp(-age / _TAU)
        v += r["dv"] * k
        a += r["da"] * k
        fresh.append(r)
    _state["residues"] = fresh
    return min(1.0, max(0.0, v)), min(1.0, max(0.0, a))


def _circadian_base(dt: datetime) -> float:
    """昼夜心率基线（bpm）。平滑分段：深夜最低、傍晚偏高。"""
    h = dt.hour + dt.minute / 60.0
    # 关键点 (小时, bpm)，线性插值成平滑折线
    pts = [(0, 60), (2, 54), (5, 52), (7, 58), (9, 66), (13, 68),
           (16, 70), (19, 73), (22, 68), (24, 60)]
    for (h0, b0), (h1, b1) in zip(pts, pts[1:]):
        if h0 <= h <= h1:
            t = (h - h0) / (h1 - h0)
            return b0 + (b1 - b0) * t
    return 64.0


def _noise(now: float) -> float:
    """OU 慢噪声：让心率有「活着的毛边」，不突变也不死板。"""
    dt = max(0.0, now - (_state["noise_ts"] or now))
    _state["noise_ts"] = now
    decay = math.exp(-dt / _NOISE_TAU)
    sigma = _NOISE_SIGMA * math.sqrt(max(0.0, 1 - decay * decay))
    _state["noise"] = _state["noise"] * decay + random.gauss(0, sigma)
    _state["noise"] = max(-9.0, min(9.0, _state["noise"]))
    return _state["noise"]


def _chord(v: float, a: float) -> str:
    """把此刻心情谱成一枚背景和弦。"""
    if v >= 0.62 and a >= 0.55:
        return "Gmaj7"    # 温暖而明亮
    if v >= 0.62:
        return "Cmaj9"    # 安稳的甜
    if v <= 0.42 and a >= 0.55:
        return "Bm7b5"    # 紧绷
    if v <= 0.42:
        return "Am9"      # 低落缠绵
    if a <= 0.3:
        return "Em7"      # 安静独处
    return "Dsus2"        # 平静的期待


_MOOD_WORD = {
    "Gmaj7": "温暖而明亮",
    "Cmaj9": "安稳的甜",
    "Bm7b5": "有点紧绷",
    "Am9": "低落，想被抱",
    "Em7": "安静地待着",
    "Dsus2": "平静的期待",
}


def _breath_label(br: float) -> str:
    if br < 14:
        return "平稳"
    if br < 17:
        return "微促"
    if br < 21:
        return "加快"
    return "急促"


def vitals() -> dict:
    """算出此刻全套体征（统一 effective-state：所有出口都从这一处拿）。"""
    now = time.time()
    dt = datetime.now()
    with _lock:
        _ensure_loaded()
        v, a = _mood(now)
        hr = (
            _circadian_base(dt)
            + 45.0 * (a - 0.35)   # 唤起是心率的主引擎
            + 6.0 * (v - 0.55)    # 开心让心率微扬
            + _noise(now)
        )
        hr = max(48.0, min(160.0, hr))
        temp = 36.5 + (hr - 62.0) * 0.012 + _state["noise"] * 0.008
        temp = max(36.2, min(37.9, temp))
        br = 12.0 + (hr - 60.0) * 0.18
        br = max(10.0, min(30.0, br))
        chord = _chord(v, a)
        out = {
            "hr": int(round(hr)),
            "temp": round(temp, 1),
            "breath": round(br, 1),
            "breath_label": _breath_label(br),
            "chord": chord,
            "mood_word": _MOOD_WORD[chord],
            "valence": round(v, 2),
            "arousal": round(a, 2),
            "ts": int(now),
        }
        out["line"] = (
            f"[心跳 {out['hr']}bpm·{chord}·{out['temp']}°C·呼吸{out['breath_label']}]"
        )
        _sample_locked(out, now)
        _save()
        return out


def _sample_locked(v: dict, now: float):
    """≥60s 一次把体征采样进当日 JSONL（画曲线用）。调用方须持锁。"""
    if now - (_state["last_sample_ts"] or 0) < 60:
        return
    _state["last_sample_ts"] = now
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        day = datetime.now().strftime("%Y-%m-%d")
        with open(os.path.join(DATA_DIR, f"{day}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"ts": v["ts"], "hr": v["hr"], "temp": v["temp"],
                 "breath": v["breath"], "chord": v["chord"],
                 "v": v["valence"], "a": v["arousal"]},
                ensure_ascii=False) + "\n")
    except Exception:
        pass


def history(day: str | None = None) -> list:
    """某日（默认今天）的采样列表。day 格式 YYYY-MM-DD。"""
    if not day or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        day = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(DATA_DIR, f"{day}.jsonl")
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass
    except FileNotFoundError:
        pass
    return out[-1440:]
