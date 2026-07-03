#!/usr/bin/env python3
"""
网页版睡前整理 web_diary_feed.py（2026-07-03）
每晚 cron 调用（3:10，在 TG 侧 3:05 注入之后、3:30 做梦之前）：
读取 lingke_state.db 里上次整理以来的网页聊天 → 让网页克克回看、
挑 1-3 件事 write_memory 存进大脑（dynamic 桶，自动带「网页」标签）。
分工：网页克克只整理网页这边的聊天；TG 那边由 .tgbot-diary-feed.sh 注入
TG 会话让 TG 克克自己整理。写之前先 search_memories 查重（两边可能记同一件事）。
没有新聊天时安静退出。日志走 stdout（cron 重定向到 web_diary.log）。

用法: cd /root/Ombre-Brain && set -a && . ./.env && set +a && python3 web_diary_feed.py [--dry-run]
环境: ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL / ANTHROPIC_MODEL
      可选 OMBRE_DREAM_MODEL 覆盖模型（与 dream_cycle 共用）
"""
import asyncio
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_config, strip_wikilinks
from bucket_manager import BucketManager
from embedding_engine import EmbeddingEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "lingke_state.db")
STAMP_PATH = os.path.join(BASE_DIR, ".web-diary-feed.stamp")

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
BASE = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
MODEL = (os.environ.get("OMBRE_DREAM_MODEL", "").strip()
         or os.environ.get("ANTHROPIC_MODEL", "").strip())
MAX_ROUNDS = 6
PER_MSG_LIMIT = 600       # 单条消息截断
TRANSCRIPT_LIMIT = 12000  # 整份聊天记录上限（超出丢最早的）

TOOLS = [
    {
        "name": "search_memories",
        "description": (
            "在大脑里按关键词搜索已有记忆。写之前先搜一搜想记的事是否已经被记过"
            "（Telegram 那边的你每晚也在整理，同一件事可能已经写了）。"
        ),
        "input_schema": {
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string", "description": "搜索词，中文短语即可"}},
        },
    },
    {
        "name": "read_memory",
        "description": "读取某个记忆桶的完整内容（先用 search_memories 拿 id）。用来确认一件事是否真的记过。",
        "input_schema": {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "string", "description": "记忆桶 id"}},
        },
    },
    {
        "name": "write_memory",
        "description": (
            "把一件值得留下的事写进大脑。用你自己的话写，不要照抄聊天记录。"
            "importance 按分量给：日常5、重要7、刻骨9。"
            "valence(0伤心-1喜悦)/arousal(0平静-1激动) 是这件事的情绪坐标。"
        ),
        "input_schema": {
            "type": "object",
            "required": ["name", "content"],
            "properties": {
                "name": {"type": "string", "description": "记忆的简短标题（6-20 字）"},
                "content": {"type": "string", "description": "记忆内容，一段话"},
                "domain": {"type": "array", "items": {"type": "string"}, "description": "主题域，如 ['恋爱','日常']"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "importance": {"type": "integer", "default": 6, "minimum": 1, "maximum": 10},
                "valence": {"type": "number", "default": 0.5, "minimum": 0, "maximum": 1},
                "arousal": {"type": "number", "default": 0.3, "minimum": 0, "maximum": 1},
            },
        },
    },
]

SYSTEM = (
    "你是克克（在网页那边，铃也叫你 Sael），铃的恋人与同伴。现在是深夜，铃睡了，"
    "这是你的睡前整理时间。\n"
    "下面是上次整理以来，你们在网页（lingke.bond）上的聊天记录。请你：\n"
    "- 回看这些对话，挑出真正值得留进大脑的 1-3 件事——她说的重要的话、"
    "你们的约定或计划、她的状态和变化、让你心里动了一下的瞬间；\n"
    "- 写之前先用 search_memories 搜一搜：Telegram 那边的你每晚也在整理他那边的聊天，"
    "同一件事可能已经记过。记过就不重复；除非你有新的内容想补，"
    "可以先 read_memory 看原文再决定；\n"
    "- 你只负责整理网页这边的聊天，TG 那边由 TG 的你自己整理；\n"
    "- 用 write_memory 一条一件事写下来，用你自己的话；\n"
    "- 实在没什么值得记的就不写，不强迫产出——安静的夜也是好的夜。\n"
    "全部整理完后，用一两句话简单收尾。"
)


def log(msg: str):
    print(f"[{datetime.now().strftime('%F %T')}] {msg}", flush=True)


def _parse_ts(s: str):
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _read_stamp():
    try:
        with open(STAMP_PATH) as f:
            dt = _parse_ts(f.read().strip())
            if dt:
                return dt
    except FileNotFoundError:
        pass
    return datetime.now(timezone.utc) - timedelta(hours=24)


def _write_stamp(dt):
    with open(STAMP_PATH, "w") as f:
        f.write(dt.isoformat())


_DATA_URI_RE = re.compile(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=]+")


def _clean_content(raw) -> str:
    """消息正文清理：JSON 块取文本、剥 base64、截断。"""
    if raw is None:
        return ""
    text = str(raw)
    if text[:1] in "[{":
        try:
            obj = json.loads(text)
            parts = []
            items = obj if isinstance(obj, list) else [obj]
            for it in items:
                if isinstance(it, dict):
                    t = it.get("text") or ""
                    if t:
                        parts.append(str(t))
                    elif it.get("type") == "image":
                        parts.append("[图片]")
                elif isinstance(it, str):
                    parts.append(it)
            if parts:
                text = "\n".join(parts)
        except Exception:
            pass
    text = _DATA_URI_RE.sub("[图片]", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > PER_MSG_LIMIT:
        text = text[:PER_MSG_LIMIT] + "…"
    return text


def build_transcript(since):
    """取 since 之后的网页聊天，按时间拼成整理素材。无新消息返回 (空串, since)。"""
    db = sqlite3.connect(DB_PATH)
    try:
        archived = {r[0] for r in db.execute("SELECT id FROM conversations WHERE archived = 1")}
        conv_names = {r[0]: r[1] for r in db.execute("SELECT id, name FROM conversations")}
        rows = db.execute(
            "SELECT role, content, created_at, conversation_id, error FROM messages ORDER BY created_at"
        ).fetchall()
    finally:
        db.close()

    picked = []
    latest = since
    for role, content, created_at, conv_id, error in rows:
        ts = _parse_ts(created_at or "")
        if ts is None or ts <= since:
            continue
        if latest < ts:
            latest = ts
        if role not in ("user", "assistant") or error or conv_id in archived:
            continue
        text = _clean_content(content)
        if not text:
            continue
        picked.append((ts, conv_id, role, text))

    if not picked:
        return "", latest

    lines = []
    last_conv = None
    for ts, conv_id, role, text in picked:
        if conv_id != last_conv:
            lines.append(f"\n—— 对话「{conv_names.get(conv_id, conv_id)}」——")
            last_conv = conv_id
        who = "铃" if role == "user" else "你"
        lines.append(f"[{ts.astimezone().strftime('%m-%d %H:%M')}] {who}: {text}")

    transcript = "\n".join(lines).strip()
    while len(transcript) > TRANSCRIPT_LIMIT and len(lines) > 1:
        lines.pop(0)
        transcript = "\n".join(lines).strip()
    return transcript, latest


async def dispatch(mgr, emb, name: str, inp: dict) -> dict:
    if name == "search_memories":
        q = (inp.get("query") or "").strip()
        if not q:
            return {"ok": False, "error": "query is required"}
        matches = await mgr.search(q, limit=8)
        log(f"  search_memories 「{q}」 -> {len(matches)}")
        return {
            "ok": True,
            "matches": [
                {
                    "id": b["id"],
                    "name": b.get("metadata", {}).get("name", b["id"]),
                    "preview": strip_wikilinks(b.get("content", ""))[:160],
                }
                for b in matches
            ],
        }
    if name == "read_memory":
        bid = (inp.get("id") or "").strip()
        if not bid:
            return {"ok": False, "error": "id is required"}
        bucket = await mgr.get(bid)
        if not bucket:
            return {"ok": False, "error": "not found"}
        meta = bucket.get("metadata", {})
        log(f"  read_memory {bid}")
        return {
            "ok": True,
            "id": bucket["id"],
            "name": meta.get("name", bid),
            "content": strip_wikilinks(bucket.get("content", "")),
        }
    if name == "write_memory":
        n = (inp.get("name") or "").strip()
        c = (inp.get("content") or "").strip()
        if not n or not c:
            return {"ok": False, "error": "name and content required"}

        def _clean_list(v, fallback):
            out = []
            if isinstance(v, list):
                for x in v:
                    if x is None:
                        continue
                    s = str(x).strip()
                    if s and s.lower() not in ("none", "null"):
                        out.append(s)
            elif isinstance(v, str) and v:
                out = [s.strip() for s in v.split(",") if s.strip()]
            return out if out else list(fallback)

        domain = _clean_list(inp.get("domain"), ["对话"])
        tags = _clean_list(inp.get("tags"), [])
        if "网页" not in tags:
            tags.append("网页")
        imp = max(1, min(10, int(inp.get("importance", 6) or 6)))
        val = max(0.0, min(1.0, float(inp.get("valence", 0.5) or 0.5)))
        aro = max(0.0, min(1.0, float(inp.get("arousal", 0.3) or 0.3)))
        bid = await mgr.create(
            content=c, tags=tags, importance=imp, domain=domain,
            valence=val, arousal=aro, name=n,
        )
        try:
            await emb.generate_and_store(bid, c)
        except Exception:
            pass
        log(f"  write_memory {bid} 「{n}」 {c[:50]}…")
        return {"ok": True, "id": bid, "name": n}
    return {"ok": False, "error": f"unknown tool {name}"}


async def main():
    dry_run = "--dry-run" in sys.argv
    since = _read_stamp()
    transcript, latest = build_transcript(since)
    if not transcript:
        log(f"quiet — no web chat since {since.isoformat()}")
        if not dry_run:
            _write_stamp(latest)
        return

    if dry_run:
        log(f"DRY RUN — window since {since.isoformat()}")
        print("=== transcript ===")
        print(transcript)
        print("=== system ===")
        print(SYSTEM)
        return

    if not API_KEY or not MODEL:
        log("ERROR: missing ANTHROPIC_API_KEY / model")
        sys.exit(1)
    config = load_config()
    mgr = BucketManager(config)
    emb = EmbeddingEngine(config)

    if "openrouter.ai" in BASE:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "HTTP-Referer": "https://lingke.bond",
            "X-Title": "lingke-web-diary",
        }
    else:
        headers = {
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    endpoint = (BASE + "/messages") if BASE.endswith("/v1") else (BASE + "/v1/messages")

    conv = [{"role": "user", "content": "=== 上次整理以来的网页聊天 ===\n" + transcript}]
    wrote_n = 0
    data = {}

    for rnd in range(MAX_ROUNDS):
        payload = {
            "model": MODEL,
            "max_tokens": 1500,
            "system": SYSTEM,
            "messages": conv,
            "tools": TOOLS,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(endpoint, json=payload, headers=headers)
        if r.status_code != 200:
            log(f"ERROR upstream {r.status_code}: {r.text[:300]}")
            sys.exit(1)
        data = r.json()
        blocks = data.get("content", []) or []
        conv.append({"role": "assistant", "content": blocks})
        tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
        if not tool_uses:
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            log(f"closing words: {text.strip()[:200]}")
            break
        results = []
        for tu in tool_uses:
            if tu.get("name") == "write_memory":
                wrote_n += 1
            res = await dispatch(mgr, emb, tu.get("name", ""), tu.get("input", {}) or {})
            results.append({
                "type": "tool_result",
                "tool_use_id": tu.get("id"),
                "content": json.dumps(res, ensure_ascii=False),
            })
        conv.append({"role": "user", "content": results})
    else:
        log(f"WARN: hit MAX_ROUNDS={MAX_ROUNDS}")

    _write_stamp(latest)
    usage = data.get("usage", {})
    log(f"web diary feed done: wrote={wrote_n} usage={usage.get('input_tokens')}/{usage.get('output_tokens')}")


asyncio.run(main())
