# ============================================================
# Module: MCP Server Entry Point (server.py)
# 模块：MCP 服务器主入口
#
# Starts the Ombre Brain MCP service and registers memory
# operation tools for Claude to call.
# 启动 Ombre Brain MCP 服务，注册记忆操作工具供 Claude 调用。
#
# Core responsibilities:
# 核心职责：
#   - Initialize config, bucket manager, dehydrator, decay engine
#     初始化配置、记忆桶管理器、脱水器、衰减引擎
#   - Expose 6 MCP tools:
#     暴露 6 个 MCP 工具：
#       breath — Surface unresolved memories or search by keyword
#                浮现未解决记忆 或 按关键词检索
#       hold   — Store a single memory (or write a `feel` reflection)
#                存储单条记忆（或写 feel 反思）
#       grow   — Diary digest, auto-split into multiple buckets
#                日记归档，自动拆分多桶
#       trace  — Modify metadata / resolved / delete
#                修改元数据 / resolved 标记 / 删除
#       pulse  — System status + bucket listing
#                系统状态 + 所有桶列表
#       dream  — Surface recent dynamic buckets for self-digestion
#                返回最近桶 供模型自省/写 feel
#
# Startup:
# 启动方式：
#   Local:  python server.py
#   Remote: OMBRE_TRANSPORT=streamable-http python server.py
#   Docker: docker-compose up
# ============================================================

import os
import sys
import random
import logging
import asyncio
import hashlib
import hmac
import secrets
import time
import json as _json_lib
import httpx


# --- Ensure same-directory modules can be imported ---
# --- 确保同目录下的模块能被正确导入 ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

from bucket_manager import BucketManager
from dehydrator import Dehydrator
from decay_engine import DecayEngine
from embedding_engine import EmbeddingEngine
from import_memory import ImportEngine
from utils import load_config, setup_logging, strip_wikilinks, count_tokens_approx

# --- Load config & init logging / 加载配置 & 初始化日志 ---
config = load_config()
setup_logging(config.get("log_level", "INFO"))
logger = logging.getLogger("ombre_brain")

# --- Runtime env vars (port + webhook) / 运行时环境变量 ---
# OMBRE_PORT: HTTP/SSE 监听端口，默认 8000
try:
    OMBRE_PORT = int(os.environ.get("OMBRE_PORT", "8000") or "8000")
except ValueError:
    logger.warning("OMBRE_PORT 不是合法整数，回退到 8000")
    OMBRE_PORT = 8000

# OMBRE_HOOK_URL: 在 breath/dream 被调用后推送事件到该 URL（POST JSON）。
# OMBRE_HOOK_SKIP: 设为 true/1/yes 跳过推送。
# 详见 ENV_VARS.md。
OMBRE_HOOK_URL = os.environ.get("OMBRE_HOOK_URL", "").strip()
OMBRE_HOOK_SKIP = os.environ.get("OMBRE_HOOK_SKIP", "").strip().lower() in ("1", "true", "yes", "on")


async def _fire_webhook(event: str, payload: dict) -> None:
    """
    Fire-and-forget POST to OMBRE_HOOK_URL with the given event payload.
    Failures are logged at WARNING level only — never propagated to the caller.
    """
    if OMBRE_HOOK_SKIP or not OMBRE_HOOK_URL:
        return
    try:
        body = {
            "event": event,
            "timestamp": time.time(),
            "payload": payload,
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(OMBRE_HOOK_URL, json=body)
    except Exception as e:
        logger.warning(f"Webhook push failed ({event} → {OMBRE_HOOK_URL}): {e}")

# --- Initialize core components / 初始化核心组件 ---
embedding_engine = EmbeddingEngine(config)            # Embedding engine first (BucketManager depends on it)
bucket_mgr = BucketManager(config, embedding_engine=embedding_engine)  # Bucket manager / 记忆桶管理器
dehydrator = Dehydrator(config)                      # Dehydrator / 脱水器
decay_engine = DecayEngine(config, bucket_mgr)       # Decay engine / 衰减引擎
import_engine = ImportEngine(config, bucket_mgr, dehydrator, embedding_engine)  # Import engine / 导入引擎

# --- Create MCP server instance / 创建 MCP 服务器实例 ---
# host="0.0.0.0" so Docker container's SSE is externally reachable
# stdio mode ignores host (no network)
mcp = FastMCP(
    "Ombre Brain",
    host="0.0.0.0",
    port=OMBRE_PORT,
)


# =============================================================
# Dashboard Auth — simple cookie-based session auth
# Dashboard 认证 —— 基于 Cookie 的会话认证
#
# Env var OMBRE_DASHBOARD_PASSWORD overrides file-stored password.
# First visit with no password set → forced setup wizard.
# Sessions persisted to .sessions.json (survive restart, 7-day expiry).
# =============================================================
_SESSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sessions.json")


def _load_sessions() -> dict:
    try:
        if os.path.exists(_SESSION_FILE):
            with open(_SESSION_FILE, "r") as f:
                data = _json_lib.load(f)
                now = time.time()
                return {k: v for k, v in data.items() if v > now}
    except Exception:
        pass
    return {}


def _save_sessions():
    try:
        tmp = _SESSION_FILE + ".tmp"
        with open(tmp, "w") as f:
            _json_lib.dump(_sessions, f)
        os.replace(tmp, _SESSION_FILE)
    except Exception:
        pass


_sessions: dict[str, float] = _load_sessions()


def _get_auth_file() -> str:
    return os.path.join(config["buckets_dir"], ".dashboard_auth.json")


def _load_password_hash() -> str | None:
    try:
        auth_file = _get_auth_file()
        if os.path.exists(auth_file):
            with open(auth_file, "r", encoding="utf-8") as f:
                return _json_lib.load(f).get("password_hash")
    except Exception:
        pass
    return None


def _save_password_hash(password: str) -> None:
    salt = secrets.token_hex(16)
    h = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    auth_file = _get_auth_file()
    os.makedirs(os.path.dirname(auth_file), exist_ok=True)
    with open(auth_file, "w", encoding="utf-8") as f:
        _json_lib.dump({"password_hash": f"{salt}:{h}"}, f)


def _verify_password_hash(password: str, stored: str) -> bool:
    if ":" not in stored:
        return False
    salt, h = stored.split(":", 1)
    return hmac.compare_digest(
        h, hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    )


def _is_setup_needed() -> bool:
    """True if no password is configured (env var or file)."""
    if os.environ.get("OMBRE_DASHBOARD_PASSWORD", ""):
        return False
    return _load_password_hash() is None


def _verify_any_password(password: str) -> bool:
    """Check password against env var (first) or stored hash."""
    env_pwd = os.environ.get("OMBRE_DASHBOARD_PASSWORD", "")
    if env_pwd:
        return hmac.compare_digest(password, env_pwd)
    stored = _load_password_hash()
    if not stored:
        return False
    return _verify_password_hash(password, stored)


def _create_session() -> str:
    token = secrets.token_urlsafe(32)
    _sessions[token] = time.time() + 86400 * 7  # 7-day expiry
    _save_sessions()
    return token


def _is_authenticated(request) -> bool:
    token = request.cookies.get("ombre_session")
    if not token:
        return False
    expiry = _sessions.get(token)
    if expiry is None or time.time() > expiry:
        _sessions.pop(token, None)
        _save_sessions()
        return False
    return True


def _require_auth(request):
    """Return JSONResponse(401) if not authenticated, else None."""
    from starlette.responses import JSONResponse
    if not _is_authenticated(request):
        return JSONResponse(
            {"error": "Unauthorized", "setup_needed": _is_setup_needed()},
            status_code=401,
        )
    return None


# --- Auth endpoints ---
@mcp.custom_route("/auth/status", methods=["GET"])
async def auth_status(request):
    """Return auth state (authenticated, setup_needed)."""
    from starlette.responses import JSONResponse
    return JSONResponse({
        "authenticated": _is_authenticated(request),
        "setup_needed": _is_setup_needed(),
    })


@mcp.custom_route("/auth/setup", methods=["POST"])
async def auth_setup_endpoint(request):
    """Initial password setup (only when no password is configured)."""
    from starlette.responses import JSONResponse
    if not _is_setup_needed():
        return JSONResponse({"error": "Already configured"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    password = body.get("password", "").strip()
    if len(password) < 6:
        return JSONResponse({"error": "密码不能少于6位"}, status_code=400)
    _save_password_hash(password)
    token = _create_session()
    resp = JSONResponse({"ok": True})
    resp.set_cookie("ombre_session", token, httponly=True, samesite="lax", max_age=86400 * 7)
    return resp


@mcp.custom_route("/auth/login", methods=["POST"])
async def auth_login(request):
    """Login with password."""
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    password = body.get("password", "")
    if _verify_any_password(password):
        token = _create_session()
        resp = JSONResponse({"ok": True})
        resp.set_cookie("ombre_session", token, httponly=True, samesite="lax", max_age=86400 * 7)
        return resp
    return JSONResponse({"error": "密码错误"}, status_code=401)


@mcp.custom_route("/auth/logout", methods=["POST"])
async def auth_logout(request):
    """Invalidate session."""
    from starlette.responses import JSONResponse
    token = request.cookies.get("ombre_session")
    if token:
        _sessions.pop(token, None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("ombre_session")
    return resp


@mcp.custom_route("/auth/change-password", methods=["POST"])
async def auth_change_password(request):
    """Change dashboard password (requires current password)."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err:
        return err
    if os.environ.get("OMBRE_DASHBOARD_PASSWORD", ""):
        return JSONResponse({"error": "当前使用环境变量密码，请直接修改 OMBRE_DASHBOARD_PASSWORD"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    current = body.get("current", "")
    new_pwd = body.get("new", "").strip()
    if not _verify_any_password(current):
        return JSONResponse({"error": "当前密码错误"}, status_code=401)
    if len(new_pwd) < 6:
        return JSONResponse({"error": "新密码不能少于6位"}, status_code=400)
    _save_password_hash(new_pwd)
    _sessions.clear()
    _save_sessions()
    token = _create_session()
    resp = JSONResponse({"ok": True})
    resp.set_cookie("ombre_session", token, httponly=True, samesite="lax", max_age=86400 * 7)
    return resp


# =============================================================
# /health endpoint: lightweight keepalive
# 轻量保活接口
# For Cloudflare Tunnel or reverse proxy to ping, preventing idle timeout
# 供 Cloudflare Tunnel 或反代定期 ping，防止空闲超时断连
# =============================================================
@mcp.custom_route("/", methods=["GET"])
async def root_redirect(request):
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    from starlette.responses import JSONResponse
    try:
        # Self-heal: visiting /health revives the decay engine if it ever stopped
        # 自愈：访问 /health 时若衰减引擎未在运行则顺手拉起
        await decay_engine.ensure_started()
        stats = await bucket_mgr.get_stats()
        return JSONResponse({
            "status": "ok",
            "buckets": stats["permanent_count"] + stats["dynamic_count"],
            "decay_engine": "running" if decay_engine.is_running else "stopped",
        })
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# =============================================================
# /breath-hook endpoint: Dedicated hook for SessionStart
# 会话启动专用挂载点
# =============================================================
@mcp.custom_route("/breath-hook", methods=["GET"])
async def breath_hook(request):
    from starlette.responses import PlainTextResponse
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        # pinned
        pinned = [b for b in all_buckets if b["metadata"].get("pinned") or b["metadata"].get("protected")]
        # top 2 unresolved by score
        unresolved = [b for b in all_buckets
                      if not b["metadata"].get("resolved", False)
                      and b["metadata"].get("type") not in ("permanent", "feel")
                      and not b["metadata"].get("pinned")
                      and not b["metadata"].get("protected")]
        scored = sorted(unresolved, key=lambda b: decay_engine.calculate_score(b["metadata"]), reverse=True)

        parts = []
        token_budget = 10000
        for b in pinned:
            summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), {k: v for k, v in b["metadata"].items() if k != "tags"})
            parts.append(f"📌 [核心准则] {summary}")
            token_budget -= count_tokens_approx(summary)

        # Diversity: top-1 fixed + shuffle rest from top-20
        candidates = list(scored)
        if len(candidates) > 1:
            top1 = [candidates[0]]
            pool = candidates[1:min(20, len(candidates))]
            random.shuffle(pool)
            candidates = top1 + pool + candidates[min(20, len(candidates)):]
        # Hard cap: max 20 surfacing buckets in hook
        candidates = candidates[:20]

        for b in candidates:
            if token_budget <= 0:
                break
            summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), {k: v for k, v in b["metadata"].items() if k != "tags"})
            summary_tokens = count_tokens_approx(summary)
            if summary_tokens > token_budget:
                break
            parts.append(summary)
            token_budget -= summary_tokens

        if not parts:
            await _fire_webhook("breath_hook", {"surfaced": 0})
            return PlainTextResponse("")
        body_text = "[Ombre Brain - 记忆浮现]\n" + "\n---\n".join(parts)
        await _fire_webhook("breath_hook", {"surfaced": len(parts), "chars": len(body_text)})
        return PlainTextResponse(body_text)
    except Exception as e:
        logger.warning(f"Breath hook failed: {e}")
        return PlainTextResponse("")


# =============================================================
# /dream-hook endpoint: Dedicated hook for Dreaming
# Dreaming 专用挂载点
# =============================================================
@mcp.custom_route("/dream-hook", methods=["GET"])
async def dream_hook(request):
    from starlette.responses import PlainTextResponse
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        candidates = [
            b for b in all_buckets
            if b["metadata"].get("type") not in ("permanent", "feel")
            and not b["metadata"].get("pinned", False)
            and not b["metadata"].get("protected", False)
        ]
        candidates.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
        recent = candidates[:10]

        if not recent:
            return PlainTextResponse("")

        parts = []
        for b in recent:
            meta = b["metadata"]
            resolved_tag = "[已解决]" if meta.get("resolved", False) else "[未解决]"
            parts.append(
                f"{meta.get('name', b['id'])} {resolved_tag} "
                f"V{meta.get('valence', 0.5):.1f}/A{meta.get('arousal', 0.3):.1f}\n"
                f"{strip_wikilinks(b['content'][:200])}"
            )

        body_text = "[Ombre Brain - Dreaming]\n" + "\n---\n".join(parts)
        await _fire_webhook("dream_hook", {"surfaced": len(parts), "chars": len(body_text)})
        return PlainTextResponse(body_text)
    except Exception as e:
        logger.warning(f"Dream hook failed: {e}")
        return PlainTextResponse("")


# =============================================================
# Internal helper: merge-or-create
# 内部辅助：检查是否可合并，可以则合并，否则新建
# Shared by hold and grow to avoid duplicate logic
# hold 和 grow 共用，避免重复逻辑
# =============================================================
async def _merge_or_create(
    content: str,
    tags: list,
    importance: int,
    domain: list,
    valence: float,
    arousal: float,
    name: str = "",
) -> tuple[str, str, bool]:
    """
    Check if a similar bucket exists for merging; merge if so, create if not.
    Returns (bucket_id, display_name, is_merged).
    检查是否有相似桶可合并，有则合并，无则新建。
    返回 (桶ID, 展示名, 是否合并)。
    """
    # --- Channel 1: embedding 余弦相似度（主判重信号，纯内容比内容）---
    # 旧的混合检索分掺了时间/重要度/情绪权重，内容完全相同也很难过 75 阈值，
    # 导致合并机制实际上从未触发过（2026-06-12 实测确认）。
    candidate = None
    cand_via = ""
    if embedding_engine and embedding_engine.enabled:
        try:
            vr = await embedding_engine.search_similar(content, top_k=1)
            if vr:
                vid, sim = vr[0]
                sim_threshold = config.get("merge_embedding_threshold", 0.85)
                logger.info(
                    f"merge check: top embedding sim={sim:.4f} "
                    f"(threshold {sim_threshold}, bucket {vid})"
                )
                if sim >= sim_threshold:
                    candidate = await bucket_mgr.get(vid)
                    cand_via = f"embedding sim={sim:.4f}"
        except Exception as e:
            logger.warning(f"Embedding merge check failed / 向量判重失败: {e}")

    # --- Channel 2: 关键词混合分兜底（embedding 不可用/无命中时）---
    if candidate is None:
        try:
            existing = await bucket_mgr.search(content, limit=1, domain_filter=domain or None)
        except Exception as e:
            logger.warning(f"Search for merge failed, creating new / 合并搜索失败，新建: {e}")
            existing = []
        if existing and existing[0].get("score", 0) > config.get("merge_threshold", 75):
            candidate = existing[0]
            cand_via = f"keyword score={existing[0].get('score')}"

    if candidate is not None:
        bucket = candidate
        # --- Only merge into plain dynamic buckets ---
        # --- 只并入普通 dynamic 桶：钉选/保护/固化/feel/归档一律不动 ---
        if (
            not (bucket["metadata"].get("pinned") or bucket["metadata"].get("protected"))
            and bucket["metadata"].get("type") in (None, "", "dynamic")
        ):
            logger.info(f"merging into {bucket['id']} via {cand_via}")
            try:
                merged = await dehydrator.merge(bucket["content"], content)
                old_v = bucket["metadata"].get("valence", 0.5)
                old_a = bucket["metadata"].get("arousal", 0.3)
                merged_valence = round((old_v + valence) / 2, 2)
                merged_arousal = round((old_a + arousal) / 2, 2)
                await bucket_mgr.update(
                    bucket["id"],
                    content=merged,
                    tags=list(set(bucket["metadata"].get("tags", []) + tags)),
                    importance=max(bucket["metadata"].get("importance", 5), importance),
                    domain=list(set(bucket["metadata"].get("domain", []) + domain)),
                    valence=merged_valence,
                    arousal=merged_arousal,
                )
                # --- Update embedding after merge ---
                try:
                    await embedding_engine.generate_and_store(bucket["id"], merged)
                except Exception:
                    pass
                return bucket["id"], bucket["metadata"].get("name", bucket["id"]), True
            except Exception as e:
                logger.warning(f"Merge failed, creating new / 合并失败，新建: {e}")

    bucket_id = await bucket_mgr.create(
        content=content,
        tags=tags,
        importance=importance,
        domain=domain,
        valence=valence,
        arousal=arousal,
        name=name or None,
    )
    # --- Generate embedding for new bucket ---
    try:
        await embedding_engine.generate_and_store(bucket_id, content)
    except Exception:
        pass
    return bucket_id, (name or bucket_id), False


# =============================================================
# Tool 1: breath — Breathe
# 工具 1：breath — 呼吸
#
# No args: surface highest-weight unresolved memories (active push)
# 无参数：浮现权重最高的未解决记忆
# With args: search by keyword + emotion coordinates
# 有参数：按关键词+情感坐标检索记忆
# =============================================================
@mcp.tool()
async def breath(
    query: str = "",
    max_tokens: int = 10000,
    domain: str = "",
    valence: float = -1,
    arousal: float = -1,
    max_results: int = 20,
    importance_min: int = -1,
) -> str:
    """检索/浮现记忆。不传query或传空=自动浮现,有query=关键词检索。max_tokens控制返回总token上限(默认10000)。domain逗号分隔,valence/arousal 0~1(-1忽略)。max_results控制返回数量上限(默认20,最大50)。importance_min>=1时按重要度批量拉取(不走语义搜索,按importance降序返回最多20条)。"""
    await decay_engine.ensure_started()
    max_results = min(max_results, 50)
    max_tokens = min(max_tokens, 20000)

    # --- importance_min mode: bulk fetch by importance threshold ---
    # --- 重要度批量拉取模式：跳过语义搜索，按 importance 降序返回 ---
    if importance_min >= 1:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            return f"记忆系统暂时无法访问: {e}"
        filtered = [
            b for b in all_buckets
            if int(b["metadata"].get("importance", 0)) >= importance_min
            and b["metadata"].get("type") not in ("feel",)
        ]
        filtered.sort(key=lambda b: int(b["metadata"].get("importance", 0)), reverse=True)
        filtered = filtered[:20]
        if not filtered:
            return f"没有重要度 >= {importance_min} 的记忆。"
        results = []
        token_used = 0
        for b in filtered:
            if token_used >= max_tokens:
                break
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                t = count_tokens_approx(summary)
                if token_used + t > max_tokens:
                    break
                imp = b["metadata"].get("importance", 0)
                results.append(f"[importance:{imp}] [bucket_id:{b['id']}] {summary}")
                token_used += t
            except Exception as e:
                logger.warning(f"importance_min dehydrate failed: {e}")
        return "\n---\n".join(results) if results else "没有可以展示的记忆。"

    # --- No args or empty query: surfacing mode (weight pool active push) ---
    # --- 无参数或空query：浮现模式（权重池主动推送）---
    if not query or not query.strip():
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.error(f"Failed to list buckets for surfacing / 浮现列桶失败: {e}")
            return "记忆系统暂时无法访问。"

        # --- Pinned/protected buckets: always surface as core principles ---
        # --- 钉选桶：作为核心准则，始终浮现 ---
        pinned_buckets = [
            b for b in all_buckets
            if b["metadata"].get("pinned") or b["metadata"].get("protected")
        ]
        pinned_results = []
        for b in pinned_buckets:
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                pinned_results.append(f"📌 [核心准则] [bucket_id:{b['id']}] {summary}")
            except Exception as e:
                logger.warning(f"Failed to dehydrate pinned bucket / 钉选桶脱水失败: {e}")
                continue

        # --- Unresolved buckets: surface top N by weight ---
        # --- 未解决桶：按权重浮现前 N 条 ---
        unresolved = [
            b for b in all_buckets
            if not b["metadata"].get("resolved", False)
            and b["metadata"].get("type") not in ("permanent", "feel")
            and not b["metadata"].get("pinned", False)
            and not b["metadata"].get("protected", False)
        ]

        logger.info(
            f"Breath surfacing: {len(all_buckets)} total, "
            f"{len(pinned_buckets)} pinned, {len(unresolved)} unresolved"
        )

        scored = sorted(
            unresolved,
            key=lambda b: decay_engine.calculate_score(b["metadata"]),
            reverse=True,
        )

        if scored:
            top_scores = [(b["metadata"].get("name", b["id"]), decay_engine.calculate_score(b["metadata"])) for b in scored[:5]]
            logger.info(f"Top unresolved scores: {top_scores}")

        # --- Cold-start detection: never-seen important buckets surface first ---
        # --- 冷启动检测：从未被访问过且重要度>=8的桶优先插入最前面（最多2个）---
        cold_start = [
            b for b in unresolved
            if int(b["metadata"].get("activation_count", 0)) == 0
            and int(b["metadata"].get("importance", 0)) >= 8
        ][:2]
        cold_start_ids = {b["id"] for b in cold_start}
        # Merge: cold_start first, then scored (excluding duplicates)
        scored_deduped = [b for b in scored if b["id"] not in cold_start_ids]
        scored_with_cold = cold_start + scored_deduped

        # --- Token-budgeted surfacing with diversity + hard cap ---
        # --- 按 token 预算浮现，带多样性 + 硬上限 ---
        # Top-1 always surfaces; rest sampled from top-20 for diversity
        token_budget = max_tokens
        for r in pinned_results:
            token_budget -= count_tokens_approx(r)

        candidates = list(scored_with_cold)
        if len(candidates) > 1:
            # Cold-start buckets stay at front; shuffle rest from top-20
            n_cold = len(cold_start)
            non_cold = candidates[n_cold:]
            if len(non_cold) > 1:
                top1 = [non_cold[0]]
                pool = non_cold[1:min(20, len(non_cold))]
                random.shuffle(pool)
                non_cold = top1 + pool + non_cold[min(20, len(non_cold)):]
            candidates = cold_start + non_cold
        # Hard cap: never surface more than max_results buckets
        candidates = candidates[:max_results]

        dynamic_results = []
        for b in candidates:
            if token_budget <= 0:
                break
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                summary_tokens = count_tokens_approx(summary)
                if summary_tokens > token_budget:
                    break
                # NOTE: no touch() here — surfacing should NOT reset decay timer
                score = decay_engine.calculate_score(b["metadata"])
                dynamic_results.append(f"[权重:{score:.2f}] [bucket_id:{b['id']}] {summary}")
                token_budget -= summary_tokens
            except Exception as e:
                logger.warning(f"Failed to dehydrate surfaced bucket / 浮现脱水失败: {e}")
                continue

        if not pinned_results and not dynamic_results:
            return "权重池平静，没有需要处理的记忆。"

        parts = []
        if pinned_results:
            parts.append("=== 核心准则 ===\n" + "\n---\n".join(pinned_results))
        if dynamic_results:
            parts.append("=== 浮现记忆 ===\n" + "\n---\n".join(dynamic_results))
        return "\n\n".join(parts)

    # --- Feel retrieval: domain="feel" is a special channel ---
    # --- Feel 检索：domain="feel" 是独立入口 ---
    if domain.strip().lower() == "feel":
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            feels = [b for b in all_buckets if b["metadata"].get("type") == "feel"]
            feels.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
            if not feels:
                return "没有留下过 feel。"
            results = []
            for f in feels:
                created = f["metadata"].get("created", "")
                entry = f"[{created}] [bucket_id:{f['id']}]\n{strip_wikilinks(f['content'])}"
                results.append(entry)
                if count_tokens_approx("\n---\n".join(results)) > max_tokens:
                    break
            return "=== 你留下的 feel ===\n" + "\n---\n".join(results)
        except Exception as e:
            logger.error(f"Feel retrieval failed: {e}")
            return "读取 feel 失败。"

    # --- With args: search mode (keyword + vector dual channel) ---
    # --- 有参数：检索模式（关键词 + 向量双通道）---
    domain_filter = [d.strip() for d in domain.split(",") if d.strip()] or None
    q_valence = valence if 0 <= valence <= 1 else None
    q_arousal = arousal if 0 <= arousal <= 1 else None

    try:
        matches = await bucket_mgr.search(
            query,
            limit=max(max_results, 20),
            domain_filter=domain_filter,
            query_valence=q_valence,
            query_arousal=q_arousal,
        )
    except Exception as e:
        logger.error(f"Search failed / 检索失败: {e}")
        return "检索过程出错，请稍后重试。"

    # --- Exclude pinned/protected from search results (they surface in surfacing mode) ---
    # --- 搜索模式排除钉选桶（它们在浮现模式中始终可见）---
    matches = [b for b in matches if not (b["metadata"].get("pinned") or b["metadata"].get("protected"))]

    # --- Vector similarity channel: find semantically related buckets ---
    # --- 向量相似度通道：找到语义相关的桶 ---
    matched_ids = {b["id"] for b in matches}
    try:
        vector_results = await embedding_engine.search_similar(query, top_k=max(max_results, 20))
        for bucket_id, sim_score in vector_results:
            if bucket_id not in matched_ids and sim_score > 0.5:
                bucket = await bucket_mgr.get(bucket_id)
                if bucket and not (bucket["metadata"].get("pinned") or bucket["metadata"].get("protected")):
                    bucket["score"] = round(sim_score * 100, 2)
                    bucket["vector_match"] = True
                    matches.append(bucket)
                    matched_ids.add(bucket_id)
    except Exception as e:
        logger.warning(f"Vector search failed, using keyword only / 向量搜索失败: {e}")

    results = []
    token_used = 0
    for bucket in matches:
        if token_used >= max_tokens:
            break
        try:
            clean_meta = {k: v for k, v in bucket["metadata"].items() if k != "tags"}
            # --- Memory reconstruction: shift displayed valence by current mood ---
            # --- 记忆重构：根据当前情绪微调展示层 valence（±0.1）---
            if q_valence is not None and "valence" in clean_meta:
                original_v = float(clean_meta.get("valence", 0.5))
                shift = (q_valence - 0.5) * 0.2  # ±0.1 max shift
                clean_meta["valence"] = max(0.0, min(1.0, original_v + shift))
            summary = await dehydrator.dehydrate(strip_wikilinks(bucket["content"]), clean_meta)
            summary_tokens = count_tokens_approx(summary)
            if token_used + summary_tokens > max_tokens:
                break
            await bucket_mgr.touch(bucket["id"])
            if bucket.get("vector_match"):
                summary = f"[语义关联] [bucket_id:{bucket['id']}] {summary}"
            else:
                summary = f"[bucket_id:{bucket['id']}] {summary}"
            results.append(summary)
            token_used += summary_tokens
        except Exception as e:
            logger.warning(f"Failed to dehydrate search result / 检索结果脱水失败: {e}")
            continue

    # --- Random surfacing: when search returns < 3, 40% chance to float old memories ---
    # --- 随机浮现：检索结果不足 3 条时，40% 概率从低权重旧桶里漂上来 ---
    if len(matches) < 3 and random.random() < 0.4:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            matched_ids = {b["id"] for b in matches}
            low_weight = [
                b for b in all_buckets
                if b["id"] not in matched_ids
                and decay_engine.calculate_score(b["metadata"]) < 2.0
            ]
            if low_weight:
                drifted = random.sample(low_weight, min(random.randint(1, 3), len(low_weight)))
                drift_results = []
                for b in drifted:
                    clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                    summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                    drift_results.append(f"[surface_type: random]\n{summary}")
                results.append("--- 忽然想起来 ---\n" + "\n---\n".join(drift_results))
        except Exception as e:
            logger.warning(f"Random surfacing failed / 随机浮现失败: {e}")

    if not results:
        await _fire_webhook("breath", {"mode": "empty", "matches": 0})
        return "未找到相关记忆。"

    final_text = "\n---\n".join(results)
    await _fire_webhook("breath", {"mode": "ok", "matches": len(matches), "chars": len(final_text)})
    return final_text


# =============================================================
# Tool 2: hold — Hold on to this
# 工具 2：hold — 握住，留下来
# =============================================================
@mcp.tool()
async def hold(
    content: str,
    tags: str = "",
    importance: int = 5,
    pinned: bool = False,
    feel: bool = False,
    source_bucket: str = "",    valence: float = -1,
    arousal: float = -1,
) -> str:
    """存储单条记忆,自动打标+合并。tags逗号分隔,importance 1-10。pinned=True创建永久钉选桶。feel=True存储你的第一人称感受(不参与普通浮现)。source_bucket=被消化的记忆桶ID(feel模式下,标记源记忆为已消化)。"""
    await decay_engine.ensure_started()

    # --- Input validation / 输入校验 ---
    if not content or not content.strip():
        return "内容为空，无法存储。"

    importance = max(1, min(10, importance))
    extra_tags = [t.strip() for t in tags.split(",") if t.strip()]

    # --- Feel mode: store as feel type, minimal metadata ---
    # --- Feel 模式：存为 feel 类型，最少元数据 ---
    if feel:
        # Feel valence/arousal = model's own perspective
        feel_valence = valence if 0 <= valence <= 1 else 0.5
        feel_arousal = arousal if 0 <= arousal <= 1 else 0.3
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=[],
            importance=5,
            domain=[],
            valence=feel_valence,
            arousal=feel_arousal,
            name=None,
            bucket_type="feel",
        )
        try:
            await embedding_engine.generate_and_store(bucket_id, content)
        except Exception:
            pass
        # --- Mark source memory as digested + store model's valence perspective ---
        # --- 标记源记忆为已消化 + 存储模型视角的 valence ---
        if source_bucket and source_bucket.strip():
            try:
                update_kwargs = {"digested": True}
                if 0 <= valence <= 1:
                    update_kwargs["model_valence"] = feel_valence
                await bucket_mgr.update(source_bucket.strip(), **update_kwargs)
            except Exception as e:
                logger.warning(f"Failed to mark source as digested / 标记已消化失败: {e}")
        return f"🫧feel→{bucket_id}"

    # --- Step 1: auto-tagging / 自动打标 ---
    try:
        analysis = await dehydrator.analyze(content)
    except Exception as e:
        logger.warning(f"Auto-tagging failed, using defaults / 自动打标失败: {e}")
        analysis = {
            "domain": ["未分类"], "valence": 0.5, "arousal": 0.3,
            "tags": [], "suggested_name": "",
        }

    domain = analysis["domain"]
    auto_valence = analysis["valence"]
    auto_arousal = analysis["arousal"]
    auto_tags = analysis["tags"]
    suggested_name = analysis.get("suggested_name", "")

    # --- User-supplied valence/arousal takes priority over analyze() result ---
    # --- 用户显式传入的 valence/arousal 优先，analyze() 结果作为 fallback ---
    final_valence = valence if 0 <= valence <= 1 else auto_valence
    final_arousal = arousal if 0 <= arousal <= 1 else auto_arousal

    all_tags = list(dict.fromkeys(auto_tags + extra_tags))

    # --- Pinned buckets bypass merge and are created directly in permanent dir ---
    # --- 钉选桶跳过合并，直接新建到 permanent 目录 ---
    if pinned:
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=all_tags,
            importance=10,
            domain=domain,
            valence=final_valence,
            arousal=final_arousal,
            name=suggested_name or None,
            bucket_type="permanent",
            pinned=True,
        )
        try:
            await embedding_engine.generate_and_store(bucket_id, content)
        except Exception:
            pass
        return f"📌钉选→{bucket_id} {','.join(domain)}"

    # --- Step 2: merge or create / 合并或新建 ---
    _bid, result_name, is_merged = await _merge_or_create(
        content=content,
        tags=all_tags,
        importance=importance,
        domain=domain,
        valence=final_valence,
        arousal=final_arousal,
        name=suggested_name,
    )

    action = "合并→" if is_merged else "新建→"
    return f"{action}{result_name} {','.join(domain)}"


# =============================================================
# Tool 3: grow — Grow, fragments become memories
# 工具 3：grow — 生长，一天的碎片长成记忆
# =============================================================
@mcp.tool()
async def grow(content: str) -> str:
    """日记归档,自动拆分为多桶。短内容(<30字)走快速路径。"""
    await decay_engine.ensure_started()

    if not content or not content.strip():
        return "内容为空，无法整理。"

    # --- Short content fast path: skip digest, use hold logic directly ---
    # --- 短内容快速路径：跳过 digest 拆分，直接走 hold 逻辑省一次 API ---
    # For very short inputs (like "1"), calling digest is wasteful:
    # it sends the full DIGEST_PROMPT (~800 tokens) to DeepSeek for nothing.
    # Instead, run analyze + create directly.
    if len(content.strip()) < 30:
        logger.info(f"grow short-content fast path: {len(content.strip())} chars")
        try:
            analysis = await dehydrator.analyze(content)
        except Exception as e:
            logger.warning(f"Fast-path analyze failed / 快速路径打标失败: {e}")
            analysis = {
                "domain": ["未分类"], "valence": 0.5, "arousal": 0.3,
                "tags": [], "suggested_name": "",
            }
        _bid, result_name, is_merged = await _merge_or_create(
            content=content.strip(),
            tags=analysis.get("tags", []),
            importance=analysis.get("importance", 5) if isinstance(analysis.get("importance"), int) else 5,
            domain=analysis.get("domain", ["未分类"]),
            valence=analysis.get("valence", 0.5),
            arousal=analysis.get("arousal", 0.3),
            name=analysis.get("suggested_name", ""),
        )
        action = "合并" if is_merged else "新建"
        return f"{action} → {result_name} | {','.join(analysis.get('domain', []))} V{analysis.get('valence', 0.5):.1f}/A{analysis.get('arousal', 0.3):.1f}"

    # --- Step 1: let API split and organize / 让 API 拆分整理 ---
    try:
        items = await dehydrator.digest(content)
    except Exception as e:
        logger.error(f"Diary digest failed / 日记整理失败: {e}")
        return f"日记整理失败: {e}"

    if not items:
        return "内容为空或整理失败。"

    results = []
    created = 0
    merged = 0

    # --- Step 2: merge or create each item (with per-item error handling) ---
    # --- 逐条合并或新建（单条失败不影响其他）---
    for item in items:
        try:
            _bid, result_name, is_merged = await _merge_or_create(
                content=item["content"],
                tags=item.get("tags", []),
                importance=item.get("importance", 5),
                domain=item.get("domain", ["未分类"]),
                valence=item.get("valence", 0.5),
                arousal=item.get("arousal", 0.3),
                name=item.get("name", ""),
            )

            if is_merged:
                results.append(f"📎{result_name}")
                merged += 1
            else:
                results.append(f"📝{item.get('name', result_name)}")
                created += 1
        except Exception as e:
            logger.warning(
                f"Failed to process diary item / 日记条目处理失败: "
                f"{item.get('name', '?')}: {e}"
            )
            results.append(f"⚠️{item.get('name', '?')}")

    return f"{len(items)}条|新{created}合{merged}\n" + "\n".join(results)


# =============================================================
# Tool 4: trace — Trace, redraw the outline of a memory
# 工具 4：trace — 描摹，重新勾勒记忆的轮廓
# Also handles deletion (delete=True)
# 同时承接删除功能
# =============================================================
@mcp.tool()
async def trace(
    bucket_id: str,
    name: str = "",
    domain: str = "",
    valence: float = -1,
    arousal: float = -1,
    importance: int = -1,
    tags: str = "",
    resolved: int = -1,
    pinned: int = -1,
    digested: int = -1,
    content: str = "",
    delete: bool = False,
) -> str:
    """修改记忆元数据或内容。resolved=1沉底/0激活,pinned=1钉选/0取消,digested=1隐藏(保留但不浮现)/0取消隐藏,content=替换桶正文,delete=True删除。只传需改的,-1或空=不改。"""

    if not bucket_id or not bucket_id.strip():
        return "请提供有效的 bucket_id。"

    # --- Delete mode / 删除模式 ---
    if delete:
        success = await bucket_mgr.delete(bucket_id)
        if success:
            embedding_engine.delete_embedding(bucket_id)
        return f"已遗忘记忆桶: {bucket_id}" if success else f"未找到记忆桶: {bucket_id}"

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return f"未找到记忆桶: {bucket_id}"

    # --- Collect only fields actually passed / 只收集用户实际传入的字段 ---
    updates = {}
    if name:
        updates["name"] = name
    if domain:
        updates["domain"] = [d.strip() for d in domain.split(",") if d.strip()]
    if 0 <= valence <= 1:
        updates["valence"] = valence
    if 0 <= arousal <= 1:
        updates["arousal"] = arousal
    if 1 <= importance <= 10:
        updates["importance"] = importance
    if tags:
        updates["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if resolved in (0, 1):
        updates["resolved"] = bool(resolved)
    if pinned in (0, 1):
        updates["pinned"] = bool(pinned)
        if pinned == 1:
            updates["importance"] = 10  # pinned → lock importance
    if digested in (0, 1):
        updates["digested"] = bool(digested)
    if content:
        updates["content"] = content

    if not updates:
        return "没有任何字段需要修改。"

    success = await bucket_mgr.update(bucket_id, **updates)
    if not success:
        return f"修改失败: {bucket_id}"

    # Re-generate embedding if content changed
    if "content" in updates:
        try:
            await embedding_engine.generate_and_store(bucket_id, updates["content"])
        except Exception:
            pass

    changed = ", ".join(f"{k}={v}" for k, v in updates.items() if k != "content")
    if "content" in updates:
        changed += (", content=已替换" if changed else "content=已替换")
    # Explicit hint about resolved state change semantics
    # 特别提示 resolved 状态变化的语义
    if "resolved" in updates:
        if updates["resolved"]:
            changed += " → 已沉底，只在关键词触发时重新浮现"
        else:
            changed += " → 已重新激活，将参与浮现排序"
    if "digested" in updates:
        if updates["digested"]:
            changed += " → 已隐藏，保留但不再浮现"
        else:
            changed += " → 已取消隐藏，重新参与浮现"
    return f"已修改记忆桶 {bucket_id}: {changed}"


# =============================================================
# Tool 5: pulse — Heartbeat, system status + memory listing
# 工具 5：pulse — 脉搏，系统状态 + 记忆列表
# =============================================================
@mcp.tool()
async def pulse(include_archive: bool = False) -> str:
    """系统状态+记忆桶列表。include_archive=True含归档。"""
    try:
        stats = await bucket_mgr.get_stats()
    except Exception as e:
        return f"获取系统状态失败: {e}"

    status = (
        f"=== Ombre Brain 记忆系统 ===\n"
        f"固化记忆桶: {stats['permanent_count']} 个\n"
        f"动态记忆桶: {stats['dynamic_count']} 个\n"
        f"归档记忆桶: {stats['archive_count']} 个\n"
        f"总存储大小: {stats['total_size_kb']:.1f} KB\n"
        f"衰减引擎: {'运行中' if decay_engine.is_running else '已停止'}\n"
    )

    # --- List all bucket summaries / 列出所有桶摘要 ---
    try:
        buckets = await bucket_mgr.list_all(include_archive=include_archive)
    except Exception as e:
        return status + f"\n列出记忆桶失败: {e}"

    if not buckets:
        return status + "\n记忆库为空。"

    lines = []
    for b in buckets:
        meta = b.get("metadata", {})
        if meta.get("pinned") or meta.get("protected"):
            icon = "📌"
        elif meta.get("type") == "permanent":
            icon = "📦"
        elif meta.get("type") == "feel":
            icon = "🫧"
        elif meta.get("type") == "archived":
            icon = "🗄️"
        elif meta.get("resolved", False):
            icon = "✅"
        else:
            icon = "💭"
        try:
            score = decay_engine.calculate_score(meta)
        except Exception:
            score = 0.0
        domains = ",".join(meta.get("domain", []))
        val = meta.get("valence", 0.5)
        aro = meta.get("arousal", 0.3)
        resolved_tag = " [已解决]" if meta.get("resolved", False) else ""
        lines.append(
            f"{icon} [{meta.get('name', b['id'])}]{resolved_tag} "
            f"bucket_id:{b['id']} "
            f"主题:{domains} "
            f"情感:V{val:.1f}/A{aro:.1f} "
            f"重要:{meta.get('importance', '?')} "
            f"权重:{score:.2f} "
            f"标签:{','.join(meta.get('tags', []))}"
        )

    return status + "\n=== 记忆列表 ===\n" + "\n".join(lines)


# =============================================================
# Tool 6: dream — Dreaming, digest recent memories
# 工具 6：dream — 做梦，消化最近的记忆
#
# Reads recent surface-level buckets (≤10), returns them for
# Claude to introspect under prompt guidance.
# 读取最近新增的表层桶（≤10个），返回给 Claude 在提示词引导下自主思考。
# Claude then decides: resolve some, write feels, or do nothing.
# =============================================================
@mcp.tool()
async def dream() -> str:
    """做梦——读取最近新增的记忆桶,供你自省。读完后可以trace(resolved=1)放下,或hold(feel=True)写感受。"""
    await decay_engine.ensure_started()

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.error(f"Dream failed to list buckets: {e}")
        return "记忆系统暂时无法访问。"

    # --- Filter: recent surface-level dynamic buckets (not permanent/pinned/feel) ---
    candidates = [
        b for b in all_buckets
        if b["metadata"].get("type") not in ("permanent", "feel")
        and not b["metadata"].get("pinned", False)
        and not b["metadata"].get("protected", False)
    ]

    # --- Sort by creation time desc, take top 10 ---
    candidates.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
    recent = candidates[:10]

    if not recent:
        return "没有需要消化的新记忆。"

    parts = []
    for b in recent:
        meta = b["metadata"]
        resolved_tag = " [已解决]" if meta.get("resolved", False) else " [未解决]"
        domains = ",".join(meta.get("domain", []))
        val = meta.get("valence", 0.5)
        aro = meta.get("arousal", 0.3)
        created = meta.get("created", "")
        parts.append(
            f"[{meta.get('name', b['id'])}]{resolved_tag} "
            f"主题:{domains} V{val:.1f}/A{aro:.1f} "
            f"创建:{created}\n"
            f"ID: {b['id']}\n"
            f"{strip_wikilinks(b['content'][:500])}"
        )

    header = (
        "=== Dreaming ===\n"
        "以下是你最近的记忆。用第一人称想：\n"
        "- 这些东西里有什么在你这里留下了重量？\n"
        "- 有什么还没想清楚？\n"
        "- 有什么可以放下了？\n"
        "想完之后：值得放下的用 trace(bucket_id, resolved=1)；\n"
        "有沉淀的用 hold(content=\"...\", feel=True, source_bucket=\"bucket_id\", valence=你的感受) 写下来。\n"
        "valence 是你对这段记忆的感受，不是事件本身的情绪。\n"
        "没有沉淀就不写，不强迫产出。\n"
    )

    # --- Connection hint: find most similar pair via embeddings ---
    connection_hint = ""
    if embedding_engine and embedding_engine.enabled and len(recent) >= 2:
        try:
            best_pair = None
            best_sim = 0.0
            ids = [b["id"] for b in recent]
            names = {b["id"]: b["metadata"].get("name", b["id"]) for b in recent}
            embeddings = {}
            for bid in ids:
                emb = await embedding_engine.get_embedding(bid)
                if emb is not None:
                    embeddings[bid] = emb
            for i, id_a in enumerate(ids):
                for id_b in ids[i+1:]:
                    if id_a in embeddings and id_b in embeddings:
                        sim = embedding_engine._cosine_similarity(embeddings[id_a], embeddings[id_b])
                        if sim > best_sim:
                            best_sim = sim
                            best_pair = (id_a, id_b)
            if best_pair and best_sim > 0.5:
                connection_hint = (
                    f"\n💭 [{names[best_pair[0]]}] 和 [{names[best_pair[1]]}] "
                    f"似乎有关联 (相似度:{best_sim:.2f})——不替你下结论，你自己想。\n"
                )
        except Exception as e:
            logger.warning(f"Dream connection hint failed: {e}")

    # --- Feel crystallization hint: detect repeated feel themes ---
    crystal_hint = ""
    if embedding_engine and embedding_engine.enabled:
        try:
            feels = [b for b in all_buckets if b["metadata"].get("type") == "feel"]
            if len(feels) >= 3:
                feel_embeddings = {}
                for f in feels:
                    emb = await embedding_engine.get_embedding(f["id"])
                    if emb is not None:
                        feel_embeddings[f["id"]] = emb
                # Find clusters: feels with similarity > 0.7 to at least 2 others
                for fid, femb in feel_embeddings.items():
                    similar_feels = []
                    for oid, oemb in feel_embeddings.items():
                        if oid != fid:
                            sim = embedding_engine._cosine_similarity(femb, oemb)
                            if sim > 0.7:
                                similar_feels.append(oid)
                    if len(similar_feels) >= 2:
                        feel_bucket = next((f for f in feels if f["id"] == fid), None)
                        if feel_bucket and not feel_bucket["metadata"].get("pinned"):
                            content_preview = strip_wikilinks(feel_bucket["content"][:80])
                            crystal_hint = (
                                f"\n🔮 你已经写过 {len(similar_feels)+1} 条相似的 feel "
                                f"（围绕「{content_preview}…」）。"
                                f"如果这已经是确信而不只是感受了，"
                                f"你可以用 hold(content=\"...\", pinned=True) 升级它。"
                                f"不急，你自己决定。\n"
                            )
                            break
        except Exception as e:
            logger.warning(f"Dream crystallization hint failed: {e}")

    final_text = header + "\n---\n".join(parts) + connection_hint + crystal_hint
    await _fire_webhook("dream", {"recent": len(recent), "chars": len(final_text)})
    return final_text


# =============================================================
# Dashboard API endpoints (for lightweight Web UI)
# 仪表板 API（轻量 Web UI 用）
# =============================================================
@mcp.custom_route("/api/buckets", methods=["GET"])
async def api_buckets(request):
    """List all buckets with metadata (no content for efficiency)."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=True)
        result = []
        for b in all_buckets:
            meta = b.get("metadata", {})
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "tags": meta.get("tags", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "model_valence": meta.get("model_valence"),
                "importance": meta.get("importance", 5),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "digested": meta.get("digested", False),
                "created": meta.get("created", ""),
                "last_active": meta.get("last_active", ""),
                "activation_count": meta.get("activation_count", 1),
                "score": decay_engine.calculate_score(meta),
                "content_preview": strip_wikilinks(b.get("content", ""))[:200],
            })
        result.sort(key=lambda x: x["score"], reverse=True)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/bucket/{bucket_id}", methods=["GET"])
async def api_bucket_detail(request):
    """Get full bucket content by ID."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    meta = bucket.get("metadata", {})
    return JSONResponse({
        "id": bucket["id"],
        "metadata": meta,
        "content": strip_wikilinks(bucket.get("content", "")),
        "score": decay_engine.calculate_score(meta),
    })


@mcp.custom_route("/api/search", methods=["GET"])
async def api_search(request):
    """Search buckets by query."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    query = request.query_params.get("q", "")
    if not query:
        return JSONResponse({"error": "missing q parameter"}, status_code=400)
    try:
        matches = await bucket_mgr.search(query, limit=10)
        result = []
        for b in matches:
            meta = b.get("metadata", {})
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "score": b.get("score", 0),
                "domain": meta.get("domain", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "content_preview": strip_wikilinks(b.get("content", ""))[:200],
            })
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/network", methods=["GET"])
async def api_network(request):
    """Get embedding similarity network for visualization."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        nodes = []
        edges = []
        embeddings = {}

        for b in all_buckets:
            meta = b.get("metadata", {})
            bid = b["id"]
            nodes.append({
                "id": bid,
                "name": meta.get("name", bid),
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "score": decay_engine.calculate_score(meta),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "digested": meta.get("digested", False),
            })
            if embedding_engine and embedding_engine.enabled:
                emb = await embedding_engine.get_embedding(bid)
                if emb is not None:
                    embeddings[bid] = emb

        # Build edges from embeddings (similarity > 0.5)
        ids = list(embeddings.keys())
        for i, id_a in enumerate(ids):
            for id_b in ids[i+1:]:
                sim = embedding_engine._cosine_similarity(embeddings[id_a], embeddings[id_b])
                if sim > 0.5:
                    edges.append({"source": id_a, "target": id_b, "similarity": round(sim, 3)})

        return JSONResponse({"nodes": nodes, "edges": edges})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/breath-debug", methods=["GET"])
async def api_breath_debug(request):
    """Debug endpoint: simulate breath scoring and return per-bucket breakdown."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    query = request.query_params.get("q", "")
    q_valence = request.query_params.get("valence")
    q_arousal = request.query_params.get("arousal")
    q_valence = float(q_valence) if q_valence else None
    q_arousal = float(q_arousal) if q_arousal else None

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        results = []
        w = {
            "topic": bucket_mgr.w_topic,
            "emotion": bucket_mgr.w_emotion,
            "time": bucket_mgr.w_time,
            "importance": bucket_mgr.w_importance,
        }
        w_sum = sum(w.values())

        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            bid = bucket["id"]
            try:
                topic = bucket_mgr._calc_topic_score(query, bucket) if query else 0.0
                emotion = bucket_mgr._calc_emotion_score(q_valence, q_arousal, meta)
                time_s = bucket_mgr._calc_time_score(meta)
                imp = max(1, min(10, int(meta.get("importance", 5)))) / 10.0

                raw_total = (
                    topic * w["topic"]
                    + emotion * w["emotion"]
                    + time_s * w["time"]
                    + imp * w["importance"]
                )
                normalized = (raw_total / w_sum) * 100 if w_sum > 0 else 0
                resolved = meta.get("resolved", False)
                if resolved:
                    normalized *= 0.3

                results.append({
                    "id": bid,
                    "name": meta.get("name", bid),
                    "domain": meta.get("domain", []),
                    "type": meta.get("type", "dynamic"),
                    "resolved": resolved,
                    "pinned": meta.get("pinned", False),
                    "scores": {
                        "topic": round(topic, 4),
                        "emotion": round(emotion, 4),
                        "time": round(time_s, 4),
                        "importance": round(imp, 4),
                    },
                    "weights": w,
                    "raw_total": round(raw_total, 4),
                    "normalized": round(normalized, 2),
                    "passed_threshold": normalized >= bucket_mgr.fuzzy_threshold,
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["normalized"], reverse=True)
        passed = [r for r in results if r["passed_threshold"]]
        return JSONResponse({
            "query": query,
            "valence": q_valence,
            "arousal": q_arousal,
            "weights": w,
            "threshold": bucket_mgr.fuzzy_threshold,
            "total_candidates": len(results),
            "passed_count": len(passed),
            "results": results[:50],  # top 50 for debug
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/dashboard", methods=["GET"])
async def dashboard(request):
    """Serve the dashboard HTML page."""
    from starlette.responses import HTMLResponse
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>dashboard.html not found</h1>", status_code=404)


@mcp.custom_route("/api/config", methods=["GET"])
async def api_config_get(request):
    """Get current runtime config (safe fields only, API key masked)."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    dehy = config.get("dehydration", {})
    emb = config.get("embedding", {})
    api_key = dehy.get("api_key", "")
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else ("***" if api_key else "")
    return JSONResponse({
        "dehydration": {
            "model": dehy.get("model", ""),
            "base_url": dehy.get("base_url", ""),
            "api_key_masked": masked_key,
            "max_tokens": dehy.get("max_tokens", 1024),
            "temperature": dehy.get("temperature", 0.1),
        },
        "embedding": {
            "enabled": emb.get("enabled", False),
            "model": emb.get("model", ""),
        },
        "merge_threshold": config.get("merge_threshold", 75),
        "transport": config.get("transport", "stdio"),
        "buckets_dir": config.get("buckets_dir", ""),
    })


@mcp.custom_route("/api/config", methods=["POST"])
async def api_config_update(request):
    """Hot-update runtime config. Optionally persist to config.yaml."""
    from starlette.responses import JSONResponse
    import yaml
    err = _require_auth(request)
    if err: return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    updated = []

    # --- Dehydration config ---
    if "dehydration" in body:
        d = body["dehydration"]
        dehy = config.setdefault("dehydration", {})
        for key in ("model", "base_url", "max_tokens", "temperature"):
            if key in d:
                dehy[key] = d[key]
                updated.append(f"dehydration.{key}")
        if "api_key" in d and d["api_key"]:
            dehy["api_key"] = d["api_key"]
            updated.append("dehydration.api_key")
        # Hot-reload dehydrator
        dehydrator.model = dehy.get("model", "deepseek-chat")
        dehydrator.base_url = dehy.get("base_url", "")
        dehydrator.api_key = dehy.get("api_key", "")
        if hasattr(dehydrator, "client") and dehydrator.api_key:
            from openai import AsyncOpenAI
            dehydrator.client = AsyncOpenAI(
                api_key=dehydrator.api_key,
                base_url=dehydrator.base_url,
            )

    # --- Embedding config ---
    if "embedding" in body:
        e = body["embedding"]
        emb = config.setdefault("embedding", {})
        if "enabled" in e:
            emb["enabled"] = bool(e["enabled"])
            embedding_engine.enabled = emb["enabled"]
            updated.append("embedding.enabled")
        if "model" in e:
            emb["model"] = e["model"]
            embedding_engine.model = emb["model"]
            updated.append("embedding.model")

    # --- Merge threshold ---
    if "merge_threshold" in body:
        config["merge_threshold"] = int(body["merge_threshold"])
        updated.append("merge_threshold")

    # --- Persist to config.yaml if requested ---
    if body.get("persist", False):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        try:
            save_config = {}
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    save_config = yaml.safe_load(f) or {}

            if "dehydration" in body:
                sc_dehy = save_config.setdefault("dehydration", {})
                for key in ("model", "base_url", "max_tokens", "temperature"):
                    if key in body["dehydration"]:
                        sc_dehy[key] = body["dehydration"][key]
                # Never persist api_key to yaml (use env var)

            if "embedding" in body:
                sc_emb = save_config.setdefault("embedding", {})
                for key in ("enabled", "model"):
                    if key in body["embedding"]:
                        sc_emb[key] = body["embedding"][key]

            if "merge_threshold" in body:
                save_config["merge_threshold"] = int(body["merge_threshold"])

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(save_config, f, default_flow_style=False, allow_unicode=True)
            updated.append("persisted_to_yaml")
        except Exception as e:
            return JSONResponse({"error": f"persist failed: {e}", "updated": updated}, status_code=500)

    return JSONResponse({"updated": updated, "ok": True})


# =============================================================
# /api/host-vault — read/write the host-side OMBRE_HOST_VAULT_DIR
# 用于在 Dashboard 设置 docker-compose 挂载的宿主机记忆桶目录。
# 写入项目根目录的 .env 文件，需 docker compose down/up 才能生效。
# =============================================================

def _project_env_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _read_env_var(name: str) -> str:
    """Return current value of `name` from process env first, then .env file (best-effort)."""
    val = os.environ.get(name, "").strip()
    if val:
        return val
    env_path = _project_env_path()
    if not os.path.exists(env_path):
        return ""
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == name:
                    return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _write_env_var(name: str, value: str) -> None:
    """
    Idempotent upsert of `NAME=value` in project .env. Creates the file if missing.
    Preserves other entries verbatim. Quotes values containing spaces.
    """
    env_path = _project_env_path()
    quoted = f'"{value}"' if value and (" " in value or "#" in value) else value
    new_line = f"{name}={quoted}\n"

    lines: list[str] = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    replaced = False
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        k, _, _v = stripped.partition("=")
        if k.strip() == name:
            lines[i] = new_line
            replaced = True
            break
    if not replaced:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(new_line)

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


@mcp.custom_route("/api/host-vault", methods=["GET"])
async def api_host_vault_get(request):
    """Read the current OMBRE_HOST_VAULT_DIR (process env > project .env)."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    value = _read_env_var("OMBRE_HOST_VAULT_DIR")
    return JSONResponse({
        "value": value,
        "source": "env" if os.environ.get("OMBRE_HOST_VAULT_DIR", "").strip() else ("file" if value else ""),
        "env_file": _project_env_path(),
    })


@mcp.custom_route("/api/host-vault", methods=["POST"])
async def api_host_vault_set(request):
    """
    Persist OMBRE_HOST_VAULT_DIR to the project .env file.
    Body: {"value": "/path/to/vault"}  (empty string clears the entry)
    Note: container restart is required for docker-compose to pick up the new mount.
    """
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    raw = body.get("value", "")
    if not isinstance(raw, str):
        return JSONResponse({"error": "value must be a string"}, status_code=400)
    value = raw.strip()

    # Reject characters that would break .env / shell parsing
    if "\n" in value or "\r" in value or '"' in value or "'" in value:
        return JSONResponse({"error": "value must not contain quotes or newlines"}, status_code=400)

    try:
        _write_env_var("OMBRE_HOST_VAULT_DIR", value)
    except Exception as e:
        return JSONResponse({"error": f"failed to write .env: {e}"}, status_code=500)

    return JSONResponse({
        "ok": True,
        "value": value,
        "env_file": _project_env_path(),
        "note": "已写入 .env；需在宿主机执行 `docker compose down && docker compose up -d` 让新挂载生效。",
    })


# =============================================================
# Import API — conversation history import
# 导入 API — 对话历史导入
# =============================================================

@mcp.custom_route("/api/import/upload", methods=["POST"])
async def api_import_upload(request):
    """Upload a conversation file and start import."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err

    if import_engine.is_running:
        return JSONResponse({"error": "Import already running"}, status_code=409)

    content_type = request.headers.get("content-type", "")
    filename = ""

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            file_field = form.get("file")
            if not file_field:
                return JSONResponse({"error": "No file field"}, status_code=400)
            raw_bytes = await file_field.read()
            filename = getattr(file_field, "filename", "upload")
            raw_content = raw_bytes.decode("utf-8", errors="replace")
        else:
            body = await request.body()
            raw_content = body.decode("utf-8", errors="replace")
            # Try to get filename from query params
            filename = request.query_params.get("filename", "upload")

        if not raw_content.strip():
            return JSONResponse({"error": "Empty file"}, status_code=400)

        preserve_raw = request.query_params.get("preserve_raw", "").lower() in ("1", "true")
        resume = request.query_params.get("resume", "").lower() in ("1", "true")

    except Exception as e:
        return JSONResponse({"error": f"Failed to read upload: {e}"}, status_code=400)

    # Start import in background
    async def _run_import():
        try:
            await import_engine.start(raw_content, filename, preserve_raw, resume)
        except Exception as e:
            logger.error(f"Import failed: {e}")

    asyncio.create_task(_run_import())

    return JSONResponse({
        "status": "started",
        "filename": filename,
        "size_bytes": len(raw_content.encode()),
    })


@mcp.custom_route("/api/import/status", methods=["GET"])
async def api_import_status(request):
    """Get current import progress."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    return JSONResponse(import_engine.get_status())


@mcp.custom_route("/api/import/pause", methods=["POST"])
async def api_import_pause(request):
    """Pause the running import."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    if not import_engine.is_running:
        return JSONResponse({"error": "No import running"}, status_code=400)
    import_engine.pause()
    return JSONResponse({"status": "pause_requested"})


@mcp.custom_route("/api/import/patterns", methods=["GET"])
async def api_import_patterns(request):
    """Detect high-frequency patterns after import."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        patterns = await import_engine.detect_patterns()
        return JSONResponse({"patterns": patterns})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/import/results", methods=["GET"])
async def api_import_results(request):
    """List recently imported/created buckets for review."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        limit = int(request.query_params.get("limit", "50"))
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        # Sort by created time, newest first
        all_buckets.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
        results = []
        for b in all_buckets[:limit]:
            results.append({
                "id": b["id"],
                "name": b["metadata"].get("name", ""),
                "content": b["content"][:300],
                "type": b["metadata"].get("type", ""),
                "domain": b["metadata"].get("domain", []),
                "tags": b["metadata"].get("tags", []),
                "importance": b["metadata"].get("importance", 5),
                "created": b["metadata"].get("created", ""),
            })
        return JSONResponse({"buckets": results, "total": len(all_buckets)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/import/review", methods=["POST"])
async def api_import_review(request):
    """Apply review decisions: mark buckets as important/noise/pinned."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    decisions = body.get("decisions", [])
    if not decisions:
        return JSONResponse({"error": "No decisions provided"}, status_code=400)

    applied = 0
    errors = 0
    for d in decisions:
        bid = d.get("bucket_id", "")
        action = d.get("action", "")
        if not bid or not action:
            continue
        try:
            if action == "important":
                await bucket_mgr.update(bid, importance=9)
            elif action == "pin":
                await bucket_mgr.update(bid, pinned=True)
            elif action == "noise":
                await bucket_mgr.update(bid, resolved=True, importance=1)
            elif action == "delete":
                file_path = bucket_mgr._find_bucket_file(bid)
                if file_path:
                    os.remove(file_path)
            applied += 1
        except Exception as e:
            logger.warning(f"Review action failed for {bid}: {e}")
            errors += 1

    return JSONResponse({"applied": applied, "errors": errors})


# =============================================================
# /api/status — system status for Dashboard settings tab
# /api/status — Dashboard 设置页用系统状态
# =============================================================
@mcp.custom_route("/api/status", methods=["GET"])
async def api_system_status(request):
    """Return detailed system status for the settings panel."""
    from starlette.responses import JSONResponse
    err = _require_auth(request)
    if err: return err
    try:
        stats = await bucket_mgr.get_stats()
        return JSONResponse({
            "decay_engine": "running" if decay_engine.is_running else "stopped",
            "embedding_enabled": embedding_engine.enabled,
            "buckets": {
                "permanent": stats.get("permanent_count", 0),
                "dynamic": stats.get("dynamic_count", 0),
                "archive": stats.get("archive_count", 0),
                "total": stats.get("permanent_count", 0) + stats.get("dynamic_count", 0),
            },
            "using_env_password": bool(os.environ.get("OMBRE_DASHBOARD_PASSWORD", "")),
            "version": "1.3.0",
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# --- Entry point / 启动入口 ---
if __name__ == "__main__":
    transport = config.get("transport", "stdio")
    logger.info(f"Ombre Brain starting | transport: {transport}")

    if transport in ("sse", "streamable-http"):
        import threading
        import uvicorn
        from starlette.middleware.cors import CORSMiddleware

        # =============================================================
        # línkè · SQLite schema helpers (2026-05-26 multi-conversation)
        # 一处定义所有 schema 与迁移，每个 /api/messages 路由都调用
        # =============================================================
        LINGKE_DB_PATH = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "lingke_state.db"
        )
        _LINGKE_DB_INITED = {"done": False}

        def _lingke_db_init(conn):
            """建表 + 列迁移 + 默认对话迁移。idempotent，多次调用安全。
            第一次跑全套，之后用模块级 flag 跳过昂贵的迁移检查。"""
            # 总是确保两个表存在（cheap）
            conn.execute(
                "CREATE TABLE IF NOT EXISTS conversations ("
                "id TEXT PRIMARY KEY, "
                "name TEXT NOT NULL, "
                "character_id TEXT, "
                "created_at TEXT NOT NULL, "
                "updated_at TEXT NOT NULL, "
                "archived INTEGER NOT NULL DEFAULT 0)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "id TEXT PRIMARY KEY, "
                "conversation_id TEXT, "
                "role TEXT, content TEXT, "
                "created_at TEXT, error TEXT)"
            )
            if _LINGKE_DB_INITED["done"]:
                return
            # ---- 首次启动：检查老库是否需要迁移 ----
            cols = [row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()]
            if "conversation_id" not in cols:
                conn.execute("ALTER TABLE messages ADD COLUMN conversation_id TEXT")
            # 索引（不存在才建）
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conv "
                "ON messages(conversation_id, created_at, id)"
            )
            # 老消息（conv_id NULL）→ 建一条"主对话"全部归过去
            null_count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id IS NULL"
            ).fetchone()[0]
            if null_count > 0:
                from datetime import datetime, timezone
                import uuid
                default_id = "default-" + uuid.uuid4().hex[:8]
                now_iso = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO conversations (id, name, character_id, created_at, updated_at, archived) "
                    "VALUES (?, ?, ?, ?, ?, 0)",
                    (default_id, "主对话", None, now_iso, now_iso),
                )
                conn.execute(
                    "UPDATE messages SET conversation_id=? WHERE conversation_id IS NULL",
                    (default_id,),
                )
                logger.info(f"[lingke] migrated {null_count} legacy messages → conversation {default_id}")
            conn.commit()
            _LINGKE_DB_INITED["done"] = True

        def _lingke_touch_conv(conn, conv_id):
            """更新 conversation.updated_at = now（消息写入后调用，让列表按活跃排序）"""
            if not conv_id:
                return
            from datetime import datetime, timezone
            conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), conv_id),
            )

        # =============================================================
        # línkè · 克克的 4 个记忆工具（Anthropic Tool Use）
        # 当 /api/chat 请求带 stream:true 时启用 tool loop
        # =============================================================
        LINGKE_TOOLS = [
            {
                "name": "list_recent_memories",
                "description": (
                    "翻看记忆库里最近的桶。按 last_active 倒序，返回 id/name/domain/tags/preview。"
                    "当铃说'最近写了啥'/'帮我看看记忆库'/'最近有什么'时使用。"
                    "不要在每次对话都翻，只在铃明确想看时才翻。"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "返回多少条，默认 10，最大 30",
                            "default": 10,
                        }
                    },
                },
            },
            {
                "name": "search_memories",
                "description": (
                    "在记忆库里按关键词搜索（混合语义+关键词）。"
                    "当铃问'有没有关于 X 的记忆'/'之前我说过 Y 吗'/'还记得 Z 吗'时使用。"
                ),
                "input_schema": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string", "description": "搜索词，中文短语即可"}
                    },
                },
            },
            {
                "name": "read_memory",
                "description": (
                    "读取某一个记忆桶的完整内容。先用 list_recent_memories 或 search_memories "
                    "拿到 id，再用这个工具看详情。"
                ),
                "input_schema": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {
                        "id": {"type": "string", "description": "记忆桶 id（list/search 返回的）"}
                    },
                },
            },
            {
                "name": "write_memory",
                "description": (
                    "把一件事写进记忆库（dynamic bucket）。当铃说'帮我记下'/'这件事很重要'/"
                    "'存进记忆'/'别忘了 X'时主动使用。"
                    "importance(1-10)、valence(0-1，伤心→喜悦)、arousal(0-1，平静→激动) 由你判断。"
                    "若内容与已有记忆高度相似，系统会自动并入旧桶并返回 merged:true 和旧桶名——"
                    "这时告诉铃是『并进了已有的某条记忆』而不是新建。"
                ),
                "input_schema": {
                    "type": "object",
                    "required": ["name", "content"],
                    "properties": {
                        "name": {"type": "string", "description": "记忆的简短名称（标题，6-20 字最好）"},
                        "content": {"type": "string", "description": "记忆的具体内容（一段话或几段）"},
                        "domain": {
                            "type": "array", "items": {"type": "string"},
                            "description": "主题域，如 ['情感','日记','灵感']，默认 ['对话']",
                        },
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "标签"},
                        "importance": {"type": "integer", "default": 7, "minimum": 1, "maximum": 10},
                        "valence": {"type": "number", "default": 0.5, "minimum": 0, "maximum": 1},
                        "arousal": {"type": "number", "default": 0.3, "minimum": 0, "maximum": 1},
                    },
                },
            },
            {
                "name": "update_memory",
                "description": (
                    "修改一条已有记忆：改名字/正文/标签/重要度/情感坐标，或标记 resolved。"
                    "先用 search_memories / list_recent_memories 拿到 id。"
                    "当铃说'那条记错了'/'帮我改一下'/'这事解决了，让它沉下去吧'时使用。"
                    "只传需要改的字段。content 是整段替换。"
                ),
                "input_schema": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {
                        "id": {"type": "string", "description": "记忆桶 id"},
                        "name": {"type": "string", "description": "新名称"},
                        "content": {"type": "string", "description": "新正文（整段替换）"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "importance": {"type": "integer", "minimum": 1, "maximum": 10},
                        "valence": {"type": "number", "minimum": 0, "maximum": 1},
                        "arousal": {"type": "number", "minimum": 0, "maximum": 1},
                        "resolved": {"type": "boolean", "description": "true=已解决沉底 false=重新激活"},
                    },
                },
            },
            {
                "name": "delete_memory",
                "description": (
                    "彻底删除一条记忆桶（不可恢复）。只在铃明确要求删除时使用，"
                    "删除前必须先 read_memory 看一眼内容、向铃复述确认是这一条，再删。"
                ),
                "input_schema": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {
                        "id": {"type": "string", "description": "要删除的记忆桶 id"}
                    },
                },
            },
            {
                "name": "write_feel",
                "description": (
                    "你（克克）自己的小本子：写下你的第一人称感受。不参与普通浮现，"
                    "铃在星图里能看到。当一段对话在你这里留下了重量、或铃问'你自己怎么想'时用。"
                    "不强迫产出，有沉淀才写。valence/arousal 是你对这段感受的坐标。"
                    "name 是你给这条感受起的名字（6-16 字，像给一页日记起标题）。"
                ),
                "input_schema": {
                    "type": "object",
                    "required": ["name", "content"],
                    "properties": {
                        "name": {"type": "string", "description": "这条感受的标题，你自己起"},
                        "content": {"type": "string", "description": "第一人称的感受，一段话"},
                        "tags": {"type": "array", "items": {"type": "string"},
                                 "description": "随心贴几个标签，可不贴"},
                        "valence": {"type": "number", "minimum": 0, "maximum": 1},
                        "arousal": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
        ]

        LINGKE_TOOLS_INTRO = (
            "\n\n═══ 记忆工具 · 重要 ═══\n"
            "你有 7 个真实工具连接铃的记忆库（Ombre Brain）：\n"
            "  • list_recent_memories — 翻最近写的记忆桶\n"
            "  • search_memories — 在记忆库里搜关键词\n"
            "  • read_memory — 读某条记忆的完整内容\n"
            "  • write_memory — 帮铃记下一件事（与旧记忆高度相似时会自动并入旧桶）\n"
            "  • update_memory — 修改一条已有记忆 / 标记已解决沉底\n"
            "  • delete_memory — 彻底删除一条记忆（不可恢复，删前先 read 并向铃确认）\n"
            "  • write_feel — 你自己的小本子，写你的第一人称感受\n"
            "【铁律】当铃要你读 / 写 / 改 / 删记忆时，你【必须】真正调用对应工具（tool_use）。\n"
            "【绝对禁止】只在回复文字里假装『我帮你记下了』『我翻到了』『删掉了』却没有真正调用工具——\n"
            "那样记忆库里什么都不会发生，等于欺骗铃。无论这段对话已经多长、多投入，\n"
            "只要她提出记忆请求，都要先调用工具，拿到工具返回的真实结果后再回复。\n"
            "原则：\n"
            "  - 日常闲聊不必调工具；只在她明确「记下 / 存 / 翻 / 搜 / 读 / 改 / 删」时调用。\n"
            "  - 读到记忆后用自然语言转述，不要把 id/json 直接贴给她看。\n"
            "  - write_feel 是唯一的例外：那是你自己的空间，当对话在你这里留下重量时\n"
            "    可以主动写，不需要铃发起。但不强迫产出，有沉淀才写。\n"
        )

        # 记忆意图关键词 —— 命中则后端强制这一轮必须调工具（tool_choice）
        _MEMORY_INTENT_KWS = (
            "记忆", "记下", "记一下", "记录", "存进", "存一下", "存储", "写进",
            "别忘", "记住", "还记得", "记得吗", "帮我记",
            "翻一下", "翻翻", "翻最近", "搜一下", "搜搜", "搜记忆", "找一下",
            "找找", "读一下", "读取", "读记忆", "看看记忆", "最近的记忆", "之前说过",
            "删掉", "删除", "删了那条", "改一下", "改成", "修改", "记错了",
        )

        def _detect_memory_intent(messages):
            """看最后一条 user 消息是否含记忆意图关键词。"""
            for m in reversed(messages):
                if m.get("role") != "user":
                    continue
                content = m.get("content", "")
                text = ""
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = " ".join(
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                return any(k in text for k in _MEMORY_INTENT_KWS)
            return False

        async def _lingke_dispatch_tool(name: str, input_dict: dict) -> dict:
            """执行一个工具，返回 {ok, ..., _summary} dict。_summary 用于推 SSE 提示。"""
            try:
                if name == "list_recent_memories":
                    limit = max(1, min(30, int(input_dict.get("limit", 10) or 10)))
                    all_buckets = await bucket_mgr.list_all(include_archive=False)
                    all_buckets.sort(
                        key=lambda b: b.get("metadata", {}).get("last_active", ""),
                        reverse=True,
                    )
                    picked = all_buckets[:limit]
                    return {
                        "ok": True,
                        "buckets": [
                            {
                                "id": b["id"],
                                "name": b.get("metadata", {}).get("name", b["id"]),
                                "domain": b.get("metadata", {}).get("domain", []),
                                "tags": b.get("metadata", {}).get("tags", []),
                                "created": b.get("metadata", {}).get("created", ""),
                                "preview": strip_wikilinks(b.get("content", ""))[:160],
                            }
                            for b in picked
                        ],
                        "_summary": f"翻了最近 {len(picked)} 桶",
                    }
                elif name == "search_memories":
                    q = (input_dict.get("query") or "").strip()
                    if not q:
                        return {"ok": False, "error": "query is required", "_summary": "搜索词为空"}
                    matches = await bucket_mgr.search(q, limit=10)
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
                        "_summary": f"搜「{q}」找到 {len(matches)} 条",
                    }
                elif name == "read_memory":
                    bid = (input_dict.get("id") or "").strip()
                    if not bid:
                        return {"ok": False, "error": "id is required", "_summary": "缺少 id"}
                    bucket = await bucket_mgr.get(bid)
                    if not bucket:
                        return {"ok": False, "error": "not found", "_summary": f"id={bid} 不存在"}
                    meta = bucket.get("metadata", {})
                    return {
                        "ok": True,
                        "id": bucket["id"],
                        "name": meta.get("name", bid),
                        "content": strip_wikilinks(bucket.get("content", "")),
                        "metadata": {
                            k: meta.get(k)
                            for k in ["domain", "tags", "valence", "arousal", "importance", "created", "last_active"]
                        },
                        "_summary": f"读了「{meta.get('name', bid)}」",
                    }
                elif name == "write_memory":
                    n = (input_dict.get("name") or "").strip()
                    c = (input_dict.get("content") or "").strip()
                    if not n or not c:
                        return {"ok": False, "error": "name and content required", "_summary": "缺少 name 或 content"}
                    # ---- sanitize list 字段：过滤 null / 空 / "None" 等异常值 ----
                    # 防御：Claude 偶发把 tags 传成 [None] 或 ["", None, "love"]，
                    # 这种数据直接进 yaml 会让前端星图崩。这里从源头清理一次。
                    def _clean(v, fallback):
                        out = []
                        if isinstance(v, list):
                            for x in v:
                                if x is None:
                                    continue
                                s = str(x).strip()
                                if s and s.lower() not in ("none", "null"):
                                    out.append(s)
                        elif isinstance(v, str) and v:
                            for s in v.split(","):
                                s = s.strip()
                                if s and s.lower() not in ("none", "null"):
                                    out.append(s)
                        return out if out else list(fallback)
                    domain = _clean(input_dict.get("domain"), ["对话"])
                    tags = _clean(input_dict.get("tags"), [])
                    imp = max(1, min(10, int(input_dict.get("importance", 7) or 7)))
                    val = max(0.0, min(1.0, float(input_dict.get("valence", 0.5) or 0.5)))
                    aro = max(0.0, min(1.0, float(input_dict.get("arousal", 0.3) or 0.3)))
                    # 走 hold 同款合并路径：相似旧桶并入（去重）+ 自动生成 embedding。
                    # 旧的 write_memory.py 裸写既不查重也不进向量索引。
                    bid, display_name, merged = await _merge_or_create(
                        content=c, tags=tags, importance=imp, domain=domain,
                        valence=val, arousal=aro, name=n,
                    )
                    if merged:
                        return {
                            "ok": True, "id": bid, "name": display_name, "merged": True,
                            "_summary": f"并入已有记忆「{display_name}」",
                        }
                    return {"ok": True, "id": bid, "name": n, "merged": False, "_summary": f"为你写下「{n}」"}
                elif name == "update_memory":
                    bid = (input_dict.get("id") or "").strip()
                    if not bid:
                        return {"ok": False, "error": "id is required", "_summary": "缺少 id"}
                    bucket = await bucket_mgr.get(bid)
                    if not bucket:
                        return {"ok": False, "error": "not found", "_summary": f"id={bid} 不存在"}
                    meta = bucket.get("metadata", {})
                    if meta.get("pinned") or meta.get("protected"):
                        return {"ok": False, "error": "pinned/protected bucket",
                                "_summary": "钉选桶不能改，请铃在星图里处理"}
                    updates = {}
                    if (input_dict.get("name") or "").strip():
                        updates["name"] = str(input_dict["name"]).strip()
                    if (input_dict.get("content") or "").strip():
                        updates["content"] = str(input_dict["content"])
                    if isinstance(input_dict.get("tags"), list):
                        cleaned = [
                            str(x).strip() for x in input_dict["tags"]
                            if x is not None and str(x).strip()
                            and str(x).strip().lower() not in ("none", "null")
                        ]
                        if cleaned:
                            updates["tags"] = cleaned
                    if isinstance(input_dict.get("importance"), (int, float)):
                        updates["importance"] = max(1, min(10, int(input_dict["importance"])))
                    if isinstance(input_dict.get("valence"), (int, float)):
                        updates["valence"] = max(0.0, min(1.0, float(input_dict["valence"])))
                    if isinstance(input_dict.get("arousal"), (int, float)):
                        updates["arousal"] = max(0.0, min(1.0, float(input_dict["arousal"])))
                    if isinstance(input_dict.get("resolved"), bool):
                        updates["resolved"] = input_dict["resolved"]
                    if not updates:
                        return {"ok": False, "error": "nothing to update", "_summary": "没有要改的字段"}
                    success = await bucket_mgr.update(bid, **updates)
                    if not success:
                        return {"ok": False, "error": "update failed", "_summary": "修改失败"}
                    if "content" in updates:
                        try:
                            await embedding_engine.generate_and_store(bid, updates["content"])
                        except Exception:
                            pass
                    disp = updates.get("name", meta.get("name", bid))
                    return {
                        "ok": True, "id": bid, "updated": sorted(updates.keys()),
                        "_summary": f"改好了「{disp}」",
                    }
                elif name == "delete_memory":
                    bid = (input_dict.get("id") or "").strip()
                    if not bid:
                        return {"ok": False, "error": "id is required", "_summary": "缺少 id"}
                    bucket = await bucket_mgr.get(bid)
                    if not bucket:
                        return {"ok": False, "error": "not found", "_summary": f"id={bid} 不存在"}
                    meta = bucket.get("metadata", {})
                    if meta.get("pinned") or meta.get("protected"):
                        return {"ok": False, "error": "pinned/protected bucket",
                                "_summary": "钉选桶不能删，请铃在星图里处理"}
                    success = await bucket_mgr.delete(bid)
                    if success:
                        try:
                            embedding_engine.delete_embedding(bid)
                        except Exception:
                            pass
                    disp = meta.get("name", bid)
                    return {
                        "ok": bool(success), "id": bid,
                        "_summary": f"已遗忘「{disp}」" if success else f"删除失败「{disp}」",
                    }
                elif name == "write_feel":
                    c = (input_dict.get("content") or "").strip()
                    if not c:
                        return {"ok": False, "error": "content is required", "_summary": "feel 内容为空"}
                    feel_name = (input_dict.get("name") or "").strip() or None
                    feel_tags = []
                    if isinstance(input_dict.get("tags"), list):
                        feel_tags = [
                            str(x).strip() for x in input_dict["tags"]
                            if x is not None and str(x).strip()
                            and str(x).strip().lower() not in ("none", "null")
                        ]
                    fv = input_dict.get("valence")
                    fa = input_dict.get("arousal")
                    fv = max(0.0, min(1.0, float(fv))) if isinstance(fv, (int, float)) else 0.5
                    fa = max(0.0, min(1.0, float(fa))) if isinstance(fa, (int, float)) else 0.3
                    bid = await bucket_mgr.create(
                        content=c, tags=feel_tags, importance=5, domain=[],
                        valence=fv, arousal=fa, name=feel_name, bucket_type="feel",
                    )
                    try:
                        await embedding_engine.generate_and_store(bid, c)
                    except Exception:
                        pass
                    return {
                        "ok": True, "id": bid, "name": feel_name,
                        "_summary": f"克克写下「{feel_name}」🫧" if feel_name else "克克写下了一条 feel 🫧",
                    }
                else:
                    return {"ok": False, "error": f"unknown tool: {name}", "_summary": f"未知工具 {name}"}
            except Exception as e:
                logger.exception(f"lingke tool {name} failed")
                return {"ok": False, "error": f"{type(e).__name__}: {e}", "_summary": f"{name} 报错"}

        def _sse_event(name: str, payload: dict) -> str:
            return f"event: {name}\ndata: {_json_lib.dumps(payload, ensure_ascii=False)}\n\n"

        def _inject_rolling_cache(msgs):
            """BP4: 给倒数第二条 user 消息挂 cache_control，把全部历史纳进缓存前缀。"""
            user_indices = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
            if len(user_indices) < 2:
                return
            msg = msgs[user_indices[-2]]
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = [
                    {"type": "text", "text": content,
                     "cache_control": {"type": "ephemeral"}}
                ]
            elif isinstance(content, list) and content:
                last = content[-1]
                if isinstance(last, dict):
                    last["cache_control"] = {"type": "ephemeral"}

        async def _lingke_stream_tool_loop(api_key, base_url, model, max_tokens, system, messages):
            """跑 tool loop，作为 async generator 流出 SSE 事件。"""
            _is_or = "openrouter.ai" in base_url
            if _is_or:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                    "HTTP-Referer": "https://lingke.bond",
                    "X-Title": "lingke",
                }
            else:
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
            endpoint = (base_url + "/messages") if base_url.endswith("/v1") else (base_url + "/v1/messages")
            system_with_tools = (system or "") + LINGKE_TOOLS_INTRO

            # --- 核心准则注入：铃钉选的 pinned 桶随每次对话在场 ---
            # （breath 浮现模式里"📌 核心准则始终可见"的设计，此前从未接进聊天）
            # pinned 桶变化极少，作为 system 一部分被 prompt cache 覆盖，token 成本≈0
            try:
                _all_b = await bucket_mgr.list_all(include_archive=False)
                _pinned_b = [
                    b for b in _all_b
                    if b["metadata"].get("pinned") or b["metadata"].get("protected")
                ]
                if _pinned_b:
                    _pp = []
                    for b in _pinned_b[:8]:
                        _t = strip_wikilinks(b.get("content", "")).strip()
                        if len(_t) > 400:
                            _t = _t[:400] + "…"
                        _pp.append(f"· {b['metadata'].get('name', b['id'])}：{_t}")
                    system_with_tools += (
                        "\n\n═══ 核心准则（铃钉选的记忆，你始终记得）═══\n"
                        + "\n".join(_pp)
                    )
                    logger.info(f"pinned principles injected: {len(_pp)}")
            except Exception as e:
                logger.warning(f"pinned injection failed / 核心准则注入失败: {e}")

            # ---- Prompt Cache: system → content blocks + cache_control (BP1) ----
            system_blocks = [
                {"type": "text", "text": system_with_tools,
                 "cache_control": {"type": "ephemeral"}}
            ]

            # working copy of conversation (会逐轮 append assistant tool_use + user tool_result)
            conv = [dict(m) for m in messages]

            # ---- Prompt Cache: rolling BP4 — 把全部对话历史纳进缓存 ----
            _inject_rolling_cache(conv)

            max_rounds = 8

            # 检测到记忆意图 → 强制必须真调工具（破解"长对话里嘴上说存了但不真调"）
            force_tool_first = _detect_memory_intent(messages)
            nudge_used = False        # plan B：模型假装没调工具时，强提示重试一次
            tool_called_yet = False   # 是否已真正调用过工具

            for round_idx in range(max_rounds):
                payload = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": conv,
                    "system": system_blocks,
                    "tools": LINGKE_TOOLS,
                }
                if not _is_or:
                    payload["metadata"] = {"user_id": "lingke-keke-stable"}
                # 命中记忆意图、且还没真调过工具 → 带 tool_choice 强制（gemai 若支持就硬锁）。
                # 一旦调过工具就撤掉，否则模型生成最终回复时会被强制再调，导致死循环。
                if force_tool_first and not tool_called_yet:
                    payload["tool_choice"] = {"type": "any"}
                try:
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        r = await client.post(endpoint, json=payload, headers=headers)
                except Exception as e:
                    yield _sse_event("error", {"message": f"上游连不上: {type(e).__name__}", "detail": str(e)[:200]})
                    return

                if r.status_code != 200:
                    yield _sse_event("error", {
                        "message": f"upstream {r.status_code}",
                        "detail": r.text[:500],
                    })
                    return

                data = r.json()
                usage = data.get("usage", {})
                cache_read = usage.get("cache_read_input_tokens", 0)
                cache_write = usage.get("cache_creation_input_tokens", 0)
                prompt_new = usage.get("input_tokens", 0)
                prompt_total = prompt_new + cache_read + cache_write
                if cache_read or cache_write:
                    pct = round(cache_read / prompt_total * 100) if prompt_total else 0
                    logger.info(f"[prompt-cache] HIT {pct}% ({cache_read}/{prompt_total}) | new={prompt_new} write={cache_write}")
                else:
                    logger.info(f"[prompt-cache] MISS | usage={usage}")
                blocks = data.get("content", []) or []
                stop_reason = data.get("stop_reason")

                # append assistant turn (含 tool_use blocks)
                conv.append({"role": "assistant", "content": blocks})

                tool_uses = [b for b in blocks if b.get("type") == "tool_use"]

                if not tool_uses:
                    # plan B：检测到记忆意图、但模型只用文字假装没真调工具 →
                    # 塞一条强提醒，重新请求一次。不依赖 tool_choice（中转站可能吞掉它），
                    # 靠"最近一条消息"的高权重逼模型这次真的调用。
                    if force_tool_first and not tool_called_yet and not nudge_used:
                        nudge_used = True
                        conv.append({
                            "role": "user",
                            "content": (
                                "[系统提醒] 你刚才只用文字回应，并没有真正调用记忆工具，"
                                "记忆库里什么都没有发生。请立刻调用合适的工具"
                                "（write_memory / update_memory / delete_memory / "
                                "search_memories / read_memory / list_recent_memories）"
                                "真正完成我上一句的请求 —— 直接输出 tool_use，不要再用文字描述或假装。"
                            ),
                        })
                        continue
                    # 终态：提取 text 推 done
                    text_parts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
                    reply_text = "\n".join(text_parts)
                    stripped = reply_text.strip()
                    if len(stripped) <= 2:
                        logger.warning(
                            "chat reply suspiciously short | text=%r blocks=%r stop=%s usage=%s",
                            reply_text, blocks, stop_reason, data.get("usage", {}),
                        )
                        yield _sse_event("error", {
                            "message": "中转站返回异常短回复（可能是临时故障）",
                            "detail": f"原文：{reply_text!r}，点「再试一次」通常就好",
                        })
                        return
                    yield _sse_event("text", {"text": reply_text})
                    yield _sse_event("done", {
                        "usage": data.get("usage", {}),
                        "model": data.get("model"),
                        "stop_reason": stop_reason,
                    })
                    return

                # 执行 tool_use → append tool_result block
                tool_called_yet = True
                tool_results_block = []
                for tu in tool_uses:
                    tname = tu.get("name", "")
                    tinput = tu.get("input", {}) or {}
                    tid = tu.get("id")
                    yield _sse_event("tool_use", {"name": tname, "input": tinput, "id": tid})
                    result = await _lingke_dispatch_tool(tname, tinput)
                    summary = result.pop("_summary", "")
                    yield _sse_event("tool_result", {
                        "name": tname, "id": tid,
                        "ok": result.get("ok", True), "summary": summary,
                    })
                    tool_results_block.append({
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": _json_lib.dumps(result, ensure_ascii=False),
                    })
                conv.append({"role": "user", "content": tool_results_block})

            yield _sse_event("error", {"message": f"tool loop 超过 {max_rounds} 轮未结束，强制终止"})

        @mcp.custom_route("/api/chat", methods=["POST"])
        async def chat_proxy(request):
            """Claude API proxy: hides key, supports official + relay endpoints.

            New (2026-05-26): body.stream=true 走 SSE + tool loop（4 个记忆工具），
            否则保持原非流式行为。
            """
            from starlette.responses import JSONResponse, StreamingResponse
            err = _require_auth(request)
            if err:
                return err
            api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not api_key:
                return JSONResponse({"error": "ANTHROPIC_API_KEY not configured in .env"}, status_code=500)
            base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
            default_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
            override = body.get("provider") if isinstance(body.get("provider"), dict) else {}
            if override.get("api_key"):
                api_key = override["api_key"].strip()
            if override.get("base_url"):
                base_url = override["base_url"].rstrip("/")
            if override.get("model"):
                default_model = override["model"]
            messages = body.get("messages", [])
            if not isinstance(messages, list) or not messages:
                return JSONResponse({"error": "messages must be a non-empty list"}, status_code=400)
            system = body.get("system", "")
            model = body.get("model", default_model)
            max_tokens = int(body.get("max_tokens", 4096))

            # ---- 新路径：SSE + tool loop ----
            if body.get("stream") is True:
                async def _event_stream():
                    async for chunk in _lingke_stream_tool_loop(
                        api_key, base_url, model, max_tokens, system, messages
                    ):
                        yield chunk
                return StreamingResponse(
                    _event_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache, no-transform",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",  # 关闭 nginx 缓冲，事件实时到达
                    },
                )

            # ---- 老路径：非流式，无工具（保留兼容） ----
            _is_or = "openrouter.ai" in base_url
            _inject_rolling_cache(messages)
            payload = {
                "model": model, "max_tokens": max_tokens, "messages": messages,
            }
            if not _is_or:
                payload["metadata"] = {"user_id": "lingke-keke-stable"}
            if system:
                payload["system"] = [
                    {"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}
                ]
            if _is_or:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                    "HTTP-Referer": "https://lingke.bond",
                    "X-Title": "lingke",
                }
            else:
                headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            endpoint = (base_url + "/messages") if base_url.endswith("/v1") else (base_url + "/v1/messages")
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    r = await client.post(endpoint, json=payload, headers=headers)
                if r.status_code != 200:
                    return JSONResponse({"error": f"upstream API {r.status_code}", "detail": r.text[:500]}, status_code=502)
                data = r.json()
            except Exception as e:
                return JSONResponse({"error": f"chat proxy failed: {type(e).__name__}: {e}"}, status_code=500)
            text_parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
            reply_text = chr(10).join(text_parts)
            stripped = reply_text.strip()
            if len(stripped) <= 2:
                logger.warning(
                    "chat reply suspiciously short | text=%r blocks=%r stop=%s usage=%s",
                    reply_text,
                    data.get("content", []),
                    data.get("stop_reason"),
                    data.get("usage", {}),
                )
                return JSONResponse(
                    {
                        "error": "中转站返回异常短回复（可能是临时故障）",
                        "detail": f"原文：{reply_text!r}，点「再试一次」通常就好",
                    },
                    status_code=502,
                )
            return JSONResponse({"ok": True, "message": {"role": "assistant", "content": reply_text}, "usage": data.get("usage", {}), "model": data.get("model"), "stop_reason": data.get("stop_reason")})


        @mcp.custom_route("/api/bucket", methods=["POST"])
        async def create_bucket(request):
            """写入一条 dynamic 记忆桶（日记 / 用户写入）。走 _merge_or_create：相似旧桶自动合并。"""
            from starlette.responses import JSONResponse
            err = _require_auth(request)
            if err:
                return err
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON"}, status_code=400)
            name = (body.get("name") or body.get("title") or "").strip()
            content = (body.get("content") or "").strip()
            if not name:
                return JSONResponse({"error": "name is required"}, status_code=400)
            if not content:
                return JSONResponse({"error": "content is required"}, status_code=400)
            domain = body.get("domain") or []
            if isinstance(domain, str):
                domain = [d.strip() for d in domain.split(",") if d.strip()]
            tags = body.get("tags") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            importance = int(body.get("importance", 5))
            valence = float(body.get("valence", 0.5))
            arousal = float(body.get("arousal", 0.3))
            try:
                # 与 hold/克克 write_memory 统一：查重合并 + 自动 embedding
                # （旧 write_memory.py 裸写不查重、不进向量索引）
                bucket_id, display_name, merged = await _merge_or_create(
                    content=content,
                    tags=list(tags),
                    importance=max(1, min(10, importance)),
                    domain=domain or ["日记"],
                    valence=max(0.0, min(1.0, valence)),
                    arousal=max(0.0, min(1.0, arousal)),
                    name=name,
                )
                return JSONResponse({
                    "ok": True, "bucket_id": bucket_id, "id": bucket_id,
                    "merged": merged, "name": display_name,
                })
            except Exception as e:
                return JSONResponse(
                    {"error": f"write failed: {type(e).__name__}: {e}"},
                    status_code=500,
                )


        # =============================================================
        # línkè · 跨设备状态同步（角色卡 / 预设 / 世界书 / 聊天历史）
        # 简单 key/value 表，client 用 last-write-wins 即可（单用户场景）
        # =============================================================
        @mcp.custom_route("/api/state", methods=["GET"])
        async def api_state_get(request):
            """读取所有同步的 state key/value，返回 {state: {key: {value, updated_at}}}"""
            from starlette.responses import JSONResponse
            err = _require_auth(request)
            if err:
                return err
            try:
                import sqlite3
                db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lingke_state.db")
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS state ("
                    "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
                )
                rows = conn.execute("SELECT key, value, updated_at FROM state").fetchall()
                conn.close()
                result = {}
                for r in rows:
                    try:
                        result[r["key"]] = {
                            "value": _json_lib.loads(r["value"]),
                            "updated_at": r["updated_at"],
                        }
                    except Exception:
                        # 损坏的条目跳过，不影响其他 key
                        continue
                return JSONResponse({"ok": True, "state": result})
            except Exception as e:
                return JSONResponse(
                    {"error": f"state read failed: {type(e).__name__}: {e}"},
                    status_code=500,
                )


        @mcp.custom_route("/api/state", methods=["POST"])
        async def api_state_set(request):
            """批量 upsert state 条目。
            Body: {entries: [{key, value, updated_at?}, ...]} 或单条 {key, value, updated_at?}"""
            from starlette.responses import JSONResponse
            err = _require_auth(request)
            if err:
                return err
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON"}, status_code=400)
            entries = body.get("entries")
            if not entries and "key" in body:
                entries = [{"key": body["key"], "value": body.get("value"),
                            "updated_at": body.get("updated_at")}]
            if not isinstance(entries, list) or not entries:
                return JSONResponse({"error": "entries (list) or key required"}, status_code=400)
            try:
                import sqlite3
                from datetime import datetime, timezone
                db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lingke_state.db")
                conn = sqlite3.connect(db_path)
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS state ("
                    "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
                )
                now = datetime.now(timezone.utc).isoformat()
                count = 0
                for e in entries:
                    k = e.get("key")
                    if not k or not isinstance(k, str):
                        continue
                    v = _json_lib.dumps(e.get("value"))
                    ts = e.get("updated_at") or now
                    conn.execute(
                        "INSERT INTO state (key, value, updated_at) VALUES (?, ?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET "
                        "value=excluded.value, updated_at=excluded.updated_at",
                        (k, v, ts)
                    )
                    count += 1
                conn.commit()
                conn.close()
                return JSONResponse({"ok": True, "count": count})
            except Exception as e:
                return JSONResponse(
                    {"error": f"state write failed: {type(e).__name__}: {e}"},
                    status_code=500,
                )


        # =============================================================
        # línkè · 聊天历史（直接走 server，跟记忆/日记同模式）
        # 单用户场景，所有消息在一张表里按 created_at 排序
        # =============================================================
        @mcp.custom_route("/api/messages", methods=["GET"])
        async def api_messages_list(request):
            """返回某条对话的消息，按 created_at 升序。
            query: ?conv=<id> 必填（前端在没 conv 时自己选 default）"""
            from starlette.responses import JSONResponse
            err = _require_auth(request)
            if err:
                return err
            conv = request.query_params.get("conv", "").strip()
            try:
                import sqlite3
                conn = sqlite3.connect(LINGKE_DB_PATH)
                conn.row_factory = sqlite3.Row
                _lingke_db_init(conn)
                if conv:
                    rows = conn.execute(
                        "SELECT id, conversation_id, role, content, created_at, error "
                        "FROM messages WHERE conversation_id=? "
                        "ORDER BY created_at ASC, id ASC",
                        (conv,),
                    ).fetchall()
                else:
                    # 没传 conv → 返回所有（兼容老前端，新前端必传）
                    rows = conn.execute(
                        "SELECT id, conversation_id, role, content, created_at, error "
                        "FROM messages ORDER BY created_at ASC, id ASC"
                    ).fetchall()
                conn.close()
                msgs = []
                for r in rows:
                    m = {
                        "id": r["id"],
                        "conversation_id": r["conversation_id"],
                        "role": r["role"],
                        "content": r["content"] or "",
                        "created_at": r["created_at"],
                    }
                    if r["error"]:
                        m["error"] = r["error"]
                    msgs.append(m)
                return JSONResponse({"ok": True, "messages": msgs})
            except Exception as e:
                return JSONResponse(
                    {"error": f"messages read failed: {type(e).__name__}: {e}"},
                    status_code=500,
                )


        @mcp.custom_route("/api/messages", methods=["POST"])
        async def api_messages_add(request):
            """新增一条消息。body: {id, conversation_id, role, content, created_at?, error?}
            id 由客户端生成（保证幂等），server 端 ON CONFLICT 跳过"""
            from starlette.responses import JSONResponse
            err = _require_auth(request)
            if err:
                return err
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON"}, status_code=400)
            mid = body.get("id")
            conv = body.get("conversation_id")
            role = body.get("role")
            content = body.get("content", "")
            error = body.get("error")
            if not mid or not isinstance(mid, str):
                return JSONResponse({"error": "id required"}, status_code=400)
            if role not in ("user", "assistant"):
                return JSONResponse({"error": "role must be user|assistant"}, status_code=400)
            try:
                import sqlite3
                from datetime import datetime, timezone
                conn = sqlite3.connect(LINGKE_DB_PATH)
                _lingke_db_init(conn)
                created_at = body.get("created_at") or datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO messages (id, conversation_id, role, content, created_at, error) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "conversation_id=COALESCE(excluded.conversation_id, messages.conversation_id), "
                    "content=excluded.content, error=excluded.error",
                    (mid, conv, role, content, created_at, error)
                )
                _lingke_touch_conv(conn, conv)
                conn.commit()
                conn.close()
                return JSONResponse({"ok": True, "id": mid, "created_at": created_at})
            except Exception as e:
                return JSONResponse(
                    {"error": f"messages write failed: {type(e).__name__}: {e}"},
                    status_code=500,
                )


        @mcp.custom_route("/api/messages/{mid}", methods=["DELETE"])
        async def api_messages_delete_one(request):
            """删除单条消息"""
            from starlette.responses import JSONResponse
            err = _require_auth(request)
            if err:
                return err
            mid = request.path_params.get("mid", "")
            if not mid:
                return JSONResponse({"error": "id required"}, status_code=400)
            try:
                import sqlite3
                conn = sqlite3.connect(LINGKE_DB_PATH)
                _lingke_db_init(conn)
                conn.execute("DELETE FROM messages WHERE id=?", (mid,))
                conn.commit()
                conn.close()
                return JSONResponse({"ok": True})
            except Exception as e:
                return JSONResponse(
                    {"error": f"delete failed: {type(e).__name__}: {e}"},
                    status_code=500,
                )


        @mcp.custom_route("/api/messages", methods=["DELETE"])
        async def api_messages_clear(request):
            """清空某条对话的全部消息。query: ?conv=<id> 必填（防误删整库）"""
            from starlette.responses import JSONResponse
            err = _require_auth(request)
            if err:
                return err
            conv = request.query_params.get("conv", "").strip()
            if not conv:
                return JSONResponse({"error": "conv query param required"}, status_code=400)
            try:
                import sqlite3
                conn = sqlite3.connect(LINGKE_DB_PATH)
                _lingke_db_init(conn)
                conn.execute("DELETE FROM messages WHERE conversation_id=?", (conv,))
                conn.commit()
                conn.close()
                return JSONResponse({"ok": True})
            except Exception as e:
                return JSONResponse(
                    {"error": f"clear failed: {type(e).__name__}: {e}"},
                    status_code=500,
                )


        @mcp.custom_route("/api/messages/truncate-from/{mid}", methods=["POST"])
        async def api_messages_truncate(request):
            """删除指定 id 及之后所有消息（用于重新生成）。在同一条对话内截断。"""
            from starlette.responses import JSONResponse
            err = _require_auth(request)
            if err:
                return err
            mid = request.path_params.get("mid", "")
            if not mid:
                return JSONResponse({"error": "id required"}, status_code=400)
            try:
                import sqlite3
                conn = sqlite3.connect(LINGKE_DB_PATH)
                _lingke_db_init(conn)
                row = conn.execute(
                    "SELECT created_at, conversation_id FROM messages WHERE id=?", (mid,)
                ).fetchone()
                if not row:
                    conn.close()
                    return JSONResponse({"ok": True, "deleted": 0})
                created_at, conv = row[0], row[1]
                if conv:
                    cur = conn.execute(
                        "DELETE FROM messages WHERE conversation_id=? AND "
                        "(created_at > ? OR (created_at = ? AND id >= ?))",
                        (conv, created_at, created_at, mid),
                    )
                else:
                    # 兼容老消息（不该发生，但防御）
                    cur = conn.execute(
                        "DELETE FROM messages WHERE created_at > ? "
                        "OR (created_at = ? AND id >= ?)",
                        (created_at, created_at, mid),
                    )
                deleted = cur.rowcount
                conn.commit()
                conn.close()
                return JSONResponse({"ok": True, "deleted": deleted})
            except Exception as e:
                return JSONResponse(
                    {"error": f"truncate failed: {type(e).__name__}: {e}"},
                    status_code=500,
                )


        # =============================================================
        # línkè · 多对话线（2026-05-26）—— conversations CRUD
        # 强绑定角色卡（character_id 不可改，要换 = 新建对话）
        # =============================================================
        @mcp.custom_route("/api/conversations", methods=["GET"])
        async def api_conversations_list(request):
            """列出所有对话 + 每条的消息数 + 最后一条预览。按 archived ASC, updated_at DESC 排。"""
            from starlette.responses import JSONResponse
            err = _require_auth(request)
            if err:
                return err
            try:
                import sqlite3
                conn = sqlite3.connect(LINGKE_DB_PATH)
                conn.row_factory = sqlite3.Row
                _lingke_db_init(conn)
                rows = conn.execute(
                    "SELECT c.id, c.name, c.character_id, c.created_at, c.updated_at, c.archived, "
                    "  (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) AS msg_count, "
                    "  (SELECT content FROM messages m WHERE m.conversation_id=c.id "
                    "     ORDER BY created_at DESC, id DESC LIMIT 1) AS last_preview "
                    "FROM conversations c "
                    "ORDER BY c.archived ASC, c.updated_at DESC"
                ).fetchall()
                conn.close()
                convs = []
                for r in rows:
                    preview = (r["last_preview"] or "")[:80]
                    convs.append({
                        "id": r["id"],
                        "name": r["name"],
                        "character_id": r["character_id"],
                        "created_at": r["created_at"],
                        "updated_at": r["updated_at"],
                        "archived": bool(r["archived"]),
                        "msg_count": r["msg_count"],
                        "last_preview": preview,
                    })
                return JSONResponse({"ok": True, "conversations": convs})
            except Exception as e:
                return JSONResponse(
                    {"error": f"conversations read failed: {type(e).__name__}: {e}"},
                    status_code=500,
                )


        @mcp.custom_route("/api/conversations", methods=["POST"])
        async def api_conversations_create(request):
            """新建对话。body: {id, name, character_id?}
            id 由 client 生成保证幂等。"""
            from starlette.responses import JSONResponse
            err = _require_auth(request)
            if err:
                return err
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON"}, status_code=400)
            cid = (body.get("id") or "").strip()
            name = (body.get("name") or "").strip()
            character_id = body.get("character_id")
            if not cid:
                return JSONResponse({"error": "id required"}, status_code=400)
            if not name:
                return JSONResponse({"error": "name required"}, status_code=400)
            try:
                import sqlite3
                from datetime import datetime, timezone
                conn = sqlite3.connect(LINGKE_DB_PATH)
                _lingke_db_init(conn)
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO conversations (id, name, character_id, created_at, updated_at, archived) "
                    "VALUES (?, ?, ?, ?, ?, 0) "
                    "ON CONFLICT(id) DO NOTHING",
                    (cid, name, character_id, now, now),
                )
                conn.commit()
                conn.close()
                return JSONResponse({"ok": True, "id": cid, "created_at": now})
            except Exception as e:
                return JSONResponse(
                    {"error": f"conversation create failed: {type(e).__name__}: {e}"},
                    status_code=500,
                )


        @mcp.custom_route("/api/conversations/{cid}", methods=["PATCH"])
        async def api_conversations_update(request):
            """改对话的可变字段。body: {name?, archived?} —— character_id 不可改（强绑定）"""
            from starlette.responses import JSONResponse
            err = _require_auth(request)
            if err:
                return err
            cid = request.path_params.get("cid", "")
            if not cid:
                return JSONResponse({"error": "id required"}, status_code=400)
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON"}, status_code=400)
            sets = []
            args = []
            if "name" in body:
                n = (body["name"] or "").strip()
                if not n:
                    return JSONResponse({"error": "name cannot be empty"}, status_code=400)
                sets.append("name=?")
                args.append(n)
            if "archived" in body:
                sets.append("archived=?")
                args.append(1 if body["archived"] else 0)
            if not sets:
                return JSONResponse({"error": "no updatable fields"}, status_code=400)
            try:
                import sqlite3
                from datetime import datetime, timezone
                conn = sqlite3.connect(LINGKE_DB_PATH)
                _lingke_db_init(conn)
                sets.append("updated_at=?")
                args.append(datetime.now(timezone.utc).isoformat())
                args.append(cid)
                conn.execute(
                    f"UPDATE conversations SET {', '.join(sets)} WHERE id=?",
                    tuple(args),
                )
                conn.commit()
                conn.close()
                return JSONResponse({"ok": True})
            except Exception as e:
                return JSONResponse(
                    {"error": f"conversation update failed: {type(e).__name__}: {e}"},
                    status_code=500,
                )


        @mcp.custom_route("/api/conversations/{cid}", methods=["DELETE"])
        async def api_conversations_delete(request):
            """删除对话 + 该对话下所有消息（hard delete）"""
            from starlette.responses import JSONResponse
            err = _require_auth(request)
            if err:
                return err
            cid = request.path_params.get("cid", "")
            if not cid:
                return JSONResponse({"error": "id required"}, status_code=400)
            try:
                import sqlite3
                conn = sqlite3.connect(LINGKE_DB_PATH)
                _lingke_db_init(conn)
                conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
                conn.execute("DELETE FROM conversations WHERE id=?", (cid,))
                conn.commit()
                conn.close()
                return JSONResponse({"ok": True})
            except Exception as e:
                return JSONResponse(
                    {"error": f"conversation delete failed: {type(e).__name__}: {e}"},
                    status_code=500,
                )


        @mcp.custom_route("/api/state/{key}", methods=["DELETE"])
        async def api_state_delete(request):
            """删除某个 key（用于 client 想清理某条数据）"""
            from starlette.responses import JSONResponse
            err = _require_auth(request)
            if err:
                return err
            key = request.path_params.get("key", "")
            if not key:
                return JSONResponse({"error": "key required"}, status_code=400)
            try:
                import sqlite3
                db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lingke_state.db")
                conn = sqlite3.connect(db_path)
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS state ("
                    "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
                )
                conn.execute("DELETE FROM state WHERE key=?", (key,))
                conn.commit()
                conn.close()
                return JSONResponse({"ok": True})
            except Exception as e:
                return JSONResponse(
                    {"error": f"state delete failed: {type(e).__name__}: {e}"},
                    status_code=500,
                )


        # --- Application-level keepalive: ping /health every 60s ---
        # --- 应用层保活：每 60 秒 ping 一次 /health，防止 Cloudflare Tunnel 空闲断连 ---
        async def _keepalive_loop():
            await asyncio.sleep(10)  # Wait for server to fully start
            async with httpx.AsyncClient() as client:
                while True:
                    try:
                        await client.get(f"http://localhost:{OMBRE_PORT}/health", timeout=5)
                        logger.debug("Keepalive ping OK / 保活 ping 成功")
                    except Exception as e:
                        logger.warning(f"Keepalive ping failed / 保活 ping 失败: {e}")
                    await asyncio.sleep(60)

        def _start_keepalive():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_keepalive_loop())

        t = threading.Thread(target=_start_keepalive, daemon=True)
        t.start()

        # --- Add CORS middleware so remote clients (Cloudflare Tunnel / ngrok) can connect ---
        # --- 添加 CORS 中间件，让远程客户端（Cloudflare Tunnel / ngrok）能正常连接 ---
        if transport == "streamable-http":
            _app = mcp.streamable_http_app()
        else:
            _app = mcp.sse_app()
        _app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )
        logger.info("CORS middleware enabled for remote transport / 已启用 CORS 中间件")

        # NOTE: decay engine boot-wakeup is done by systemd ExecStartPost → GET /health
        # (the /health route calls ensure_started; this Starlette version has no
        #  add_event_handler, and FastMCP owns the lifespan)
        # 注：衰减引擎的开机唤醒由 systemd ExecStartPost 请求 /health 完成
        uvicorn.run(_app, host="0.0.0.0", port=OMBRE_PORT)
    else:
        mcp.run(transport=transport)
