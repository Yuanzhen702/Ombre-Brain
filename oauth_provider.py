"""
最小可用 OAuth 2.0 授权服务器 provider（2026-07-04 安全体检时新增）。

背景：claude.ai 现在给「新添加的」远程 MCP 连接器强制要求 OAuth（老连接器被祖父豁免，
一旦删除重加就必须走 OAuth）。上游 Ombre-Brain 本身不带 OAuth，于是铃删掉旧连接器后
再也加不回来（POST /register 405）。本模块用 MCP SDK 内置的 OAuth 能力补上这一套。

安全定位：/mcp 端点在 nginx 层已用 IP 白名单锁死（只放行 Anthropic 连接器出口段），
所以这里的 OAuth 主要是「满足 claude.ai 协议要求」的礼仪，采用【自动放行】(auto-approve)：
任何走到 /authorize 的请求都直接发码换令牌，不弹登录页。真正的访问边界是那道 IP 白名单
+ claude.ai 自己的账号体系。即便有人拿到令牌，也进不了被白名单挡住的 /mcp。

持久化：所有 client / 授权码 / 令牌存在 oauth.db（SQLite），**跨大脑重启存活**——否则每次
重启都会让铃的连接器掉线、被迫重新授权（正是我们要根除的脆弱点）。
"""
import os
import json
import time
import secrets
import sqlite3
import threading

from mcp.server.auth.provider import (
    AuthorizationParams,
    AuthorizationCode,
    RefreshToken,
    AccessToken,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oauth.db")
_LOCK = threading.Lock()

# 令牌寿命：访问令牌 30 天（够久，少触发刷新），刷新令牌不过期。都持久化，重启不失效。
_ACCESS_TTL = 30 * 24 * 3600
_CODE_TTL = 600  # 授权码 10 分钟内换取，一次性
_DEFAULT_SCOPES = ["user"]


def _conn():
    c = sqlite3.connect(_DB_PATH)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def _init_db():
    with _LOCK, _conn() as c:
        c.execute("CREATE TABLE IF NOT EXISTS clients (client_id TEXT PRIMARY KEY, data TEXT)")
        c.execute(
            "CREATE TABLE IF NOT EXISTS auth_codes (code TEXT PRIMARY KEY, data TEXT, expires_at REAL)"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS access_tokens "
            "(token TEXT PRIMARY KEY, client_id TEXT, scopes TEXT, expires_at INTEGER, resource TEXT)"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS refresh_tokens "
            "(token TEXT PRIMARY KEY, client_id TEXT, scopes TEXT, expires_at INTEGER)"
        )


class SqliteOAuthProvider:
    """自动放行 + SQLite 持久化的 OAuth 授权服务器 provider。"""

    def __init__(self):
        _init_db()

    # ---- 客户端（动态注册 DCR）----
    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        with _LOCK, _conn() as c:
            row = c.execute("SELECT data FROM clients WHERE client_id=?", (client_id,)).fetchone()
        if not row:
            return None
        return OAuthClientInformationFull.model_validate_json(row[0])

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        with _LOCK, _conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO clients (client_id, data) VALUES (?, ?)",
                (client_info.client_id, client_info.model_dump_json()),
            )

    # ---- 授权（/authorize）：自动发码，直接重定向回 redirect_uri ----
    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        code = "code_" + secrets.token_urlsafe(32)
        scopes = params.scopes or _DEFAULT_SCOPES
        record = {
            "code": code,
            "scopes": scopes,
            "expires_at": time.time() + _CODE_TTL,
            "client_id": client.client_id,
            "code_challenge": params.code_challenge,  # SDK 的 /token 处理器会自己校验 PKCE
            "redirect_uri": str(params.redirect_uri),
            "redirect_uri_provided_explicitly": params.redirect_uri_provided_explicitly,
            "resource": params.resource,
        }
        with _LOCK, _conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO auth_codes (code, data, expires_at) VALUES (?, ?, ?)",
                (code, json.dumps(record), record["expires_at"]),
            )
        return construct_redirect_uri(str(params.redirect_uri), code=code, state=params.state)

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        with _LOCK, _conn() as c:
            row = c.execute("SELECT data FROM auth_codes WHERE code=?", (authorization_code,)).fetchone()
        if not row:
            return None
        d = json.loads(row[0])
        if d["expires_at"] < time.time():
            return None
        return AuthorizationCode(
            code=d["code"],
            scopes=d["scopes"],
            expires_at=d["expires_at"],
            client_id=d["client_id"],
            code_challenge=d["code_challenge"],
            redirect_uri=d["redirect_uri"],
            redirect_uri_provided_explicitly=d["redirect_uri_provided_explicitly"],
            resource=d.get("resource"),
        )

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        # 授权码一次性：换取后立即删除
        with _LOCK, _conn() as c:
            c.execute("DELETE FROM auth_codes WHERE code=?", (authorization_code.code,))
        return self._issue_tokens(
            authorization_code.client_id, authorization_code.scopes, authorization_code.resource
        )

    # ---- 刷新令牌 ----
    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        with _LOCK, _conn() as c:
            row = c.execute(
                "SELECT client_id, scopes, expires_at FROM refresh_tokens WHERE token=?", (refresh_token,)
            ).fetchone()
        if not row:
            return None
        return RefreshToken(
            token=refresh_token, client_id=row[0], scopes=json.loads(row[1]), expires_at=row[2]
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        # 轮换：作废旧刷新令牌，签发新的一对
        with _LOCK, _conn() as c:
            c.execute("DELETE FROM refresh_tokens WHERE token=?", (refresh_token.token,))
        use_scopes = scopes or refresh_token.scopes
        return self._issue_tokens(refresh_token.client_id, use_scopes, None)

    # ---- 访问令牌校验（/mcp 每次请求都会调）----
    async def load_access_token(self, token: str) -> AccessToken | None:
        with _LOCK, _conn() as c:
            row = c.execute(
                "SELECT client_id, scopes, expires_at, resource FROM access_tokens WHERE token=?", (token,)
            ).fetchone()
        if not row:
            return None
        if row[2] and row[2] < int(time.time()):
            return None
        return AccessToken(
            token=token, client_id=row[0], scopes=json.loads(row[1]), expires_at=row[2], resource=row[3]
        )

    async def revoke_token(self, token) -> None:
        t = getattr(token, "token", None) or str(token)
        with _LOCK, _conn() as c:
            c.execute("DELETE FROM access_tokens WHERE token=?", (t,))
            c.execute("DELETE FROM refresh_tokens WHERE token=?", (t,))

    # ---- 内部：签发一对 access + refresh 令牌并持久化 ----
    def _issue_tokens(self, client_id: str, scopes, resource) -> OAuthToken:
        access = "at_" + secrets.token_urlsafe(32)
        refresh = "rt_" + secrets.token_urlsafe(32)
        exp = int(time.time()) + _ACCESS_TTL
        scopes = list(scopes) if scopes else list(_DEFAULT_SCOPES)
        with _LOCK, _conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO access_tokens (token, client_id, scopes, expires_at, resource) "
                "VALUES (?, ?, ?, ?, ?)",
                (access, client_id, json.dumps(scopes), exp, resource),
            )
            c.execute(
                "INSERT OR REPLACE INTO refresh_tokens (token, client_id, scopes, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (refresh, client_id, json.dumps(scopes), None),
            )
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=_ACCESS_TTL,
            scope=" ".join(scopes),
            refresh_token=refresh,
        )
