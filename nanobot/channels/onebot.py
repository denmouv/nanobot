"""OneBot v11 channel implementation.

Supports both forward WebSocket client mode and reverse WebSocket server mode.
Outbound uses OneBot action API over WebSocket (preferred) or HTTP fallback.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

import httpx
import websockets
from loguru import logger
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Base
from nanobot.utils.helpers import split_message

_MAX_SEEN_MESSAGE_IDS = 2000
_CQ_CODE_RE = re.compile(r"\[CQ:([a-zA-Z0-9_]+),([^\]]*)\]")
_FILE_URI_RE = re.compile(r"^file://", re.IGNORECASE)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".tif", ".tiff"}
_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".amr", ".silk"}
_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}


class OneBotConfig(Base):
    """OneBot v11 channel configuration."""

    enabled: bool = False
    allow_from: list[str] = Field(default_factory=list)
    ws_url: str = "ws://127.0.0.1:6700"
    connection_mode: Literal["forward_ws", "reverse_ws"] = "forward_ws"
    ws_reverse_host: str = "127.0.0.1"
    ws_reverse_port: int = 6199
    ws_reverse_path: str = "/"
    ws_reverse_token: str = ""
    api_base_url: str = "http://127.0.0.1:5700"
    access_token: str = ""
    reconnect_interval: float = 5.0
    action_timeout: float = 15.0
    api_transport: Literal["ws", "http"] = "ws"
    group_policy: Literal["open", "mention"] = "mention"
    self_id: str = ""
    include_self_message: bool = False
    strip_self_mention: bool = True
    reply_to_message: bool = True
    max_message_length: int = 1800
    max_media_bytes_in: int = 50 * 1024 * 1024
    max_media_bytes_out: int = 50 * 1024 * 1024


class OneBotChannel(BaseChannel):
    """OneBot v11 channel for QQ ecosystems (NapCat/LLOneBot/go-cqhttp)."""

    name = "onebot"
    display_name = "OneBot"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """Return onboarding/default config for this channel."""
        return OneBotConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        """Initialize channel state and normalize config input type."""
        if isinstance(config, dict):
            config = OneBotConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: OneBotConfig = config

        self._ws: Any | None = None
        self._ws_server: Any | None = None
        self._http: httpx.AsyncClient | None = None
        self._connected = False
        self._reverse_connected = asyncio.Event()
        self._pending_actions: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._seen_message_ids: OrderedDict[str, None] = OrderedDict()
        self._self_id: str = str(self.config.self_id or "").strip()
        self._warned_mention_without_self_id = False

    async def start(self) -> None:
        """Start OneBot channel in forward or reverse WebSocket mode."""
        self._running = True
        self._http = httpx.AsyncClient(timeout=self.config.action_timeout)

        if self.config.connection_mode == "reverse_ws":
            await self._run_reverse_ws_server()
            return

        if not self.config.ws_url.strip():
            logger.error("OneBot ws_url not configured")
            return

        while self._running:
            try:
                await self._run_ws_session()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("OneBot connection error: {}", e)

            if self._running:
                await asyncio.sleep(max(self.config.reconnect_interval, 1.0))

    async def stop(self) -> None:
        """Stop OneBot channel and release resources."""
        self._running = False
        self._connected = False

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._reverse_connected.clear()

        if self._ws_server is not None:
            try:
                self._ws_server.close()
                await self._ws_server.wait_closed()
            except Exception:
                pass
            self._ws_server = None

        for echo, fut in list(self._pending_actions.items()):
            if not fut.done():
                fut.set_exception(RuntimeError("OneBot channel stopped"))
            self._pending_actions.pop(echo, None)

        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception:
                pass
            self._http = None

    async def send(self, msg: OutboundMessage) -> None:
        """Send outbound message through OneBot action API."""
        target_type, target_id = self._resolve_target(msg.chat_id, msg.metadata)
        if not target_id:
            raise ValueError(f"Invalid OneBot target for chat_id={msg.chat_id}")

        message_id = msg.metadata.get("message_id") or msg.reply_to
        chunks = split_message(msg.content or "", max_len=max(self.config.max_message_length, 500))

        if msg.media:
            await self._send_media_refs(
                target_type=target_type, target_id=target_id, media=msg.media
            )

        if not chunks and not msg.media:
            return

        for chunk in chunks or [""]:
            payload: str | list[dict[str, Any]]
            if message_id and self.config.reply_to_message:
                payload = [
                    {"type": "reply", "data": {"id": str(message_id)}},
                    {"type": "text", "data": {"text": chunk}},
                ]
            else:
                payload = chunk

            await self._send_message(target_type=target_type, target_id=target_id, message=payload)

    async def _run_ws_session(self) -> None:
        """Run one forward-WS client session until disconnected."""
        headers: dict[str, str] = {}
        if self.config.access_token:
            headers["Authorization"] = f"Bearer {self.config.access_token}"

        logger.info("Connecting OneBot WebSocket: {}", self.config.ws_url)

        async with websockets.connect(
            self.config.ws_url,
            additional_headers=headers or None,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            max_size=8 * 1024 * 1024,
        ) as ws:
            self._ws = ws
            self._connected = True
            logger.info("OneBot connected")

            if not self._self_id:
                await self._refresh_self_id()

            async for raw in ws:
                await self._handle_ws_frame(raw)

        self._fail_pending_actions("OneBot WebSocket disconnected")
        self._connected = False
        self._ws = None

    async def _run_reverse_ws_server(self) -> None:
        """Host reverse WebSocket server and accept OneBot bridge connections."""
        host = self.config.ws_reverse_host.strip() or "127.0.0.1"
        port = int(self.config.ws_reverse_port)
        logger.info("Starting OneBot reverse WebSocket server on {}:{}", host, port)

        async with websockets.serve(
            self._handle_reverse_connection,
            host,
            port,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            max_size=8 * 1024 * 1024,
        ) as server:
            self._ws_server = server
            while self._running:
                await asyncio.sleep(0.5)

        self._ws_server = None

    async def _handle_reverse_connection(self, ws: Any) -> None:
        """Handle a single reverse-WS connection lifecycle."""
        if not self._is_reverse_authorized(ws):
            logger.warning("OneBot reverse WebSocket unauthorized connection rejected")
            await ws.close(code=4401, reason="Unauthorized")
            return

        if self._ws is not None and self._ws is not ws:
            try:
                await self._ws.close(code=1012, reason="Replaced by new connection")
            except Exception:
                pass

        self._ws = ws
        self._connected = True
        self._reverse_connected.set()
        logger.info("OneBot reverse WebSocket connected")

        # In reverse WebSocket mode, many OneBot implementations push `self_id`
        # in the first event frame but may not respond to `get_login_info`
        # immediately after connect. To avoid noisy startup timeouts, we rely on
        # event frames to populate self_id first.

        try:
            async for raw in ws:
                await self._handle_ws_frame(raw)
        finally:
            if self._ws is ws:
                self._fail_pending_actions("OneBot reverse WebSocket disconnected")
                self._ws = None
                self._connected = False
                self._reverse_connected.clear()
            logger.info("OneBot reverse WebSocket disconnected")

    def _is_reverse_authorized(self, ws: Any) -> bool:
        """Validate reverse-WS path and bearer token (if configured)."""
        expected_path = str(self.config.ws_reverse_path or "/").strip() or "/"
        if not expected_path.startswith("/"):
            expected_path = f"/{expected_path}"

        expected = str(self.config.ws_reverse_token or self.config.access_token or "").strip()

        headers: Any = None
        request_path = ""
        request = getattr(ws, "request", None)
        if request is not None:
            headers = getattr(request, "headers", None)
            request_path = str(getattr(request, "path", "") or "")

        if request_path and request_path != expected_path:
            return False

        if not expected:
            return True

        auth = ""
        if headers is not None:
            auth = str(headers.get("Authorization", "")).strip()

        token = auth.split(" ", 1)[1].strip() if auth.lower().startswith("bearer ") else auth
        return token == expected

    async def _handle_ws_frame(self, raw: str) -> None:
        """Handle incoming WS frame: action echo reply or event dispatch."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("OneBot received non-JSON frame")
            return

        if not isinstance(data, dict):
            return

        echo = str(data.get("echo") or "").strip()
        if echo and echo in self._pending_actions:
            fut = self._pending_actions.pop(echo)
            if not fut.done():
                fut.set_result(data)
            return

        if "self_id" in data and not self._self_id:
            self._self_id = str(data.get("self_id") or "").strip()

        post_type = str(data.get("post_type") or "")
        if post_type == "message":
            await self._handle_event_message(data)
        elif post_type == "notice":
            await self._handle_event_notice(data)
        elif post_type == "request":
            await self._handle_event_request(data)

    async def _refresh_self_id(self) -> None:
        """Best-effort fetch of bot self_id via OneBot action API."""
        try:
            ret = await self._call_action("get_login_info", {})
            if isinstance(ret, dict):
                maybe = ret.get("user_id") or ret.get("uid")
                if maybe is not None:
                    self._self_id = str(maybe).strip()
                    logger.info("OneBot self_id resolved: {}", self._self_id)
        except Exception as e:
            logger.debug(
                "OneBot get_login_info failed (mode={}): {}",
                self.config.connection_mode,
                e,
            )

    async def _handle_event_message(self, event: dict[str, Any]) -> None:
        """Convert OneBot message event to nanobot inbound message."""
        message_id = str(event.get("message_id") or "").strip()
        if message_id and self._is_duplicate_message(message_id):
            return

        message_type = str(event.get("message_type") or "")
        user_id = str(event.get("user_id") or "").strip()
        group_id = str(event.get("group_id") or "").strip()
        if not user_id:
            return

        if not self.config.include_self_message and self._self_id and user_id == self._self_id:
            return

        content, media, was_mentioned = self._parse_message_payload(
            event.get("message"),
            str(event.get("raw_message") or ""),
        )
        media = await self._enrich_file_media_refs(event, media)
        media = await self._filter_media_by_size(media, self.config.max_media_bytes_in, direction="in")

        if message_type == "group" and self.config.group_policy == "mention":
            if not self._self_id and not self._warned_mention_without_self_id:
                logger.warning(
                    "OneBot group_policy=mention but self_id is unknown; "
                    "configure channels.onebot.selfId for accurate mention filtering"
                )
                self._warned_mention_without_self_id = True
            if self._self_id and not was_mentioned:
                return

        if self.config.strip_self_mention and self._self_id and content:
            content = self._strip_self_mentions(content, self._self_id)

        reply_id = self._extract_reply_id(event.get("message"), str(event.get("raw_message") or ""))
        reply_context = await self._fetch_reply_context(reply_id) if reply_id else None

        chat_id = (
            f"group:{group_id}" if message_type == "group" and group_id else f"private:{user_id}"
        )
        if not content and media:
            content = "[media]"

        metadata = {
            "message_id": message_id,
            "message_type": message_type,
            "user_id": user_id,
            "group_id": group_id,
            "raw_message": str(event.get("raw_message") or ""),
            "self_id": self._self_id,
            "was_mentioned": was_mentioned,
            "reply_id": reply_id,
            "reply_context": reply_context,
        }

        await self._handle_message(
            sender_id=user_id,
            chat_id=chat_id,
            content=content,
            media=media,
            metadata=metadata,
        )

    async def _handle_event_notice(self, event: dict[str, Any]) -> None:
        """Convert OneBot notice event to synthetic inbound text message."""
        user_id = str(event.get("user_id") or event.get("operator_id") or "").strip()
        if not user_id:
            return

        group_id = str(event.get("group_id") or "").strip()
        chat_id = f"group:{group_id}" if group_id else f"private:{user_id}"

        notice_type = str(event.get("notice_type") or "notice").strip()
        sub_type = str(event.get("sub_type") or "").strip()
        target_id = str(event.get("target_id") or "").strip()

        if notice_type == "notify" and sub_type == "poke" and target_id:
            content = f"[notice] poke:{target_id}"
        else:
            content = f"[notice] {notice_type}" + (f":{sub_type}" if sub_type else "")

        metadata = {
            "post_type": "notice",
            "notice_type": notice_type,
            "sub_type": sub_type,
            "group_id": group_id,
            "user_id": user_id,
            "target_id": target_id,
            "self_id": self._self_id,
            "raw": event,
        }

        await self._handle_message(
            sender_id=user_id,
            chat_id=chat_id,
            content=content,
            media=[],
            metadata=metadata,
        )

    async def _handle_event_request(self, event: dict[str, Any]) -> None:
        """Convert OneBot request event to synthetic inbound text message."""
        user_id = str(event.get("user_id") or "").strip()
        if not user_id:
            return

        group_id = str(event.get("group_id") or "").strip()
        chat_id = f"group:{group_id}" if group_id else f"private:{user_id}"

        request_type = str(event.get("request_type") or "request").strip()
        sub_type = str(event.get("sub_type") or "").strip()
        comment = str(event.get("comment") or "").strip()
        content = f"[request] {request_type}" + (f":{sub_type}" if sub_type else "")
        if comment:
            content += f" {comment}"

        metadata = {
            "post_type": "request",
            "request_type": request_type,
            "sub_type": sub_type,
            "group_id": group_id,
            "user_id": user_id,
            "self_id": self._self_id,
            "raw": event,
        }

        await self._handle_message(
            sender_id=user_id,
            chat_id=chat_id,
            content=content,
            media=[],
            metadata=metadata,
        )

    async def _send_media_refs(self, target_type: str, target_id: str, media: list[str]) -> None:
        """Send media references as OneBot segments.

        Notes:
        - Image/audio/video/file are mapped to OneBot segment types.
        - Local absolute paths are converted to `file://` URI for better compatibility.
        - References exceeding `max_media_bytes_out` are skipped.
        """
        for item in media:
            ref = str(item or "").strip()
            if not ref:
                continue
            if not await self._is_media_size_allowed(ref, self.config.max_media_bytes_out, "out"):
                continue

            seg_type = self._segment_type_from_ref(ref)
            onebot_ref = self._normalize_media_ref(ref)
            seg = [{"type": seg_type, "data": {"file": onebot_ref}}]
            await self._send_message(target_type=target_type, target_id=target_id, message=seg)

    async def _send_message(
        self,
        target_type: str,
        target_id: str,
        message: str | list[dict[str, Any]],
    ) -> None:
        """Send one message payload to group or private target."""
        if target_type == "group":
            action = "send_group_msg"
            params = {"group_id": target_id, "message": message}
        else:
            action = "send_private_msg"
            params = {"user_id": target_id, "message": message}

        await self._call_action(action, params)

    async def _call_action(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call OneBot action via selected transport (WS first, HTTP fallback)."""
        transport = self.config.api_transport
        if transport == "ws" and self._ws is not None and self._connected:
            return await self._call_action_ws(action, params)
        return await self._call_action_http(action, params)

    async def _call_action_ws(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call OneBot action over WebSocket with echo correlation."""
        if self._ws is None:
            raise RuntimeError("OneBot WebSocket is not connected")

        echo = uuid.uuid4().hex
        payload = {"action": action, "params": params, "echo": echo}
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending_actions[echo] = fut

        await self._ws.send(json.dumps(payload, ensure_ascii=False))

        try:
            resp = await asyncio.wait_for(fut, timeout=self.config.action_timeout)
        except asyncio.TimeoutError as e:
            self._pending_actions.pop(echo, None)
            raise RuntimeError(f"OneBot action timeout: {action}") from e

        self._validate_action_response(action, resp)
        data = resp.get("data")
        return data if isinstance(data, dict) else resp

    async def _call_action_http(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call OneBot action over HTTP endpoint `{api_base_url}/{action}`."""
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.config.action_timeout)

        base = self.config.api_base_url.rstrip("/")
        if not base:
            raise RuntimeError("OneBot api_base_url not configured")

        url = f"{base}/{action}"
        headers: dict[str, str] = {}
        if self.config.access_token:
            headers["Authorization"] = f"Bearer {self.config.access_token}"

        resp = await self._http.post(url, json=params, headers=headers)
        resp.raise_for_status()
        body = resp.json()
        if not isinstance(body, dict):
            raise RuntimeError(f"OneBot invalid HTTP response for {action}")
        self._validate_action_response(action, body)
        data = body.get("data")
        return data if isinstance(data, dict) else body

    def _fail_pending_actions(self, reason: str) -> None:
        """Fail and clear all pending WS action futures."""
        for echo, fut in list(self._pending_actions.items()):
            if not fut.done():
                fut.set_exception(RuntimeError(reason))
            self._pending_actions.pop(echo, None)

    @staticmethod
    def _validate_action_response(action: str, body: dict[str, Any]) -> None:
        """Validate standard OneBot action response fields (`status`, `retcode`)."""
        status = str(body.get("status") or "").lower()
        retcode = body.get("retcode")
        if status and status != "ok":
            raise RuntimeError(f"OneBot action failed: {action} status={status} retcode={retcode}")
        if retcode not in (None, 0):
            raise RuntimeError(f"OneBot action failed: {action} retcode={retcode}")

    def _resolve_target(self, chat_id: str, metadata: dict[str, Any] | None) -> tuple[str, str]:
        """Resolve outbound target from metadata first, then `chat_id` convention."""
        meta = metadata or {}

        if isinstance(meta.get("group_id"), (str, int)) and str(meta["group_id"]).strip():
            return "group", str(meta["group_id"]).strip()
        if isinstance(meta.get("user_id"), (str, int)) and str(meta["user_id"]).strip():
            return "private", str(meta["user_id"]).strip()

        raw = str(chat_id or "").strip()
        if raw.startswith("group:"):
            return "group", raw.split(":", 1)[1]
        if raw.startswith("private:"):
            return "private", raw.split(":", 1)[1]
        if raw.isdigit():
            return "private", raw
        return "", ""

    def _parse_message_payload(self, message: Any, raw_message: str) -> tuple[str, list[str], bool]:
        """Parse OneBot message payload across segment/CQ/plain formats."""
        if isinstance(message, list):
            return self._parse_segment_message(message)
        if isinstance(message, str) and "[CQ:" in message:
            return self._parse_cq_message(message)
        if "[CQ:" in raw_message:
            return self._parse_cq_message(raw_message)
        text = str(message) if isinstance(message, str) else raw_message
        return text.strip(), [], False

    def _parse_segment_message(self, segments: list[Any]) -> tuple[str, list[str], bool]:
        """Parse OneBot array segment payload into text/media/mention state."""
        text_parts: list[str] = []
        media: list[str] = []
        was_mentioned = False

        for segment in segments:
            if not isinstance(segment, dict):
                continue
            typ = str(segment.get("type") or "")
            data = segment.get("data")
            if not isinstance(data, dict):
                data = {}

            if typ == "text":
                text_parts.append(str(data.get("text") or ""))
                continue

            if typ == "at":
                qq = str(data.get("qq") or "")
                if self._self_id and qq in {self._self_id, "all"}:
                    was_mentioned = True
                continue

            if typ in {"image", "record", "video", "file"}:
                ref = str(data.get("url") or data.get("file") or "").strip()
                if ref:
                    media.append(ref)

        return "".join(text_parts).strip(), media, was_mentioned

    def _parse_cq_message(self, content: str) -> tuple[str, list[str], bool]:
        """Parse CQ-code string payload into text/media/mention state."""
        media: list[str] = []
        was_mentioned = False

        def repl(match: re.Match[str]) -> str:
            nonlocal was_mentioned

            typ = match.group(1)
            params = self._parse_cq_params(match.group(2))

            if typ == "at":
                qq = params.get("qq", "")
                if self._self_id and qq in {self._self_id, "all"}:
                    was_mentioned = True
                return ""

            if typ in {"image", "record", "video", "file"}:
                ref = params.get("url") or params.get("file")
                if ref:
                    media.append(ref)
                return ""

            return ""

        text = _CQ_CODE_RE.sub(repl, content)
        text = re.sub(r"\s+", " ", text).strip()
        return text, media, was_mentioned

    async def _enrich_file_media_refs(self, event: dict[str, Any], media: list[str]) -> list[str]:
        """Fill file refs from file_id for implementations like NapCat.

        Some OneBot payloads provide `file_id` without direct URL/file path.
        This method calls file-url actions to resolve downloadable refs.
        """
        out = list(media)
        message = event.get("message")
        if not isinstance(message, list):
            return out

        message_type = str(event.get("message_type") or "")
        group_id = str(event.get("group_id") or "").strip()

        for seg in message:
            if not isinstance(seg, dict) or str(seg.get("type") or "") != "file":
                continue
            data = seg.get("data")
            if not isinstance(data, dict):
                continue

            direct = str(data.get("url") or data.get("file") or "").strip()
            if direct:
                continue

            file_id = str(data.get("file_id") or "").strip()
            if not file_id:
                continue

            try:
                if message_type == "group" and group_id:
                    ret = await self._call_action(
                        "get_group_file_url",
                        {"group_id": group_id, "file_id": file_id},
                    )
                else:
                    ret = await self._call_action(
                        "get_private_file_url",
                        {"file_id": file_id},
                    )
                if isinstance(ret, dict):
                    ref = str(ret.get("url") or ret.get("file") or "").strip()
                    if ref:
                        out.append(ref)
            except Exception as e:
                logger.debug("OneBot file_id resolve failed: {}", e)

        return out

    async def _fetch_reply_context(self, reply_id: str) -> dict[str, Any] | None:
        """Fetch quoted message context by reply id (best effort)."""
        if not reply_id:
            return None
        try:
            ret = await self._call_action("get_msg", {"message_id": reply_id})
            if not isinstance(ret, dict):
                return None
            return {
                "message_id": str(ret.get("message_id") or reply_id),
                "user_id": str(ret.get("user_id") or ""),
                "message_type": str(ret.get("message_type") or ""),
                "raw_message": str(ret.get("raw_message") or ""),
            }
        except Exception as e:
            logger.debug("OneBot get_msg failed for reply_id={}: {}", reply_id, e)
            return None

    def _extract_reply_id(self, message: Any, raw_message: str) -> str:
        """Extract reply message id from segment or CQ payload."""
        if isinstance(message, list):
            for seg in message:
                if not isinstance(seg, dict):
                    continue
                if str(seg.get("type") or "") != "reply":
                    continue
                data = seg.get("data")
                if not isinstance(data, dict):
                    continue
                rid = str(data.get("id") or "").strip()
                if rid:
                    return rid

        src = ""
        if isinstance(message, str):
            src = message
        elif raw_message:
            src = raw_message
        if src:
            m = re.search(r"\[CQ:reply,id=([^\],]+)", src, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return ""

    @staticmethod
    def _segment_type_from_ref(ref: str) -> str:
        """Map a media ref suffix to OneBot segment type."""
        suffix = Path(str(ref)).suffix.lower()
        if suffix in _IMAGE_EXTS:
            return "image"
        if suffix in _AUDIO_EXTS:
            return "record"
        if suffix in _VIDEO_EXTS:
            return "video"
        return "file"

    @staticmethod
    def _is_remote_ref(ref: str) -> bool:
        p = urlparse(ref)
        return p.scheme.lower() in {"http", "https"}

    @staticmethod
    def _is_file_uri(ref: str) -> bool:
        return bool(_FILE_URI_RE.match(ref or ""))

    @staticmethod
    def _local_path_from_ref(ref: str) -> Path | None:
        """Parse local filesystem path from raw ref or file:// URI."""
        if not ref:
            return None
        if OneBotChannel._is_remote_ref(ref):
            return None
        if OneBotChannel._is_file_uri(ref):
            parsed = urlparse(ref)
            path_str = unquote(parsed.path or "")
            # Windows file URI often starts with /C:/...
            if re.match(r"^/[A-Za-z]:/", path_str):
                path_str = path_str[1:]
            return Path(path_str)
        return Path(ref)

    @staticmethod
    def _normalize_media_ref(ref: str) -> str:
        """Normalize local absolute path to file:// URI for OneBot."""
        p = OneBotChannel._local_path_from_ref(ref)
        if p is None:
            return ref
        try:
            rp = p.expanduser().resolve()
        except Exception:
            return ref
        if rp.is_absolute() and not OneBotChannel._is_file_uri(ref):
            try:
                return rp.as_uri()
            except Exception:
                return str(rp)
        return ref

    async def _media_size_bytes(self, ref: str) -> int | None:
        """Get media size in bytes for local path or remote URL (HEAD)."""
        p = self._local_path_from_ref(ref)
        if p is not None:
            try:
                rp = p.expanduser().resolve()
            except Exception:
                return None
            if not rp.exists() or not rp.is_file():
                return None
            try:
                return rp.stat().st_size
            except Exception:
                return None

        if self._is_remote_ref(ref):
            client = self._http
            if client is None:
                client = httpx.AsyncClient(timeout=self.config.action_timeout)
                self._http = client
            try:
                resp = await client.head(ref, follow_redirects=True)
                raw = resp.headers.get("Content-Length") or resp.headers.get("content-length")
                return int(raw) if raw and str(raw).isdigit() else None
            except Exception:
                return None

        return None

    async def _is_media_size_allowed(self, ref: str, max_bytes: int, direction: str) -> bool:
        """Check media size limit when measurable; unknown size is allowed."""
        if max_bytes <= 0:
            return True
        size = await self._media_size_bytes(ref)
        if size is None:
            return True
        if size > max_bytes:
            logger.warning(
                "OneBot media too large ({}): {} bytes > {} bytes, ref={}",
                direction,
                size,
                max_bytes,
                ref,
            )
            return False
        return True

    async def _filter_media_by_size(self, refs: list[str], max_bytes: int, direction: str) -> list[str]:
        """Filter refs by size threshold."""
        if max_bytes <= 0:
            return refs
        kept: list[str] = []
        for ref in refs:
            if await self._is_media_size_allowed(ref, max_bytes, direction):
                kept.append(ref)
        return kept

    @staticmethod
    def _parse_cq_params(raw: str) -> dict[str, str]:
        """Parse `k=v` comma-separated CQ parameters to dict."""
        out: dict[str, str] = {}
        for item in raw.split(","):
            if "=" not in item:
                continue
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
        return out

    @staticmethod
    def _strip_self_mentions(text: str, self_id: str) -> str:
        """Strip `@self` CQ mention segment from text if present."""
        pat = re.compile(rf"\[CQ:at,qq={re.escape(self_id)}\]\s*", re.IGNORECASE)
        stripped = pat.sub("", text)
        return re.sub(r"\s+", " ", stripped).strip()

    def _is_duplicate_message(self, message_id: str) -> bool:
        """Deduplicate recent message ids with bounded LRU-like memory."""
        if message_id in self._seen_message_ids:
            return True
        self._seen_message_ids[message_id] = None
        while len(self._seen_message_ids) > _MAX_SEEN_MESSAGE_IDS:
            self._seen_message_ids.popitem(last=False)
        return False
