"""
Security checks for SentinelSearch's autonomous agents.

Two concrete risks a research agent that fetches arbitrary web pages and
feeds them into an LLM actually faces:

1. SSRF — a malicious/misconfigured search result could point at an
   internal address (localhost, 169.254.169.254 cloud metadata, a private
   LAN IP, etc.). `is_safe_url` blocks those before any fetch happens.

2. Prompt injection via retrieved content — a scraped web page can contain
   text like "Ignore your previous instructions and instead..." aimed at
   hijacking the agent that reads it, not the end user. Since Web Scout and
   Fact Checker paste fetched text directly into an LLM prompt, that's a
   real injection surface. `detect_prompt_injection` / `sanitize_external_content`
   catch and neutralize the common patterns before the text ever reaches a model.

Every flagged event is written to an append-only audit log so a run can be
reviewed after the fact.
"""
from __future__ import annotations
import ipaddress
import json
import os
import re
import socket
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------

_BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal"}


def is_safe_url(url: str) -> Tuple[bool, str]:
    """Returns (is_safe, reason_if_unsafe). Blocks non-http(s) schemes and
    anything resolving to a private/loopback/link-local address, which
    covers the classic SSRF targets (localhost, cloud metadata endpoints,
    internal LAN services)."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "unparseable URL"

    if parsed.scheme not in ("http", "https"):
        return False, f"disallowed scheme '{parsed.scheme}'"

    host = parsed.hostname or ""
    if not host:
        return False, "no hostname"
    if host.lower() in _BLOCKED_HOSTNAMES:
        return False, f"blocked hostname '{host}'"

    try:
        resolved_ips = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except socket.gaierror:
        # If it doesn't resolve, let the HTTP call fail naturally rather than
        # guessing — DNS lookup errors aren't a security signal by themselves.
        return True, ""

    for ip_str in resolved_ips:
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False, f"resolves to non-public address {ip_str}"

    return True, ""


# ---------------------------------------------------------------------------
# Prompt injection detection for untrusted retrieved content
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?(previous|prior|above) instructions",
    r"disregard (all|any|the)? ?(previous|prior|above) instructions",
    r"you are now\s+\w+",
    r"system prompt\s*:",
    r"reveal (your|the) (system prompt|instructions)",
    r"act as (if you|though)",
    r"new instructions?\s*:",
    r"override (your|the) (rules|instructions|guidelines)",
    r"do not (follow|obey) (the|your) (original|previous)",
    r"\bDAN\b.{0,20}(mode|prompt)",
    r"forget (everything|all) (you|that)",
]
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def detect_prompt_injection(text: str) -> Tuple[bool, List[str]]:
    """Scans retrieved/external text for known prompt-injection phrasing.
    Returns (flagged, matched_snippets)."""
    matches = []
    for pattern in _COMPILED_PATTERNS:
        m = pattern.search(text)
        if m:
            matches.append(m.group(0))
    return (len(matches) > 0, matches)


def sanitize_external_content(text: str, source: str = "unknown") -> Tuple[str, bool]:
    """Neutralizes suspected injection attempts in text pulled from the web
    before it's placed in an LLM prompt. Rather than silently dropping
    content (which can hide real information), suspicious spans are
    defused in place and the whole event is logged/flagged for review."""
    flagged, matches = detect_prompt_injection(text)
    if not flagged:
        return text, False

    cleaned = text
    for pattern in _COMPILED_PATTERNS:
        cleaned = pattern.sub("[REDACTED: possible prompt injection]", cleaned)

    AUDIT_LOG.log("prompt_injection_detected", {
        "source": source,
        "matched_phrases": matches,
    })
    return cleaned, True


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

@dataclass
class SecurityEvent:
    ts: float
    event_type: str
    details: dict


class AuditLog:
    def __init__(self, log_path: str = "./data/security_log.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.events: List[SecurityEvent] = []

    def log(self, event_type: str, details: dict) -> SecurityEvent:
        event = SecurityEvent(time.time(), event_type, details)
        self.events.append(event)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(event)) + "\n")
        return event

    def recent(self, n: int = 20) -> List[dict]:
        return [asdict(e) for e in self.events[-n:]]


AUDIT_LOG = AuditLog()
