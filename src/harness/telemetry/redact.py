"""Secret redaction for telemetry output.

Scrubs known secret patterns before writing to disk. This is a defense-in-depth
layer, not a complete solution: still use environment variables and key vaults
for actual secret management. The goal is to make accidental disclosure less
likely.
"""

from __future__ import annotations

import re
from typing import Any

_HEX_ONLY = re.compile(r"[0-9a-fA-F]+")


def redact(text: str) -> str:
    """Remove known secret patterns from text, replacing with [redacted:*].

    Covers: AWS keys, bearer tokens, api_key=..., password=..., etc., PEM
    private key blocks, .env-style assignments with secret names, GitHub tokens,
    and long base64-looking blobs.

    All patterns are case-insensitive.
    """
    if not text:
        return text

    result = text

    # AWS access keys: AKIA... (starts with AKIA followed by 16 alphanumeric)
    result = re.sub(
        r'\bAKIA[0-9A-Za-z]{16}\b',
        '[redacted:aws_key]',
        result
    )

    # GitHub tokens: gh[pousr]_... followed by 20+ alphanumeric/underscore characters.
    result = re.sub(
        r'\bgh[pousr]_[A-Za-z0-9_]{20,}\b',
        '[redacted:github_token]',
        result
    )

    # Bearer tokens: "Bearer <token>" or just token after Bearer keyword
    result = re.sub(
        r'(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b',
        '[redacted:bearer_token]',
        result
    )

    # .env-style assignments with secret names in key: NAME=value or NAME="value"
    # Matches: API_KEY=..., api_key=..., PASSWORD=..., etc.
    result = re.sub(
        r"(?i)\b([A-Z_]*(?:KEY|SECRET|TOKEN|PASSWORD)[A-Z_]*)\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s'\"]+)",
        r'[\1]=[redacted:\1]',
        result
    )

    # Keywords in assignments: api_key=, apikey=, password=, passwd=, token=, secret=
    # These might not match the env-style pattern above
    result = re.sub(
        r"(?i)(?:api_?key|password|passwd|token|secret)\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s'\"]+)",
        '[redacted:api_key]',
        result
    )

    # PEM private key blocks: -----BEGIN ... PRIVATE KEY-----
    result = re.sub(
        r'-----BEGIN\s+[A-Z0-9\s]+PRIVATE KEY-----.*?-----END\s+[A-Z0-9\s]+PRIVATE KEY-----',
        '[redacted:pem_private_key]',
        result,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Long base64-looking blobs: 40+ characters of alphanumeric, +, /, =
    # Avoid replacing short tokens or already-redacted markers
    def _blob(match: re.Match[str]) -> str:
        text = match.group(0)
        # A pure hexadecimal run is a digest, not a credential. Prefix hashes and
        # commit ids are exactly what makes a journal diagnosable, so redacting
        # them would destroy the observability this module exists to protect.
        if _HEX_ONLY.fullmatch(text):
            return text
        return text if text.startswith('[') else '[redacted:base64_blob]'

    result = re.sub(r'(?<!\[redacted:)[A-Za-z0-9+/]{40,}={0,2}(?!\])', _blob, result)

    return result


def redact_data(value: Any) -> Any:
    """Apply :func:`redact` to every string inside a JSON-shaped structure.

    Structure-preserving on purpose. Callers that inspect stored data need
    both a printable object and valid JSON out of the same pass, so a variant
    that rendered to text first would force one of them to parse it back.
    Each value is scrubbed on its own: running the patterns over a whole
    rendered line redacts ``max_output_tokens`` for containing the word TOKEN.
    """
    if isinstance(value, dict):
        return {key: redact_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, str):
        return redact(value)
    return value
