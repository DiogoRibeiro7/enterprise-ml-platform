"""Simple multi‑factor authentication helpers."""

from __future__ import annotations

import base64
import hmac
import struct
import time
from dataclasses import dataclass
from hashlib import sha1


@dataclass
class MultiFactorAuth:
    """Implements basic TOTP generation and verification."""

    interval: int = 30
    digits: int = 6

    def _time_counter(self, for_time: int | None = None) -> int:
        if for_time is None:
            for_time = int(time.time())
        return int(for_time / self.interval)

    def generate(self, secret: str, for_time: int | None = None) -> str:
        """Generate a TOTP code for ``secret``."""

        counter = self._time_counter(for_time)
        key = base64.b32decode(secret, casefold=True)
        msg = struct.pack("!Q", counter)
        digest = hmac.new(key, msg, sha1).digest()
        offset = digest[-1] & 0x0F
        truncated = digest[offset : offset + 4]
        code = struct.unpack("!I", truncated)[0] & 0x7FFFFFFF
        return str(code % (10**self.digits)).zfill(self.digits)

    def verify(self, secret: str, code: str, at_time: int | None = None) -> bool:
        """Verify that ``code`` is valid for ``secret``."""

        # allow one step clock skew
        for offset in (-1, 0, 1):
            counter_time = self._time_counter(at_time) + offset
            expected = self.generate(secret, counter_time * self.interval)
            if hmac.compare_digest(expected, code):
                return True
        return False
