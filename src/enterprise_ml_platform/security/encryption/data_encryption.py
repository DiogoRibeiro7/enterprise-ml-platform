"""Data encryption utilities.

This module provides simple helpers for encrypting data at rest and in
transit.  It is **not** a full featured key management solution but
offers a small abstraction that can easily be replaced by one backed by
services such as AWS KMS or Azure Key Vault.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass
class DataEncryption:
    """Helper class implementing AES‑256 encryption.

    Parameters
    ----------
    key: bytes
        Symmetric key used for encryption.  The key must be 32 bytes long
        to comply with AES‑256 requirements.  In a production deployment
        this key should be fetched from a secure key management system and
        rotated regularly.
    """

    key: bytes

    def __post_init__(self) -> None:  # pragma: no cover - simple check
        if len(self.key) != 32:
            raise ValueError("AES‑256 requires 32 byte keys")

    def encrypt_at_rest(self, data: bytes, *, associated_data: bytes | None = None) -> tuple[bytes, bytes]:
        """Encrypt ``data`` for storage.

        Parameters
        ----------
        data:
            Plaintext bytes to encrypt.
        associated_data:
            Optional associated data for authentication (e.g. file name).

        Returns
        -------
        tuple
            A tuple of ``(nonce, ciphertext)`` which must both be stored
            to allow later decryption.
        """

        aesgcm = AESGCM(self.key)
        nonce = os.urandom(12)  # 96‑bit nonce as recommended for GCM
        ciphertext = aesgcm.encrypt(nonce, data, associated_data)
        return nonce, ciphertext

    def decrypt_at_rest(self, nonce: bytes, ciphertext: bytes, *, associated_data: bytes | None = None) -> bytes:
        """Decrypt data previously encrypted with :meth:`encrypt_at_rest`."""

        aesgcm = AESGCM(self.key)
        return aesgcm.decrypt(nonce, ciphertext, associated_data)

    def wrap_for_transport(self, data: bytes) -> bytes:
        """Mock helper representing TLS 1.3 transport encryption.

        Real transport encryption is handled by the networking stack
        (e.g. ``requests`` + ``https``).  For testing we simply reuse the
        at rest encryption to simulate end‑to‑end security.
        """

        nonce, ciphertext = self.encrypt_at_rest(data)
        return nonce + ciphertext

