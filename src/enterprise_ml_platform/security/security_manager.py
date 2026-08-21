"""Top level orchestration for security features."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .access_control.rbac_manager import RBACManager
from .audit.audit_logger import AuditLogger
from .authentication.multi_factor_auth import MultiFactorAuth
from .compliance.gdpr_compliance import GDPRCompliance
from .compliance.hipaa_compliance import HIPAACompliance
from .encryption.data_encryption import DataEncryption
from .privacy.pii_detector import PiiDetector
from .scanning.vulnerability_scanner import VulnerabilityScanner


@dataclass
class SecurityManager:
    """Coordinates all security related subsystems.

    The manager wires together the various helper classes implemented in
    the ``security`` package.  Only lightweight logic lives here so that
    it can be easily swapped out for integrations with existing
    enterprise infrastructure.
    """

    encryption_key: bytes
    audit_log: Path

    rbac: RBACManager = field(default_factory=RBACManager)
    pii: PiiDetector = field(default_factory=PiiDetector)
    gdpr: GDPRCompliance = field(default_factory=GDPRCompliance)
    hipaa: HIPAACompliance = field(default_factory=HIPAACompliance)
    scanner: VulnerabilityScanner = field(default_factory=VulnerabilityScanner)
    mfa: MultiFactorAuth = field(default_factory=MultiFactorAuth)

    def __post_init__(self) -> None:
        self.encryption = DataEncryption(self.encryption_key)
        self.audit = AuditLogger(self.audit_log)

    # Encryption -----------------------------------------------------
    def encrypt(self, data: bytes) -> tuple[bytes, bytes]:
        self.audit.log("encrypt")
        return self.encryption.encrypt_at_rest(data)

    def decrypt(self, nonce: bytes, ciphertext: bytes) -> bytes:
        self.audit.log("decrypt")
        return self.encryption.decrypt_at_rest(nonce, ciphertext)

    # RBAC -----------------------------------------------------------
    def check_permission(self, user: str, perm: str) -> bool:
        allowed = self.rbac.check_access(user, perm)
        self.audit.log("check_permission", user=user, perm=perm, allowed=allowed)
        return allowed

    # PII ------------------------------------------------------------
    def anonymize_dataset(self, text: str) -> str:
        findings = self.pii.detect(text)
        if findings:
            self.audit.log("pii_detected", findings=findings)
        return self.pii.anonymize(text)

    # Compliance -----------------------------------------------------
    def record_consent(self, user_id: str, granted: bool) -> None:
        self.gdpr.record_consent(user_id, granted)
        self.audit.log("consent_recorded", user=user_id, granted=granted)

    def handle_deletion(self, user_id: str) -> None:
        if self.gdpr.should_delete(user_id):
            self.audit.log("gdpr_delete", user=user_id)
            # placeholder for deletion logic

    # Scanning -------------------------------------------------------
    def scan(self, packages: list[str]) -> dict[str, list[str]]:
        findings = self.scanner.scan_dependencies(packages)
        self.audit.log("scan", findings=findings)
        return findings

    # MFA ------------------------------------------------------------
    def generate_mfa(self, secret: str) -> str:
        return self.mfa.generate(secret)

    def verify_mfa(self, secret: str, code: str) -> bool:
        result = self.mfa.verify(secret, code)
        self.audit.log("mfa_verify", success=result)
        return result

    # Model security placeholders -----------------------------------
    def secure_model(self, model_bytes: bytes) -> tuple[bytes, bytes]:
        """Encrypt a model artifact."""

        return self.encrypt(model_bytes)
