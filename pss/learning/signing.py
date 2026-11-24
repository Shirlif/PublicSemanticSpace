"""Ed25519 signing helpers for Phase 9 packages.

sha256 proves integrity; Ed25519 proves publisher identity. The backend signs
a manifest with its private key and the edge verifies with a pinned public key.
"""

from __future__ import annotations

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def generate_keypair() -> tuple[bytes, bytes]:
    private = Ed25519PrivateKey.generate()
    private_bytes = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_bytes, public_bytes


def public_from_private(private_key: bytes) -> bytes:
    key = serialization.load_pem_private_key(private_key, password=None)
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def sign_bytes(private_key: bytes, data: bytes) -> str:
    key = serialization.load_pem_private_key(private_key, password=None)
    return key.sign(data).hex()


def verify_bytes(public_key: bytes, data: bytes, signature: str) -> bool:
    try:
        key = serialization.load_pem_public_key(public_key)
        key.verify(bytes.fromhex(signature), data)
        return True
    except Exception:
        return False


def public_key_text(public_key: bytes) -> str:
    return public_key.decode("utf-8")


def public_key_bytes(text: str) -> bytes:
    return text.encode("utf-8")
