"""
Local PII Guard — local-first privacy gateway for LLM apps and AI agents.

Detects PII, replaces it with stable reversible tokens, keeps originals
in a local Vault, and restores cloud model responses locally.

Basic usage::

    from pii_agent_pro import PIIGateway
    from pii_models import PIIGatewayPolicy

    policy = PIIGatewayPolicy(fail_closed=True)
    gateway = PIIGateway(model_path="Qwen2.5-3B-Instruct-IQ3_M.gguf", policy=policy)

    result = gateway.tokenize("session_1", "My name is John. Phone +79991234567.")
    restored = gateway.restore("session_1", result["tokenized_text"])
"""

from pii_agent_pro import PIIGateway
from pii_models import (
    PIIGatewayPolicy,
    Entity,
    ENTITY_EMAIL,
    ENTITY_PHONE,
    ENTITY_CARD,
    ENTITY_SNILS,
    ENTITY_PASSPORT,
    ENTITY_DATE,
    ENTITY_NAME,
)
from pii_vault import InMemoryPIIVault

__version__ = "0.1.0"
__all__ = [
    "PIIGateway",
    "PIIGatewayPolicy",
    "InMemoryPIIVault",
    "Entity",
    "ENTITY_EMAIL",
    "ENTITY_PHONE",
    "ENTITY_CARD",
    "ENTITY_SNILS",
    "ENTITY_PASSPORT",
    "ENTITY_DATE",
    "ENTITY_NAME",
    "__version__",
]
