"""Secreto compartido entre el backend Java y el servicio de IA.

Si `AI_SHARED_SECRET` está definida, las rutas protegidas exigen la cabecera
`X-Internal-Token` con ese mismo valor. Si no está definida (desarrollo y tests),
no se exige nada y el arranque registra una advertencia.
"""

import hmac
import logging
import os
from typing import Annotated

from fastapi import Header, HTTPException

SHARED_SECRET_ENV = "AI_SHARED_SECRET"
TOKEN_HEADER = "X-Internal-Token"
# Rutas que nunca exigen el secreto (health check de Render y documentación de FastAPI).
PUBLIC_PATHS = frozenset({"/health", "/docs", "/docs/oauth2-redirect", "/openapi.json", "/redoc"})

logger = logging.getLogger(__name__)


def configured_shared_secret() -> str | None:
    """Secreto configurado, o None si la variable falta o está vacía."""
    return os.environ.get(SHARED_SECRET_ENV, "").strip() or None


def require_shared_secret(
    x_internal_token: Annotated[str | None, Header(alias=TOKEN_HEADER)] = None,
) -> None:
    secret = configured_shared_secret()
    if secret is None:
        return
    if x_internal_token is None or not hmac.compare_digest(
        x_internal_token.encode("utf-8"), secret.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="unauthorized")


def warn_if_unprotected() -> None:
    if configured_shared_secret() is None:
        logger.warning(
            "%s no está definida: las rutas de IA aceptan peticiones sin %s. "
            "Defínela en producción.",
            SHARED_SECRET_ENV,
            TOKEN_HEADER,
        )
