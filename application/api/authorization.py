"""Authorization helpers for securing FastAPI endpoints."""

from __future__ import annotations

from typing import Awaitable, Callable, Iterable, Literal, Sequence

from fastapi import Depends, HTTPException, status

from src.admin.remote_control import AdminIdentity

from .security import verify_request_identity

__all__ = ["require_roles"]


def _normalise_roles(roles: Iterable[str]) -> tuple[str, ...]:
    normalised = []
    for role in roles:
        candidate = role.strip().lower()
        if not candidate:
            continue
        if candidate not in normalised:
            normalised.append(candidate)
    if not normalised:
        raise ValueError("At least one non-empty role must be provided")
    return tuple(normalised)


def require_roles(
    roles: Sequence[str] | str,
    *,
    identity_dependency: Callable[..., Awaitable[AdminIdentity]] | None = None,
    match: Literal["all", "any"] = "all",
) -> Callable[[AdminIdentity], AdminIdentity]:
    """Return a dependency enforcing role based access control."""

    if isinstance(roles, str):
        required = _normalise_roles([roles])
    else:
        required = _normalise_roles(roles)

    dependency = identity_dependency or verify_request_identity()

    async def _dependency(
        identity: AdminIdentity = Depends(dependency),
    ) -> AdminIdentity:
        granted = identity.role_set
        if match == "all":
            missing = [role for role in required if role not in granted]
            if missing:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "message": "Insufficient privileges for this operation.",
                        "missing_roles": missing,
                    },
                )
        elif match == "any":
            if not any(role in granted for role in required):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "message": "One of the required roles is missing.",
                        "required_roles": list(required),
                    },
                )
        else:  # pragma: no cover - defensive guard for enum evolution
            raise ValueError(f"Unsupported match strategy: {match}")
        return identity

    return _dependency
