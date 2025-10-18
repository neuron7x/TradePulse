import pytest
from fastapi import HTTPException

from application.api.authorization import _normalise_roles, require_roles
from src.admin.remote_control import AdminIdentity


class TestNormaliseRoles:
    def test_trims_and_deduplicates_roles(self) -> None:
        roles = ["  Admin  ", "admin", "Operator", " operator "]

        assert _normalise_roles(roles) == ("admin", "operator")

    def test_raises_value_error_when_no_valid_roles(self) -> None:
        with pytest.raises(ValueError, match="At least one non-empty role must be provided"):
            _normalise_roles(["   ", "\t\n"])  # only whitespace entries


@pytest.mark.anyio
class TestRequireRolesAllMatch:
    async def test_passes_when_all_roles_granted(self) -> None:
        identity = AdminIdentity(subject="alice", roles=("admin", "operator"))
        dependency = require_roles(["Admin", "Operator"], match="all")

        result = await dependency(identity)

        assert result is identity

    async def test_raises_http_403_with_missing_roles(self) -> None:
        identity = AdminIdentity(subject="bob", roles=("admin",))
        dependency = require_roles(["Admin", "Operator"], match="all")

        with pytest.raises(HTTPException) as exc_info:
            await dependency(identity)

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == {
            "message": "Insufficient privileges for this operation.",
            "missing_roles": ["operator"],
        }


@pytest.mark.anyio
class TestRequireRolesAnyMatch:
    async def test_passes_when_at_least_one_role_matches(self) -> None:
        identity = AdminIdentity(subject="carol", roles=("auditor", "operator"))
        dependency = require_roles(["Admin", "Operator"], match="any")

        result = await dependency(identity)

        assert result is identity

    async def test_raises_http_403_when_no_roles_match(self) -> None:
        identity = AdminIdentity(subject="dave", roles=("auditor",))
        dependency = require_roles(["Admin", "Operator"], match="any")

        with pytest.raises(HTTPException) as exc_info:
            await dependency(identity)

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == {
            "message": "One of the required roles is missing.",
            "required_roles": ["admin", "operator"],
        }


@pytest.mark.anyio
async def test_require_roles_invalid_match_mode() -> None:
    identity = AdminIdentity(subject="eve", roles=("admin",))
    dependency = require_roles(["Admin"], match="none")

    with pytest.raises(ValueError, match="Unsupported match strategy: none"):
        await dependency(identity)
