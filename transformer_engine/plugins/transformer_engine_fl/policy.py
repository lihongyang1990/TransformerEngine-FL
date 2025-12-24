from __future__ import annotations

from .policy_manager import SelectionPolicy, PolicyManager, _PolicyContext


def get_policy_epoch() -> int:
    return PolicyManager.get_instance().get_policy_epoch()


def bump_policy_epoch() -> int:
    return PolicyManager.get_instance().bump_policy_epoch()


def get_policy() -> SelectionPolicy:
    return PolicyManager.get_instance().get_policy()


def set_global_policy(policy: SelectionPolicy) -> SelectionPolicy:
    return PolicyManager.get_instance().set_global_policy(policy)


def reset_global_policy() -> None:
    PolicyManager.get_instance().reset_global_policy()


def policy_from_env() -> SelectionPolicy:
    return PolicyManager.get_instance()._policy_from_env()


class policy_context(_PolicyContext):

    def __init__(self, policy: SelectionPolicy):
        manager = PolicyManager.get_instance()
        super().__init__(manager, policy)


def with_strict_mode():
    current = get_policy()
    return policy_context(SelectionPolicy.from_dict(
        prefer_vendor=current.prefer_vendor,
        strict=True,
        per_op_order=dict(current.per_op_order) if current.per_op_order else None,
        deny_vendors=set(current.deny_vendors) if current.deny_vendors else None,
        allow_vendors=set(current.allow_vendors) if current.allow_vendors else None,
    ))


def with_vendor_preference(prefer: bool):
    current = get_policy()
    return policy_context(SelectionPolicy.from_dict(
        prefer_vendor=prefer,
        strict=current.strict,
        per_op_order=dict(current.per_op_order) if current.per_op_order else None,
        deny_vendors=set(current.deny_vendors) if current.deny_vendors else None,
        allow_vendors=set(current.allow_vendors) if current.allow_vendors else None,
    ))


def with_allowed_vendors(*vendors: str):
    current = get_policy()
    return policy_context(SelectionPolicy.from_dict(
        prefer_vendor=current.prefer_vendor,
        strict=current.strict,
        per_op_order=dict(current.per_op_order) if current.per_op_order else None,
        deny_vendors=set(current.deny_vendors) if current.deny_vendors else None,
        allow_vendors=set(vendors),
    ))


def with_denied_vendors(*vendors: str):
    current = get_policy()
    denied = set(current.deny_vendors) if current.deny_vendors else set()
    denied.update(vendors)
    return policy_context(SelectionPolicy.from_dict(
        prefer_vendor=current.prefer_vendor,
        strict=current.strict,
        per_op_order=dict(current.per_op_order) if current.per_op_order else None,
        deny_vendors=denied,
        allow_vendors=set(current.allow_vendors) if current.allow_vendors else None,
    ))


__all__ = [
    "SelectionPolicy",
    "PolicyManager",
    "get_policy",
    "set_global_policy",
    "reset_global_policy",
    "policy_context",
    "policy_from_env",
    "get_policy_epoch",
    "bump_policy_epoch",
    "with_strict_mode",
    "with_vendor_preference",
    "with_allowed_vendors",
    "with_denied_vendors",
]
