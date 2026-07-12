"""netguard — provider-agnostic in-guest egress enforcement for a sandbox.

A per-task ``NetworkPolicy`` (from ``task_card.json`` ``vm.network``) is applied
to the guest OS, not the provider: :func:`apply_network_policy` dispatches on
``handle.os`` and drives the guest through the provider-agnostic
``SandboxHandle`` API. Linux uses nftables + the ``aleguard`` transparent proxy;
Windows uses Windows Firewall + the ``winguard`` CONNECT proxy. Both filter by
hostname (TLS SNI / HTTP Host / CONNECT target) without decrypting.
"""
from ._apply import apply_network_policy

__all__ = ["apply_network_policy"]
