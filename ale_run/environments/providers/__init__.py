"""Provider implementations for VM lifecycle.

``Provider`` ABC + ``EnvSpec`` + ``VMHandle`` + ``ReleaseMode`` are in
:mod:`provider`. Backends:

  - :class:`GcloudProvider` (``gcloud.py``): ephemeral GCE VMs.
  - :class:`StaticProvider` (``static.py``): a pre-existing VM endpoint.
"""

from .gcloud import GcloudProvider, GcloudProviderConfig
from .provider import EnvSpec, Provider, ReleaseMode, VMHandle
from .static import StaticProvider, StaticProviderConfig

__all__ = [
    "EnvSpec",
    "GcloudProvider",
    "GcloudProviderConfig",
    "Provider",
    "ReleaseMode",
    "StaticProvider",
    "StaticProviderConfig",
    "VMHandle",
]
