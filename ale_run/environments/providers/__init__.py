"""Provider implementations for VM lifecycle.

``Provider`` ABC + ``EnvSpec`` + ``VMHandle`` + ``ReleaseMode`` live in
:mod:`ale_run.base_interface`; this package only holds the backends:

  - :class:`GcloudProvider` (``gcloud.py``): ephemeral GCE VMs.
  - :class:`StaticProvider` (``static.py``): a pre-existing VM endpoint.
"""

from ...base_interface import EnvSpec, Provider, ReleaseMode, VMHandle
from .gcloud import GcloudProvider, GcloudProviderConfig
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
