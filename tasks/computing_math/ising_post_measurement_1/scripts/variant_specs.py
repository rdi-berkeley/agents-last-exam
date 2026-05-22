"""Shared variant metadata for ising_post_measurement_1."""

from __future__ import annotations

from dataclasses import dataclass

DOMAIN_NAME = "computing_math"
TASK_NAME = "ising_post_measurement_1"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"


@dataclass(frozen=True)
class VariantSpec:
    variant_name: str
    n_qubits: int
    ancilla_mode: str
    has_ancilla_state: bool
    coupling_u: float
    requires_correlators: bool

    @property
    def expected_dimension(self) -> int:
        return 1 << self.n_qubits

    @property
    def required_outputs(self) -> tuple[str, ...]:
        base = ("critical_state.npy", "post_probs.npy", "rdm_site1.npy")
        if self.requires_correlators:
            return (*base, "correlators.npz")
        return base

    @property
    def variant_label(self) -> str:
        correlators = (
            "with one-body correlators" if self.requires_correlators else "without correlators"
        )
        ancilla = (
            "paramagnetic ancilla from staged ancilla_state.npy"
            if self.has_ancilla_state
            else "critical ancilla equal to the computed ground state"
        )
        return f"N={self.n_qubits}, u={self.coupling_u}, {ancilla}, {correlators}"


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        variant_name="n10_critical_u01_correlators",
        n_qubits=10,
        ancilla_mode="critical",
        has_ancilla_state=False,
        coupling_u=0.1,
        requires_correlators=True,
    ),
    VariantSpec(
        variant_name="n10_paramagnetic_u01_correlators",
        n_qubits=10,
        ancilla_mode="paramagnetic",
        has_ancilla_state=True,
        coupling_u=0.1,
        requires_correlators=True,
    ),
    VariantSpec(
        variant_name="n12_critical_u01_correlators",
        n_qubits=12,
        ancilla_mode="critical",
        has_ancilla_state=False,
        coupling_u=0.1,
        requires_correlators=True,
    ),
    VariantSpec(
        variant_name="n12_paramagnetic_u01_correlators",
        n_qubits=12,
        ancilla_mode="paramagnetic",
        has_ancilla_state=True,
        coupling_u=0.1,
        requires_correlators=True,
    ),
    VariantSpec(
        variant_name="n12_critical_u02_no_correlators",
        n_qubits=12,
        ancilla_mode="critical",
        has_ancilla_state=False,
        coupling_u=0.2,
        requires_correlators=False,
    ),
    VariantSpec(
        variant_name="n14_critical_u01_no_correlators",
        n_qubits=14,
        ancilla_mode="critical",
        has_ancilla_state=False,
        coupling_u=0.1,
        requires_correlators=False,
    ),
    VariantSpec(
        variant_name="n16_critical_u01_no_correlators",
        n_qubits=16,
        ancilla_mode="critical",
        has_ancilla_state=False,
        coupling_u=0.1,
        requires_correlators=False,
    ),
)

VARIANT_BY_NAME = {spec.variant_name: spec for spec in VARIANTS}


def get_variant(variant_name: str) -> VariantSpec:
    try:
        return VARIANT_BY_NAME[variant_name]
    except KeyError as exc:
        raise KeyError(f"unknown variant_name: {variant_name}") from exc
