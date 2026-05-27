"""Canonical variant metadata for colabfold_protein_structure_prediction."""

from __future__ import annotations

from dataclasses import dataclass

DOMAIN_NAME = "life_sciences"
TASK_NAME = "colabfold_protein_structure_prediction"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"


@dataclass(frozen=True)
class VariantSpec:
    variant_name: str
    variant_label: str
    raw_reference_gcs_path: str
    residue_count: int
    min_complete_residues: int


VARIANTS = (
    VariantSpec(
        variant_name="variant_01",
        variant_label="Protein 01",
        raw_reference_gcs_path=(
            "jtu22/2026-04-15_21-38-17_5130093c/variants/"
            "1_2026-04-15_22-15-38_ddae8089/_Reference_Output/"
            "AF-0000000365833162-model_v1.pdb"
        ),
        residue_count=460,
        min_complete_residues=452,
    ),
    VariantSpec(
        variant_name="variant_02",
        variant_label="Protein 02",
        raw_reference_gcs_path=(
            "jtu22/2026-04-15_21-38-17_5130093c/variants/"
            "2_2026-04-15_22-18-11_e505b904/_Reference_Output/"
            "AF-0000000365760600-model_v1.pdb"
        ),
        residue_count=506,
        min_complete_residues=498,
    ),
    VariantSpec(
        variant_name="variant_03",
        variant_label="Protein 03",
        raw_reference_gcs_path=(
            "jtu22/2026-04-15_21-38-17_5130093c/variants/"
            "3_2026-04-15_22-19-13_92eeb2ed/_Reference_Output/"
            "AF-0000000365761745-model_v1.pdb"
        ),
        residue_count=493,
        min_complete_residues=486,
    ),
    VariantSpec(
        variant_name="variant_04",
        variant_label="Protein 04",
        raw_reference_gcs_path=(
            "jtu22/2026-04-15_21-38-17_5130093c/variants/"
            "4_2026-04-15_22-20-29_6eb8e7b9/_Reference_Output/"
            "AF-0000000365762403-model_v1.pdb"
        ),
        residue_count=489,
        min_complete_residues=482,
    ),
    VariantSpec(
        variant_name="variant_05",
        variant_label="Protein 05",
        raw_reference_gcs_path=(
            "jtu22/2026-04-15_21-38-17_5130093c/variants/"
            "5_2026-04-15_22-21-18_1ad5326c/_Reference_Output/"
            "AF-0000000365827148-model_v1.pdb"
        ),
        residue_count=514,
        min_complete_residues=507,
    ),
    VariantSpec(
        variant_name="variant_06",
        variant_label="Protein 06",
        raw_reference_gcs_path=(
            "jtu22/2026-04-15_21-38-17_5130093c/variants/"
            "6_2026-04-15_22-22-01_d970e513/_Reference_Output/"
            "AF-0000000365783378-model_v1.pdb"
        ),
        residue_count=503,
        min_complete_residues=496,
    ),
    VariantSpec(
        variant_name="variant_07",
        variant_label="Protein 07",
        raw_reference_gcs_path=(
            "jtu22/2026-04-15_21-38-17_5130093c/variants/"
            "7_2026-04-15_22-22-49_bf492861/_Reference_Output/"
            "AF-0000000365794080-model_v1.pdb"
        ),
        residue_count=447,
        min_complete_residues=440,
    ),
    VariantSpec(
        variant_name="variant_08",
        variant_label="Protein 08",
        raw_reference_gcs_path=(
            "jtu22/2026-04-15_21-38-17_5130093c/variants/"
            "8_2026-04-15_22-23-16_40daea89/_Reference_Output/"
            "AF-0000000365780227-model_v1.pdb"
        ),
        residue_count=448,
        min_complete_residues=441,
    ),
)


def get_variant(name: str) -> VariantSpec:
    for variant in VARIANTS:
        if variant.variant_name == name:
            return variant
    raise KeyError(f"unknown variant {name!r}")
