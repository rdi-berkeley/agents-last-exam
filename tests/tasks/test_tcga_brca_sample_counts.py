import pytest

from tasks.life_sciences.tcga_brca_deg_analysis.scripts.score_outputs import _parse_sample_counts


@pytest.mark.parametrize("put_gene_counts_first", [False, True])
def test_gene_counts_do_not_change_reported_sample_counts(put_gene_counts_first):
    samples = "Tumor samples matched: 1032\nMatched normal samples: 96"
    genes = "Significant DEGs: 2500 (up in tumor: 1420; down in tumor: 1080)\nUp in normal: 132"
    parts = [genes, samples] if put_gene_counts_first else [samples, genes]
    assert _parse_sample_counts("\n".join(parts)) == (1032, 96)


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        ("Primary Tumor samples used: 1,032\nSolid Tissue Normal samples used: 96", (1032, 96)),
        (
            "| Sample type | Count |\n| Solid Tissue Normal | 96 |\n| Primary Tumor | 1032 |",
            (1032, 96),
        ),
        ("Normal: 96; Tumor: 1032", (1032, 96)),
        ("Sample counts: tumor=1032, normal=96", (1032, 96)),
        ("Matched samples: 1,032 tumor, 96 normal", (1032, 96)),
        ("Tumor and normal samples: 1032 and 96, respectively.", (1032, 96)),
        ("Normal / tumor samples: 96 / 1,032.", (1032, 96)),
        ("Tumor samples matched 1032; Normal samples (matched): 96", (1032, 96)),
        ("We matched 1,032 primary tumor and 96 solid tissue normal samples.", (1032, 96)),
        ("Tumor (n=1032), normal (n=96)", (1032, 96)),
        ("Matched tumor samples: 1032\nTumor samples used: 1032", (1032, None)),
        ("Tumor samples: 1032\nTumor samples: 1040\nNormal samples: 96", (None, 96)),
        ("Matched tumor samples: 411\nUp in tumor: 1032\nUp in normal: 96", (411, None)),
        ("Tumor samples: 96; normal samples: 1032", (96, 1032)),
        ("Genes up in tumor: 1032; down in normal: 96", (None, None)),
        ("Genes expressed in both tumor and normal: 96", (None, None)),
        ("Tumor samples excluded: 1032; excluded normal samples: 96", (None, None)),
        (
            "We excluded 35 tumor samples and 12 normal samples; tumor samples matched: 1032",
            (1032, None),
        ),
        ("35 tumor samples and 12 normal samples were excluded.", (None, None)),
    ],
)
def test_counts_stay_with_their_cohort(summary, expected):
    assert _parse_sample_counts(summary) == expected
