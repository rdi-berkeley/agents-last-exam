# Evaluator data

Set `WGS_VARIANT_CALLING_EVAL_DATA_DIR` to an evaluator-local directory containing:

- `chr17.fa`
- `truth_chr17.vcf.gz`
- `truth_chr17.vcf.gz.tbi`
- `eval_region_confident.bed`

This directory must not be staged on the agent VM or stored in a public repository. The evaluator
downloads only the submitted VCF and compares it against these local files.
