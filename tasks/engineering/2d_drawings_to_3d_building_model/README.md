# Building visual judge

The native geometry gate and fourteen rendered views are unchanged. The eight
original questions remain equally weighted. Each question receives only the
following reference/candidate pairs, in that order:

| Criterion | Views | Images per request |
| --- | --- | --- |
| Hall/Tower proportions | axon_NE, axon_SW | 4 |
| Floor levels | section_NS, section_EW, elevation_east, elevation_west | 8 |
| Tower placement | plan_hall_ground, plan_tower_typical, axon_NE | 6 |
| Solid north Hall facade | elevation_north, axon_NW | 4 |
| Open south structural frame | elevation_south, axon_SE | 4 |
| Tower curtain-wall mullions | elevation_east, elevation_west, axon_NE | 6 |
| Workshop-module placement | plan_hall_ground | 2 |
| Four elevation silhouettes | elevation_north, elevation_south, elevation_east, elevation_west | 8 |

The mapping uses the original question text, so reordering questions preserves
their evidence. A changed, omitted or duplicated criterion is a configuration
error. Missing renders and incomplete/non-binary replies are evaluator errors;
they do not become candidate NO answers or scores of zero.

## Configuration

The scorer reads `secret/.env` from the repository and overlays the current
environment without changing global environment variables. Resolution order:

- Model: explicit `model`/`--model`, `D2T3B_JUDGE_MODEL`, `LLM_JUDGE_MODEL`,
  then `gpt-6-astra`.
- Endpoint: `D2T3B_JUDGE_BASE_URL`, `OPENAI_BASE_URL`, `OPENAI_API_BASE`,
  then `https://api.openai.com/v1`.
- Key: `D2T3B_JUDGE_API_KEY`, then `OPENAI_API_KEY`.

The API route is `/chat/completions`, with `max_completion_tokens=2048` and no
temperature override. Task-specific settings permit a paired endpoint/key
override without changing other tasks. Keys are excluded from reports.

## Evidence

Every individual reply is checkpointed in full, including the API response
object, requested model/endpoint, original question, selected views and parsed
answer. The task writes these checkpoints to the guest evaluation directory's
`judge-report.json` before issuing the next question. Completed answers survive
a later API failure. Invalid replies are saved before raising an evaluator error.
The score remains null until all eight answers are valid.

The standalone scorer's `--output-json` uses the same incremental checkpoints.
Normal logs report the aggregate score and guest report path rather than the
reply bodies. The two small live controls used for the October 6 validation
remain separately recorded in the retained guest; historical solver rewards
were not replaced.
