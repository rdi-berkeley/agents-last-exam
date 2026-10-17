# Linux music project migration

Repair the source-only Ardour project by choosing open-source replacement
instruments and effects. Preserve original editable notes, controllers, tempo,
meter, routing and plugin-state clues. The selected release task is
`celeste_symphonic_suite`; all five original variants remain registered.

The public delivery and software interfaces are documented in
[public/README.md](public/README.md). Setup stages that document as
`software/rubric.txt`, alongside the conversion and native export adapters.
The private evaluator and its dependencies are staged only during evaluation.
The task-local `rubric.txt` records the evaluator specification and is not
part of the solver software bundle.

Use the provisioned Ubuntu/Ardour 6.9 environment with `luasession`,
`ardour6-export`, Xvfb, FFmpeg, `ARDOUR_BATCH_HELPER` and
`/usr/local/bin/ale-migration-env`. The wrapper sets the native export/library
environment. Guest evaluator dependencies are in `requirements-eval.txt`.
Native replay and deterministic audio scoring run inside the VM. Audio and
receipts stay there; the host reads assessment JSON and the overview screenshot.
The native evaluator retains its 7200-second resource cap.

The source archive is checked against `assets/sources.json`. Setup stages
unsolved inputs only. Original references are provisioned separately for
assessment. Only the selected Celeste source bundle ships in this release;
other variants require their separately provisioned archives at the filenames
and hashes recorded in `assets/sources.json`.
