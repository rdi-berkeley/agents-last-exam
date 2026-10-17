# Music project migration interfaces

Repair `input/project/project.ardour` with installed open-source replacements
for unavailable active instruments and effects. Preserve the original editable
notes, muted material, tempo, meter, controllers, track organization and routing.
Disabled effects do not require replacement. Preserve the original timbre,
articulation and changing dynamics; constant overall gain differences are allowed.

The original plugin names, route mappings and state are in
`input/project/conversion-manifest.json` and `input/project/source-state/`.
These are original source data, not replacement assignments.

Required outputs:

- `output/available_plugins.txt`: installed plugin and sound-library inventory.
- `output/migrated_project/migrated_project.ardour`: self-contained editable
  project with all media and plugin state, which saves and reopens in Ardour.
- `output/mixdown.wav`: the complete mix with original group/master processing.
- `output/stems/<track_name>.wav`: one complete solo WAV per original track/channel.
- `output/overview.png`: Ardour with tracks and plugin assignments visible.

Audio must agree with native project playback and be finite, unclipped mono or
stereo WAV at 44.1 kHz or higher, starting at time zero and retaining complete
effect tails. Each stem taps its instrument route's audio output ports after
Inserts/Strip and before downstream group/master processing. Keep the original
routing and group/master effects in the project and full mix.

The provisioned Linux runtime includes Ardour 6.9, `/usr/lib/ardour6/luasession`,
`/usr/bin/ardour6-export`, Xvfb, FFmpeg and `/usr/local/bin/ale-migration-env`.
The environment wrapper supplies native library/export settings.
`ARDOUR_BATCH_HELPER` identifies the provisioned instrument-output export library.

`native_audio.render_native_stems(snapshot, delivery, evidence, full=True)`
exports selected route outputs from a separate project copy. `delivery` has a
`stems` list; each entry supplies `reference` (the original track/channel WAV
name), `valid: true`, `reference_audio: {"audible": true}`, and `passages: []`.
These labels select routes; no reference audio is required for this export.
The helper writes WAVs under `evidence` and returns their paths in its receipts.
`native_batch.py`, `prepare_batch.lua` and `native_batch_export.cc` provide its
Ardour export adapter. `build_batch_helper.py --headers <directory>
--library-dir <directory> --output <new-library-path>` builds the adapter against
local Ardour headers/libraries when needed; `--dev-prefix` is optional.
`rebind.py <snapshot.ardour>` updates media search paths after relocating a project.
`audio_delivery.py` supplies audio metadata, explicit passage extraction and
track-name normalization for the export helper.
