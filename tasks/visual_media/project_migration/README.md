# Linux music project migration

## Current stem and audio-judge contract

The guarded joint audio protocol and instrument-output tap were accepted for
the October validation. The completed delivery and evaluator corrections were
reviewed by 2026-10-07: all 20 native stems and 60 excerpts passed, and the final
corrected score is **0.65**. The exact decoded-identity correction reused the
same 60 saved model responses without new judge calls or native rendering;
102 migration tests passed. The initial failed/null result and subsequent 0.64
evaluation remain preserved. See the [release inventory](../../../docs/releases/v1.1-tasks.md).

Raw audio judgments can vary; deterministic guards cover only proved identity
or equivalence. Exact zero was not a predeclared negative-control criterion, so
the retained 0.25 oboe/piano rating alone does not establish a rubric violation.
The completed review found no remaining confirmed scoring defect in the checked
delivery; it does not establish universal perceptual calibration.

Scored stems use the instrument channel output with its Inserts/Strip, before
downstream group/master processing. In Ardour this is the instrument route's
mono/stereo audio output ports, after its instrument and track inserts. Preserve
original music, controllers, routing and all group/master effects in the editable
project and complete full mix. Those effects must not be deleted to obtain stems.
Both delivery and evaluator renders start at zero and preserve tails at their
respective tap. Reference-selected excerpt windows and 20% delivery / 80% timbre
weighting are unchanged.

`native_audio.render_native_stems(..., full=True)` is the public full-stem helper;
run it on a separate copy of the project. Supply `delivery={"stems": [...]}` with
one record per instrument route: `{"valid": True, "reference": "Harp.wav",
"reference_audio": {"audible": True}, "passages": []}`. The reference field here
is a route-matching filename, not access to hidden reference audio. Keep the
complete session range including tails. The returned receipts identify the full
WAVs to copy into the requested output stems directory. Official evaluation uses
the same helper with fixed passages and `full=False`, cropping after continuous
native playback. Use stock `ardour6-export` for the complete master mix.

The provisioned `ARDOUR_BATCH_HELPER` is now required for both batched and isolated
instrument-output export. Rebuild `native_batch_export.cc`; the older binary
rejects one-output manifests. Missing/broken helpers are infrastructure errors,
never a fallback to the wrong master tap. Python preparation preserves every
route, insert, automation curve and downstream plugin state. Batching with
coupled inputs, sends, sidechains or VCA/groups falls back to one source at a time
using the same instrument tap. Downstream external plugin state needs no cloning.

Each unproved window uses one `gpt-audio-1.5` request containing Reference,
Candidate A (delivered) and Candidate B (native), with neutral candidate labels
and separate audio messages. One reference description and two ordinal ratings
are required. The window takes the lower score. Exact PCM and existing analytic
quantization proofs remain unchanged; proved identical/equivalent candidates use
the predetermined delivery rating, never the higher rating. Zero/DC handling,
constant-gain normalization, denominator and fixed-window rules remain intact.
Receipts bind all three hashes, protocol, model and rubric, retain raw HTTP bodies,
and record normalized payload hashes and exact serialized request hashes. Only
malformed JSON receives one identical-request retry; provider/unassessable failures
are unscored. Successful negative judgments are not retried.

The [official API schema](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
and [audio guide](https://developers.openai.com/api/docs/guides/audio-chat-completions)
support audio content parts; installed OpenAI Python 2.43.0 exposes their iterable
shape. This establishes request shape, not comparative-listening accuracy.
Bounded controls use `tests/tasks/migration_judge_controls.py` and
`tests/tasks/migration_native_preflight.py`. Their acceptance covers the guarded
protocol and native tap; an HTTP 200 alone does not validate the perceptual metric.

The r21/r22/r23 results below are historical evidence under the earlier
solo-through-master / separate-pair protocols, not acceptance of this contract.

The entry point now registers all five original variants on Linux/Ardour.
Preparation supplies original music and unresolved plugin placeholders. Choosing
replacement instruments, presets, effects and controller adapters remains the task.
No private replacement project, rendered answer or completed reference CPR is
included in the source bundles.

Historical r23 integration check: all20 complete private submission stems and the
full mix are retained separately from r22's cropped workload evidence. Successful
new native rendering totals1761.014s; the completeHarp and mix were reused. Only
the failed final seven were retried after a60s control confirmed restored native
solo propagation through their downstream buses. All1468 retained MIDI files and
the submitted snapshot are unchanged. The actual Ardour mixer overview is retained.
The real task hook's native worker exited0 in37min24.692s at CPU3/2GiB: two native
readbacks each contain41259 notes,0 missing/extra, and20/20 fresh evaluator stems
with60 passages are retained. Source/reopen checks preserve1195 muted notes,
46 regions and76 routes. Full delivery component is1.0; the overview gate passed.
The original `main.evaluate()` then failed at comparison80 with provider
`JSONDecodeError` and returned no score. Resuming the actual `score_delivery()`
phase from the preserved gates/passages completed all120 delivered/native
comparisons: **delivery1.0000, timbre0.533333, weighted0.626667**. It reused77
successful provider judgments and2 deterministic silent zeros, requiring41 new
responses and147.417s with no native rerender. This is recovered complete scoring,
not a second successful complete hook invocation. All music jobs are terminal.
An audit of the actual embedded program adapter confirms sourcePC0 is replaced
before synthesis; region labels do not identify the downstream replacement patch.
This is an approximate private candidate, not timbral acceptance or a
solver benchmark. The actual runtime adapter preflight and47 task tests pass;
subsequent inventory/batch fixes pass their4/14 targeted tests respectively.
Provider recovery and Linux task-hook controls pass23 focused tests. The subsequent
consistency correction passes42 tests, including the three retained native witnesses.

The initial frozen full scoring result exposed concrete reliability defects. Two Mallets
native windows contain only constant DC (peak5.06415e-10) but received0.75 after
normalization; their exactly silent PCM16 deliveries keep the final windows at0
under the existing lower-of-two rule. Trombones at603.45s received0.25 delivered
versus0.75 native although every sample differs by at most half a PCM16 step
(1.52587890625e-5). Both defects are now corrected without new judgments or renders:
silence/DC checks precede normalization and cached scores, and proved delivered/native
quantization equivalence reuses the delivered judgment. Reaggregation of all60
windows finds6 proved equivalent pairs. Mallets native scores become0; the cited
Trombones pair becomes0.25/0.25. The overall0.626667 is unchanged, since the earlier
lower-of-two rule already selected these values. Six unproved disagreements remain
independently scored; no quality threshold was tuned. The measured total still
cannot certify source parity or general fairness of the80% perceptual component.
Latest evidence: `music-r23/scoring-gain-invariance-terminal.json` and
`scoring-gain-invariance/scored.json`. The unnecessary amplitude-based silence
rule was removed; faint varying FLOAT/PCM24 signals preserve gain invariance.
Previous private terminal evidence is
`music-r23/scoring-resume-terminal.json`, `scoring-resume-v2/scored.json` and
`scorer-limit-witnesses.json`; the initial failure remains `full-hook.json`.

Latest bounded native validation (r22): all **20 distinct evaluator stems and 60
fixed passages** completed in **2474.785 seconds**, including storage pause and
recovery. The 41,259-note private approximation retains all MIDI bytes,
tempo/meter and measured downstream routing/effect state. This establishes the
complete passage workload on the retained Ardour 6.9 guest, not full submitted
WAV delivery, complete task-hook acceptance or source-parity sound. One frozen
alternate-library Harp challenge received 3/4 against the original and 3/4 against
the earlier library. The fixed scorer remains a candidate with the reliability
limits below. No solver benchmark was run.

Reference audio remains evaluator-only under the original staging contract. The
original task exposed the CPR input, not a listening-reference path. ALE stages
input/software before execution and reference data afterward; baked references
remain encrypted until evaluation. The Linux brief does not promise access to
`reference_stems_dir`, and setup does not populate it. The rubric and passage
selection rule are public; original reference stems are provisioned for scoring.
An accessible old audit VM or a publicly reachable archive does not change this
task visibility contract.

Public sound-design clues come only from the original input: the conversion
manifest maps plugin names/IDs and route assignments to hashed preset/state files;
`source-state/native-dawproject.xml`, `original-track-archive.xml.gz` and
`plugins/` preserve the original metadata and state. Binary preset data is not
claimed to be a decoded human-readable preset name. No hidden rendered stem or
completed-reference setting is used to fill in an unknown source value. This
matches `NECESSITY_AUDIT_20260930.md`'s preparation/solver boundary.

| Variant | Original editable notes |
| --- | ---: |
| celeste_symphonic_suite | 41,259 |
| eora | 1,461 |
| hollow_knight_symphonic_suite | 39,804 |
| twilight_princess_credits | 10,351 |
| undertale_medley | 8,002 |

`assets/sources.json` identifies each verified source snapshot, source archive,
bundle and member SHA256. `prepare_sources.py --private-root <retained-evidence>`
rebuilds only these source bundles. Setup rebinds media paths to the actual staged
copy. Original notes, muted material, channel overrides, tempo/meter, routes,
initializers and original plugin state survive. Hollow's original velocity-zero
record remains explicitly muted and editable with original velocity metadata.

The selected public release includes only `assets/celeste_symphonic_suite.tar.gz`.
All five variants remain registered in code and `assets/sources.json`. The Eora,
Hollow Knight, Twilight Princess and Undertale bundles are provisioned separately
from the private data distribution. To run one of those variants, place its
archive in this package's `assets/` directory under the filename and SHA256
recorded in the registry. Setup and evaluation both require that immutable source
archive. Tests require the selected Celeste bundle and check each nonselected
bundle when present; absent nonselected bundles skip only their integrity checks.

## Source parity and the private approximation

| Evidence | Meaning | What it does not establish |
| --- | --- | --- |
| Original reference stems | The target sound, retaining the original instrument/effect character | An open-source replacement has achieved that sound |
| Exact decoded PCM or analytically proven gain/quantization equivalence | Expected timbre score 1 for audio within the guard's validity | A generally similar GM patch is equivalent |
| Five source-only conversions and native note readbacks | Original editable music/controller/routing preservation | Audible source parity while original plugins remain missing |
| Private GM/FluidSynth mix and complete Harp | Actual playable approximation, with Harp ratings 0.75/0.75/0.75 on the fixed excerpts | A source-parity render, a perfect positive, or acceptance of the complete migration |

The 1269.9-second mix and Harp are **Ardour renders of the private GM approximation**,
not native Cubase renders of the original instruments. A functionally faithful
replacement should retain the reference's audible role and character; it need not
use the same proprietary library or produce the same waveform. The fixed rubric
already allows ordinary library variation at rating 4 and close functional
replacement with minor audible differences at rating 3. No complete functionally
faithful open-source replacement positive has been demonstrated. A note-preservation pass
cannot supply that missing evidence. The observed 0.75 ratings are neither a tuned
pass threshold nor a reason to relax the original 80% timbre objective.

## Evaluator contract

The original weighting remains **20% delivery / 80% timbre**. There is no approved
instrument list, automatic GM acceptance or legacy MFCC fallback. The old
`scripts/score_audio_remote.py` remains a diagnostic for prior evidence and is not
staged or invoked by the Linux task.

1. Check the self-contained Ardour project against the trusted source bundle.
   Compare notes, mute state, region placement, channel overrides, initial CCs,
   tempo/meter and original routing. Simple CC1/2/7/10/11 setters may omit repeated
   identical values. Other controller/program messages retain their order and
   values, including bank selection, sustain, RPN/NRPN and relative data entry.
   Report controller movement up to five ticks at 1920 PPQ, the observed cumulative
   native-save rounding. This is not a count-equality requirement. Original
   inactive EQ does not require replacement. Unsupported active-automation
   translations are unscored verifier limitations, not presumed musical failures.
2. On an evaluator copy, use Ardour 6's real Lua API to read all notes, save, reopen
   and read again. Check for unavailable active processors. Do not save diagnostics
   over the submitted or source project. Check the Ardour overview screenshot.
3. Stream every full WAV and the full mix. Require finite, unclipped mono/stereo
   audio at at least 44.1 kHz, covering the complete reference duration from zero.
   Extra tails are allowed. Missing, ambiguous, silent-for-active-reference or
   truncated stems remain zero entries in the full denominator. The originally
   requested plugin inventory must exist and contain non-whitespace text. This
   file-presence gate imposes no approved plugin or instrument list.
4. Select passages before inspecting candidate audio: three 12-second windows
   centered on the 25/50/75% quantiles of active 100 ms reference RMS bins. Activity
   means above 0.001 times that reference's loudest bin; this selects windows only,
   not timbre quality. Clamp to the reference bounds and merge duplicate windows.
5. Re-render the submitted native project at instrument output through actual native solo snapshots,
   always from zero so sustained notes and control history survive. Crop only after
   rendering. For each window, score both delivered and evaluator-rendered audio
   in one request against the same reference and take the lower score. Reuse the delivered
   judgment when decoded identity or the analytic PCM check proves equivalence
   to native playback; the first judgment is selected before knowing its rating.
   Thus copying reference
   WAVs cannot conceal wrong native instruments. Average windows per active stem,
   then stems equally. The overall score is `0.2 * delivery + 0.8 * timbre`.

The complete public prompt is `rubric.txt`. Exact non-silent decoded PCM identity
or analytically justified integer-PCM/common-positive-gain equivalence receives
timbre 1 before inference. The guard uses one scalar across time and channels;
it does not align, resample or accept time-varying/channel-specific gains. The
optional quantization-history witness is evaluator-owned, never submission-declared.
Float PCM and lossy MP3 have no automatic gain-equivalence claim. For delivered
versus evaluator-rendered native playback only, the guard additionally accepts
lossless float WAV against integer PCM when every sample lies in its corresponding
nearest-rounding cell at gain1. Rate, frames and channels must agree, and the
existing duration, precision and saturation bounds apply. This does not allow
float gain fitting, alignment, resampling or an arbitrary noise tolerance.
Quantization-cell consistency is a bounded numerical result, not validation of all
perceptual scores.

Before cached judgments, exact-identity positives or gain normalization, candidates
with zero samples or exact constant DC per channel receive0. There is no amplitude
floor: a faint nonconstant float/PCM24 signal may be amplified by the same scalar
across all samples/channels before transport, preserving the public gain-invariance
rule. Tests verify identical normalized PCM16 payloads for faint and louder copies.
An already encoded PCM16 delivery containing only zero samples remains zero.
Normalization refuses zero/DC inputs, so a DC reference produces infrastructure
failure instead of an invented semantic judgment.

Unproved pairs use official `gpt-audio-1.5`, the fixed public 0–4 rubric and strict
JSON validation. Normalization uses a single positive scalar, preserving dynamics
and channel balance. The model may downmix; it does not certify stereo fidelity.
Credentials stay on the evaluator host. HTTP failures, absent references/runtime,
unassessable judgments and unsupported verification raise infrastructure errors
instead of returning a candidate zero. Raw judgments and hashes are retained.
The format contract is prompt-requested JSON with strict local field validation,
not provider-enforced JSON schema. Invalid JSON gets one identical-request retry;
responses are retained before parsing. All41 responses in the completed recovery
were valid on the first attempt, finish reason `stop`, using147–242 completion
tokens under the unchanged1600 cap. The original failed response body was not
retained, so its specific formatting or truncation cause remains unknown.

## Runtime and validation boundary

Tested native runtime: Ubuntu, Ardour 6.9, `luasession`, `ardour6-export`, Xvfb,
FFmpeg and Python with NumPy, SoundFile and Mido. Host evaluator dependencies are
listed in `requirements-eval.txt`, in addition to the repository's judge dependencies.
The guest must provide executable `/usr/local/bin/ale-migration-env`, which
executes its arguments with the provisioned Python `PATH`, `LV2_PATH`,
`LD_PRELOAD` export-startup shim and `ARDOUR_BATCH_HELPER`. Setup commands,
including source rebinding, and the detached evaluator run through this wrapper;
its environment reaches native child processes. A missing or non-executable
wrapper is an explicit infrastructure error. Provision runtime files separately
from task inputs and private references; batch size still defaults to 2.
Provision the original reference WAV stems separately. Task setup stages no solved
instrument configuration. The evaluator retains a private source copy, so edits to
the public input do not redefine the baseline. Newer Ardour APIs are not verified.

Evaluator references also accept lossless FLAC with the same decoded PCM; ambiguous
duplicate encodings are infrastructure errors. Submitted audio remains full WAV.
Long native evaluation launches through an idempotent detached worker with short
status polls, avoiding the installed SDK's120-second WebSocket response limit.
The7200-second native worker cap remains. Evaluator copies keep repeated sample
library aliases compact within the new copy, isolated from submitted files.
Completed diagnostic render groups are hashed and cropped before the next group
allocates full WAVs; required complete submitted stems remain untouched.

All five full source inventories pass the new host checks. The retained private
Celeste replacement passes actual save/reopen with 41,259 notes both times,
1,195 muted editable notes, 46 regions and 76 routes. Its complete 1269.9-second
mix passes file checks. The three old 345-second stems are correctly rejected as
incomplete against all 20 full reference stems. These are actual retained-artifact
controls, not a solver run or accepted task delivery.

Remaining acceptance boundaries are explicit: the observed DC and quantization
consistency defects are fixed, but bounded controls do not certify general
cross-library musical fairness. Provision/verify source bundles, runtime and
hidden references on the target image before deployment. The other four source
conversions remain preserved; their complete
replacement audio is unverified and is outside this round. The bounded full-Harp
native render completed in
434.624 seconds: 1269.9 seconds, 48 kHz stereo float, finite/unclipped, peak 0.07405.
The fixed three reference-selected windows each received 3/4 from the official
audio endpoint. One valid stem over the unchanged 20-stem denominator gives an
audio-only diagnostic value of 0.04. That historical partial control was not a task
grade. The current complete r23 delivery supersedes its missing-file status, but
does not change the retained Harp judgments. The frozen wrong-instrument judgment remains zero;
it was not queried again. The private diagnostic uses an auditable FluidSynth dynamic-loading
patch and export-startup shim; these are not a declared generally validated image
or a solved public input. Native-rendering memory/time and plugin loading must be
checked for each eventual runtime.

Prior official-provider controls rejected the same-note wrong-instrument pair four
times, but under-scored identical and gain/encoding-equivalent positives at 0.75.
The deterministic guard covers its proven subset only. Broader perceptual reliability,
all articulations, and whole-project replacement quality remain unaccepted. Do not
describe this implemented scorer as a validated 80% perceptual metric.

## Historical batch rendering and acceptance evidence

The earlier optional multi-stem path used `scripts/native_batch.py`,
`prepare_batch.lua` and `native_batch_export.cc`. An evaluator-provisioned
`ARDOUR_BATCH_HELPER` enabled it; without that dependency, the old serial master
path remained active. That fallback and tap are superseded by the current
instrument-output contract above. The helper interposes the stock export
configuration call and queues multiple outputs through the real Ardour 6.9 API.
It uses upstream shared-pointer/XML headers and exported symbols, not object
layout offsets. The tested binary and header/build provenance are retained in
private r21 evidence; target-image compilation/provisioning remains pending.

`scripts/build_batch_helper.py` provides the reproducible local build entry. It
checks all six installed Ardour API symbols before compilation and records source,
upstream headers, installed libraries, compiler, command and output hashes in
`<output>.build.json`. It refuses to overwrite a previous binary or receipt.
Supply the retained official Ardour 6.9 header tree and installed development
headers, or an extracted development-package prefix; no download or installation
is performed by the script. For example, on the retained native guest:

```sh
python3 tools/build_batch_helper.py --headers ../music-r21/include \
  --dev-prefix ../music-r21/build/prefix --output batch_export.so
```

This build succeeded against the retained guest's actual libraries in r22. Set
`ARDOUR_BATCH_HELPER` to the absolute output path in the evaluator environment.
The receipt is ABI/build evidence, not a claim of compatibility with another
Ardour release. Deployment on a release image still requires its own native check.

Actual Celeste control: both60s streams exported in one30.106s pass, peak716MiB.
Independent solos took27.818s and26.234s. Cross-process PCM differed, but repeated
solos also differed (Harp2.24%, Oboe6.32% relative RMS), so raw differences did not
establish a batching error. In the decisive control, the **same instrument signal**
fed the original and copied downstream chains simultaneously. Harp direct/master
and Oboe group/master outputs each matched exactly across2,880,000 stereo frames,
with gain1 and offset0. This establishes the measured chain transformation;
it does not prove every library's cross-render determinism or all-song timbre.

The actual task-native rendering module then produced two finite, unclipped,
non-silent60s outputs through `native_batch` in43.207s including preparation.
Connected sidechains, sends/returns, split/fanout routing, cycles, VCA/group
coupling and downstream external plugin state use serial fallback. The evaluator's
`ARDOUR_BATCH_SIZE` controls group size (default 2, minimum 2); it is a resource
setting, not a task requirement. A remaining single stem uses serial export. Native
failures retain diagnostics and fall back to isolated rendering. Forty-two
focused tests pass, including these boundaries and the existing task-hook tests.
The initial private preparation's close139 is retained; releasing Lua route
references before closing succeeds in the integrated run. Solo preparation also
waits the native readback's six seconds for plugin workers before saving. The
alternate-library control saves its actual SF3 state and exits cleanly with this
fix. This is measured startup behavior, not a generic readiness guarantee for
every asynchronous plugin.

Cloning assigns new IDs to Ardour route, IO, processor, automation-list and
controllable objects. It preserves plugin parameter indices, including numeric
`Port id` fields even when they coincide with an object ID. Parameter values,
automation events and IO connections remain subject to the same checks. Distinct
names/IDs of copied, unconnected sidechain IOs do not themselves change sound.

Ardour 6.9 supports multiple output files in one transport pass. Its actual C++
API is `Session::get_export_handler()`, `add_timespan()`, `add_channel_config()`,
`PortExportChannel::add_port()`, `register_channel()`, `add_export_config()` and
one `do_export()`. Reuse the **same timespan object** for every stereo output;
`ExportHandler::start_timespan()` groups configurations by that object and adds
them to one export graph before starting playback. The stock command-line tool
registers only the master output, and the inspected Lua bindings do not expose
this export handler. A small C++ extension of the existing utility is the concrete
batch route, not an assumed Lua method or an existing CLI multistem flag.
See the upstream [export utility](https://github.com/Ardour/ardour/blob/6.9/session_utils/export.cc)
and [handler](https://github.com/Ardour/ardour/blob/6.9/libs/ardour/export_handler.cc).

The earlier implementation used isolated solo-through-master exports. Direct instrument
output taps bypass downstream effects. The retained private Celeste master has
two active compressors and an EQ; woodwind/brass buses also have compression and
reverb. A single full-mix master cannot provide the isolated response for all
stems. On an evaluator copy, construct independent downstream bus/master branches
from the submitted states, preserving automation, gains, panning, latency and
tails, then queue their stereo outputs in one timespan starting at zero. The
first native control compares two such branches (one direct-to-master,
one through a group bus) with isolated exports and a same-signal control.
If sidechains, feedback, shared state or a processor's behavior cannot be shown
equivalent, retain isolated export for that topology. The two measured branches
pass; generic topology coverage remains limited to the explicit fallback rules.

The observed full Harp export cost gives `20 * 434.624 = 8692.48` seconds
(144.9 minutes) as a rough serial full-length estimate, already beyond the
framework's 7200-second evaluation bound. Other stems and the current shorter
evaluation render ranges cost differently; the complete evaluator-passage
workload now passes as recorded below.
A one-pass helper still incurs all synthesis, duplicated effects and output I/O;
do not claim a 20-fold speedup. Twenty full stereo float WAVs at the measured
length require about 9.75 GB. Keep bulk audio on guest persistent storage and
transfer only the fixed passages. Evaluator rendering may end after the last
fixed window, but must process continuously from zero. Full submitted WAVs
remain required. Window-only sink capture is a further optimization, not an
already implemented Ardour API behavior.

The task-hook subset includes **27 passing tests** using the real installed
`cua_bench.computers.remote.RemoteDesktopSession` for task setup and the
evaluation hook. That runtime adapter accepts `run_command(command, check=False)`
and returns a dictionary; its lower interface receives only `command`. Tests
exercise the adapter with local/fake transport, a real staged Eora source bundle,
the deterministic audio guard, nonzero worker exit and missing evidence/provider
failures. They do not start a VM or claim a complete native task-hook run.

## Complete passage workload and scoring recommendation

r22 exercises actual `render_native_stems()` in groups of 2, 2, 4, 6 and 7, with
20 unique accepted stems and 60 complete reference-selected windows. The six-
output unit hit its 600s wall limit at 98.8%, including a 152.422s storage freeze.
That process remains a failed infrastructure unit. Five stems already contained
all selected windows and were recovered without replay. Its incomplete
Percussion High stem joined the six remaining stems in the final seven-output
unit, which exited0 in 491.736s. No complete stem was counted or rendered twice.

Every unit used one CPU and a 2GiB/no-swap cgroup. The last unit's anonymous peak
was 721,698,816 bytes; its total cgroup reached 2GiB with file-cache reclaim and
zero OOM kills. All native streams were finite, unclipped and audible. All 46
active MIDI source files were byte-identical, preserving 41,259 notes and their
controllers. Playlists, tempo/meter, cloned downstream chains, effect parameters,
automation and pan pass conservation checks after mapping only required object
IDs/IO namespaces. This extends the prior same-input exact-PCM branch controls.

The elapsed 2474.785s includes recovery and fits the remote worker's 7200s bound.
Only 276,485,280 bytes of fixed passages remain in guest storage. Stream metadata
and passage hashes are preserved. Full-render SHA capture was added before
deletion for the last 12 accepted stems; the first eight were already removed
and their missing whole-file hashes are an explicit evidence gap. New diagnostic
units support a 900s finite cap; the failed live extension of the earlier unit
was not treated as effective. Host reserve was 8GiB; sync/fstrim reclaimed only
deleted derivative blocks, with original projects and unique evidence retained.

The finite fairness challenge changes only installed FluidR3_GM Harp to installed
MuseScore_General_Lite Harp, preserving original MIDI, bank/program adapter and
downstream processing. Native save/export succeeds in 108.236s at 910,741,504-byte
peak. The frozen 12s window receives 3/4 both against the hidden original and
against the earlier FluidR3 version, with exactly two new blind calls and no
score-based selection or rubric change. The deliberate same-note Oboe/Piano
negative remains zero from prior raw judgments; proven numerical positives
remain one through the deterministic guard.

This is one accepted library-variation candidate, not independent ground truth
that its proper rating is 3/4. The identical alternate candidate is described as
sustained in one context and plucked in the other. Together with prior raw
self-comparison hallucinations, that limits explanation/calibration confidence.
Do not turn 0.75 into a passing threshold or describe the full 80% perceptual
metric as validated. Library names, audible output and correct notes alone do
not certify source-parity timbre.

Historical review recommendation, before the common-tap correction: retain all five source-only Linux inputs and the
full original deliverables; keep hidden original audio evaluator-only. Preserve
20% delivery / 80% timbre, public passage/rubric rules, analytic equality for its
proved subset, and frozen semantic judgments for other audio. Runtime/provider
failures remain infrastructure. Deploy the actual helper on the target image
and choose batch size from measured resource headroom, using the existing serial
fallback for unsupported topology. Celeste's complete submitted stems/overview,
native readbacks and recovered full provider workload now have terminal evidence;
the initial hook infrastructure failure remains recorded separately. Exact-zero/DC
rejection and proved delivered/native judgment reuse are now
implemented and tested against retained clips. Reaggregation changes three
component judgments and preserves the weighted0.626667 result with no provider
call or native rendering. Broader independently anchored musical reliability
remains limited to the retained controls; no repeated source
conversion, patch whitelist or threshold tuning is required for this review.

Run the targeted checks with:

```sh
uv run --with soundfile --with librosa pytest -q tests/tasks/test_project_migration_linux.py tests/tasks/test_project_migration_audio.py tests/tasks/test_project_migration_batch.py tests/tasks/test_project_migration_batch_build.py
uv run --with soundfile --with mido pytest -q tests/tasks/test_project_migration_scoring_consistency.py tests/tasks/test_project_migration_judge_resume.py tests/tasks/test_project_migration_linux.py
uv run ruff check tasks/visual_media/project_migration tests/tasks/test_project_migration_linux.py
uv run ruff format --check tasks/visual_media/project_migration tests/tasks/test_project_migration_linux.py
```

Librosa is needed only by the retained legacy diagnostic test, not the live evaluator.
Set evaluator-only `MIGRATION_FROZEN_SCORE` to the retained r23 scored JSON to run
the three actual-clip regressions; otherwise those private witness tests skip.
