import xml.etree.ElementTree as ET

import mido
import pytest

from tasks.visual_media.project_migration.scripts.project_check import compare_project


def project(directory, *, cached, setter_tick=0, setter_channel=0, setter_value=80,
            note_tick=10, cropped=False, muted=False, controller=7):
    media = directory / "interchange/test/midifiles"
    media.mkdir(parents=True)
    midi = mido.MidiFile(ticks_per_beat=1920)
    events = [
        (setter_tick, mido.Message("control_change", channel=setter_channel,
                                   control=controller, value=setter_value)),
        (note_tick, mido.Message("note_on", channel=0, note=60, velocity=90)),
        (note_tick + 100, mido.Message("note_off", channel=0, note=60, velocity=0)),
    ]
    track = mido.MidiTrack()
    previous = 0
    for tick, message in sorted(events, key=lambda event: event[0]):
        track.append(message.copy(time=tick - previous))
        previous = tick
    midi.tracks.append(track)
    midi.save(media / "notes.mid")
    tree = ET.fromstring(f'''<Session>
      <Sources><Source id="1" name="notes.mid"/></Sources>
      <Routes><Route name="Piano" midi-playlist="2">
        <Controllable name="midicc-0-{controller}" value="{cached}"/>
      </Route></Routes>
      <Playlists><Playlist id="2"><Region name="notes" source-0="1"
        muted="{int(muted)}" position="0" length="48000" start="0"
        beat="0" start-beats="{0.002 if cropped else 0}" length-beats="1"/>
      </Playlist></Playlists>
    </Session>''')
    snapshot = directory / "project.ardour"
    ET.ElementTree(tree).write(snapshot)
    return snapshot


def test_playback_cache_does_not_override_explicit_midi_initializer(tmp_path):
    original = project(tmp_path / "original", cached=80)
    saved = project(tmp_path / "saved", cached=119)
    assert compare_project(original, saved)["passed"]


@pytest.mark.parametrize("changes", [
    {"setter_tick": 20},
    {"setter_channel": 1},
    {"cropped": True},
    {"muted": True},
    {"controller": 64},
])
def test_uninitialized_or_stateful_controller_still_rejects_changed_state(tmp_path, changes):
    original = project(tmp_path / "original", cached=80, **changes)
    saved = project(tmp_path / "saved", cached=119, **changes)
    result = compare_project(original, saved)
    assert not result["passed"]
    assert any("initial state" in failure for failure in result["failures"])


def test_changed_effective_volume_is_rejected(tmp_path):
    original = project(tmp_path / "original", cached=80)
    saved = project(tmp_path / "saved", cached=119, setter_value=119)
    assert not compare_project(original, saved)["passed"]
