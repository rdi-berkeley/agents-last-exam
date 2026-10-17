import asyncio
import io
import json
import shlex
import uuid
from fractions import Fraction
from pathlib import PurePosixPath
from xml.etree import ElementTree
from zipfile import ZipFile, is_zipfile

import mido


def validate_native_score(content):
    if len(content) > 32 * 1024 * 1024:
        raise ValueError("Native score exceeds 32 MiB")
    if is_zipfile(io.BytesIO(content)):
        with ZipFile(io.BytesIO(content)) as archive:
            container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
            entries = [node.get("full-path") for node in container.iter() if node.tag.rsplit("}", 1)[-1] == "rootfile"]
            entries = [path for path in entries if path and path.endswith(".mscx")]
            if len(entries) != 1 or archive.getinfo(entries[0]).file_size > 32 * 1024 * 1024:
                raise ValueError("Expected one bounded native score document")
            content = archive.read(entries[0])
    document = ElementTree.fromstring(content)
    if document.tag != "museScore" or document.find("Score/Staff/Measure") is None:
        raise ValueError("Expected a MuseScore score with measures")


async def export_native_score(session, score_path, output_dir):
    validate_native_score(await session.read_bytes(score_path))
    location = str(PurePosixPath(output_dir).parent / f".music-native-{uuid.uuid4().hex}")
    await session.interface.create_dir(location)
    notation_path = f"{location}/expanded.musicxml"
    pdf_path = f"{location}/original.pdf"
    midi_path = f"{location}/playback.mid"
    status_path = f"{location}/status.json"
    log_path = f"{location}/export.log"
    commands = [
        ["musescore4", "--unroll-repeats", "-o", notation_path, score_path],
        ["musescore4", "-o", pdf_path, score_path],
        ["musescore4", "-o", midi_path, score_path],
    ]
    script = (
        "import json, os, subprocess\n"
        "environment = dict(os.environ, QT_QPA_PLATFORM='offscreen')\n"
        "status = {'returncode': 0}\n"
        f"with open({log_path!r}, 'wb') as output:\n"
        "    try:\n"
        f"        for command in {commands!r}:\n"
        "            result = subprocess.run(command, env=environment, stdin=subprocess.DEVNULL, "
        "stdout=output, stderr=output, timeout=150)\n"
        "            status['returncode'] = result.returncode\n"
        "            if result.returncode:\n"
        "                break\n"
        "    except Exception as error:\n"
        "        status = {'error': str(error)}\n"
        f"with open({status_path + '.partial'!r}, 'w') as output:\n"
        "    json.dump(status, output)\n"
        f"os.replace({status_path + '.partial'!r}, {status_path!r})\n"
    )
    launcher = (
        "import subprocess; "
        f"process = subprocess.Popen(['python3', '-c', {script!r}], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, "
        "start_new_session=True, close_fds=True); print(process.pid)"
    )
    result = await session.run_command("python3 -c " + shlex.quote(launcher))
    if not result.get("stdout", "").strip().isdigit():
        raise RuntimeError(f"Could not launch native score exporter: {result}")
    deadline = asyncio.get_running_loop().time() + 480
    while asyncio.get_running_loop().time() < deadline:
        if await session.file_exists(status_path):
            status = json.loads(await session.read_bytes(status_path))
            if status.get("returncode") != 0:
                raise RuntimeError(f"Native score export failed: {status}; inspect {log_path}")
            missing = [path for path in (notation_path, pdf_path, midi_path) if not await session.file_exists(path)]
            if missing:
                raise RuntimeError(f"Native score exporter omitted files: {missing}")
            try:
                playback = mido.MidiFile(file=io.BytesIO(await session.read_bytes(midi_path)))
                ticks = 0
                tempos = {Fraction(0): 120.0}
                for message in mido.merge_tracks(playback.tracks):
                    ticks += message.time
                    if message.type == "set_tempo":
                        tempos[Fraction(ticks, playback.ticks_per_beat)] = mido.tempo2bpm(message.tempo)
            except (ValueError, OSError, EOFError, ZeroDivisionError) as error:
                raise RuntimeError("Native score exporter produced invalid playback") from error
            return await session.read_bytes(notation_path), await session.read_bytes(pdf_path), sorted(tempos.items())
        await asyncio.sleep(2)
    raise RuntimeError(f"Native score exporter did not report completion: {location}")
