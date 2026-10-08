import io
from xml.etree import ElementTree
from zipfile import ZipFile

import pytest

from tasks.visual_media.music_transcription.native_export import validate_native_score


NATIVE = b'<museScore version="3.01"><Score><Staff id="1"><Measure><voice/></Measure></Staff></Score></museScore>'


def test_plain_and_compressed_native_scores_are_accepted():
    validate_native_score(NATIVE)
    stream = io.BytesIO()
    with ZipFile(stream, 'w') as archive:
        archive.writestr('META-INF/container.xml', '<container><rootfiles><rootfile full-path="transcription.mscx"/></rootfiles></container>')
        archive.writestr('transcription.mscx', NATIVE)
    validate_native_score(stream.getvalue())


@pytest.mark.parametrize('content', [b'<broken', b'<score-partwise/>', b'<museScore><Score/></museScore>'])
def test_non_scores_are_rejected(content):
    with pytest.raises((ValueError, ElementTree.ParseError)):
        validate_native_score(content)


def test_ambiguous_archive_is_rejected():
    stream = io.BytesIO()
    with ZipFile(stream, 'w') as archive:
        archive.writestr('META-INF/container.xml', '<container><rootfiles><rootfile full-path="first.mscx"/><rootfile full-path="second.mscx"/></rootfiles></container>')
        archive.writestr('first.mscx', NATIVE)
        archive.writestr('second.mscx', NATIVE)
    with pytest.raises(ValueError, match='one bounded'):
        validate_native_score(stream.getvalue())
