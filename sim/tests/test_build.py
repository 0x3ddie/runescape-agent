"""sim/build.py's rules for writing an archive: nothing writes the recorded archive (DATA) under any spelling of its
path, and a content refresh keeps the extracted tables only while their pins equal the checkout's. A temporary file
stands in for DATA, and the checkout's pins and content are fixed stand-ins, so no test reads or writes logs/."""
import json
import os

import numpy as np
import pytest

from sim import build


def archive(path, commit):
    meta = dict(pins=dict(vendor_commit=commit, content='old'), content=dict(rows=1))
    np.savez(path, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), walks=np.arange(4))
    return path


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    path = archive(tmp_path / 'recorded.npz', 'a')
    monkeypatch.setattr(build, 'DATA', path)
    monkeypatch.setattr(build, 'pins', lambda: dict(vendor_commit='a', content='new'))
    monkeypatch.setattr(build, 'content', lambda meta: dict(rows=2))
    monkeypatch.setattr(build, 'WORK', tmp_path / 'work')  # a build that passed the guard would stop at once, in here

    def no_extraction(*args, **kwargs):
        raise AssertionError('the guard let a build start')
    monkeypatch.setattr(build.subprocess, 'run', no_extraction)
    return path


def refused(tmp_path, name):
    """A build and a content refresh (from another archive's tables) both refuse `name`, and no file changes or appears."""
    other = archive(tmp_path / 'other.npz', 'a')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    for write in (lambda: build.build(tmp_path / name), lambda: build.refresh_content(tmp_path / name, source=other)):
        with pytest.raises(ValueError, match='recorded runs'):
            write()
    return {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize('present', [True, False], ids=['present', 'absent'])
@pytest.mark.parametrize('spelling', ['recorded.npz', 'recorded', 'RECORDED.npz', 'Recorded'])
def test_no_spelling_of_the_recorded_archive_is_written(recorded, tmp_path, spelling, present):
    """np.savez adds a missing .npz, and a Mac volume takes names that differ only in letter case for one file, so the
    guard decides on the file that would be written. While the archive is absent only its name can decide."""
    if not present:
        recorded.unlink()
    assert refused(tmp_path, spelling)


def test_a_second_name_of_the_recorded_archive_is_not_written(recorded, tmp_path):
    """A hard link shares no name with the recorded archive, and a write to it would change that archive's bytes."""
    os.link(recorded, tmp_path / 'link.npz')
    assert refused(tmp_path, 'link.npz')


def test_a_content_refresh_keeps_the_extracted_tables_and_refuses_other_pins(recorded, tmp_path, monkeypatch):
    """--content-only writes the file np.savez names, with the source's arrays, the content read again and the
    checkout's pins; with no source it reads that same file. An archive extracted from other inputs is refused, with
    nothing written."""
    before = recorded.read_bytes()
    for source, rows in ((recorded, 2), (None, 3)):
        monkeypatch.setattr(build, 'content', lambda meta, rows=rows: dict(rows=rows))
        build.refresh_content(tmp_path / 'next', source=source)
        with np.load(tmp_path / 'next.npz') as data:
            meta, walks = json.loads(data['meta'].tobytes()), data['walks']
        assert meta == dict(pins=dict(vendor_commit='a', content='new'), content=dict(rows=rows))
        assert np.array_equal(walks, np.arange(4)) and recorded.read_bytes() == before
    assert not (tmp_path / 'next').exists()
    monkeypatch.setattr(build, 'pins', lambda: dict(vendor_commit='b', content='new'))
    written = (tmp_path / 'next.npz').read_bytes()
    for output, source in ((tmp_path / 'again.npz', recorded), (tmp_path / 'next.npz', None)):
        with pytest.raises(ValueError, match='other inputs'):
            build.refresh_content(output, source=source)
    assert not (tmp_path / 'again.npz').exists() and (tmp_path / 'next.npz').read_bytes() == written
