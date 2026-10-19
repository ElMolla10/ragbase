"""The development client rejects unsupported/oversized files before upload."""

from pathlib import Path

import pytest

from scripts.try_enterprise import MAX_BYTES, read_upload


def test_upload_reader_preserves_bytes_and_accepts_uppercase_suffix(tmp_path):
    path = tmp_path / "sample.CSV"
    path.write_bytes(b"id,value\n001,120\n")
    assert read_upload(path) == path.read_bytes()


@pytest.mark.parametrize("size", [0, MAX_BYTES + 1])
def test_upload_reader_checks_nonempty_size_before_network(tmp_path, size):
    path = tmp_path / "file.pdf"
    path.write_bytes(b"x" * size)
    with pytest.raises(ValueError, match="5 MB"):
        read_upload(path)


def test_upload_reader_rejects_office_and_macro_inputs_before_opening():
    with pytest.raises(ValueError, match="Choose a PDF"):
        read_upload(Path("not-opened.xlsm"))
