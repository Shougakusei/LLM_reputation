from __future__ import annotations

from src.core.jsonextract import extract_json_obj


def test_raw_object_parses():
    assert extract_json_obj('{"number": 4}') == {"number": 4}


def test_literal_newline_inside_string_is_tolerated():
    # Models write multi-line notes with raw line breaks instead of \n — strict JSON
    # rejects control characters in strings, our extraction must not.
    reply = '{\n"note": "line one.\nline two."}'
    assert extract_json_obj(reply) == {"note": "line one.\nline two."}


def test_non_json_returns_none():
    assert extract_json_obj("I will pick four.") is None
