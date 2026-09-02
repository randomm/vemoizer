"""Output stages of the consensus pipeline (issue #10).

Two halves:

- ``naming`` — NFC filename normalization so batch output paths
  collapse correctly on APFS.
- ``formatters`` — pure string builders for txt / json / srt / vtt
  over the ``TranscriptionResult`` contract (``vemoizer.transcriber.
  TranscriptionResult``). No I/O, no model imports, no network. The
  CLI wiring (task D) and file-write helpers live in sibling modules so
  the format selection and the batch loop stay decoupled from the
  text layout.
"""

from vemoizer.output.formatters import (
    FORMAT_EXTENSIONS,
    OUTPUT_FORMATS,
    format_json,
    format_srt,
    format_transcript,
    format_txt,
    format_vtt,
    srt_timestamp,
    vtt_timestamp,
)

__all__ = [
    "FORMAT_EXTENSIONS",
    "OUTPUT_FORMATS",
    "format_json",
    "format_srt",
    "format_txt",
    "format_transcript",
    "format_vtt",
    "srt_timestamp",
    "vtt_timestamp",
]
