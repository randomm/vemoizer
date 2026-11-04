"""The post-call meeting decode never draws an mlx-whisper progress bar (issue #236)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from _whisper_helpers import _audio, _raw, _seg

from vemoizer.whisper_transcriber import decode_meeting


def test_post_call_meeting_decode_never_draws_a_progress_bar() -> None:
    """issue #236: the meeting pass has its own stage display (``✓ decode
    (124s)``), so EVERY mlx-whisper call it makes must run with verbose=None —
    the flag that disables mlx-whisper's tqdm bar. The self-heal re-decodes
    run after the progress shim has exited, so their bars are the ones that
    leaked into the terminal next to the stage lines. Nothing but ``verbose``
    may differ from the plain decode options (invariant 2: no decode option
    rides along with a display change)."""
    sr = 16_000
    wall_words = [
        [{"word": " Kiitos.", "start": 10.0 + i, "end": 10.5 + i}] for i in range(8)
    ]
    whole = _raw(
        [
            _seg(
                "alussa ihan oikeaa puhetta tässä on",
                [
                    {"word": " alussa", "start": 0.0, "end": 0.4},
                    {"word": " ihan", "start": 0.5, "end": 0.7},
                    {"word": " oikeaa", "start": 0.8, "end": 1.1},
                    {"word": " puhetta", "start": 1.2, "end": 1.5},
                    {"word": " tässä", "start": 1.6, "end": 1.8},
                    {"word": " on", "start": 1.9, "end": 2.0},
                ],
            )
        ]
        + [_seg("Kiitos.", w) for w in wall_words]
    )
    fixed = _raw(
        [
            _seg(
                "demossa näytettiin ihan oikeita lukuja kaikille",
                [
                    {"word": " demossa", "start": 0.5, "end": 1.0},
                    {"word": " näytettiin", "start": 1.1, "end": 1.6},
                    {"word": " oikeita", "start": 2.0, "end": 2.4},
                ],
            )
        ]
    )
    mock = MagicMock()
    mock.transcribe = MagicMock(side_effect=[whole, fixed])
    slices = [
        (0, np.zeros(8 * sr, dtype=np.float32)),
        (9 * sr, np.zeros(11 * sr, dtype=np.float32)),
    ]
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        result = decode_meeting(_audio(20.0), slices, initial_prompt="Sanasto: Pia.")

    assert result is not None
    # One window-loop call plus the self-heal re-decode of the wall's slice.
    assert mock.transcribe.call_count == 2
    for call in mock.transcribe.call_args_list:
        assert call.kwargs["verbose"] is None, (
            "the post-call pass drew a tqdm bar next to its own stage display"
        )
    main = mock.transcribe.call_args_list[0].kwargs
    assert main == {
        "path_or_hf_repo": "/tmp/turbo",
        "word_timestamps": True,
        "language": None,
        "task": "transcribe",
        "temperature": (0.0, 0.2, 0.4),
        "condition_on_previous_text": True,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "initial_prompt": "Sanasto: Pia.",
        "verbose": None,
    }
    heal = mock.transcribe.call_args_list[1].kwargs
    assert heal["condition_on_previous_text"] is False
    assert heal["initial_prompt"] == "Sanasto: Pia."
    assert set(heal) == set(main)  # the same option surface, no extras
