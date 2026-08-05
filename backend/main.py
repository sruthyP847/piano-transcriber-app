import json
import math
import uuid
from pathlib import Path

import abjad
import cv2
import librosa
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from moviepy import VideoFileClip

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/api/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

ALLOWED_CONTENT_TYPES = {"video/mp4", "video/quicktime", "video/x-m4v"}
BASE_URL = "http://localhost:8000"


def generate_placeholder_pdf(pdf_path: Path, title: str) -> None:
    # Hand-rolled minimal single-page PDF (no external dependency) with a
    # correctly computed xref table so browsers can render it directly.
    safe_title = title.replace("(", r"\(").replace(")", r"\)")
    stream_content = f"BT /F1 20 Tf 72 700 Td ({safe_title}) Tj ET".encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(stream_content)
        + stream_content
        + b"\nendstream",
    ]

    buffer = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(buffer))
        buffer += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"

    xref_offset = len(buffer)
    buffer += f"xref\n0 {len(objects) + 1}\n".encode()
    buffer += b"0000000000 65535 f \n"
    for offset in offsets:
        buffer += f"{offset:010d} 00000 n \n".encode()
    buffer += (
        b"trailer\n"
        + f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
        + b"startxref\n"
        + f"{xref_offset}\n".encode()
        + b"%%EOF"
    )

    pdf_path.write_bytes(bytes(buffer))


def generate_placeholder_musicxml(musicxml_path: Path) -> None:
    # Minimal valid single-measure MusicXML skeleton, standing in until the
    # real transcription pipeline produces actual notation.
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name>Piano</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note>
        <rest/>
        <duration>4</duration>
      </note>
    </measure>
  </part>
</score-partwise>
"""
    musicxml_path.write_text(xml, encoding="utf-8")


# --- Grand-staff notation rendering (validated against hardcoded data in
# Stages A/B before being wired to real pipeline output here) ---

NOTATION_BEATS_PER_BAR = 4
NOTATION_SIXTEENTHS_PER_BEAT = 4
NOTATION_BAR_SIZE = NOTATION_BEATS_PER_BAR * NOTATION_SIXTEENTHS_PER_BEAT  # 16

# (value_in_sixteenths, lilypond_code), largest to smallest. Only values that
# stay integer on a sixteenth-note grid are included — e.g. double-dotted
# eighth (3.5) is not representable here and is excluded.
NOTATION_STANDARD_VALUES = [
    (16, "1"),    # whole
    (14, "2.."),  # double-dotted half
    (12, "2."),   # dotted half
    (8, "2"),     # half
    (7, "4.."),   # double-dotted quarter
    (6, "4."),    # dotted quarter
    (4, "4"),     # quarter
    (3, "8."),    # dotted eighth
    (2, "8"),     # eighth
    (1, "16"),    # sixteenth
]
NOTATION_VALUE_TO_CODE = {value: code for value, code in NOTATION_STANDARD_VALUES}


def _largest_fitting_value(n: int) -> int:
    for value, _ in NOTATION_STANDARD_VALUES:
        if value <= n:
            return value
    raise ValueError(f"no standard duration value fits {n} sixteenths")


def _tie_last(pieces: list[tuple[int, bool]]) -> list[tuple[int, bool]]:
    if not pieces:
        return pieces
    last_value, _ = pieces[-1]
    return pieces[:-1] + [(last_value, True)]


def spell_rhythm(
    start: int,
    duration: int,
    beats_per_bar: float = NOTATION_BEATS_PER_BAR,
    sixteenths_per_beat: int = NOTATION_SIXTEENTHS_PER_BEAT,
) -> list[tuple[int, bool]]:
    """Beat-respecting rhythm decomposition, in sixteenth-note integer units.

    Returns [(value_in_sixteenths, is_tied_to_next), ...]. Applies identically
    to notes and rests (rests just ignore the tie flag when rendered).
    """
    if duration <= 0:
        return []

    # beats_per_bar can be non-integer for compound meters (e.g. 6/8 -> 3.0
    # quarter-note-equivalent beats), but the product with sixteenths_per_beat
    # is always a whole number of sixteenths for the meters this app supports.
    bar_size = round(beats_per_bar * sixteenths_per_beat)
    end = start + duration

    # Rule 1 (strongest): never let a single value cross a bar line.
    this_bar_end = (start // bar_size + 1) * bar_size
    if end > this_bar_end:
        first_len = this_bar_end - start
        first = spell_rhythm(start, first_len, beats_per_bar, sixteenths_per_beat)
        rest = spell_rhythm(this_bar_end, end - this_bar_end, beats_per_bar, sixteenths_per_beat)
        return _tie_last(first) + rest

    # Confined to one bar now.
    starts_on_beat = start % sixteenths_per_beat == 0

    # Rule 2 (middle): a value starting on a beat boundary can always be
    # notated as a single standard value (this legally covers dotted/
    # double-dotted values too — e.g. a dotted quarter starting on a beat is
    # standard notation even though it extends past the next beat boundary).
    # Note: this is intentionally wider than "duration is a whole multiple of
    # a beat" — that stricter gate would incorrectly force a beat-aligned
    # double-dotted quarter (7 sixteenths, not a whole-beat multiple) to
    # split into tied pieces instead of rendering as one notehead.
    if starts_on_beat:
        value = _largest_fitting_value(duration)
        if value == duration:
            return [(value, False)]
        remainder = duration - value
        rest = spell_rhythm(start + value, remainder, beats_per_bar, sixteenths_per_beat)
        return [(value, True)] + rest

    # Rule 2 else-branch: starts mid-beat. If it doesn't cross the next beat
    # boundary, handle it as a single within-beat greedy pick (rule 3).
    this_beat_end = (start // sixteenths_per_beat + 1) * sixteenths_per_beat
    if end <= this_beat_end:
        value = _largest_fitting_value(duration)
        if value == duration:
            return [(value, False)]
        remainder = duration - value
        rest = spell_rhythm(start + value, remainder, beats_per_bar, sixteenths_per_beat)
        return [(value, True)] + rest

    # Crosses the next beat boundary -> split there.
    first_len = this_beat_end - start
    first = spell_rhythm(start, first_len, beats_per_bar, sixteenths_per_beat)
    rest = spell_rhythm(this_beat_end, end - this_beat_end, beats_per_bar, sixteenths_per_beat)
    return _tie_last(first) + rest


NOTE_LETTER_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHARP_SYMBOLS = {"♯", "#"}
_FLAT_SYMBOLS = {"♭", "b"}


def _parse_note_name(note_name: str) -> tuple[str, int, int]:
    """Returns (letter, accidental_semitones, octave). Handles both the
    unicode sharp/flat symbols librosa.midi_to_note() actually produces
    (♯/♭) and plain ASCII (#/b) for robustness."""
    letter = note_name[0].upper()
    rest = note_name[1:]
    accidental = 0
    if rest and rest[0] in _SHARP_SYMBOLS:
        accidental = 1
        rest = rest[1:]
    elif rest and rest[0] in _FLAT_SYMBOLS:
        accidental = -1
        rest = rest[1:]
    octave = int(rest)
    return letter, accidental, octave


def note_name_to_midi(note_name: str) -> int:
    letter, accidental, octave = _parse_note_name(note_name)
    return (octave + 1) * 12 + NOTE_LETTER_SEMITONES[letter] + accidental


def note_name_to_lilypond_pitch(note_name: str) -> str:
    letter, accidental, octave = _parse_note_name(note_name)
    pitch = letter.lower()
    # abjad.Staff() parses with language="english" by default, where sharp/flat
    # are the plain suffixes "s"/"f" (e.g. "cs"=C-sharp) -- NOT the Dutch-style
    # "is"/"es" used in Stages A/B, which never actually exercised an
    # accidental and so never caught this. Verified against abjad's own
    # NamedPitch.number for all seven letters before fixing.
    if accidental == 1:
        pitch += "s"
    elif accidental == -1:
        pitch += "f"
    if octave >= 4:
        pitch += "'" * (octave - 3)
    elif octave < 3:
        pitch += "," * (3 - octave)
    return pitch


# --- Key signature model and key-aware enharmonic spelling ---
#
# Two problems that HAD to be fixed together: no key signature was attached to
# the score (so LilyPond defaulted to C major and printed every accidental
# inline), and librosa.midi_to_note() always returns SHARP spellings. Attaching
# a flat key signature while still emitting sharp note names would have been
# worse than doing nothing -- the score would show inline sharps contradicting
# a flat signature.
#
# Accidental count per key, positive = sharps, negative = flats. All 30 keys
# (15 major + 15 minor) enumerated explicitly rather than derived, so the table
# can be checked against the circle of fifths by eye. Minor keys carry their
# relative major's signature (A minor = C major = 0, and so on up/down the
# circle) -- note A# minor (7 sharps) exists and is easy to omit by accident.
KEY_SIGNATURE_ACCIDENTALS: dict[tuple[str, str], int] = {
    ("C", "major"): 0, ("G", "major"): 1, ("D", "major"): 2, ("A", "major"): 3,
    ("E", "major"): 4, ("B", "major"): 5, ("F#", "major"): 6, ("C#", "major"): 7,
    ("F", "major"): -1, ("Bb", "major"): -2, ("Eb", "major"): -3, ("Ab", "major"): -4,
    ("Db", "major"): -5, ("Gb", "major"): -6, ("Cb", "major"): -7,
    ("A", "minor"): 0, ("E", "minor"): 1, ("B", "minor"): 2, ("F#", "minor"): 3,
    ("C#", "minor"): 4, ("G#", "minor"): 5, ("D#", "minor"): 6, ("A#", "minor"): 7,
    ("D", "minor"): -1, ("G", "minor"): -2, ("C", "minor"): -3, ("F", "minor"): -4,
    ("Bb", "minor"): -5, ("Eb", "minor"): -6, ("Ab", "minor"): -7,
}

# Order in which accidentals are added around the circle of fifths.
SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]

DEFAULT_KEY_SIGNATURE = ("C", "major")


def parse_key_signature(text: str | None) -> tuple[str, str]:
    """Parse "d major" / "Eb minor" / "F# Major" into a canonical (tonic, mode).

    Falls back to C major for None/blank so callers that never send the field
    behave exactly as before. Raises ValueError on a genuinely bad value so the
    endpoint can turn it into a 400 rather than silently transposing the score.
    """
    if text is None or not text.strip():
        return DEFAULT_KEY_SIGNATURE

    parts = text.strip().split()
    if len(parts) != 2:
        raise ValueError(f"key_signature {text!r} must look like '<tonic> <major|minor>'")
    raw_tonic, raw_mode = parts

    mode = raw_mode.lower()
    if mode not in ("major", "minor"):
        raise ValueError(f"key_signature mode {raw_mode!r} must be 'major' or 'minor'")

    # Normalise "eb"/"EB"/"E♭" -> "Eb", "f#"/"F♯" -> "F#".
    tonic = raw_tonic[0].upper()
    for suffix in raw_tonic[1:]:
        if suffix in _SHARP_SYMBOLS:
            tonic += "#"
        elif suffix in _FLAT_SYMBOLS:
            tonic += "b"
        else:
            raise ValueError(f"key_signature tonic {raw_tonic!r} is not a valid note name")

    if (tonic, mode) not in KEY_SIGNATURE_ACCIDENTALS:
        raise ValueError(
            f"key_signature {tonic} {mode} is not one of the 30 standard keys "
            "(a key like D# major exists only as an enharmonic respelling)"
        )
    return tonic, mode


def key_letter_accidentals(accidental_count: int) -> dict[str, int]:
    """Per-letter accidental implied by a key signature, e.g. 2 sharps ->
    {F: +1, C: +1, rest 0}."""
    letters = {letter: 0 for letter in NOTE_LETTER_SEMITONES}
    if accidental_count > 0:
        for letter in SHARP_ORDER[:accidental_count]:
            letters[letter] = 1
    elif accidental_count < 0:
        for letter in FLAT_ORDER[:-accidental_count]:
            letters[letter] = -1
    return letters


def spell_midi_in_key(midi: int, letter_accidentals: dict[str, int], prefer_flats: bool) -> str:
    """MIDI number -> note name spelled for the key, e.g. 63 -> "Eb4" in Eb
    major (not "D#4", which is what librosa.midi_to_note would give).

    Three rules, applied in this order:

    1. Diatonic -- take the spelling the key signature dictates (in Eb major
       MIDI 63 is Eb, not D#).
    2. Chromatic but a NATURAL letter exists at that pitch class -- spell it
       natural, and let the engraver add the natural sign. This rule is not
       optional: without it, C natural in D major comes out as B#, and B
       natural in F major as Cb, because rule 3 would blindly alter a
       neighbouring letter. Those are the notes an accidental-heavy passage
       hits most often, so getting this wrong is very visible.
    3. Otherwise (one of the five genuinely-black-key pitch classes, not in
       the key) -- follow the key's accidental DIRECTION: sharp keys spell it
       as a sharp, flat keys as a flat.

    Rule 3 is a HEURISTIC, and its limits are worth being explicit about: the
    musically-correct spelling of a chromatic note often depends on harmonic
    function we do not have. A raised fourth leading to the dominant genuinely
    wants a sharp even in a flat key (F# in C minor), and a borrowed flat sixth
    wants a flat even in a sharp key. Getting those right needs chord/function
    analysis (the unbuilt music-theory engine), so this picks the direction
    that is right most often within a key rather than pretending to know the
    harmony.
    """
    pitch_class = midi % 12

    # 1. Diatonic: exactly one letter in the key produces this pitch class.
    for letter, accidental in letter_accidentals.items():
        if (NOTE_LETTER_SEMITONES[letter] + accidental) % 12 == pitch_class:
            return _format_note_name(letter, accidental, midi)

    # 2. A natural letter sits on this pitch class -- prefer it over altering
    #    a neighbour, even if the key alters that letter.
    for letter, semitones in NOTE_LETTER_SEMITONES.items():
        if semitones % 12 == pitch_class:
            return _format_note_name(letter, 0, midi)

    # 3. A black-key pitch class outside the key: follow the key's direction.
    step = -1 if prefer_flats else 1
    for letter, semitones in NOTE_LETTER_SEMITONES.items():
        if (semitones + step) % 12 == pitch_class:
            return _format_note_name(letter, step, midi)

    raise ValueError(f"could not spell MIDI {midi}")


def _format_note_name(letter: str, accidental: int, midi: int) -> str:
    """Assemble "Eb4"-style names. The octave is derived from the SPELLING, not
    from midi // 12, so enharmonics that cross an octave boundary stay correct:
    Cb4 and B3 are the same pitch (MIDI 59) but belong to different octaves."""
    octave = (midi - NOTE_LETTER_SEMITONES[letter] - accidental) // 12 - 1
    suffix = "#" if accidental == 1 else "b" if accidental == -1 else ""
    return f"{letter}{suffix}{octave}"


def tonic_to_lilypond(tonic: str) -> str:
    """"Eb" -> "ef", "F#" -> "fs" -- abjad's English naming (verified against
    the installed abjad, which rejects Dutch "cis"/"ces" outright)."""
    pitch = tonic[0].lower()
    if len(tonic) > 1:
        if tonic[1] in _SHARP_SYMBOLS:
            pitch += "s"
        elif tonic[1] in _FLAT_SYMBOLS:
            pitch += "f"
    return pitch


def _full_measure_rest_token(bar_size_sixteenths: int) -> str:
    """A whole bar of silence is one full-measure rest -- LilyPond's R, not r.

    Deliberately NOT combined with LilyPond's multi-measure compression
    (\\compressMMRests), which collapses several empty bars into one symbol
    with a count above it. That is right for a single-staff part but wrong
    here: on a PianoStaff the other hand is usually still playing, and
    compressing one staff would desynchronise the two.
    """
    code = NOTATION_VALUE_TO_CODE.get(bar_size_sixteenths)
    if code is not None:
        return f"R{code}"
    # Bars that aren't a single standard value (9/8 = 18 sixteenths, 12/8 = 24)
    # use LilyPond's scaled form, e.g. R1*9/8.
    divisor = math.gcd(bar_size_sixteenths, NOTATION_BAR_SIZE)
    return f"R1*{bar_size_sixteenths // divisor}/{NOTATION_BAR_SIZE // divisor}"


def _spell_rests(start: int, length: int, bar_length_beats: float) -> list[str]:
    """Spell a stretch of silence as the fewest, largest rests that fit.

    Silence follows the same beat-respecting rules as notes (via spell_rhythm:
    no value crosses a bar line, and a value starting off-beat can't swallow
    the next beat boundary), plus one convention notes don't have -- a bar
    silent from its first sixteenth to its last is a single full-measure rest.

    The caller must pass ONE contiguous span of silence. Spelling each silent
    event separately is exactly the bug this replaced: adjacent silences never
    merged, so an empty 4/4 bar came out as four quarter rests, or as thirty
    eighth rests across an empty staff.
    """
    bar_size = round(bar_length_beats * NOTATION_SIXTEENTHS_PER_BEAT)
    tokens: list[str] = []
    position = start
    remaining = length
    while remaining > 0:
        bar_start = (position // bar_size) * bar_size
        chunk = min(remaining, bar_start + bar_size - position)
        if position == bar_start and chunk == bar_size:
            tokens.append(_full_measure_rest_token(bar_size))
        else:
            for value, _tied in spell_rhythm(position, chunk, bar_length_beats):
                tokens.append(f"r{NOTATION_VALUE_TO_CODE[value]}")
        position += chunk
        remaining -= chunk
    return tokens


def _build_staff_input(
    events: list[dict],
    is_treble: bool,
    total_sixteenths: int,
    bar_length_beats: float = NOTATION_BEATS_PER_BAR,
    letter_accidentals: dict[str, int] | None = None,
    prefer_flats: bool = False,
) -> str:
    # Pass 1 -- work out which spans carry notes ON THIS STAFF. The accept/skip
    # rule below depends only on event start/end, never on pitch, so both
    # staves walk an identical timeline and no note can shift between them.
    filled: list[tuple[int, int, str]] = []
    position = 0
    for event in events:
        absolute_beat = (event["bar"] - 1) * bar_length_beats + event["beat_in_bar"]
        start = round(absolute_beat * NOTATION_SIXTEENTHS_PER_BEAT)
        # Defensive floor: two very close real onsets can quantize to the same
        # beat, leaving calculate_note_durations to report a zero/negative gap.
        # A note can't be notated with zero duration, so floor it to a
        # sixteenth rather than silently dropping it.
        duration_beats = max(event["duration_beats"], 0.25)
        end = start + round(duration_beats * NOTATION_SIXTEENTHS_PER_BEAT)

        if start < position:
            # Quantization collapsed this event earlier than where the previous
            # event already filled to -- nothing sensible to notate, skip it.
            continue

        # Spelling is key-aware, but the treble/bass split is unchanged: it is
        # still purely pitch-based on the same MIDI value as before. Respelling
        # can't move a note between staves -- Cb4 and B3 are the same MIDI
        # number, so they land on the same staff either way.
        pitches = [
            note_name_to_lilypond_pitch(
                spell_midi_in_key(
                    note_name_to_midi(note["note"]), letter_accidentals, prefer_flats
                )
                if letter_accidentals is not None
                else note["note"]
            )
            for note in event["notes"]
            if (note_name_to_midi(note["note"]) >= 60) == is_treble
        ]
        if pitches:
            # Multiple notes on one staff (e.g. two notes in different octaves
            # both landing treble) render as a normal simultaneous chord.
            pitch_str = pitches[0] if len(pitches) == 1 else "<" + " ".join(pitches) + ">"
            filled.append((start, end, pitch_str))
        position = end

    # Pass 2 -- emit. Everything between two notes on this staff is one
    # contiguous silence, however many events it spans, so it consolidates.
    tokens: list[str] = []
    position = 0
    for start, end, pitch_str in filled:
        if start > position:
            tokens.extend(_spell_rests(position, start - position, bar_length_beats))
        for value, tied in spell_rhythm(start, end - start, bar_length_beats):
            tie = "~" if tied else ""
            tokens.append(f"{pitch_str}{NOTATION_VALUE_TO_CODE[value]}{tie}")
        position = end

    if total_sixteenths > position:
        tokens.extend(_spell_rests(position, total_sixteenths - position, bar_length_beats))

    return " ".join(tokens) if tokens else _full_measure_rest_token(
        round(bar_length_beats * NOTATION_SIXTEENTHS_PER_BEAT)
    )


def generate_notation_pdf(
    events: list[dict],
    output_path: Path,
    time_signature: tuple[int, int] = (4, 4),
    key_signature: tuple[str, str] = DEFAULT_KEY_SIGNATURE,
) -> None:
    tonic, mode = key_signature
    accidental_count = KEY_SIGNATURE_ACCIDENTALS[(tonic, mode)]
    letter_accidentals = key_letter_accidentals(accidental_count)
    # Flat keys spell chromatic notes as flats. C major / A minor have no
    # accidentals and so no direction of their own -- they keep sharps, which
    # is what librosa.midi_to_note produced before this existed, so their
    # output is unchanged.
    prefer_flats = accidental_count < 0

    bar_length_beats = time_signature_bar_length_beats(*time_signature)
    bar_size_sixteenths = round(bar_length_beats * NOTATION_SIXTEENTHS_PER_BEAT)

    if events:
        last_event = events[-1]
        last_start_beats = (last_event["bar"] - 1) * bar_length_beats + last_event["beat_in_bar"]
        total_beats_needed = last_start_beats + max(last_event["duration_beats"], 0.25)
    else:
        total_beats_needed = 0

    total_sixteenths_needed = total_beats_needed * NOTATION_SIXTEENTHS_PER_BEAT
    total_sixteenths = max(
        bar_size_sixteenths,
        math.ceil(total_sixteenths_needed / bar_size_sixteenths) * bar_size_sixteenths,
    )

    treble_input = _build_staff_input(
        events,
        is_treble=True,
        total_sixteenths=total_sixteenths,
        bar_length_beats=bar_length_beats,
        letter_accidentals=letter_accidentals,
        prefer_flats=prefer_flats,
    )
    bass_input = _build_staff_input(
        events,
        is_treble=False,
        total_sixteenths=total_sixteenths,
        bar_length_beats=bar_length_beats,
        letter_accidentals=letter_accidentals,
        prefer_flats=prefer_flats,
    )

    treble_staff = abjad.Staff(treble_input, name="Treble")
    bass_staff = abjad.Staff(bass_input, name="Bass")

    # Attached to BOTH staves: on a PianoStaff, LilyPond does not propagate a
    # key signature from one staff to the other, so a single attach would
    # engrave the accidentals on the treble stave only.
    abjad_key = abjad.KeySignature(abjad.NamedPitchClass(tonic_to_lilypond(tonic)), abjad.Mode(mode))
    abjad.attach(abjad.Clef("treble"), abjad.select.leaves(treble_staff)[0])
    abjad.attach(abjad_key, abjad.select.leaves(treble_staff)[0])
    abjad.attach(abjad.TimeSignature(time_signature), abjad.select.leaves(treble_staff)[0])
    abjad.attach(abjad.Clef("bass"), abjad.select.leaves(bass_staff)[0])
    # A fresh instance: abjad forbids attaching one indicator to two leaves.
    abjad.attach(
        abjad.KeySignature(abjad.NamedPitchClass(tonic_to_lilypond(tonic)), abjad.Mode(mode)),
        abjad.select.leaves(bass_staff)[0],
    )
    abjad.attach(abjad.TimeSignature(time_signature), abjad.select.leaves(bass_staff)[0])

    piano_staff_group = abjad.StaffGroup(
        [treble_staff, bass_staff], lilypond_type="PianoStaff", name="Piano"
    )
    score = abjad.Score([piano_staff_group], name="Score")
    lilypond_file = abjad.LilyPondFile([score])

    abjad.persist.as_pdf(lilypond_file, str(output_path))


def extract_frame_at_time(video_path: Path, timestamp_seconds: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            return None

        frame_index = int(timestamp_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = cap.read()
        if not success or frame is None:
            return None

        return frame
    finally:
        # cv2.VideoCapture holds an open file handle until explicitly released.
        cap.release()


# --- Keyboard calibration (video/CV layer, Stage 1) ---
#
# Real key geometry, sourced from Wikimedia Commons "Pianoteilung.svg"
# (https://commons.wikimedia.org/wiki/File:Pianoteilung.svg), the diagram
# used on German Wikipedia's "Klaviatur" article to illustrate real piano
# key spacing ("Pianoteilung" = piano key division). It's a to-scale
# technical drawing with real coordinate paths, not a stylized approximation.
# Cross-checked against real-world figures independently: the drawing's
# white key width works out to 23.6mm, matching DIN 8995 (118.0cm across
# 50 white keys / 7 octaves = 23.6mm/white key), and 7 * 23.6mm = 165.2mm,
# matching the well-established ~165.1mm (6.5") standard octave span.
# The ratios below were computed directly from the SVG's raw path
# coordinates (in white-key-width units): all 5 black keys independently
# work out to the identical 0.53814 width fraction, which cross-validates
# the source data as internally consistent, not a transcription error.
WHITE_KEY_PITCH_CLASSES = {0: "C", 2: "D", 4: "E", 5: "F", 7: "G", 9: "A", 11: "B"}
BLACK_KEY_WIDTH_FRACTION = 0.53814  # fraction of white-key-width
# Left-edge offset of each black key, in white-key-width units, measured
# from the left edge of its octave's C -- asymmetric by design, not evenly
# spaced (e.g. C# sits well into the C-D gap, not centered on it).
BLACK_KEY_LEFT_OFFSETS = {
    1: ("C#", 0.65683),
    3: ("D#", 1.84313),
    6: ("F#", 3.61002),
    8: ("G#", 4.74998),
    10: ("A#", 5.88994),
}


def _key_geometry(midi: int) -> tuple[str, float, float]:
    """Returns (note_name, left_offset_in_white_widths, width_in_white_widths)
    for a MIDI note, positioned on an absolute scale where octave N (MIDI
    12*N..12*N+11) occupies white-key-width units [7N, 7N+7)."""
    octave_index = midi // 12
    pitch_class = midi % 12
    octave_start = octave_index * 7
    octave_number = octave_index - 1  # MIDI 60 (pitch class 0) -> C4

    if pitch_class in WHITE_KEY_PITCH_CLASSES:
        letter = WHITE_KEY_PITCH_CLASSES[pitch_class]
        white_index = list(WHITE_KEY_PITCH_CLASSES).index(pitch_class)
        return f"{letter}{octave_number}", octave_start + white_index, 1.0

    letter, left_offset = BLACK_KEY_LEFT_OFFSETS[pitch_class]
    return f"{letter}{octave_number}", octave_start + left_offset, BLACK_KEY_WIDTH_FRACTION


# --- Perspective-corrected calibration (Stage 1.5) ---
#
# Real key DEPTH dimensions, from the same source as the width geometry
# above (Wikimedia Commons "Pianoteilung.svg"). That diagram carries its
# own vertical dimension lines and embedded mm labels ("100" and "45") on
# the depth axis -- an independent measurement channel within the same
# source. Decoding the raw SVG rect coordinates directly (white key rect:
# y=3528.82, height=7161.12; black key rect: y=5751.27, height=4938.67;
# both flush at the back edge y=10689.94) and converting through the same
# raw-unit-to-mm scale factor established from the width axis (23.6mm per
# 1165.5 raw units) gives white-only front zone = 45.00mm and black key
# depth = 100.00mm -- both within 0.001mm of the diagram's own printed
# labels, confirming the decode rather than coincidence.
# Cross-validated against a second, independent source: a PianoWorld forum
# thread (https://forum.pianoworld.com/ubbthreads.php/topics/154457.html)
# with real grand-piano fallboard-to-tip measurements -- Schimmel CC213
# and Estonia grands at 5 7/8" (149.2mm), Baldwin Model L at 5 5/8"
# (142.9mm). This diagram's total white key depth (45+100=145mm) falls
# squarely inside that real-world 142.9-149.2mm range.
WHITE_KEY_TOTAL_DEPTH_MM = 145.0
BLACK_KEY_DEPTH_MM = 100.0
WHITE_ONLY_ZONE_DEPTH_MM = WHITE_KEY_TOTAL_DEPTH_MM - BLACK_KEY_DEPTH_MM  # 45.0
WHITE_KEY_WIDTH_MM = 23.6  # DIN 8995, established in Stage 1

# Canonical 2D physical keyboard space, in white-key-width (W) units for
# both axes: u = lateral position (Stage 1's model, unchanged), v = depth
# from the front edge (v=0) toward the back of the visible keybed.
TOTAL_DEPTH_W = WHITE_KEY_TOTAL_DEPTH_MM / WHITE_KEY_WIDTH_MM  # ~6.144
WHITE_ONLY_ZONE_DEPTH_W = WHITE_ONLY_ZONE_DEPTH_MM / WHITE_KEY_WIDTH_MM  # ~1.907


def build_keyboard_calibration(
    leftmost_note: str,
    rightmost_note: str,
    front_left_px: tuple[float, float],
    front_right_px: tuple[float, float],
    back_left_px: tuple[float, float],
    back_right_px: tuple[float, float],
) -> dict:
    """Perspective-corrected calibration (Stage 1.5) -- supersedes Stage 1's
    2-point horizontal-only version, which assumed an overhead camera and
    couldn't distinguish "over a white key" from "over a black key" by
    depth. Takes the visible keybed's 4 real corners instead of 2 edges."""
    leftmost_midi = note_name_to_midi(leftmost_note)
    rightmost_midi = note_name_to_midi(rightmost_note)
    if rightmost_midi <= leftmost_midi:
        raise ValueError("rightmost_note must be higher than leftmost_note")

    _, left_anchor_offset, _ = _key_geometry(leftmost_midi)
    _, right_anchor_offset, right_anchor_width = _key_geometry(rightmost_midi)
    u_left = left_anchor_offset
    u_right = right_anchor_offset + right_anchor_width

    src_px = np.array([front_left_px, front_right_px, back_left_px, back_right_px], dtype=np.float32)
    dst_canonical = np.array(
        [[u_left, 0.0], [u_right, 0.0], [u_left, TOTAL_DEPTH_W], [u_right, TOTAL_DEPTH_W]],
        dtype=np.float32,
    )
    pixel_to_canonical_matrix = cv2.getPerspectiveTransform(src_px, dst_canonical)
    canonical_to_pixel_matrix = cv2.getPerspectiveTransform(dst_canonical, src_px)

    def _transform(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
        point = np.array([[[x, y]]], dtype=np.float32)
        result = cv2.perspectiveTransform(point, matrix)
        return float(result[0, 0, 0]), float(result[0, 0, 1])

    # Per-key lateral (u) ranges, shared by the forward pixel-quadrilateral
    # build-out below and the reverse lookup closure.
    u_ranges = {}
    for midi in range(leftmost_midi, rightmost_midi + 1):
        note_name, left_offset, width = _key_geometry(midi)
        is_white = (midi % 12) in WHITE_KEY_PITCH_CLASSES
        u_ranges[note_name] = (left_offset, left_offset + width, is_white, midi)

    keys = {}
    for note_name, (u0, u1, is_white, midi) in u_ranges.items():
        v0, v1 = (0.0, TOTAL_DEPTH_W) if is_white else (WHITE_ONLY_ZONE_DEPTH_W, TOTAL_DEPTH_W)
        corners_px = [
            _transform(canonical_to_pixel_matrix, u0, v0),
            _transform(canonical_to_pixel_matrix, u1, v0),
            _transform(canonical_to_pixel_matrix, u1, v1),
            _transform(canonical_to_pixel_matrix, u0, v1),
        ]
        center_px = (
            sum(p[0] for p in corners_px) / 4,
            sum(p[1] for p in corners_px) / 4,
        )
        keys[note_name] = {
            "midi": midi,
            "is_white": is_white,
            "corners_px": corners_px,  # [front-left, front-right, back-right, back-left]
            "center_px": center_px,
            # Canonical lateral extent/centre. Kept alongside the pixel
            # geometry so proximity checks can work in real millimetres
            # instead of pixels, which vary with perspective across the frame.
            "u_range": (u0, u1),
            "u_center": (u0 + u1) / 2,
        }

    def pixel_to_note(x: float, y: float) -> str:
        u, v = _transform(pixel_to_canonical_matrix, x, y)
        # Physical reachability gate: within the white-only front zone, a
        # black key literally isn't there to press, regardless of how close
        # u is to one -- the finger can only be touching the white key.
        force_white = v < WHITE_ONLY_ZONE_DEPTH_W

        for is_white_pass in (False, True):
            if is_white_pass is False and force_white:
                continue
            for note_name, (u0, u1, is_white, _midi) in u_ranges.items():
                if is_white != is_white_pass:
                    continue
                if u0 <= u < u1:
                    return note_name

        # Outside the calibrated lateral range -- clamp to the nearest key
        # by u, restricted to white keys if depth forces it.
        candidates = u_ranges.items()
        if force_white:
            candidates = [(n, r) for n, r in candidates if r[2]]
        nearest = min(candidates, key=lambda item: abs((item[1][0] + item[1][1]) / 2 - u))
        return nearest[0]

    def pixel_to_canonical(x: float, y: float) -> tuple[float, float]:
        """Raw (u, v) in white-key-width units -- unlike pixel_to_note this
        doesn't snap to a key, so callers can measure real distances."""
        return _transform(pixel_to_canonical_matrix, x, y)

    return {
        "keys": keys,
        # Indexed by MIDI as well as name: librosa.midi_to_note() emits
        # unicode sharps ("A♯4") while these key names are ASCII ("A#4"), so
        # name lookups from pipeline data would silently miss every sharp.
        "keys_by_midi": {info["midi"]: info for info in keys.values()},
        "pixel_to_note": pixel_to_note,
        "pixel_to_canonical": pixel_to_canonical,
        "u_bounds": (u_left, u_right),
    }


# --- Motion-based key-press confirmation (video/CV layer, Stage 2) ---
#
# Uses MediaPipe's Tasks API (mp.solutions.hands is gone entirely as of
# mediapipe 0.10.35, not just deprecated -- only HandLandmarker exists now).
# Requires a separately-downloaded model file, same as basic-pitch's
# checkpoint; see backend/models/hand_landmarker.task.

HAND_LANDMARKER_MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
FINGERTIP_LANDMARK_INDICES = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips

# Tunable -- empirically set against test.wav.mp4's known onsets. The 0.015
# default first tried (a round-number guess) was too strict: real observed
# peak z-deviations for genuine presses in that clip mostly fell in the
# 0.004-0.011 range, well under it, so every timestamp came back empty.
# Lowering the threshold to 0.006 was tested against several window sizes
# (0.12s, 0.2s, 0.25s) with materially the same ~4/15 accuracy each time --
# see the validation report for the honest accuracy figure and failure
# characterization; this is a first-pass value, not a fully solved one.
KEY_PRESS_WINDOW_SECONDS = 0.2
KEY_PRESS_DIP_THRESHOLD = 0.006

_hand_landmarker: mp_vision.HandLandmarker | None = None


def _get_hand_landmarker() -> mp_vision.HandLandmarker:
    global _hand_landmarker
    if _hand_landmarker is None:
        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_MODEL_PATH)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
        )
        _hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)
    return _hand_landmarker


def _extract_frames_in_window(
    video_path: Path, center_timestamp: float, window_seconds: float, fps: float | None = None
) -> list[tuple[float, np.ndarray]]:
    """Extends extract_frame_at_time's seek-by-frame-index approach to pull
    every frame across a time window instead of a single instant."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            return []

        start_frame = max(0, int((center_timestamp - window_seconds) * fps))
        end_frame = int((center_timestamp + window_seconds) * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        frame_index = start_frame
        while frame_index <= end_frame:
            success, frame = cap.read()
            if not success or frame is None:
                break
            frames.append((frame_index / fps, frame))
            frame_index += 1
        return frames
    finally:
        cap.release()


def _dip_confidence(z_values: list[float]) -> float:
    """A real key press is a dip AND a recovery -- baseline from the
    window's edges, a real deviation in the middle, then back toward
    baseline. Sign-agnostic: which direction counts as "down" depends on
    camera angle/hand orientation, so this looks at deviation magnitude,
    not a hardcoded sign."""
    n = len(z_values)
    if n < 3:
        return 0.0

    edge_count = max(1, n // 4)
    if n <= 2 * edge_count:
        return 0.0
    baseline = (sum(z_values[:edge_count]) + sum(z_values[-edge_count:])) / (2 * edge_count)

    middle = z_values[edge_count:-edge_count]
    deviations = [abs(v - baseline) for v in middle]
    peak_deviation = max(deviations)
    peak_index = deviations.index(peak_deviation)

    if peak_deviation < KEY_PRESS_DIP_THRESHOLD:
        return 0.0

    # Require recovery after the peak -- otherwise this is drift, not a
    # press-and-release.
    after_peak = middle[peak_index:]
    if len(after_peak) >= 2 and abs(after_peak[-1] - baseline) >= peak_deviation:
        return 0.0

    return peak_deviation


def detect_key_press_at_timestamp(
    video_path: Path,
    timestamp: float,
    calibration: dict,
    window_seconds: float = KEY_PRESS_WINDOW_SECONDS,
    fps: float | None = None,
) -> list[dict]:
    frames = _extract_frames_in_window(video_path, timestamp, window_seconds, fps=fps)
    if len(frames) < 3:
        return []

    frame_height, frame_width = frames[0][1].shape[:2]
    landmarker = _get_hand_landmarker()

    detections = []
    for frame_ts, frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        detections.append((frame_ts, result.hand_landmarks))

    anchor_index = min(range(len(detections)), key=lambda i: abs(detections[i][0] - timestamp))
    _, anchor_hands = detections[anchor_index]
    if not anchor_hands:
        return []

    pixel_to_note = calibration["pixel_to_note"]

    results = []
    for hand in anchor_hands:
        for finger_idx in FINGERTIP_LANDMARK_INDICES:
            anchor_lm = hand[finger_idx]
            anchor_x_px = anchor_lm.x * frame_width
            anchor_y_px = anchor_lm.y * frame_height

            # MediaPipe's IMAGE mode re-detects hands independently per
            # frame -- there's no persistent hand/finger identity across
            # frames the way VIDEO/LIVE_STREAM mode's internal tracker
            # would provide. Match the same finger index on whichever
            # detected hand sits closest to the anchor position each frame.
            trajectory = []
            for frame_ts, hands in detections:
                candidates = [h[finger_idx] for h in hands]
                if not candidates:
                    continue
                best = min(
                    candidates,
                    key=lambda lm: (lm.x * frame_width - anchor_x_px) ** 2
                    + (lm.y * frame_height - anchor_y_px) ** 2,
                )
                trajectory.append((frame_ts, best.z))

            z_values = [z for _, z in trajectory]
            confidence = _dip_confidence(z_values)
            if confidence <= 0:
                continue

            results.append(
                {
                    "note": pixel_to_note(anchor_x_px, anchor_y_px),
                    "confidence": confidence,
                    "finger_track": trajectory,
                }
            )

    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results


# --- Lateral proximity measurement (video/CV layer, Stage 2.5) ---
#
# Stages 2 and 1.5 between them established that MediaPipe's z signal cannot
# reliably resolve a ~10mm key press (accuracy sat at 3-4/15 whether or not
# the calibration geometry was perspective-correct). So this drops motion and
# z entirely and uses only what MediaPipe is genuinely reliable at: 2D lateral
# hand position. One frame, no window, no trajectory, no dip detection.

ALL_HAND_LANDMARK_INDICES = list(range(21))


def detect_hand_canonical_positions(
    video_path: Path,
    timestamp: float,
    calibration: dict,
    landmark_indices: list[int] | None = None,
) -> dict:
    """Project hand landmarks visible at `timestamp` into canonical keyboard
    space, using the single frame nearest that instant."""
    if landmark_indices is None:
        landmark_indices = FINGERTIP_LANDMARK_INDICES

    frame = extract_frame_at_time(video_path, timestamp)
    if frame is None:
        return {"hands_detected": 0, "points": [], "frame_read": False}

    frame_height, frame_width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = _get_hand_landmarker().detect(mp_image)

    pixel_to_canonical = calibration["pixel_to_canonical"]
    points = []
    for hand_index, hand in enumerate(result.hand_landmarks):
        for index in landmark_indices:
            landmark = hand[index]
            u, v = pixel_to_canonical(landmark.x * frame_width, landmark.y * frame_height)
            points.append({"hand": hand_index, "landmark": index, "u": u, "v": v})

    return {
        "hands_detected": len(result.hand_landmarks),
        "points": points,
        "frame_read": True,
    }


def lateral_distance_to_nearest_landmark_mm(
    calibration: dict, midi: int, points: list[dict]
) -> float | None:
    """Lateral (u-axis) distance in mm from a note's key centre to the nearest
    projected landmark. Returns None when the note falls outside the calibrated
    range -- that's unverifiable, which is a different thing from "far away"
    and must not be collapsed into a large distance."""
    key = calibration["keys_by_midi"].get(midi)
    if key is None or not points:
        return None
    return min(abs(point["u"] - key["u_center"]) * WHITE_KEY_WIDTH_MM for point in points)


def convert_seconds_to_beats(onset_times: list[float], tempo_bpm: float) -> list[float]:
    return [round(timestamp * (tempo_bpm / 60.0), 2) for timestamp in onset_times]


def quantize_beats(detected_beats: list[float], resolution: float = 0.25) -> list[float]:
    return [round(round(beat / resolution) * resolution, 2) for beat in detected_beats]


def select_best_quantization_resolution(
    detected_beats: list[float],
    candidate_resolutions: list[float] = [1.0, 0.5, 0.25],
    fit_tolerance_fraction: float = 0.3,
) -> float:
    for resolution in candidate_resolutions:
        quantized = quantize_beats(detected_beats, resolution=resolution)
        deviations = [abs(beat - q) for beat, q in zip(detected_beats, quantized)]
        average_deviation = sum(deviations) / len(deviations) if deviations else 0.0
        # A resolution that fits on average can still collapse two distinct
        # onsets onto the same beat -- that's not "denser than ideal", it's
        # wrong, so any collision disqualifies the resolution outright.
        collides = len(set(quantized)) != len(quantized)
        if average_deviation <= fit_tolerance_fraction * resolution and not collides:
            return resolution

    return min(candidate_resolutions)


COMPOUND_TIME_SIGNATURES = {"6/8", "9/8", "12/8"}


def parse_time_signature(time_signature: str) -> tuple[int, int]:
    numerator_str, denominator_str = time_signature.split("/")
    return int(numerator_str), int(denominator_str)


def time_signature_bar_length_beats(numerator: int, denominator: int) -> float:
    """Bar length in quarter-note-equivalent beats -- e.g. 6/8 -> 3.0."""
    return numerator * 4 / denominator


def calculate_bar_structures(
    quantized_beats: list[float], bar_length_beats: float = 4.0
) -> tuple[list[int], list[float]]:
    detected_bars = []
    measure_beats = []

    for beat in quantized_beats:
        bar_number = int(beat // bar_length_beats) + 1
        beat_in_bar = round(beat % bar_length_beats, 2)
        detected_bars.append(bar_number)
        measure_beats.append(beat_in_bar)

    return detected_bars, measure_beats


STANDARD_NOTE_DURATIONS = [
    (0.25, "sixteenth"),
    (0.50, "eighth"),
    (1.00, "quarter"),
    (2.00, "half"),
    (4.00, "whole"),
]


def _closest_note_type(duration: float) -> str:
    if duration <= 0:
        return "complex"

    closest_value, closest_name = min(
        STANDARD_NOTE_DURATIONS, key=lambda item: abs(item[0] - duration)
    )
    # Anything within half the closest standard value's own length counts as
    # that note type; further off doesn't cleanly fit the standard grid.
    if abs(duration - closest_value) <= closest_value * 0.5:
        return closest_name
    return "complex"


def calculate_note_durations(
    quantized_beats: list[float], total_duration_beats: float
) -> tuple[list[float], list[str]]:
    note_durations = []
    note_types = []

    for i, beat in enumerate(quantized_beats):
        if i < len(quantized_beats) - 1:
            duration = quantized_beats[i + 1] - beat
        else:
            duration = total_duration_beats - beat

        duration = round(duration, 2)
        note_durations.append(duration)
        note_types.append(_closest_note_type(duration))

    return note_durations, note_types


def detect_notes_basic_pitch(audio_path: Path) -> list[dict]:
    _, _, note_events = predict(str(audio_path), ICASSP_2022_MODEL_PATH)

    notes = [
        {
            "onset": float(onset),
            "offset": float(offset),
            "note": librosa.midi_to_note(midi_pitch),
            "midi": int(midi_pitch),
            "confidence": float(amplitude),
        }
        for onset, offset, midi_pitch, amplitude, _pitch_bends in note_events
    ]

    return sorted(notes, key=lambda note: note["onset"])


DEDUP_GAP_SECONDS = 0.05  # max gap between offset and next onset, same pitch, to treat as one fragmented note
# Cap on total merged-note duration, expressed in BEATS because it describes a
# musical note length, not a physical constant. It was previously 2.0 SECONDS,
# justified by "the longest genuine note observed in testing is ~1.65s" -- but
# that observation is tempo-dependent, so encoding it as an absolute duration
# was wrong: a whole note at 60 BPM lasts 4s and would have been split.
# 4.0 beats = a whole note in 4/4, which is the longest value the notation
# vocabulary (NOTATION_STANDARD_VALUES) can spell, so a fragment chain longer
# than this is almost certainly separate re-attacks rather than one note.
# Empirically chosen, not guessed: sweeping the cap shows the reference clips
# keep byte-identical output only for 1.57-2.06s on chords-notes-mix.mp4
# (= 3.22-4.22 beats at its 123 BPM), and for any value on test.wav.mp4
# (nothing there merges >1 fragment, so the cap never binds). 4.0 beats is the
# only musically meaningful value inside that intersection -- 3.0 beats would
# be 1.46s at 123 BPM and would split a real 1.569s two-fragment merge.
DEDUP_MAX_SPAN_BEATS = 4.0
GROUPING_WINDOW_SECONDS = 0.15  # same value as the old onset grouping
SOLID_THRESHOLD_SECONDS = 0.03  # same as before, for style classification
RELATIVE_CONFIDENCE_FRACTION = 0.6  # a note must reach this fraction of its event's strongest note to survive
ABSOLUTE_CONFIDENCE_FLOOR = 0.35  # hard floor below which nothing survives regardless of group


def deduplicate_notes(
    notes: list[dict],
    gap_threshold: float = DEDUP_GAP_SECONDS,
    *,
    max_span_seconds: float,
) -> list[dict]:
    """`max_span_seconds` is keyword-only and deliberately has NO default: it
    must be derived from the piece's tempo by the caller (see
    DEDUP_MAX_SPAN_BEATS). A default here would be a tempo-blind constant,
    which is exactly the bug this signature exists to prevent."""
    if not notes:
        return []

    # Group by pitch first so same-pitch fragments merge correctly even when
    # an unrelated note's onset sorts between them chronologically — merging
    # only adjacent entries in one global onset-sorted list would miss those.
    by_pitch: dict[int, list[dict]] = {}
    for note in notes:
        by_pitch.setdefault(note["midi"], []).append(note)

    merged = []
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda note: note["onset"])

        current = dict(pitch_notes[0])
        current["fragments_merged"] = 1

        for note in pitch_notes[1:]:
            gap = note["onset"] - current["offset"]
            prospective_span = note["offset"] - current["onset"]
            if gap <= gap_threshold and prospective_span <= max_span_seconds:
                # Chain absorption: current keeps extending as long as the next
                # fragment matches, so a note split into 3+ pieces still merges
                # into a single entry rather than just pairwise. The span cap
                # stops this from chaining across genuinely separate re-attacks
                # of the same pitch (e.g. the same note in two different chords)
                # that happen to have near-zero gaps between them.
                current["offset"] = max(current["offset"], note["offset"])
                current["confidence"] = max(current["confidence"], note["confidence"])
                current["fragments_merged"] += 1
            else:
                merged.append(current)
                current = dict(note)
                current["fragments_merged"] = 1

        merged.append(current)

    return sorted(merged, key=lambda note: note["onset"])


def group_notes_into_events(
    notes: list[dict],
    grouping_window: float = GROUPING_WINDOW_SECONDS,
    solid_threshold: float = SOLID_THRESHOLD_SECONDS,
) -> list[dict]:
    if not notes:
        return []

    groups: list[list[dict]] = []
    current_group: list[dict] = []

    for note in notes:
        # Chained distance: compare against the previous note already in the
        # group, not the group's first note, so a run of closely-spaced
        # attacks can drift further apart than grouping_window in total.
        if current_group and (note["onset"] - current_group[-1]["onset"]) > grouping_window:
            groups.append(current_group)
            current_group = [note]
        else:
            current_group.append(note)

    if current_group:
        groups.append(current_group)

    events = []
    for group in groups:
        onsets = [note["onset"] for note in group]
        span = round(max(onsets) - min(onsets), 3)

        if len(group) == 1:
            style = "single"
        elif span <= solid_threshold:
            style = "solid"
        else:
            style = "rolled"

        events.append(
            {
                "event_time": min(onsets),
                "span": span,
                "style": style,
                "notes": group,
            }
        )

    return events


def filter_event_notes(
    events: list[dict],
    relative_fraction: float = RELATIVE_CONFIDENCE_FRACTION,
    absolute_floor: float = ABSOLUTE_CONFIDENCE_FLOOR,
) -> list[dict]:
    filtered_events = []

    for event in events:
        max_confidence = max(note["confidence"] for note in event["notes"])

        surviving = []
        dropped = []
        for note in event["notes"]:
            if note["confidence"] >= absolute_floor and note["confidence"] >= relative_fraction * max_confidence:
                surviving.append(note)
            else:
                dropped.append(note)

        filtered_events.append(
            {
                "event_time": event["event_time"],
                "span": event["span"],
                "style": event["style"],
                "notes": surviving,
                "dropped_notes": dropped,
            }
        )

    return filtered_events


DECAY_TAIL_DURATION_THRESHOLD = 0.4
DECAY_TAIL_PITCH_LOOKBACK_EVENTS = 2
DECAY_TAIL_DECAY_MARGIN_SECONDS = 0.5


def suppress_decay_tail_notes(events: list[dict]) -> list[dict]:
    result = []

    for i, event in enumerate(events):
        # Lookback uses the original filtered events, not the progressively
        # rebuilt result, so each event is judged against what actually
        # survived filter_event_notes — not against this function's own
        # earlier decisions.
        lookback_events = events[max(0, i - DECAY_TAIL_PITCH_LOOKBACK_EVENTS) : i]

        surviving = []
        dropped = list(event["dropped_notes"])

        for note in event["notes"]:
            duration = note["offset"] - note["onset"]
            is_short = duration < DECAY_TAIL_DURATION_THRESHOLD

            matches_recent_pitch = False
            if is_short:
                for prev_event in lookback_events:
                    for prev_note in prev_event["notes"]:
                        same_pitch_class = (prev_note["midi"] % 12) == (note["midi"] % 12)
                        within_margin = note["onset"] <= prev_note["offset"] + DECAY_TAIL_DECAY_MARGIN_SECONDS
                        if same_pitch_class and within_margin:
                            matches_recent_pitch = True
                            break
                    if matches_recent_pitch:
                        break

            if is_short and matches_recent_pitch:
                dropped.append(note)
            else:
                surviving.append(note)

        result.append(
            {
                "event_time": event["event_time"],
                "span": event["span"],
                "style": event["style"],
                "notes": surviving,
                "dropped_notes": dropped,
            }
        )

    return result


def analyze_audio(
    audio_path: Path,
    video_path: Path,
    time_signature: str | None = None,
    tempo_bpm_override: float | None = None,
) -> dict:
    # sr=None preserves the file's native sample rate instead of resampling to 22.05kHz.
    waveform, sample_rate = librosa.load(str(audio_path), sr=None)
    duration_seconds = librosa.get_duration(y=waveform, sr=sample_rate)

    if time_signature:
        numerator, denominator = parse_time_signature(time_signature)
    else:
        # "Auto" doesn't attempt time-signature detection (a separate, harder
        # problem) -- it just assumes 4/4, matching prior behavior exactly.
        numerator, denominator = 4, 4
    bar_length_beats = time_signature_bar_length_beats(numerator, denominator)
    is_compound = time_signature in COMPOUND_TIME_SIGNATURES

    if tempo_bpm_override is not None:
        # A compound-meter tempo marking is conventionally given as the
        # dotted-quarter-note value; internally everything is quarter-note
        # beats, so convert (dotted quarter = 1.5 quarter notes) before use.
        tempo_bpm = tempo_bpm_override * 1.5 if is_compound else tempo_bpm_override
        tempo_source = "user"
    else:
        tempo, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate)
        # librosa returns tempo as a 1-element array rather than a bare scalar.
        tempo_bpm = float(np.asarray(tempo).reshape(-1)[0])
        tempo_source = "auto"

    notes = detect_notes_basic_pitch(audio_path)
    # Convert the beat-relative dedup cap to seconds here, where the piece's
    # tempo (user-supplied or auto-detected above) is known -- deduplicate_notes
    # itself works in raw seconds because that's what onsets/offsets are in.
    dedup_max_span_seconds = DEDUP_MAX_SPAN_BEATS * 60.0 / tempo_bpm
    deduped = deduplicate_notes(notes, max_span_seconds=dedup_max_span_seconds)
    events = group_notes_into_events(deduped)
    filtered_events = filter_event_notes(events)
    filtered_events = suppress_decay_tail_notes(filtered_events)

    # Reintegrate bar/beat/duration onto the final (deduped/grouped/filtered)
    # events -- these functions predate the basic-pitch migration and were
    # unused dead code since then; the event structure they operate on
    # (a plain list of onset times) is unchanged, so this is a direct
    # drop-in against event_times instead of the old CQT event onsets.
    event_times = [event["event_time"] for event in filtered_events]
    detected_beats = convert_seconds_to_beats(event_times, tempo_bpm)
    resolution = select_best_quantization_resolution(detected_beats)
    quantized_beats = quantize_beats(detected_beats, resolution=resolution)
    detected_bars, measure_beats = calculate_bar_structures(
        quantized_beats, bar_length_beats=bar_length_beats
    )
    total_duration_beats = duration_seconds * (tempo_bpm / 60.0)
    note_durations, note_types = calculate_note_durations(quantized_beats, total_duration_beats)

    for event, bar, beat_in_bar, duration_beats, note_type in zip(
        filtered_events, detected_bars, measure_beats, note_durations, note_types
    ):
        event["bar"] = bar
        event["beat_in_bar"] = beat_in_bar
        event["duration_beats"] = duration_beats
        event["note_type"] = note_type

    # Sanity-check the audio-to-video frame targeting math against the first
    # few note onsets before it's relied on for real multimodal analysis.
    for note in notes[:3]:
        onset_time = note["onset"]
        frame = extract_frame_at_time(video_path, onset_time)
        if frame is not None:
            print(f"[frame check] onset={onset_time}s -> frame shape {frame.shape}")
        else:
            print(f"[frame check] onset={onset_time}s -> FAILED to read frame")

    return {
        "duration_seconds": round(float(duration_seconds), 3),
        "sample_rate": int(sample_rate),
        "tempo_bpm": round(tempo_bpm, 1),
        "raw_notes": notes,
        "events": filtered_events,
        "quantization_resolution": resolution,
        "time_signature": f"{numerator}/{denominator}",
        "tempo_source": tempo_source,
    }


@app.get("/api/hello")
def read_hello():
    return {"message": "Hello World"}


# --- Calibration intake (Stage 2.75) ---
#
# The frontend has always captured 4 keybed corners and a leftmost/rightmost
# note range and then thrown them away. Everything downstream therefore had to
# *infer* the absolute octave anchor from finger spacing, which is a weak
# constraint and produced a confirmed ~80mm (tritone) lateral offset on
# chords-notes-mix.mp4 that invalidated a whole session of measurements
# (PROJECT_STATE §3 item 15). A user-supplied note range removes that
# inference entirely, so it is worth strict validation here: a calibration
# that is silently wrong is far more damaging than one that is rejected.
#
# Sent as a single JSON field rather than ~12 flat Form fields on purpose --
# calibration is all-or-nothing, and one field can only be present or absent,
# whereas separate fields admit partially-populated states that would each
# need their own ambiguous failure handling.

# Both guards are in source-frame px^2 and sit orders of magnitude below any
# real calibration (a genuine keybed quad is ~10^4-10^5 px^2).
CALIBRATION_MIN_QUAD_AREA_PX = 100.0
CALIBRATION_MIN_TRIANGLE_AREA_PX = 25.0
CALIBRATION_CORNER_NAMES = ("front_left", "front_right", "back_left", "back_right")


def _video_frame_size(video_path: Path) -> tuple[int, int] | None:
    cap = cv2.VideoCapture(str(video_path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height) if width > 0 and height > 0 else None
    finally:
        cap.release()


def _triangle_area(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0


def parse_calibration_payload(payload: str, video_path: Path) -> dict:
    """Validate the frontend's calibration blob, returning corner tuples and
    note names ready for build_keyboard_calibration.

    Raises HTTPException(400) with a specific message for every rejection --
    the alternative is an opaque cv2 error from a singular homography, or
    worse, a plausible-looking result that is quietly misprojected.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"calibration is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="calibration must be a JSON object.")

    raw_corners = data.get("corners")
    if not isinstance(raw_corners, dict):
        raise HTTPException(status_code=400, detail="calibration.corners is missing or not an object.")

    corners: dict[str, tuple[float, float]] = {}
    for name in CALIBRATION_CORNER_NAMES:
        point = raw_corners.get(name)
        # bool is a subclass of int, so exclude it explicitly rather than
        # letting `true` silently become the coordinate 1.
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in point
            )
        ):
            raise HTTPException(
                status_code=400,
                detail=f"calibration.corners.{name} must be an [x, y] pair of finite numbers.",
            )
        corners[name] = (float(point[0]), float(point[1]))

    leftmost = data.get("leftmost_note")
    rightmost = data.get("rightmost_note")
    if not isinstance(leftmost, str) or not isinstance(rightmost, str):
        raise HTTPException(
            status_code=400,
            detail="calibration.leftmost_note and calibration.rightmost_note are required strings.",
        )
    midis = {}
    for label, name in (("leftmost_note", leftmost), ("rightmost_note", rightmost)):
        try:
            midis[label] = note_name_to_midi(name)
        except (ValueError, KeyError, IndexError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"calibration.{label} {name!r} is not a valid note name (expected e.g. 'C4', 'A#3').",
            ) from exc
    if midis["rightmost_note"] <= midis["leftmost_note"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"calibration.rightmost_note ({rightmost}) must be higher than "
                f"calibration.leftmost_note ({leftmost})."
            ),
        )

    # Corners are in the source frame's own pixel space. If the client measured
    # against a differently-sized frame, every projected point is wrong by that
    # scale factor -- exactly the class of silent systematic offset that caused
    # the tritone bug -- so refuse rather than guess.
    frame_width = data.get("frame_width")
    frame_height = data.get("frame_height")
    if frame_width is not None and frame_height is not None:
        actual = _video_frame_size(video_path)
        if actual is not None and (int(frame_width), int(frame_height)) != actual:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"calibration frame size {int(frame_width)}x{int(frame_height)} does not match "
                    f"the uploaded video's actual {actual[0]}x{actual[1]}."
                ),
            )

    # Degenerate geometry check. cv2.getPerspectiveTransform is singular when
    # any three of the four points are collinear, which a plain "are the points
    # distinct?" test would miss entirely.
    quad = [corners[n] for n in ("front_left", "front_right", "back_right", "back_left")]
    shoelace = sum(
        quad[i][0] * quad[(i + 1) % 4][1] - quad[(i + 1) % 4][0] * quad[i][1] for i in range(4)
    )
    quad_area = abs(shoelace) / 2.0
    if quad_area < CALIBRATION_MIN_QUAD_AREA_PX:
        raise HTTPException(
            status_code=400,
            detail=(
                f"calibration corners are degenerate: quadrilateral area {quad_area:.1f}px^2 is too "
                "small (corners are coincident or collinear)."
            ),
        )
    for i in range(4):
        for j in range(i + 1, 4):
            for k in range(j + 1, 4):
                if _triangle_area(quad[i], quad[j], quad[k]) < CALIBRATION_MIN_TRIANGLE_AREA_PX:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "calibration corners are degenerate: three corners are collinear, which "
                            "makes the perspective transform singular."
                        ),
                    )

    return {"corners": corners, "leftmost_note": leftmost, "rightmost_note": rightmost}


def build_calibration_summary(
    keyboard_calibration: dict,
    leftmost_note: str,
    rightmost_note: str,
    frame_size: tuple[int, int] | None,
) -> dict:
    """JSON-safe view of a calibration. Deliberately excludes the closures
    (pixel_to_note / pixel_to_canonical) the calibration dict also carries."""
    keys = keyboard_calibration["keys"]
    u_low, u_high = keyboard_calibration["u_bounds"]
    ordered = sorted(keys.items(), key=lambda item: item[1]["midi"])
    return {
        "leftmost_note": leftmost_note,
        "rightmost_note": rightmost_note,
        "frame_width": frame_size[0] if frame_size else None,
        "frame_height": frame_size[1] if frame_size else None,
        "u_bounds": [round(u_low, 4), round(u_high, 4)],
        "key_count": len(keys),
        "white_key_count": sum(1 for info in keys.values() if info["is_white"]),
        "white_key_width_mm": WHITE_KEY_WIDTH_MM,
        "keys": [
            {
                "note": name,
                "midi": info["midi"],
                "is_white": info["is_white"],
                "u_center": round(info["u_center"], 4),
                "center_px": [round(info["center_px"][0], 2), round(info["center_px"][1], 2)],
                "corners_px": [[round(x, 2), round(y, 2)] for x, y in info["corners_px"]],
            }
            for name, info in ordered
        ],
    }


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile,
    time_signature: str | None = Form(None),
    tempo_bpm: float | None = Form(None),
    # Optional; absent means C major, which is what the score defaulted to
    # before key signatures existed.
    key_signature: str | None = Form(None),
    # Optional, and must stay optional: every caller without calibration data
    # (including the pipeline's own regression fixtures) has to keep working
    # byte-identically. When absent, nothing below touches the analysis at all.
    calibration: str | None = Form(None),
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Please upload a video file.",
        )

    # Parsed up front so a bad key fails before any file is written or analysed.
    try:
        resolved_key = parse_key_signature(key_signature)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    file_id = uuid.uuid4()
    extension = Path(file.filename).suffix
    saved_filename = f"{file_id}{extension}"
    destination = UPLOAD_DIR / saved_filename

    size_bytes = 0
    with destination.open("wb") as out_file:
        while chunk := await file.read(1024 * 1024):
            size_bytes += len(chunk)
            out_file.write(chunk)

    audio_filename = f"{file_id}.wav"
    audio_destination = UPLOAD_DIR / audio_filename

    video_clip = None
    try:
        video_clip = VideoFileClip(str(destination))

        if video_clip.audio is None:
            raise HTTPException(
                status_code=422,
                detail="Uploaded video has no audio track to extract.",
            )

        # 44.1kHz / 16-bit PCM: uncompressed, high-quality audio for downstream transcription.
        video_clip.audio.write_audiofile(
            str(audio_destination),
            fps=44100,
            codec="pcm_s16le",
            logger=None,
        )
    finally:
        # Explicitly release the ffmpeg subprocess/file handles moviepy opens,
        # otherwise repeated uploads leak processes and can hang on macOS.
        if video_clip is not None:
            video_clip.close()

    # Built before the (expensive) analysis so a bad calibration fails fast.
    # NOTE: this only *builds and reports* the calibration -- it deliberately
    # does not influence note detection, filtering, or confidences in any way.
    # Applying it to the notes (proximity vetoing) is a separate decision.
    calibration_summary = None
    if calibration is not None:
        spec = parse_calibration_payload(calibration, destination)
        try:
            keyboard_calibration = build_keyboard_calibration(
                spec["leftmost_note"],
                spec["rightmost_note"],
                spec["corners"]["front_left"],
                spec["corners"]["front_right"],
                spec["corners"]["back_left"],
                spec["corners"]["back_right"],
            )
        except (ValueError, cv2.error) as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not build keyboard calibration: {exc}"
            ) from exc
        calibration_summary = build_calibration_summary(
            keyboard_calibration,
            spec["leftmost_note"],
            spec["rightmost_note"],
            _video_frame_size(destination),
        )

    audio_analysis = analyze_audio(
        audio_destination,
        destination,
        time_signature=time_signature,
        tempo_bpm_override=tempo_bpm,
    )

    pdf_filename = f"{file_id}.pdf"
    musicxml_filename = f"{file_id}.musicxml"
    generate_notation_pdf(
        audio_analysis["events"],
        UPLOAD_DIR / pdf_filename,
        time_signature=parse_time_signature(audio_analysis["time_signature"]),
        key_signature=resolved_key,
    )
    generate_placeholder_musicxml(UPLOAD_DIR / musicxml_filename)

    response = {
        "status": "success",
        "message": "File ingested and audio extracted successfully.",
        "original_filename": file.filename,
        "saved_as": saved_filename,
        "audio_filename": audio_filename,
        "size_bytes": size_bytes,
        "content_type": file.content_type,
        "pdf_url": f"{BASE_URL}/api/uploads/{pdf_filename}",
        "musicxml_url": f"{BASE_URL}/api/uploads/{musicxml_filename}",
        **audio_analysis,
        "key_signature": {
            "tonic": resolved_key[0],
            "mode": resolved_key[1],
            "accidentals": KEY_SIGNATURE_ACCIDENTALS[resolved_key],
        },
    }
    # Added only when calibration was supplied, so the no-calibration response
    # keeps exactly the shape it has today rather than gaining a null field.
    if calibration_summary is not None:
        response["calibration"] = calibration_summary
    return response
