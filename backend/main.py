import math
import uuid
from pathlib import Path

import abjad
import cv2
import librosa
import numpy as np
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
from fastapi import FastAPI, HTTPException, UploadFile
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
    beats_per_bar: int = NOTATION_BEATS_PER_BAR,
    sixteenths_per_beat: int = NOTATION_SIXTEENTHS_PER_BEAT,
) -> list[tuple[int, bool]]:
    """Beat-respecting rhythm decomposition, in sixteenth-note integer units.

    Returns [(value_in_sixteenths, is_tied_to_next), ...]. Applies identically
    to notes and rests (rests just ignore the tie flag when rendered).
    """
    if duration <= 0:
        return []

    bar_size = beats_per_bar * sixteenths_per_beat
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


def _build_staff_input(events: list[dict], is_treble: bool, total_sixteenths: int) -> str:
    tokens: list[str] = []
    position = 0

    for event in events:
        absolute_beat = (event["bar"] - 1) * NOTATION_BEATS_PER_BAR + event["beat_in_bar"]
        start = round(absolute_beat * NOTATION_SIXTEENTHS_PER_BEAT)
        # Defensive floor: two very close real onsets can quantize to the same
        # beat, leaving calculate_note_durations to report a zero/negative gap.
        # A note can't be notated with zero duration, so floor it to a
        # sixteenth rather than silently dropping it.
        duration_beats = max(event["duration_beats"], 0.25)
        duration_sixteenths = round(duration_beats * NOTATION_SIXTEENTHS_PER_BEAT)
        end = start + duration_sixteenths

        if start > position:
            for value, _tied in spell_rhythm(position, start - position):
                tokens.append(f"r{NOTATION_VALUE_TO_CODE[value]}")
            position = start
        elif start < position:
            # Quantization collapsed this event earlier than where the previous
            # event already filled to -- nothing sensible to notate, skip it.
            continue

        pitches = [
            note_name_to_lilypond_pitch(note["note"])
            for note in event["notes"]
            if (note_name_to_midi(note["note"]) >= 60) == is_treble
        ]

        pieces = spell_rhythm(position, end - position)
        if pitches:
            # Multiple notes on one staff (e.g. two notes in different octaves
            # both landing treble) render as a normal simultaneous chord.
            pitch_str = pitches[0] if len(pitches) == 1 else "<" + " ".join(pitches) + ">"
            for value, tied in pieces:
                tie = "~" if tied else ""
                tokens.append(f"{pitch_str}{NOTATION_VALUE_TO_CODE[value]}{tie}")
        else:
            for value, _tied in pieces:
                tokens.append(f"r{NOTATION_VALUE_TO_CODE[value]}")
        position = end

    if total_sixteenths > position:
        for value, _tied in spell_rhythm(position, total_sixteenths - position):
            tokens.append(f"r{NOTATION_VALUE_TO_CODE[value]}")
        position = total_sixteenths

    return " ".join(tokens) if tokens else f"r{NOTATION_VALUE_TO_CODE[NOTATION_BAR_SIZE]}"


def generate_notation_pdf(events: list[dict], output_path: Path) -> None:
    if events:
        last_event = events[-1]
        last_start_beats = (last_event["bar"] - 1) * NOTATION_BEATS_PER_BAR + last_event["beat_in_bar"]
        total_beats_needed = last_start_beats + max(last_event["duration_beats"], 0.25)
    else:
        total_beats_needed = 0

    total_sixteenths_needed = total_beats_needed * NOTATION_SIXTEENTHS_PER_BEAT
    total_sixteenths = max(
        NOTATION_BAR_SIZE,
        math.ceil(total_sixteenths_needed / NOTATION_BAR_SIZE) * NOTATION_BAR_SIZE,
    )

    treble_input = _build_staff_input(events, is_treble=True, total_sixteenths=total_sixteenths)
    bass_input = _build_staff_input(events, is_treble=False, total_sixteenths=total_sixteenths)

    treble_staff = abjad.Staff(treble_input, name="Treble")
    bass_staff = abjad.Staff(bass_input, name="Bass")

    abjad.attach(abjad.Clef("treble"), abjad.select.leaves(treble_staff)[0])
    abjad.attach(abjad.TimeSignature((4, 4)), abjad.select.leaves(treble_staff)[0])
    abjad.attach(abjad.Clef("bass"), abjad.select.leaves(bass_staff)[0])
    abjad.attach(abjad.TimeSignature((4, 4)), abjad.select.leaves(bass_staff)[0])

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


def convert_seconds_to_beats(onset_times: list[float], tempo_bpm: float) -> list[float]:
    return [round(timestamp * (tempo_bpm / 60.0), 2) for timestamp in onset_times]


def quantize_beats(detected_beats: list[float], resolution: float = 0.25) -> list[float]:
    return [round(round(beat / resolution) * resolution, 2) for beat in detected_beats]


def calculate_bar_structures(
    quantized_beats: list[float], beats_per_bar: int = 4
) -> tuple[list[int], list[float]]:
    detected_bars = []
    measure_beats = []

    for beat in quantized_beats:
        bar_number = int(beat // beats_per_bar) + 1
        beat_in_bar = round(beat % beats_per_bar, 2)
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
DEDUP_MAX_SPAN_SECONDS = 2.0  # cap on total merged-note duration; the longest genuine single-attack note observed in testing is ~1.65s, so a chain exceeding this is almost certainly separate re-attacks, not one fragmented note
GROUPING_WINDOW_SECONDS = 0.15  # same value as the old onset grouping
SOLID_THRESHOLD_SECONDS = 0.03  # same as before, for style classification
RELATIVE_CONFIDENCE_FRACTION = 0.6  # a note must reach this fraction of its event's strongest note to survive
ABSOLUTE_CONFIDENCE_FLOOR = 0.35  # hard floor below which nothing survives regardless of group


def deduplicate_notes(
    notes: list[dict],
    gap_threshold: float = DEDUP_GAP_SECONDS,
    max_span: float = DEDUP_MAX_SPAN_SECONDS,
) -> list[dict]:
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
            if gap <= gap_threshold and prospective_span <= max_span:
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


def analyze_audio(audio_path: Path, video_path: Path) -> dict:
    # sr=None preserves the file's native sample rate instead of resampling to 22.05kHz.
    waveform, sample_rate = librosa.load(str(audio_path), sr=None)
    duration_seconds = librosa.get_duration(y=waveform, sr=sample_rate)
    tempo, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate)

    # librosa returns tempo as a 1-element array rather than a bare scalar.
    tempo_bpm = float(np.asarray(tempo).reshape(-1)[0])

    notes = detect_notes_basic_pitch(audio_path)
    deduped = deduplicate_notes(notes)
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
    quantized_beats = quantize_beats(detected_beats)
    detected_bars, measure_beats = calculate_bar_structures(quantized_beats)
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
    }


@app.get("/api/hello")
def read_hello():
    return {"message": "Hello World"}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Please upload a video file.",
        )

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

    audio_analysis = analyze_audio(audio_destination, destination)

    pdf_filename = f"{file_id}.pdf"
    musicxml_filename = f"{file_id}.musicxml"
    generate_notation_pdf(audio_analysis["events"], UPLOAD_DIR / pdf_filename)
    generate_placeholder_musicxml(UPLOAD_DIR / musicxml_filename)

    return {
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
    }
