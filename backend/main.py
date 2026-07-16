import uuid
from pathlib import Path

import cv2
import librosa
import numpy as np
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


NOTE_SLICE_SECONDS = 0.3
GROUPING_WINDOW_SECONDS = 0.15
SOLID_THRESHOLD_SECONDS = 0.03
WINDOW_SAFETY_MARGIN_SECONDS = 0.02  # stop this far before next onset
MIN_WINDOW_SECONDS = 0.10  # floor so windows are never too short for a stable pitch read


def group_onsets_into_events(
    onset_times: np.ndarray,
    grouping_window: float = GROUPING_WINDOW_SECONDS,
    solid_threshold: float = SOLID_THRESHOLD_SECONDS,
) -> list[dict]:
    sorted_onsets = sorted(float(t) for t in onset_times)

    groups: list[list[float]] = []
    current_group: list[float] = []

    for onset in sorted_onsets:
        # Chained distance: compare against the previous onset already in the
        # group, not the group's first onset, so a run of closely-spaced
        # attacks can drift further apart than grouping_window in total.
        if current_group and (onset - current_group[-1]) > grouping_window:
            groups.append(current_group)
            current_group = [onset]
        else:
            current_group.append(onset)

    if current_group:
        groups.append(current_group)

    events = []
    for group in groups:
        span = round(max(group) - min(group), 3)

        if len(group) == 1:
            style = "single"
        elif span <= solid_threshold:
            style = "solid"
        else:
            style = "rolled"

        events.append(
            {
                "event_time": min(group),
                "onset_times": group,
                "span": span,
                "style": style,
            }
        )

    return events


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


CQT_HOP_LENGTH = 512


def detect_chords_cqt(
    waveform: np.ndarray,
    sample_rate: int,
    events: list[dict],
    duration_seconds: float,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    relative_threshold: float = 0.25,
) -> list[list[str]]:
    fmin = librosa.midi_to_hz(36)  # C1

    cqt_magnitude = np.abs(
        librosa.cqt(
            y=waveform,
            sr=sample_rate,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            hop_length=CQT_HOP_LENGTH,
        )
    )
    frame_times = librosa.frames_to_time(
        np.arange(cqt_magnitude.shape[1]), sr=sample_rate, hop_length=CQT_HOP_LENGTH
    )

    detected_chords = []
    for i, event in enumerate(events):
        window_start = event["event_time"]

        # Onset-to-onset windowing: read pitch only up to just before the next
        # event's attack, so its transient never contaminates this chord.
        if i + 1 < len(events):
            window_end = events[i + 1]["event_time"] - WINDOW_SAFETY_MARGIN_SECONDS
        else:
            window_end = duration_seconds

        # Floor: never so short that the pitch read is unstable.
        if window_end - window_start < MIN_WINDOW_SECONDS:
            window_end = window_start + MIN_WINDOW_SECONDS
        # Ceiling: for long gaps, don't average in seconds of decay/silence.
        max_window_end = window_start + event["span"] + NOTE_SLICE_SECONDS + 0.2
        window_end = min(window_end, max_window_end)
        # Never read past the end of the audio.
        window_end = min(window_end, duration_seconds)

        window_indices = np.where(
            (frame_times >= window_start) & (frame_times < window_end)
        )[0]

        if window_indices.size == 0:
            detected_chords.append([])
            continue

        magnitude = cqt_magnitude[:, window_indices].mean(axis=1)
        peak_magnitude = magnitude.max()

        if peak_magnitude <= 0:
            detected_chords.append([])
            continue

        # Iterating bins low-to-high means active_bins comes out pitch-sorted
        # for free (bin index i == MIDI 36 + i, monotonically increasing).
        active_bins = []
        for i in range(n_bins):
            if magnitude[i] <= relative_threshold * peak_magnitude:
                continue
            left = magnitude[i - 1] if i > 0 else -np.inf
            right = magnitude[i + 1] if i < n_bins - 1 else -np.inf
            if magnitude[i] >= left and magnitude[i] >= right:
                active_bins.append(i)

        chord_notes = [librosa.midi_to_note(36 + i) for i in active_bins]
        detected_chords.append(chord_notes)

    return detected_chords


def analyze_audio(audio_path: Path, video_path: Path) -> dict:
    # sr=None preserves the file's native sample rate instead of resampling to 22.05kHz.
    waveform, sample_rate = librosa.load(str(audio_path), sr=None)
    duration_seconds = librosa.get_duration(y=waveform, sr=sample_rate)
    tempo, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate)

    # librosa returns tempo as a 1-element array rather than a bare scalar.
    tempo_bpm = float(np.asarray(tempo).reshape(-1)[0])

    onset_frames = librosa.onset.onset_detect(y=waveform, sr=sample_rate, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)

    events = group_onsets_into_events(onset_times)
    event_times = [event["event_time"] for event in events]
    chord_styles = [event["style"] for event in events]

    detected_onsets = [round(float(t), 2) for t in event_times]
    detected_beats = convert_seconds_to_beats(detected_onsets, tempo_bpm)
    quantized_beats = quantize_beats(detected_beats)
    detected_bars, measure_beats = calculate_bar_structures(quantized_beats)

    total_duration_beats = duration_seconds * (tempo_bpm / 60.0)
    note_durations, note_types = calculate_note_durations(quantized_beats, total_duration_beats)

    detected_chords = detect_chords_cqt(waveform, sample_rate, events, duration_seconds)

    # Sanity-check the audio-to-video frame targeting math against the first
    # few onsets before it's relied on for real multimodal analysis.
    for onset_time in detected_onsets[:3]:
        frame = extract_frame_at_time(video_path, onset_time)
        if frame is not None:
            print(f"[frame check] onset={onset_time}s -> frame shape {frame.shape}")
        else:
            print(f"[frame check] onset={onset_time}s -> FAILED to read frame")

    return {
        "duration_seconds": round(float(duration_seconds), 3),
        "sample_rate": int(sample_rate),
        "tempo_bpm": round(tempo_bpm, 1),
        "detected_onsets": detected_onsets,
        "detected_beats": detected_beats,
        "quantized_beats": quantized_beats,
        "detected_bars": detected_bars,
        "measure_beats": measure_beats,
        "note_durations": note_durations,
        "note_types": note_types,
        "detected_chords": detected_chords,
        "chord_styles": chord_styles,
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
    generate_placeholder_pdf(UPLOAD_DIR / pdf_filename, file.filename or "Piano Transcriber")
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
