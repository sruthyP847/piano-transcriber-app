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


PIANO_FMIN = 65  # ~C2
PIANO_FMAX = 2093  # ~C7
NOTE_SLICE_SECONDS = 0.3
YIN_MIN_SAMPLES = 512  # below this, yin has too little signal to estimate a frame


def detect_notes(waveform: np.ndarray, sample_rate: int, onset_times: np.ndarray) -> list[str]:
    slice_samples = int(NOTE_SLICE_SECONDS * sample_rate)
    total_samples = len(waveform)
    notes = []

    for onset_time in onset_times:
        start_sample = int(onset_time * sample_rate)
        end_sample = min(start_sample + slice_samples, total_samples)
        segment = waveform[start_sample:end_sample]

        note = "Rest"
        if len(segment) >= YIN_MIN_SAMPLES:
            f0 = librosa.yin(segment, fmin=PIANO_FMIN, fmax=PIANO_FMAX, sr=sample_rate)
            f0 = f0[~np.isnan(f0)]
            if f0.size > 0:
                median_freq = float(np.median(f0))
                if median_freq > 0:
                    note = librosa.hz_to_note(median_freq)

        notes.append(note)

    return notes


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


PITCH_MAGNITUDE_THRESHOLD = 0.1


def detect_pitches_at_timestamps(y: np.ndarray, sr: int, timestamps: list[float]) -> list[str]:
    detected_pitches = []

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    num_frames = magnitudes.shape[1]

    for timestamp in timestamps:
        frame = int(librosa.time_to_frames(timestamp, sr=sr))
        frame = max(0, min(frame, num_frames - 1))

        idx = magnitudes[:, frame].argmax()
        frequency = pitches[idx, frame]
        magnitude = magnitudes[idx, frame]

        if frequency > 0 and magnitude > PITCH_MAGNITUDE_THRESHOLD:
            note = librosa.hz_to_note(frequency)
        else:
            note = "unknown"

        detected_pitches.append(note)

    return detected_pitches


def analyze_audio(audio_path: Path, video_path: Path) -> dict:
    # sr=None preserves the file's native sample rate instead of resampling to 22.05kHz.
    waveform, sample_rate = librosa.load(str(audio_path), sr=None)
    duration_seconds = librosa.get_duration(y=waveform, sr=sample_rate)
    tempo, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate)

    # librosa returns tempo as a 1-element array rather than a bare scalar.
    tempo_bpm = float(np.asarray(tempo).reshape(-1)[0])

    onset_frames = librosa.onset.onset_detect(y=waveform, sr=sample_rate, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
    detected_onsets = [round(float(t), 2) for t in onset_times]
    detected_beats = convert_seconds_to_beats(detected_onsets, tempo_bpm)
    quantized_beats = quantize_beats(detected_beats)
    detected_bars, measure_beats = calculate_bar_structures(quantized_beats)

    total_duration_beats = duration_seconds * (tempo_bpm / 60.0)
    note_durations, note_types = calculate_note_durations(quantized_beats, total_duration_beats)

    # Note: using detected_onsets (seconds) here, not detected_beats (which are
    # scaled by tempo/60 and would misalign frame lookups for any tempo != 60 BPM).
    detected_pitches = detect_pitches_at_timestamps(waveform, sample_rate, detected_onsets)

    detected_notes = detect_notes(waveform, sample_rate, onset_times)

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
        "detected_pitches": detected_pitches,
        "detected_notes": detected_notes,
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
