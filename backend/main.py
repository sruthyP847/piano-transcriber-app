import uuid
from pathlib import Path

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


def analyze_audio(audio_path: Path) -> dict:
    # sr=None preserves the file's native sample rate instead of resampling to 22.05kHz.
    waveform, sample_rate = librosa.load(str(audio_path), sr=None)
    duration_seconds = librosa.get_duration(y=waveform, sr=sample_rate)
    tempo, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate)

    # librosa returns tempo as a 1-element array rather than a bare scalar.
    tempo_bpm = float(np.asarray(tempo).reshape(-1)[0])

    return {
        "duration_seconds": round(float(duration_seconds), 3),
        "sample_rate": int(sample_rate),
        "tempo_bpm": round(tempo_bpm, 1),
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

    audio_analysis = analyze_audio(audio_destination)

    return {
        "status": "success",
        "message": "File ingested and audio extracted successfully.",
        "original_filename": file.filename,
        "saved_as": saved_filename,
        "audio_filename": audio_filename,
        "size_bytes": size_bytes,
        "content_type": file.content_type,
        **audio_analysis,
    }
