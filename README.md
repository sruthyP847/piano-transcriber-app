# 🎹 Piano Transcriber App

Turn a video of a piano performance into publication-quality sheet music — automatically.

## Overview

Piano Transcriber App takes a **bird's-eye view video** of someone playing piano and generates a **fully fleshed-out, print-ready PDF of the sheet music** — no manual transcription required.

Upload a performance, and the app handles the rest: detecting which keys are played, when, and how, then rendering that into clean, readable musical notation.

## How It Works

1. **Upload** — Provide a top-down video of a piano performance
2. **Detection** — The app analyzes the video to identify key presses, timing, and note duration
3. **Transcription** — Detected notes are converted into structured musical data
4. **Sheet Music Generation** — The app renders a polished, properly formatted PDF of the sheet music

## Tech Stack

- **Backend:** Python
- **Frontend:** TypeScript / JavaScript
- **Styling:** CSS

## Project Structure

```
piano-transcriber-app/
├── backend/     # Video processing, note detection, and transcription logic
├── frontend/    # User interface for uploading videos and viewing/downloading sheet music
└── .gitignore
```

## Getting Started

```bash
# Clone the repo
git clone https://github.com/sruthyP847/piano-transcriber-app.git
cd piano-transcriber-app

# Backend setup
cd backend
# TODO: add install/run instructions

# Frontend setup
cd ../frontend
# TODO: add install/run instructions
```

## Roadmap

- [ ] Improve key-detection accuracy across lighting/camera angles
- [ ] Support for pedal detection (sustain/soft pedal)
- [ ] Multi-hand / chord disambiguation
- [ ] Export options beyond PDF (MusicXML, MIDI)

## License

TODO: add a license (e.g. MIT) if you want this open source.
