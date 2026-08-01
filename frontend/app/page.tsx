"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Status = "idle" | "uploading" | "processing" | "success" | "error";

type RawNote = {
  onset: number;
  offset: number;
  note: string;
  midi: number;
  confidence: number;
};

type EventNote = RawNote & {
  fragments_merged: number;
};

type EventData = {
  event_time: number;
  span: number;
  style: string;
  notes: EventNote[];
  dropped_notes: EventNote[];
};

// Present in the response only when calibration was supplied. It is reported
// back purely so the mapping can be verified -- it does not affect detection.
type KeySignatureInfo = {
  tonic: string;
  mode: string;
  accidentals: number;
};

type CalibrationSummary = {
  leftmost_note: string;
  rightmost_note: string;
  frame_width: number | null;
  frame_height: number | null;
  key_count: number;
  white_key_count: number;
};

const ACCEPTED_TYPES = ["video/mp4", "video/quicktime", "video/x-m4v"];
const API_BASE = "http://localhost:8000";

type TimeSignatureMode = "auto" | "specify";
const SIMPLE_METERS = ["4/4", "3/4", "2/4"];
const COMPOUND_METERS = ["6/8", "9/8", "12/8"];

// All 30 standard keys, ordered around the circle of fifths (naturals first,
// then sharps outward, then flats outward) so the dropdown reads the way a
// musician expects rather than alphabetically. `value` is the wire format the
// backend parses; `label` also states the accidental count, because "Gb major"
// alone doesn't tell you it's six flats.
// Note the minor list runs A E B F# C# G# D# A# on the sharp side -- A# minor
// (7 sharps) is the one that's easy to leave out.
type KeyOption = { value: string; label: string };
function keyLabel(tonic: string, mode: string, n: number): string {
  const count = Math.abs(n);
  const kind = n > 0 ? "sharp" : "flat";
  const detail = count === 0 ? "no accidentals" : `${count} ${kind}${count > 1 ? "s" : ""}`;
  return `${tonic} ${mode} (${detail})`;
}
const KEY_SIGNATURE_OPTIONS: KeyOption[] = (
  [
    ["C", "major", 0], ["G", "major", 1], ["D", "major", 2], ["A", "major", 3],
    ["E", "major", 4], ["B", "major", 5], ["F#", "major", 6], ["C#", "major", 7],
    ["F", "major", -1], ["Bb", "major", -2], ["Eb", "major", -3], ["Ab", "major", -4],
    ["Db", "major", -5], ["Gb", "major", -6], ["Cb", "major", -7],
    ["A", "minor", 0], ["E", "minor", 1], ["B", "minor", 2], ["F#", "minor", 3],
    ["C#", "minor", 4], ["G#", "minor", 5], ["D#", "minor", 6], ["A#", "minor", 7],
    ["D", "minor", -1], ["G", "minor", -2], ["C", "minor", -3], ["F", "minor", -4],
    ["Bb", "minor", -5], ["Eb", "minor", -6], ["Ab", "minor", -7],
  ] as [string, string, number][]
).map(([tonic, mode, n]) => ({
  value: `${tonic} ${mode}`,
  label: keyLabel(tonic, mode, n),
}));
const DEFAULT_KEY_SIGNATURE = "C major";

// Standard 88-key range, A0..C8.
const NOTE_LETTERS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
function midiToNoteName(midi: number): string {
  return `${NOTE_LETTERS[midi % 12]}${Math.floor(midi / 12) - 1}`;
}
const PIANO_NOTE_RANGE = Array.from({ length: 108 - 21 + 1 }, (_, i) => midiToNoteName(21 + i));

// 4-corner keybed calibration (Stage 1.5) -- replaces Stage 1's 2-handle
// horizontal-only crop, which assumed a purely overhead camera. Fractions
// are relative to the extracted frame, not raw pixels, so they survive the
// frame being displayed at any size.
type CornerKey = "frontLeft" | "frontRight" | "backLeft" | "backRight";
type Corner = { x: number; y: number };
type Corners = Record<CornerKey, Corner>;

// Sensible default trapezoid for typical bird's-eye framing: the back edge
// (further from the camera) reads narrower and higher in frame than the
// front edge (closer to the camera, where the keys are struck).
const DEFAULT_CORNERS: Corners = {
  frontLeft: { x: 0.15, y: 0.85 },
  frontRight: { x: 0.85, y: 0.85 },
  backLeft: { x: 0.3, y: 0.15 },
  backRight: { x: 0.7, y: 0.15 },
};

const CORNER_LABELS: { key: CornerKey; label: string }[] = [
  { key: "frontLeft", label: "Front-left" },
  { key: "frontRight", label: "Front-right" },
  { key: "backLeft", label: "Back-left" },
  { key: "backRight", label: "Back-right" },
];

// Handles used to be clamped to [0,1], i.e. to the frame itself -- which made
// a correct calibration literally unexpressible for close-up shots, where the
// keybed runs off the edge. Both reference clips hit this: test.wav.mp4's
// front-left corner is at x=-17.6px on a 640px frame, and
// chords-notes-mix.mp4's front-right is at x=259.5px on a 240px frame. So the
// draggable stage extends a margin past every edge and the preview is
// letterboxed inside it. Bounded rather than unbounded so a stray drag can't
// fling a corner somewhere that yields a wildly unstable homography.
const HANDLE_MARGIN = 0.25; // in normalized frame units, per side
const HANDLE_SPAN = 1 + 2 * HANDLE_MARGIN;

// normalized frame coordinate -> % position within the padded stage
const toStagePercent = (frameCoord: number) =>
  ((frameCoord + HANDLE_MARGIN) / HANDLE_SPAN) * 100;

// fraction across the padded stage -> normalized frame coordinate
const toFrameCoord = (stageFraction: number) =>
  Math.min(
    1 + HANDLE_MARGIN,
    Math.max(-HANDLE_MARGIN, stageFraction * HANDLE_SPAN - HANDLE_MARGIN)
  );

export default function Home() {
  const [status, setStatus] = useState<Status>("idle");
  const [progress, setProgress] = useState(0);
  const [fileName, setFileName] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [savedFilename, setSavedFilename] = useState<string | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [musicxmlUrl, setMusicxmlUrl] = useState<string | null>(null);
  const [durationSeconds, setDurationSeconds] = useState<number | null>(null);
  const [sampleRate, setSampleRate] = useState<number | null>(null);
  const [tempoBpm, setTempoBpm] = useState<number | null>(null);
  const [rawNotes, setRawNotes] = useState<RawNote[]>([]);
  const [events, setEvents] = useState<EventData[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [timeSignatureMode, setTimeSignatureMode] = useState<TimeSignatureMode>("auto");
  const [simpleMeter, setSimpleMeter] = useState("");
  const [compoundMeter, setCompoundMeter] = useState("");
  const [tempoBpmInput, setTempoBpmInput] = useState("");
  const [keySignature, setKeySignature] = useState(DEFAULT_KEY_SIGNATURE);
  const [hasPickup, setHasPickup] = useState(false);
  const [pickupBeatsInput, setPickupBeatsInput] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewFrameUrl, setPreviewFrameUrl] = useState<string | null>(null);
  const [previewFrameWidth, setPreviewFrameWidth] = useState(0);
  const [previewFrameHeight, setPreviewFrameHeight] = useState(0);
  const [frameExtractionError, setFrameExtractionError] = useState<string | null>(null);
  const [corners, setCorners] = useState<Corners>(DEFAULT_CORNERS);
  const [leftmostNote, setLeftmostNote] = useState("");
  const [rightmostNote, setRightmostNote] = useState("");
  const [draggingCorner, setDraggingCorner] = useState<CornerKey | null>(null);
  const [calibrationSummary, setCalibrationSummary] = useState<CalibrationSummary | null>(null);
  const [resolvedKey, setResolvedKey] = useState<KeySignatureInfo | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const progressIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const cropContainerRef = useRef<HTMLDivElement>(null);

  // Drag handling for the 4 corner handles -- each corner moves
  // independently in both x and y, no cross-corner constraint needed.
  useEffect(() => {
    if (!draggingCorner) return;

    const handleMove = (e: MouseEvent) => {
      const container = cropContainerRef.current;
      if (!container) return;
      // rect is the padded stage, not the frame, so convert through
      // toFrameCoord -- corners stay stored in frame-normalized units.
      const rect = container.getBoundingClientRect();
      const x = toFrameCoord((e.clientX - rect.left) / rect.width);
      const y = toFrameCoord((e.clientY - rect.top) / rect.height);
      setCorners((prev) => ({ ...prev, [draggingCorner]: { x, y } }));
    };
    const handleUp = () => setDraggingCorner(null);

    // Mouse events (not Pointer Events) for broadest compatibility with both
    // real users and automated input synthesis.
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, [draggingCorner]);

  const resetState = () => {
    setStatus("idle");
    setProgress(0);
    setFileName(null);
    setErrorMessage(null);
    setSavedFilename(null);
    setPdfUrl(null);
    setMusicxmlUrl(null);
    setDurationSeconds(null);
    setSampleRate(null);
    setTempoBpm(null);
    setRawNotes([]);
    setEvents([]);
    setSelectedFile(null);
    setPreviewFrameUrl(null);
    setPreviewFrameWidth(0);
    setPreviewFrameHeight(0);
    setFrameExtractionError(null);
    setCorners(DEFAULT_CORNERS);
    setLeftmostNote("");
    setRightmostNote("");
    setCalibrationSummary(null);
  };

  // Extracts a single preview frame from a locally-selected video file,
  // entirely client-side (off-DOM <video> + <canvas>) -- no backend
  // round-trip needed just to show the user their keyboard for calibration.
  const extractPreviewFrame = useCallback((file: File) => {
    setFrameExtractionError(null);
    setPreviewFrameUrl(null);

    const videoUrl = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    video.src = videoUrl;

    const cleanup = () => URL.revokeObjectURL(videoUrl);

    video.addEventListener("loadedmetadata", () => {
      video.currentTime = Math.min(0.1, video.duration / 2);
    });

    video.addEventListener("seeked", () => {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        setFrameExtractionError("Could not extract a preview frame from this video.");
        cleanup();
        return;
      }
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      setPreviewFrameUrl(canvas.toDataURL("image/png"));
      setPreviewFrameWidth(canvas.width);
      setPreviewFrameHeight(canvas.height);
      cleanup();
    });

    video.addEventListener("error", () => {
      setFrameExtractionError("Could not read this video file to extract a preview frame.");
      cleanup();
    });
  }, []);

  const handleFileChosen = useCallback(
    (file: File) => {
      if (!ACCEPTED_TYPES.includes(file.type)) {
        setStatus("error");
        setErrorMessage("Unsupported file type. Please upload an MP4 or MOV video.");
        return;
      }
      setStatus("idle");
      setErrorMessage(null);
      setSelectedFile(file);
      setFileName(file.name);
      setCorners(DEFAULT_CORNERS);
      setLeftmostNote("");
      setRightmostNote("");
      extractPreviewFrame(file);
    },
    [extractPreviewFrame]
  );

  const clearSelectedFile = () => {
    setSelectedFile(null);
    setFileName(null);
    setPreviewFrameUrl(null);
    setPreviewFrameWidth(0);
    setPreviewFrameHeight(0);
    setFrameExtractionError(null);
    setLeftmostNote("");
    setRightmostNote("");
    setCorners(DEFAULT_CORNERS);
  };

  const uploadFile = useCallback(async (file: File) => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      setStatus("error");
      setErrorMessage("Unsupported file type. Please upload an MP4 or MOV video.");
      return;
    }

    setStatus("uploading");
    setFileName(file.name);
    setErrorMessage(null);
    setSavedFilename(null);
    setPdfUrl(null);
    setMusicxmlUrl(null);
    setDurationSeconds(null);
    setSampleRate(null);
    setTempoBpm(null);
    setRawNotes([]);
    setEvents([]);
    setCalibrationSummary(null);
    setProgress(0);

    // Simulated progress while the real upload happens in the background.
    progressIntervalRef.current = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) {
          return prev;
        }
        return prev + Math.random() * 15;
      });
    }, 300);

    try {
      const formData = new FormData();
      formData.append("file", file);

      if (timeSignatureMode === "specify") {
        const chosenMeter = simpleMeter || compoundMeter;
        if (chosenMeter) {
          formData.append("time_signature", chosenMeter);
        }
      }

      const parsedTempo = tempoBpmInput.trim() === "" ? null : Number(tempoBpmInput);
      if (parsedTempo !== null && !Number.isNaN(parsedTempo)) {
        formData.append("tempo_bpm", String(parsedTempo));
      }

      // Always sent, since the dropdown always holds a valid key. The backend
      // treats an absent field as C major anyway, so older clients still work.
      formData.append("key_signature", keySignature);

      // Pickup-measure UI stays interactive, but the backend doesn't support
      // it yet -- has_pickup/pickup_beats are intentionally not sent.

      // Keyboard calibration. Corners go as SOURCE VIDEO PIXELS, never
      // displayed-element pixels: the preview is scaled to fit the card (and
      // now letterboxed inside a padded stage too), so anything measured
      // against the rendered element would be off by the display scale.
      // `corners` is stored normalized to the extracted frame, and
      // previewFrameWidth/Height are that frame's natural dimensions
      // (canvas.width = video.videoWidth), so this multiplication lands in
      // exactly the pixel space OpenCV sees server-side. Values outside
      // [0,width] are expected and correct for keybeds that run off-frame.
      // Sent as one JSON field because calibration is all-or-nothing.
      if (previewFrameWidth > 0 && previewFrameHeight > 0 && leftmostNote && rightmostNote) {
        const toSourcePixels = (corner: Corner): [number, number] => [
          corner.x * previewFrameWidth,
          corner.y * previewFrameHeight,
        ];
        formData.append(
          "calibration",
          JSON.stringify({
            frame_width: previewFrameWidth,
            frame_height: previewFrameHeight,
            leftmost_note: leftmostNote,
            rightmost_note: rightmostNote,
            corners: {
              front_left: toSourcePixels(corners.frontLeft),
              front_right: toSourcePixels(corners.frontRight),
              back_left: toSourcePixels(corners.backLeft),
              back_right: toSourcePixels(corners.backRight),
            },
          })
        );
      }

      const response = await fetch(`${API_BASE}/api/transcribe`, {
        method: "POST",
        body: formData,
      });

      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
      setProgress(100);
      setStatus("processing");

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? "Upload failed.");
      }

      // Brief pause so the "processing" state is visible before showing the result.
      setTimeout(() => {
        setStatus("success");
        setSavedFilename(data.saved_as);
        setPdfUrl(data.pdf_url);
        setMusicxmlUrl(data.musicxml_url);
        setDurationSeconds(data.duration_seconds);
        setSampleRate(data.sample_rate);
        setTempoBpm(data.tempo_bpm);
        setRawNotes(Array.isArray(data.raw_notes) ? data.raw_notes : []);
        setEvents(Array.isArray(data.events) ? data.events : []);
        setCalibrationSummary(data.calibration ?? null);
        setResolvedKey(data.key_signature ?? null);
      }, 600);
    } catch (err) {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
      setStatus("error");
      setErrorMessage(
        err instanceof Error ? err.message : "Something went wrong while uploading."
      );
    }
  }, [
    timeSignatureMode,
    simpleMeter,
    compoundMeter,
    tempoBpmInput,
    keySignature,
    corners,
    previewFrameWidth,
    previewFrameHeight,
    leftmostNote,
    rightmostNote,
  ]);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) {
        handleFileChosen(file);
      }
    },
    [handleFileChosen]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        handleFileChosen(file);
      }
      e.target.value = "";
    },
    [handleFileChosen]
  );

  const isBusy = status === "uploading" || status === "processing";
  const cornerPixels: Record<CornerKey, Corner> | null =
    previewFrameWidth && previewFrameHeight
      ? {
          frontLeft: { x: corners.frontLeft.x * previewFrameWidth, y: corners.frontLeft.y * previewFrameHeight },
          frontRight: { x: corners.frontRight.x * previewFrameWidth, y: corners.frontRight.y * previewFrameHeight },
          backLeft: { x: corners.backLeft.x * previewFrameWidth, y: corners.backLeft.y * previewFrameHeight },
          backRight: { x: corners.backRight.x * previewFrameWidth, y: corners.backRight.y * previewFrameHeight },
        }
      : null;
  const calibrationComplete = Boolean(
    selectedFile &&
      previewFrameUrl &&
      leftmostNote &&
      rightmostNote &&
      PIANO_NOTE_RANGE.indexOf(leftmostNote) < PIANO_NOTE_RANGE.indexOf(rightmostNote)
  );

  if (status === "success" && savedFilename && pdfUrl && musicxmlUrl) {
    const videoUrl = `${API_BASE}/api/uploads/${savedFilename}`;

    return (
      <main className="min-h-screen bg-gray-950 px-4 py-8 text-gray-100 md:px-8">
        <div className="mb-6 text-center">
          <h1 className="text-3xl font-bold text-white">Piano Transcriber</h1>
          <p className="mt-2 text-gray-400">{fileName}</p>
        </div>

        <div className="mx-auto flex max-w-7xl flex-col gap-6 lg:flex-row lg:items-stretch">
          {/* Left column: video + audio properties + downloads, 40% */}
          <div className="flex flex-col gap-6 lg:w-[40%]">
            <div className="overflow-hidden rounded-2xl bg-black shadow-lg shadow-black/40">
              <video key={videoUrl} src={videoUrl} controls className="aspect-video w-full" />
            </div>

            <div className="rounded-2xl border border-gray-800 bg-gray-900 p-6 shadow-lg shadow-black/40">
              <h2 className="text-lg font-semibold text-white">Audio Properties</h2>

              <dl className="mt-4 grid grid-cols-3 gap-3 text-center">
                <div className="rounded-xl bg-gray-800/60 p-3">
                  <dt className="text-xs uppercase tracking-wide text-gray-400">Tempo</dt>
                  <dd className="mt-1 text-xl font-semibold text-indigo-400">
                    {tempoBpm !== null ? tempoBpm.toFixed(1) : "—"}
                  </dd>
                  <dd className="text-xs text-gray-500">BPM</dd>
                </div>
                <div className="rounded-xl bg-gray-800/60 p-3">
                  <dt className="text-xs uppercase tracking-wide text-gray-400">Duration</dt>
                  <dd className="mt-1 text-xl font-semibold text-indigo-400">
                    {durationSeconds !== null ? durationSeconds.toFixed(2) : "—"}
                  </dd>
                  <dd className="text-xs text-gray-500">seconds</dd>
                </div>
                <div className="rounded-xl bg-gray-800/60 p-3">
                  <dt className="text-xs uppercase tracking-wide text-gray-400">Sample Rate</dt>
                  <dd className="mt-1 text-xl font-semibold text-indigo-400">
                    {sampleRate !== null ? (sampleRate / 1000).toFixed(1) : "—"}
                  </dd>
                  <dd className="text-xs text-gray-500">kHz</dd>
                </div>
              </dl>

              {resolvedKey && (
                <p className="mt-3 text-center text-sm text-gray-400">
                  Key signature{" "}
                  <span className="font-semibold text-indigo-400">
                    {resolvedKey.tonic} {resolvedKey.mode}
                  </span>{" "}
                  <span className="text-xs text-gray-500">
                    (
                    {resolvedKey.accidentals === 0
                      ? "no accidentals"
                      : `${Math.abs(resolvedKey.accidentals)} ${
                          resolvedKey.accidentals > 0 ? "sharp" : "flat"
                        }${Math.abs(resolvedKey.accidentals) > 1 ? "s" : ""}`}
                    )
                  </span>
                </p>
              )}

              <div className="mt-6 flex flex-col gap-3">
                <a
                  href={pdfUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  download
                  className="flex items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-3 text-sm font-semibold text-white shadow-md shadow-indigo-950/50 transition-colors hover:bg-indigo-500"
                >
                  Download Printable PDF
                </a>
                <a
                  href={musicxmlUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  download
                  className="flex items-center justify-center gap-2 rounded-xl border border-gray-700 bg-gray-800 px-4 py-3 text-sm font-semibold text-gray-100 transition-colors hover:bg-gray-700"
                >
                  Download MusicXML Data
                </a>
              </div>
            </div>

            {calibrationSummary && (
              <div className="rounded-2xl border border-gray-800 bg-gray-900 p-6 shadow-lg shadow-black/40">
                <h2 className="text-lg font-semibold text-white">Keyboard Calibration</h2>
                <p className="mt-1 text-sm text-gray-400">
                  Accepted by the backend and reported back for verification. It does not
                  affect note detection.
                </p>
                <dl className="mt-4 space-y-1 text-sm">
                  <div className="flex justify-between rounded-lg bg-gray-800/60 px-3 py-2">
                    <dt className="text-gray-400">Calibrated range</dt>
                    <dd className="font-medium text-indigo-400">
                      {calibrationSummary.leftmost_note} – {calibrationSummary.rightmost_note}
                    </dd>
                  </div>
                  <div className="flex justify-between rounded-lg bg-gray-800/60 px-3 py-2">
                    <dt className="text-gray-400">Keys mapped</dt>
                    <dd className="font-medium text-indigo-400">
                      {calibrationSummary.key_count} ({calibrationSummary.white_key_count} white)
                    </dd>
                  </div>
                  <div className="flex justify-between rounded-lg bg-gray-800/60 px-3 py-2">
                    <dt className="text-gray-400">Frame</dt>
                    <dd className="font-medium text-indigo-400">
                      {calibrationSummary.frame_width}×{calibrationSummary.frame_height}px
                    </dd>
                  </div>
                </dl>
              </div>
            )}

            <div className="rounded-2xl border border-gray-800 bg-gray-900 p-6 shadow-lg shadow-black/40">
              <h2 className="text-lg font-semibold text-white">Detected Events</h2>
              <p className="mt-1 text-sm text-gray-400">
                Notes grouped into events, filtered by confidence relative to each events
                strongest note.
              </p>

              <ul className="mt-4 max-h-64 space-y-1 overflow-y-auto pr-1 text-sm">
                {(events ?? []).length === 0 ? (
                  <li className="text-gray-500">No events detected.</li>
                ) : (
                  (events ?? []).map((event, i) => (
                    <li
                      key={i}
                      className="flex items-center justify-between rounded-lg bg-gray-800/60 px-3 py-2"
                    >
                      <span className="text-gray-400">
                        Event {i} ({event.style})
                      </span>
                      <span className="font-medium text-indigo-400">
                        {event.notes.length ? event.notes.map((n) => n.note).join(", ") : "rest"}
                      </span>
                    </li>
                  ))
                )}
              </ul>
            </div>

            <div className="rounded-2xl border border-gray-800 bg-gray-900 p-6 shadow-lg shadow-black/40">
              <h2 className="text-lg font-semibold text-white">Raw Notes</h2>
              <p className="mt-1 text-sm text-gray-400">
                Unfiltered per-note output from the pitch-detection model — no grouping or
                dedup yet.
              </p>

              <ul className="mt-4 max-h-64 space-y-1 overflow-y-auto pr-1 text-sm">
                {(rawNotes ?? []).length === 0 ? (
                  <li className="text-gray-500">No notes detected.</li>
                ) : (
                  (rawNotes ?? []).map((note, i) => (
                    <li
                      key={i}
                      className="flex items-center justify-between rounded-lg bg-gray-800/60 px-3 py-2"
                    >
                      <span className="text-gray-400">{note.onset.toFixed(2)}s</span>
                      <span className="font-medium text-indigo-400">
                        {note.note} ({note.confidence.toFixed(2)})
                      </span>
                    </li>
                  ))
                )}
              </ul>
            </div>

            <button
              onClick={resetState}
              className="rounded-lg border border-gray-700 bg-gray-900 py-2 text-sm font-medium text-gray-300 transition-colors hover:bg-gray-800"
            >
              Upload a New Video
            </button>
          </div>

          {/* Right column: PDF preview, 60% */}
          <div className="flex flex-col lg:w-[60%]">
            <div className="flex flex-1 flex-col rounded-2xl border border-gray-800 bg-gray-900 p-6 shadow-lg shadow-black/40">
              <h2 className="text-lg font-semibold text-white">
                Generated Sheet Music Preview
              </h2>
              <p className="mt-1 text-sm text-gray-400">
                Rendered directly from the generated PDF.
              </p>

              <div className="mt-4 min-h-150 flex-1 overflow-hidden rounded-xl bg-white">
                <iframe
                  src={pdfUrl}
                  title="Generated sheet music PDF preview"
                  className="h-full min-h-150 w-full"
                />
              </div>
            </div>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-50 px-4">
      <div className="w-full max-w-xl">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-gray-900">Piano Transcriber</h1>
          <p className="mt-2 text-gray-500">
            Upload a video of a piano performance to generate sheet music.
          </p>
        </div>

        {!isBusy && (
          <div className="mb-6 rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-gray-900">
              Optional: help us get the rhythm right
            </h2>
            <p className="mt-1 text-xs text-gray-500">
              Auto-detected tempo and time signature can be off, especially on short clips. Fill
              in what you know, or leave everything on auto-detect.
            </p>

            <div className="mt-4">
              <span className="block text-xs font-medium uppercase tracking-wide text-gray-500">
                Time signature
              </span>
              <div className="mt-2 flex gap-4">
                <label className="flex items-center gap-2 text-sm text-gray-700">
                  <input
                    type="radio"
                    name="time-signature-mode"
                    checked={timeSignatureMode === "auto"}
                    onChange={() => {
                      setTimeSignatureMode("auto");
                      setSimpleMeter("");
                      setCompoundMeter("");
                    }}
                  />
                  Auto-detect
                </label>
                <label className="flex items-center gap-2 text-sm text-gray-700">
                  <input
                    type="radio"
                    name="time-signature-mode"
                    checked={timeSignatureMode === "specify"}
                    onChange={() => setTimeSignatureMode("specify")}
                  />
                  I&apos;ll specify
                </label>
              </div>

              {timeSignatureMode === "specify" && (
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs text-gray-500">Simple meters</label>
                    <select
                      value={simpleMeter}
                      onChange={(e) => {
                        setSimpleMeter(e.target.value);
                        setCompoundMeter("");
                      }}
                      className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900"
                    >
                      <option value="">—</option>
                      {SIMPLE_METERS.map((meter) => (
                        <option key={meter} value={meter}>
                          {meter}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500">Compound meters</label>
                    <select
                      value={compoundMeter}
                      onChange={(e) => {
                        setCompoundMeter(e.target.value);
                        setSimpleMeter("");
                      }}
                      className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900"
                    >
                      <option value="">—</option>
                      {COMPOUND_METERS.map((meter) => (
                        <option key={meter} value={meter}>
                          {meter}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              )}
            </div>

            <div className="mt-4">
              <label className="block text-xs font-medium uppercase tracking-wide text-gray-500">
                Key signature
              </label>
              <select
                value={keySignature}
                onChange={(e) => setKeySignature(e.target.value)}
                className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900"
              >
                {KEY_SIGNATURE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <p className="mt-1 text-xs text-gray-500">
                We don&apos;t detect the key — pick it and we&apos;ll spell the notes to
                match (E flat rather than D sharp in flat keys).
              </p>
            </div>

            <div className="mt-4">
              <label className="block text-xs font-medium uppercase tracking-wide text-gray-500">
                Tempo (BPM)
              </label>
              <input
                type="number"
                inputMode="decimal"
                value={tempoBpmInput}
                onChange={(e) => setTempoBpmInput(e.target.value)}
                placeholder="leave blank to auto-detect"
                className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400"
              />
              {compoundMeter !== "" && (
                <p className="mt-1 text-xs text-gray-500">
                  For compound meters, enter the dotted-quarter-note tempo (e.g. the
                  &quot;quarter-note-dot equals X&quot; marking).
                </p>
              )}
            </div>

            <div className="mt-4">
              <span className="block text-xs font-medium uppercase tracking-wide text-gray-500">
                Pickup measure
              </span>
              <div className="mt-2 flex gap-4">
                <label className="flex items-center gap-2 text-sm text-gray-700">
                  <input
                    type="radio"
                    name="has-pickup"
                    checked={!hasPickup}
                    onChange={() => {
                      setHasPickup(false);
                      setPickupBeatsInput("");
                    }}
                  />
                  No
                </label>
                <label className="flex items-center gap-2 text-sm text-gray-700">
                  <input
                    type="radio"
                    name="has-pickup"
                    checked={hasPickup}
                    onChange={() => setHasPickup(true)}
                  />
                  Yes
                </label>
              </div>

              {hasPickup && (
                <div className="mt-3">
                  <label className="block text-xs text-gray-500">
                    How many beats is the pickup?
                  </label>
                  <input
                    type="number"
                    inputMode="decimal"
                    value={pickupBeatsInput}
                    onChange={(e) => setPickupBeatsInput(e.target.value)}
                    placeholder="leave blank if unsure"
                    className="mt-1 w-full max-w-40 rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400"
                  />
                </div>
              )}
            </div>
          </div>
        )}

        {isBusy && (
          <div className="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-gray-200 bg-gray-100 p-12 text-center">
            <div className="w-full max-w-xs">
              <p className="mb-2 truncate text-sm font-medium text-gray-700">{fileName}</p>
              <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
                <div
                  className="h-full rounded-full bg-indigo-500 transition-all duration-300 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="mt-2 text-xs text-gray-500">
                {status === "uploading" ? `Uploading... ${Math.round(progress)}%` : "Processing..."}
              </p>
            </div>
          </div>
        )}

        {!isBusy && (!selectedFile || status === "error") && (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-12 text-center transition-colors cursor-pointer border-gray-300 bg-white hover:border-indigo-400 hover:bg-indigo-50 ${
              isDragging ? "border-indigo-500 bg-indigo-50" : ""
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/quicktime,video/x-m4v"
              className="hidden"
              onChange={handleFileSelect}
            />

            <svg
              className="mb-4 h-12 w-12 text-gray-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>

            {status === "idle" && (
              <>
                <p className="text-sm font-medium text-gray-700">
                  Drag and drop your video here, or click to browse
                </p>
                <p className="mt-1 text-xs text-gray-400">MP4 or MOV, up to your backend&apos;s limit</p>
              </>
            )}

            {status === "error" && (
              <>
                <p className="text-sm font-medium text-red-600">{errorMessage}</p>
                <p className="mt-1 text-xs text-gray-400">Click to try again</p>
              </>
            )}
          </div>
        )}

        {!isBusy && selectedFile && status !== "error" && (
          <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-gray-900">Mark the keyboard</h2>
              <button
                onClick={clearSelectedFile}
                className="text-xs text-gray-500 underline hover:text-gray-700"
              >
                Choose a different video
              </button>
            </div>
            <p className="mt-1 text-xs text-gray-500">
              Drag the four corner handles onto the keybed&apos;s actual front-left, front-right,
              back-left, and back-right corners, then tell us which notes are visible there.
              If the keyboard runs past the edge of the video, drag the handles out into the
              grey margin — corners outside the frame are expected and are sent correctly.
            </p>

            {frameExtractionError && (
              <p className="mt-3 text-sm text-red-600">{frameExtractionError}</p>
            )}

            {previewFrameUrl && previewFrameWidth > 0 && previewFrameHeight > 0 && (
              <div
                ref={cropContainerRef}
                className="relative mt-4 w-full touch-none select-none rounded-xl bg-gray-100 ring-1 ring-gray-200 ring-inset"
                // Stage spans HANDLE_SPAN frame-widths by HANDLE_SPAN
                // frame-heights, so its aspect ratio is still the frame's.
                style={{ aspectRatio: `${previewFrameWidth} / ${previewFrameHeight}` }}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={previewFrameUrl}
                  alt="Video preview frame"
                  className="absolute block ring-1 ring-gray-400/60"
                  draggable={false}
                  style={{
                    left: `${toStagePercent(0)}%`,
                    top: `${toStagePercent(0)}%`,
                    width: `${(1 / HANDLE_SPAN) * 100}%`,
                    height: `${(1 / HANDLE_SPAN) * 100}%`,
                  }}
                />
                <svg
                  className="pointer-events-none absolute inset-0 h-full w-full"
                  preserveAspectRatio="none"
                  viewBox="0 0 100 100"
                >
                  <polygon
                    points={`${toStagePercent(corners.frontLeft.x)},${toStagePercent(corners.frontLeft.y)} ${toStagePercent(corners.frontRight.x)},${toStagePercent(corners.frontRight.y)} ${toStagePercent(corners.backRight.x)},${toStagePercent(corners.backRight.y)} ${toStagePercent(corners.backLeft.x)},${toStagePercent(corners.backLeft.y)}`}
                    fill="rgba(99,102,241,0.15)"
                    stroke="rgb(99,102,241)"
                    strokeWidth="0.5"
                    vectorEffect="non-scaling-stroke"
                  />
                </svg>
                {CORNER_LABELS.map(({ key, label }) => (
                  <div
                    key={key}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      setDraggingCorner(key);
                    }}
                    title={label}
                    className="absolute -ml-2 -mt-2 h-4 w-4 cursor-grab rounded-full border-2 border-white bg-indigo-500 shadow active:cursor-grabbing"
                    style={{
                      left: `${toStagePercent(corners[key].x)}%`,
                      top: `${toStagePercent(corners[key].y)}%`,
                    }}
                  />
                ))}
              </div>
            )}

            {!previewFrameUrl && !frameExtractionError && (
              <p className="mt-4 text-sm text-gray-500">Extracting a preview frame…</p>
            )}

            <p className="mt-2 text-xs text-gray-500">
              Front-left:{" "}
              {cornerPixels ? `(${cornerPixels.frontLeft.x.toFixed(1)}, ${cornerPixels.frontLeft.y.toFixed(1)})` : "—"}
              {" · "}Front-right:{" "}
              {cornerPixels ? `(${cornerPixels.frontRight.x.toFixed(1)}, ${cornerPixels.frontRight.y.toFixed(1)})` : "—"}
              {" · "}Back-left:{" "}
              {cornerPixels ? `(${cornerPixels.backLeft.x.toFixed(1)}, ${cornerPixels.backLeft.y.toFixed(1)})` : "—"}
              {" · "}Back-right:{" "}
              {cornerPixels ? `(${cornerPixels.backRight.x.toFixed(1)}, ${cornerPixels.backRight.y.toFixed(1)})` : "—"}
              {previewFrameWidth ? ` (frame ${previewFrameWidth}x${previewFrameHeight}px)` : ""}
            </p>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-gray-500">Leftmost visible note</label>
                <select
                  value={leftmostNote}
                  onChange={(e) => setLeftmostNote(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900"
                >
                  <option value="">Select a note…</option>
                  {PIANO_NOTE_RANGE.map((note) => (
                    <option key={note} value={note}>
                      {note}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-500">Rightmost visible note</label>
                <select
                  value={rightmostNote}
                  onChange={(e) => setRightmostNote(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900"
                >
                  <option value="">Select a note…</option>
                  {PIANO_NOTE_RANGE.map((note) => (
                    <option key={note} value={note}>
                      {note}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <button
              onClick={() => selectedFile && uploadFile(selectedFile)}
              disabled={!calibrationComplete}
              className="mt-4 w-full rounded-xl bg-indigo-600 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-gray-300"
            >
              Transcribe
            </button>
          </div>
        )}

        {status === "error" && (
          <button
            onClick={resetState}
            className="mt-4 w-full rounded-lg border border-gray-300 bg-white py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50"
          >
            Try again
          </button>
        )}
      </div>
    </main>
  );
}
