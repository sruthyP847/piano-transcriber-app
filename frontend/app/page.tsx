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

const ACCEPTED_TYPES = ["video/mp4", "video/quicktime", "video/x-m4v"];
const API_BASE = "http://localhost:8000";

type TimeSignatureMode = "auto" | "specify";
const SIMPLE_METERS = ["4/4", "3/4", "2/4"];
const COMPOUND_METERS = ["6/8", "9/8", "12/8"];

// Standard 88-key range, A0..C8.
const NOTE_LETTERS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
function midiToNoteName(midi: number): string {
  return `${NOTE_LETTERS[midi % 12]}${Math.floor(midi / 12) - 1}`;
}
const PIANO_NOTE_RANGE = Array.from({ length: 108 - 21 + 1 }, (_, i) => midiToNoteName(21 + i));

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
  const [hasPickup, setHasPickup] = useState(false);
  const [pickupBeatsInput, setPickupBeatsInput] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewFrameUrl, setPreviewFrameUrl] = useState<string | null>(null);
  const [previewFrameWidth, setPreviewFrameWidth] = useState(0);
  const [frameExtractionError, setFrameExtractionError] = useState<string | null>(null);
  const [leftBoundaryFraction, setLeftBoundaryFraction] = useState(0.1);
  const [rightBoundaryFraction, setRightBoundaryFraction] = useState(0.9);
  const [leftmostNote, setLeftmostNote] = useState("");
  const [rightmostNote, setRightmostNote] = useState("");
  const [draggingHandle, setDraggingHandle] = useState<"left" | "right" | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const progressIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const cropContainerRef = useRef<HTMLDivElement>(null);
  const leftFractionRef = useRef(leftBoundaryFraction);
  const rightFractionRef = useRef(rightBoundaryFraction);

  useEffect(() => {
    leftFractionRef.current = leftBoundaryFraction;
  }, [leftBoundaryFraction]);
  useEffect(() => {
    rightFractionRef.current = rightBoundaryFraction;
  }, [rightBoundaryFraction]);

  // Drag handling for the crop handles -- subscribes once per drag session
  // (reading the other handle's latest position via ref, not state, so the
  // listeners don't need to be torn down and re-added on every pixel of
  // movement).
  useEffect(() => {
    if (!draggingHandle) return;

    const handleMove = (e: MouseEvent) => {
      const container = cropContainerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const fraction = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
      if (draggingHandle === "left") {
        setLeftBoundaryFraction(Math.min(fraction, rightFractionRef.current - 0.02));
      } else {
        setRightBoundaryFraction(Math.max(fraction, leftFractionRef.current + 0.02));
      }
    };
    const handleUp = () => setDraggingHandle(null);

    // Mouse events (not Pointer Events) for broadest compatibility with both
    // real users and automated input synthesis.
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, [draggingHandle]);

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
    setFrameExtractionError(null);
    setLeftBoundaryFraction(0.1);
    setRightBoundaryFraction(0.9);
    setLeftmostNote("");
    setRightmostNote("");
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
      setLeftBoundaryFraction(0.1);
      setRightBoundaryFraction(0.9);
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
    setFrameExtractionError(null);
    setLeftmostNote("");
    setRightmostNote("");
    setLeftBoundaryFraction(0.1);
    setRightBoundaryFraction(0.9);
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

      // Pickup-measure UI stays interactive, but the backend doesn't support
      // it yet -- has_pickup/pickup_beats are intentionally not sent.

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
  }, [timeSignatureMode, simpleMeter, compoundMeter, tempoBpmInput]);

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
  const keyboardPixelLeft = previewFrameWidth ? leftBoundaryFraction * previewFrameWidth : null;
  const keyboardPixelRight = previewFrameWidth ? rightBoundaryFraction * previewFrameWidth : null;
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
              Drag the two handles to the keyboard&apos;s left and right edges, then tell us which
              notes are visible there.
            </p>

            {frameExtractionError && (
              <p className="mt-3 text-sm text-red-600">{frameExtractionError}</p>
            )}

            {previewFrameUrl && (
              <div
                ref={cropContainerRef}
                className="relative mt-4 w-full touch-none select-none overflow-hidden rounded-xl"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={previewFrameUrl}
                  alt="Video preview frame"
                  className="block h-auto w-full"
                  draggable={false}
                />
                <div
                  className="absolute inset-y-0 left-0 bg-black/40"
                  style={{ width: `${leftBoundaryFraction * 100}%` }}
                />
                <div
                  className="absolute inset-y-0 right-0 bg-black/40"
                  style={{ width: `${(1 - rightBoundaryFraction) * 100}%` }}
                />
                <div
                  onMouseDown={(e) => {
                    e.preventDefault();
                    setDraggingHandle("left");
                  }}
                  className="absolute inset-y-0 -ml-1.5 w-3 cursor-ew-resize bg-indigo-500"
                  style={{ left: `${leftBoundaryFraction * 100}%` }}
                />
                <div
                  onMouseDown={(e) => {
                    e.preventDefault();
                    setDraggingHandle("right");
                  }}
                  className="absolute inset-y-0 -ml-1.5 w-3 cursor-ew-resize bg-indigo-500"
                  style={{ left: `${rightBoundaryFraction * 100}%` }}
                />
              </div>
            )}

            {!previewFrameUrl && !frameExtractionError && (
              <p className="mt-4 text-sm text-gray-500">Extracting a preview frame…</p>
            )}

            <p className="mt-2 text-xs text-gray-500">
              Left edge:{" "}
              {keyboardPixelLeft !== null ? `${keyboardPixelLeft.toFixed(1)}px` : "—"} · Right
              edge: {keyboardPixelRight !== null ? `${keyboardPixelRight.toFixed(1)}px` : "—"}
              {previewFrameWidth ? ` (of ${previewFrameWidth}px wide frame)` : ""}
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
