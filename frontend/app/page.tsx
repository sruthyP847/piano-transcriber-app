"use client";

import { useCallback, useRef, useState } from "react";

type Status = "idle" | "uploading" | "processing" | "success" | "error";

type RawNote = {
  onset: number;
  offset: number;
  note: string;
  midi: number;
  confidence: number;
};

const ACCEPTED_TYPES = ["video/mp4", "video/quicktime", "video/x-m4v"];
const API_BASE = "http://localhost:8000";

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
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const progressIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

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
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) {
        uploadFile(file);
      }
    },
    [uploadFile]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        uploadFile(file);
      }
      e.target.value = "";
    },
    [uploadFile]
  );

  const isBusy = status === "uploading" || status === "processing";

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

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => !isBusy && fileInputRef.current?.click()}
          className={`flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-12 text-center transition-colors ${
            isBusy
              ? "cursor-not-allowed border-gray-200 bg-gray-100"
              : "cursor-pointer border-gray-300 bg-white hover:border-indigo-400 hover:bg-indigo-50"
          } ${isDragging ? "border-indigo-500 bg-indigo-50" : ""}`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/quicktime,video/x-m4v"
            className="hidden"
            onChange={handleFileSelect}
            disabled={isBusy}
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

          {isBusy && (
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
          )}

          {status === "error" && (
            <>
              <p className="text-sm font-medium text-red-600">{errorMessage}</p>
              <p className="mt-1 text-xs text-gray-400">Click to try again</p>
            </>
          )}
        </div>

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
