"use client";

import { useCallback, useRef, useState } from "react";

type Status = "idle" | "uploading" | "processing" | "success" | "error";

const ACCEPTED_TYPES = ["video/mp4", "video/quicktime", "video/x-m4v"];
const API_BASE = "http://localhost:8000";

export default function Home() {
  const [status, setStatus] = useState<Status>("idle");
  const [progress, setProgress] = useState(0);
  const [fileName, setFileName] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [savedFilename, setSavedFilename] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const progressIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const resetState = () => {
    setStatus("idle");
    setProgress(0);
    setFileName(null);
    setErrorMessage(null);
    setSavedFilename(null);
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

  if (status === "success" && savedFilename) {
    const videoUrl = `${API_BASE}/api/uploads/${savedFilename}`;

    return (
      <main className="min-h-screen bg-gray-50 px-4 py-8 md:px-8">
        <div className="mb-6 text-center">
          <h1 className="text-3xl font-bold text-gray-900">Piano Transcriber</h1>
          <p className="mt-2 text-gray-500">{fileName}</p>
        </div>

        <div className="mx-auto flex max-w-7xl flex-col gap-6 md:flex-row">
          {/* Left: video player, 60% */}
          <div className="md:w-[60%]">
            <div className="overflow-hidden rounded-2xl bg-black shadow-lg">
              <video
                key={videoUrl}
                src={videoUrl}
                controls
                className="aspect-video w-full"
              />
            </div>
          </div>

          {/* Right: transcription dashboard, 40% */}
          <div className="md:w-[40%]">
            <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
              <h2 className="text-lg font-semibold text-gray-900">
                Piano Transcription Feed
              </h2>
              <p className="mt-1 text-sm text-gray-400">
                Sheet music will appear here once processing begins.
              </p>

              <div className="mt-6 space-y-3">
                <div className="h-4 w-3/4 animate-pulse rounded bg-gray-200" />
                <div className="h-4 w-full animate-pulse rounded bg-gray-200" />
                <div className="h-4 w-5/6 animate-pulse rounded bg-gray-200" />
                <div className="h-32 w-full animate-pulse rounded-lg bg-gray-200" />
                <div className="h-4 w-2/3 animate-pulse rounded bg-gray-200" />
              </div>
            </div>

            <button
              onClick={resetState}
              className="mt-4 w-full rounded-lg border border-gray-300 bg-white py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50"
            >
              Upload a New Video
            </button>
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
