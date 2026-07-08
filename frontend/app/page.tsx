"use client";

import { useCallback, useRef, useState } from "react";

type Status = "idle" | "uploading" | "processing" | "success" | "error";

const ACCEPTED_TYPES = ["video/mp4", "video/quicktime", "video/x-m4v"];

export default function Home() {
  const [status, setStatus] = useState<Status>("idle");
  const [progress, setProgress] = useState(0);
  const [fileName, setFileName] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [resultMessage, setResultMessage] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const progressIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const resetState = () => {
    setStatus("idle");
    setProgress(0);
    setFileName(null);
    setErrorMessage(null);
    setResultMessage(null);
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
    setResultMessage(null);
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

      const response = await fetch("http://localhost:8000/api/transcribe", {
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
        setResultMessage(data.message ?? "File ingested successfully.");
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

          {status === "success" && (
            <>
              <p className="text-sm font-medium text-green-600">{resultMessage}</p>
              <p className="mt-1 truncate text-xs text-gray-400">{fileName}</p>
            </>
          )}

          {status === "error" && (
            <>
              <p className="text-sm font-medium text-red-600">{errorMessage}</p>
              <p className="mt-1 text-xs text-gray-400">Click to try again</p>
            </>
          )}
        </div>

        {(status === "success" || status === "error") && (
          <button
            onClick={resetState}
            className="mt-4 w-full rounded-lg border border-gray-300 bg-white py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50"
          >
            Upload another file
          </button>
        )}
      </div>
    </main>
  );
}
