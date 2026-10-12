import { useCallback, useRef, useState } from "react";
import { MAX_UPLOAD_MB } from "../../api/client";

const ACCEPTED_EXTENSIONS = [".wav", ".mp3", ".flac"];

export default function UploadPanel({
  file,
  onFileSelected,
}: {
  file: File | null;
  onFileSelected: (file: File) => void;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [rejection, setRejection] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const f = files[0];
      const ext = "." + f.name.split(".").pop()?.toLowerCase();
      if (!ACCEPTED_EXTENSIONS.includes(ext)) {
        setRejection(`"${f.name}" is not a supported type. Use ${ACCEPTED_EXTENSIONS.join(", ")}.`);
        return;
      }
      if (f.size === 0) {
        setRejection(`"${f.name}" is empty.`);
        return;
      }
      if (f.size > MAX_UPLOAD_MB * 1024 * 1024) {
        setRejection(`"${f.name}" is ${(f.size / 1048576).toFixed(1)} MB; the limit is ${MAX_UPLOAD_MB} MB.`);
        return;
      }
      setRejection(null);
      onFileSelected(f);
    },
    [onFileSelected]
  );

  return (
    <div
      className={`upload-panel${isDragging ? " upload-panel-dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      aria-label="Upload an audio recording"
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_EXTENSIONS.join(",")}
        className="visually-hidden"
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = ""; // allow re-selecting the same file after Clear
        }}
      />
      <svg width="30" height="30" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path
          d="M3 13H5.5L7 6.5L10 18.5L12.5 9L14 15L15.5 8L17 14L18.5 13H21"
          stroke="var(--color-accent)"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <p className="upload-title">{file ? file.name : "Drop a recording here, or click to browse"}</p>
      <p className="eyebrow">WAV · MP3 · FLAC · up to {MAX_UPLOAD_MB} MB</p>
      {rejection && (
        <p role="alert" style={{ color: "var(--color-critical)", fontSize: 12.5 }}>
          {rejection}
        </p>
      )}

      <style>{`
        .upload-panel {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 10px;
          padding: 40px 20px;
          border: 1.5px dashed var(--color-border-strong);
          border-radius: var(--radius-lg);
          background: var(--color-canvas-sunken);
          cursor: pointer;
          transition: border-color 0.15s ease, background 0.15s ease;
        }
        .upload-panel:hover, .upload-panel-dragging {
          border-color: var(--color-accent);
          background: var(--color-accent-soft);
        }
        .upload-title {
          font-size: 14px;
          font-weight: 600;
          color: var(--color-ink);
          text-align: center;
          word-break: break-word;
          max-width: 90%;
        }
      `}</style>
    </div>
  );
}
