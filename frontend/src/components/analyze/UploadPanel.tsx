import { useCallback, useRef, useState } from "react";

const ACCEPTED_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a"];

export default function UploadPanel({
  file,
  onFileSelected,
}: {
  file: File | null;
  onFileSelected: (file: File) => void;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const f = files[0];
      const ext = "." + f.name.split(".").pop()?.toLowerCase();
      if (!ACCEPTED_EXTENSIONS.includes(ext)) return;
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
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_EXTENSIONS.join(",")}
        className="visually-hidden"
        onChange={(e) => handleFiles(e.target.files)}
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
      <p className="eyebrow">WAV · MP3 · FLAC · M4A</p>

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
