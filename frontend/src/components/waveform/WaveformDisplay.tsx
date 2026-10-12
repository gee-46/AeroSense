import { useEffect, useRef, useState } from "react";

export default function WaveformDisplay({ file }: { file: File }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setReady(false);
    setError(null);

    async function render() {
      let ctx: AudioContext | null = null;
      try {
        const arrayBuffer = await file.arrayBuffer();
        const AudioCtx =
          window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        ctx = new AudioCtx();
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
        if (cancelled) return;

        const channelData = audioBuffer.getChannelData(0);
        const canvas = canvasRef.current;
        if (!canvas) return;
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        const c2d = canvas.getContext("2d");
        if (!c2d) return;
        c2d.setTransform(dpr, 0, 0, dpr, 0, 0);
        c2d.clearRect(0, 0, width, height);

        const samplesPerPixel = Math.max(1, Math.floor(channelData.length / width));
        const mid = height / 2;
        const accent = getComputedStyle(document.documentElement).getPropertyValue("--color-accent").trim() || "#3e4c7a";
        c2d.fillStyle = accent;

        for (let x = 0; x < width; x++) {
          let min = 1;
          let max = -1;
          const start = x * samplesPerPixel;
          const end = Math.min(start + samplesPerPixel, channelData.length);
          for (let i = start; i < end; i++) {
            const v = channelData[i];
            if (v < min) min = v;
            if (v > max) max = v;
          }
          const y1 = mid + min * mid * 0.92;
          const y2 = mid + max * mid * 0.92;
          c2d.globalAlpha = 0.85;
          c2d.fillRect(x, Math.min(y1, y2), 1, Math.max(1, Math.abs(y2 - y1)));
        }

        if (!cancelled) setReady(true);
      } catch {
        if (!cancelled) setError("Could not decode audio for waveform preview (the server can still analyze it).");
      } finally {
        // Browsers cap the number of live AudioContexts; always release ours.
        void ctx?.close().catch(() => undefined);
      }
    }

    render();
    return () => {
      cancelled = true;
    };
  }, [file]);

  return (
    <div className="waveform-display">
      <canvas ref={canvasRef} className="waveform-canvas" role="img" aria-label="Waveform preview of the selected recording" />
      {!ready && !error && <div className="waveform-loading eyebrow">decoding waveform…</div>}
      {error && <div className="waveform-error eyebrow">{error}</div>}
      <style>{`
        .waveform-display {
          position: relative;
          height: 96px;
          background: var(--color-canvas-sunken);
          border-radius: var(--radius-md);
          border: 1px solid var(--color-border);
          overflow: hidden;
        }
        .waveform-canvas {
          width: 100%;
          height: 100%;
          display: block;
        }
        .waveform-loading, .waveform-error {
          position: absolute;
          inset: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--color-ink-muted);
          text-align: center;
          padding: 0 16px;
        }
      `}</style>
    </div>
  );
}
