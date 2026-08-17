import { useEffect, useRef } from "react";

/**
 * The AeroSense "living instrument" background.
 *
 * Three quiet layers, all derived from the acoustic-monitoring subject
 * rather than decoration for its own sake:
 *   1. A fine grid that drifts almost imperceptibly, like a scope readout.
 *   2. A single slow sine-ish trace that crosses the canvas — a heartbeat,
 *      not a chart.
 *   3. A soft radial glow that breathes very slowly near the accent color.
 *
 * Respects prefers-reduced-motion by freezing on a single static frame.
 */
export default function LivingBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let width = 0;
    let height = 0;
    let dpr = Math.min(window.devicePixelRatio || 1, 2);

    function resize() {
      const canvasEl = canvasRef.current;
      if (!canvasEl) return;
      width = window.innerWidth;
      height = window.innerHeight;
      canvasEl.width = width * dpr;
      canvasEl.height = height * dpr;
      canvasEl.style.width = `${width}px`;
      canvasEl.style.height = `${height}px`;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    window.addEventListener("resize", resize);

    let raf = 0;

    const gridSize = 42;
    const inkGrid = "rgba(23, 22, 15, 0.035)";
    const traceColor = "rgba(62, 76, 122, 0.28)";
    const glowColor = "rgba(62, 76, 122, 0.10)";

    function drawGrid(offset: number) {
      ctx!.strokeStyle = inkGrid;
      ctx!.lineWidth = 1;
      const offX = offset % gridSize;
      for (let x = -gridSize + offX; x < width + gridSize; x += gridSize) {
        ctx!.beginPath();
        ctx!.moveTo(x, 0);
        ctx!.lineTo(x, height);
        ctx!.stroke();
      }
      for (let y = 0; y < height; y += gridSize) {
        ctx!.beginPath();
        ctx!.moveTo(0, y);
        ctx!.lineTo(width, y);
        ctx!.stroke();
      }
    }

    function drawGlow(time: number) {
      const cx = width * (0.72 + 0.06 * Math.sin(time * 0.00007));
      const cy = height * (0.22 + 0.04 * Math.cos(time * 0.00005));
      const r = Math.max(width, height) * 0.5;
      const gradient = ctx!.createRadialGradient(cx, cy, 0, cx, cy, r);
      gradient.addColorStop(0, glowColor);
      gradient.addColorStop(1, "rgba(62, 76, 122, 0)");
      ctx!.fillStyle = gradient;
      ctx!.fillRect(0, 0, width, height);
    }

    function drawTrace(time: number) {
      const baseline = height * 0.62;
      const amplitude = 18;
      const wavelength = 340;
      const speed = 0.00004;

      ctx!.beginPath();
      ctx!.strokeStyle = traceColor;
      ctx!.lineWidth = 1.4;
      for (let x = 0; x <= width; x += 4) {
        const envelope = Math.sin((x / width) * Math.PI); // fades at edges
        const y =
          baseline +
          Math.sin(x / wavelength + time * speed) *
            amplitude *
            envelope *
            (0.6 + 0.4 * Math.sin(time * 0.00003));
        if (x === 0) ctx!.moveTo(x, y);
        else ctx!.lineTo(x, y);
      }
      ctx!.stroke();
    }

    function frame(time: number) {
      ctx!.clearRect(0, 0, width, height);
      drawGlow(time);
      drawGrid(time * 0.004);
      drawTrace(time);
      if (!reduceMotion) {
        raf = requestAnimationFrame(frame);
      }
    }

    if (reduceMotion) {
      frame(0);
    } else {
      raf = requestAnimationFrame(frame);
    }

    return () => {
      window.removeEventListener("resize", resize);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 0,
        pointerEvents: "none",
      }}
    />
  );
}
