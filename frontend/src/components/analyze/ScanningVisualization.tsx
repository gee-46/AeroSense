import { motion } from "framer-motion";

export default function ScanningVisualization() {
  const bars = Array.from({ length: 48 });
  return (
    <div className="scanning">
      <div className="scanning-bars">
        {bars.map((_, i) => (
          <motion.span
            key={i}
            className="scan-bar"
            initial={{ scaleY: 0.15 }}
            animate={{ scaleY: [0.15, 0.9, 0.3, 0.7, 0.15] }}
            transition={{
              duration: 1.4 + (i % 5) * 0.15,
              repeat: Infinity,
              ease: "easeInOut",
              delay: (i % 12) * 0.05,
            }}
          />
        ))}
      </div>
      <motion.div
        className="scan-line"
        initial={{ left: "0%" }}
        animate={{ left: ["0%", "100%"] }}
        transition={{ duration: 2.2, repeat: Infinity, ease: "linear" }}
      />
      <div className="scanning-label eyebrow">running inference — CNN classifier + autoencoder</div>
      <style>{`
        .scanning {
          position: relative;
          height: 160px;
          border-radius: var(--radius-md);
          border: 1px solid var(--color-border);
          background: var(--color-canvas-sunken);
          overflow: hidden;
          display: flex;
          flex-direction: column;
          justify-content: center;
        }
        .scanning-bars {
          display: flex;
          align-items: center;
          gap: 3px;
          height: 90px;
          padding: 0 20px;
        }
        .scan-bar {
          flex: 1;
          height: 100%;
          background: linear-gradient(180deg, var(--color-accent), color-mix(in srgb, var(--color-accent) 30%, transparent));
          border-radius: 2px;
          transform-origin: center;
          opacity: 0.75;
        }
        .scan-line {
          position: absolute;
          top: 0;
          bottom: 24px;
          width: 2px;
          background: linear-gradient(180deg, transparent, var(--color-accent-strong), transparent);
          box-shadow: 0 0 12px var(--color-accent);
        }
        .scanning-label {
          text-align: center;
          margin-top: 10px;
          color: var(--color-accent-strong);
        }
      `}</style>
    </div>
  );
}
