import type { ReactNode } from "react";
import LivingBackground from "../background/LivingBackground";
import TopNav from "./TopNav";

export default function PageShell({ children }: { children: ReactNode }) {
  return (
    <div style={{ position: "relative", minHeight: "100%" }}>
      <LivingBackground />
      <div style={{ position: "relative", zIndex: 1 }}>
        <TopNav />
        <main className="content-container">{children}</main>
      </div>
      <style>{`
        .content-container {
          max-width: 1240px;
          margin: 0 auto;
          padding: 0 28px;
        }
        @media (max-width: 700px) {
          .content-container { padding: 0 16px; }
        }
      `}</style>
    </div>
  );
}
