export default function WaveformMark({ size = 22 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <rect x="0.5" y="0.5" width="23" height="23" rx="6" stroke="var(--color-border-strong)" />
      <path
        d="M4 12H6.2L7.6 6.5L10.4 17.5L12.4 9L13.6 15L15.2 8L16.6 14L18 12H20"
        stroke="var(--color-accent)"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}
