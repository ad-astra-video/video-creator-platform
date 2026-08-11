interface LtxLogoProps {
  className?: string
}

// Video Creator wordmark (branded "VC"). Kept as an SVG with a viewBox so callers'
// sizing classes (h-6 w-auto) scale it like the original LTX wordmark did.
export function LtxLogo({ className = "h-6" }: LtxLogoProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 48 32"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      aria-label="Video Creator"
      role="img"
    >
      <text
        x="2"
        y="26"
        fontFamily="inherit"
        fontSize="26"
        fontWeight="800"
        letterSpacing="-0.5"
        fill="currentColor"
      >
        V
      </text>
      <text
        x="22"
        y="26"
        fontFamily="inherit"
        fontSize="26"
        fontWeight="800"
        letterSpacing="-0.5"
        fill="#3b82f6"
      >
        C
      </text>
    </svg>
  )
}
