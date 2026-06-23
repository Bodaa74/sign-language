// ─── LetterDisplay ──────────────────────────────────────────────────────────
export default function LetterDisplay({ letter }) {
  return (
    <div className="letter-display">
      <label className="section-label">Detected Letter</label>
      <div className={`letter-box ${letter ? "letter-box--active" : ""}`}>
        <span className="letter-char">{letter || "—"}</span>
      </div>
    </div>
  );
}
