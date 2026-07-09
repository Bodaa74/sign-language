export default function LetterDisplay({ letter, candidateLetter, candidateConfidence }) {
  const hasCandidate = !letter && candidateLetter;

  return (
    <div className="letter-display">
      <label className="section-label">Detected Letter</label>
      <div className={`letter-box ${letter ? "letter-box--active" : ""} ${hasCandidate ? "letter-box--candidate" : ""}`}>
        <span className="letter-char">{letter || candidateLetter || "-"}</span>
        {hasCandidate && (
          <span className="letter-hint">
            stabilizing {Math.round(candidateConfidence * 100)}%
          </span>
        )}
      </div>
    </div>
  );
}
