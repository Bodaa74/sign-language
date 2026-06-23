export default function ConfidenceBar({ confidence, top3 }) {
  const pct   = Math.round(confidence * 100);
  const color = confidence > 0.85 ? "#00e5a0" : confidence > 0.60 ? "#f7c948" : "#ff6b6b";

  return (
    <div className="confidence-section">
      <label className="section-label">Confidence</label>
      <div className="conf-bar-track">
        <div
          className="conf-bar-fill"
          style={{ width: `${pct}%`, background: color }}
        />
        <span className="conf-pct">{pct}%</span>
      </div>

      {top3?.length > 0 && (
        <div className="top3">
          {top3.map(({ letter, confidence: c }) => (
            <div className="top3-item" key={letter}>
              <span className="top3-letter">{letter}</span>
              <div className="top3-bar-track">
                <div
                  className="top3-bar-fill"
                  style={{ width: `${Math.round(c * 100)}%` }}
                />
              </div>
              <span className="top3-pct">{Math.round(c * 100)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
