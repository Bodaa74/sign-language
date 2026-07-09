export function WordDisplay({ word, sentence, onClearSentence }) {
  return (
    <div className="word-section">
      <div className="word-row">
        <label className="section-label">Current Word</label>
        <div className="word-box">
          <span className="word-text">
            {word || <span className="placeholder">start signing...</span>}
          </span>
          <span className="cursor-blink">|</span>
        </div>
      </div>

      {sentence && (
        <div className="sentence-row">
          <label className="section-label">Sentence</label>
          <div className="sentence-box">
            <p className="sentence-text">{sentence}</p>
            <button className="btn-clear-sentence" onClick={onClearSentence} title="Clear sentence">
              x
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

const COMMON_WORDS = [
  "APPLE","BALL","CAT","DOG","EGG","FISH","GOOD","HELLO","ICE","JAM",
  "KITE","LOVE","MORE","NICE","OPEN","PEN","QUEEN","RED","SUN","TOP",
  "UP","VERY","WATER","XRAY","YELLOW","ZERO","ABLE","ABOVE","AFTER",
  "AGAIN","BACK","BAD","BEST","BIRD","BLACK","BLUE","BOOK","COME",
  "DOWN","EARLY","FACE","FAST","FEEL","FIND","FIRE","FIRST","FOOD",
  "GAME","GIVE","GLAD","GOING","GONE","GREAT","GREEN","GROW","HAND",
  "HARD","HAVE","HEAD","HIGH","HOME","HOW","HURT",
];

export function Suggestions({ word, onSelect }) {
  if (!word) return null;
  const prefix = word.toUpperCase();
  const matches = COMMON_WORDS.filter((w) => w.startsWith(prefix)).slice(0, 4);
  if (!matches.length) return null;

  return (
    <div className="suggestions">
      <label className="section-label">Suggestions</label>
      <div className="suggestion-chips">
        {matches.map((w) => (
          <button key={w} className="suggestion-chip" onClick={() => onSelect(w)}>
            <span className="chip-prefix">{w.slice(0, prefix.length)}</span>
            <span className="chip-rest">{w.slice(prefix.length)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

const STATUS_CONFIG = {
  open: { color: "#00e5a0", label: "Connected" },
  connecting: { color: "#f7c948", label: "Connecting..." },
  closed: { color: "#ff6b6b", label: "Disconnected" },
  error: { color: "#ff6b6b", label: "Error" },
};

export function StatusBar({ wsStatus, isActive }) {
  const cfg = STATUS_CONFIG[wsStatus] || STATUS_CONFIG.closed;
  return (
    <div className="status-bar">
      <span className="status-dot" style={{ backgroundColor: cfg.color }} />
      <span className="status-label" style={{ color: cfg.color }}>{cfg.label}</span>
      {isActive && <span className="status-active">LIVE</span>}
    </div>
  );
}

export function ControlPanel({ isActive, onToggle, onSpace, onBackspace, onClear }) {
  return (
    <div className="control-panel">
      <button className={`btn-toggle ${isActive ? "btn-toggle--stop" : "btn-toggle--start"}`} onClick={onToggle}>
        {isActive ? "STOP" : "START"}
      </button>
      <div className="control-group">
        <button className="btn-ctrl" onClick={onSpace} title="Space (finalize word)">SPACE</button>
        <button className="btn-ctrl" onClick={onBackspace} title="Backspace">DEL</button>
        <button className="btn-ctrl" onClick={onClear} title="Clear word">CLR</button>
      </div>
      <p className="kbd-hint">Keyboard: <kbd>Space</kbd> <kbd>Backspace</kbd> <kbd>Enter</kbd></p>
    </div>
  );
}
