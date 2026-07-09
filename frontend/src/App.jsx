import { useState, useRef, useEffect, useCallback } from "react";
import Webcam from "react-webcam";
import { useWebSocket } from "./hooks/useWebSocket";
import { useWordBuilder } from "./hooks/useWordBuilder";
import LetterDisplay from "./components/LetterDisplay";
import WordDisplay from "./components/WordDisplay";
import ConfidenceBar from "./components/ConfidenceBar";
import Suggestions from "./components/Suggestions";
import StatusBar from "./components/StatusBar";
import ControlPanel from "./components/ControlPanel";
import "./index.css";

const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/predict";
const FRAME_INTERVAL_MS = 80;
const STABLE_FRAMES_TO_APPEND = 18;
const MIN_APPEND_CONFIDENCE = 0.65;

export default function App() {
  const webcamRef = useRef(null);
  const intervalRef = useRef(null);
  const waitingForReleaseRef = useRef(false);

  const [isActive, setIsActive] = useState(false);
  const [prediction, setPrediction] = useState({
    letter: "",
    confidence: 0,
    top3: [],
    candidateLetter: "",
    candidateConfidence: 0,
    handDetected: false,
    frameId: 0,
  });
  const [sentence, setSentence] = useState("");

  const { word, appendLetter, deleteLetter, clearWord } = useWordBuilder();

  const resetPrediction = useCallback(() => {
    setPrediction({
      letter: "",
      confidence: 0,
      top3: [],
      candidateLetter: "",
      candidateConfidence: 0,
      handDetected: false,
      frameId: 0,
    });
  }, []);

  const { sendMessage, status: wsStatus } = useWebSocket(WS_URL, {
    onMessage: (data) => {
      let parsed;
      try {
        parsed = JSON.parse(data);
      } catch {
        return;
      }

      if (parsed.error) return;

      if (waitingForReleaseRef.current) {
        if (!parsed.hand_detected) {
          waitingForReleaseRef.current = false;
        }
        resetPrediction();
        return;
      }

      if (parsed.letter !== undefined) {
        setPrediction({
          letter: parsed.letter,
          confidence: parsed.confidence,
          top3: parsed.top3 || [],
          candidateLetter: parsed.candidate_letter || parsed.top3?.[0]?.letter || "",
          candidateConfidence: parsed.candidate_confidence || parsed.top3?.[0]?.confidence || 0,
          handDetected: Boolean(parsed.hand_detected),
          frameId: Date.now(),
        });
      }
    },
  });

  const captureFrame = useCallback(() => {
    if (!webcamRef.current || wsStatus !== "open") return;
    const imageSrc = webcamRef.current.getScreenshot({ width: 640, height: 480 });
    if (!imageSrc) return;
    sendMessage(JSON.stringify({ image: imageSrc }));
  }, [sendMessage, wsStatus]);

  useEffect(() => {
    if (isActive) {
      intervalRef.current = setInterval(captureFrame, FRAME_INTERVAL_MS);
    } else {
      clearInterval(intervalRef.current);
      waitingForReleaseRef.current = false;
      resetPrediction();
    }
    return () => clearInterval(intervalRef.current);
  }, [isActive, captureFrame, resetPrediction]);

  useEffect(() => {
    const handler = (e) => {
      if (e.code === "Space") {
        e.preventDefault();
        if (word) {
          setSentence((s) => s + word + " ");
          clearWord();
        }
      } else if (e.code === "Backspace") {
        deleteLetter();
      } else if (e.code === "Enter") {
        clearWord();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [word, clearWord, deleteLetter]);

  const stableRef = useRef({ letter: "", count: 0 });
  useEffect(() => {
    if (!prediction.handDetected || !prediction.letter || prediction.confidence < MIN_APPEND_CONFIDENCE) {
      stableRef.current = { letter: "", count: 0 };
      return;
    }

    if (prediction.letter === stableRef.current.letter) {
      stableRef.current.count++;
      if (stableRef.current.count === STABLE_FRAMES_TO_APPEND) {
        appendLetter(prediction.letter);
        waitingForReleaseRef.current = true;
        stableRef.current = { letter: "", count: 0 };
        resetPrediction();
      }
    } else {
      stableRef.current = { letter: prediction.letter, count: 0 };
    }
  }, [prediction.letter, prediction.frameId, appendLetter]);

  const finalizeWord = () => {
    if (!word) return;
    setSentence((s) => s + word + " ");
    clearWord();
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo">
          <span className="logo-icon">ASL</span>
          <h1>ASL Translator</h1>
          <span className="logo-sub">A-Z</span>
        </div>
        <StatusBar wsStatus={wsStatus} isActive={isActive} />
      </header>

      <main className="app-main">
        <section className="camera-panel">
          <div className="camera-wrapper">
            <Webcam
              ref={webcamRef}
              mirrored={true}
              screenshotFormat="image/jpeg"
              className="webcam-feed"
              videoConstraints={{ width: 1280, height: 720, facingMode: "user" }}
            />
            {!isActive && (
              <div className="camera-overlay">
                <span className="camera-overlay-icon">CAM</span>
                <p>Click START to begin detection</p>
              </div>
            )}
            {isActive && prediction.letter && (
              <div className="live-badge">
                <span>{prediction.letter}</span>
              </div>
            )}
          </div>

          <ControlPanel
            isActive={isActive}
            onToggle={() => setIsActive((a) => !a)}
            onSpace={finalizeWord}
            onBackspace={deleteLetter}
            onClear={clearWord}
          />
        </section>

        <section className="results-panel">
          <LetterDisplay
            letter={prediction.letter}
            candidateLetter={prediction.candidateLetter}
            candidateConfidence={prediction.candidateConfidence}
          />
          <ConfidenceBar confidence={prediction.confidence} top3={prediction.top3} />
          <WordDisplay word={word} sentence={sentence} onClearSentence={() => setSentence("")} />
          <Suggestions
            word={word}
            onSelect={(w) => {
              setSentence((s) => s + w + " ");
              clearWord();
            }}
          />

          <div className="hand-guide">
            <h3>ASL Reference</h3>
            <div className="alphabet-grid">
              {"ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("").map((l) => (
                <span
                  key={l}
                  className={`alpha-chip ${prediction.letter === l ? "alpha-chip--active" : ""}`}
                >
                  {l}
                </span>
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
