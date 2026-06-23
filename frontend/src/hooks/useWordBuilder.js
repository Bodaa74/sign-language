import { useState, useCallback } from "react";

export function useWordBuilder() {
  const [word, setWord] = useState("");

  const appendLetter = useCallback((letter) => {
    setWord(w => w + letter);
  }, []);

  const deleteLetter = useCallback(() => {
    setWord(w => w.slice(0, -1));
  }, []);

  const clearWord = useCallback(() => setWord(""), []);

  const spaceWord = useCallback(() => {
    setWord(w => w + " ");
  }, []);

  return { word, appendLetter, deleteLetter, clearWord, spaceWord };
}
