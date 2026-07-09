import { useEffect, useRef, useState, useCallback } from "react";

export function useWebSocket(url, { onMessage } = {}) {
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const [status, setStatus] = useState("closed");   // closed | connecting | open | error

  useEffect(() => {
    if (!url) return;
    let shouldReconnect = true;

    const connect = () => {
      setStatus("connecting");
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => setStatus("open");
      ws.onclose = () => {
        setStatus("closed");
        if (shouldReconnect) {
          reconnectTimerRef.current = setTimeout(connect, 3000);
        }
      };
      ws.onerror = () => setStatus("error");
      ws.onmessage = (e) => onMessage?.(e.data);
    };

    connect();
    return () => {
      shouldReconnect = false;
      clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  const sendMessage = useCallback((msg) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(msg);
    }
  }, []);

  return { sendMessage, status };
}
