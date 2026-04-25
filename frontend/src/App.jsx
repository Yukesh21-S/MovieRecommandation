import { useMemo, useState } from "react";

const API_BASE = "http://localhost:8000";
const SESSION_KEY = "movie_chatbot_session_id";

export default function App() {
  const [input, setInput] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sessionId, setSessionId] = useState(() => localStorage.getItem(SESSION_KEY) || "");

  const messages = useMemo(() => history, [history]);

  const onSend = async (event) => {
    event.preventDefault();
    const query = input.trim();
    if (!query || loading) return;

    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, session_id: sessionId || null })
      });
      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }
      const data = await response.json();
      setHistory(data.history || []);
      if (data.session_id) {
        setSessionId(data.session_id);
        localStorage.setItem(SESSION_KEY, data.session_id);
      }
      setInput("");
    } catch (err) {
      setError(err.message || "Failed to reach backend.");
    } finally {
      setLoading(false);
    }
  };

  const onReset = async () => {
    setError("");
    try {
      if (sessionId) {
        await fetch(`${API_BASE}/reset`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId })
        });
      }
    } catch {
      // ignore reset errors
    } finally {
      localStorage.removeItem(SESSION_KEY);
      setSessionId("");
      setHistory([]);
      setInput("");
    }
  };

  return (
    <div className="page">
      <div className="card">
        <div className="headerRow">
          <h1>Movie Chatbot</h1>
          <button className="ghost" type="button" onClick={onReset} disabled={loading}>
            New chat
          </button>
        </div>
        <p className="subtitle">Ask for movie search, mood-based recommendations, or follow-up details.</p>

        <div className="chatbox">
          {messages.length === 0 ? (
            <div className="placeholder">Try: latest tamil movies</div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`msg ${msg.role === "user" ? "user" : "bot"}`}>
                <div className="role">{msg.role === "user" ? "You" : "Bot"}</div>
                <div className="content">{msg.content}</div>
              </div>
            ))
          )}
        </div>

        <form className="inputRow" onSubmit={onSend}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            disabled={loading}
          />
          <button type="submit" disabled={loading || !input.trim()}>
            {loading ? "Sending..." : "Send"}
          </button>
        </form>

        {error ? <div className="error">{error}</div> : null}
      </div>
    </div>
  );
}

