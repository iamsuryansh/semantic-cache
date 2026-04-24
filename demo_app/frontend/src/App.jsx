import React, { useMemo, useState } from "react";
import { askQuestion, resetSession, startSession } from "./api";

const DEMO_PROMPTS = [
  "What is semantic caching in simple terms?",
  "Can you explain semantic cache in plain language?",
];

function money(value) {
  return `$${Number(value || 0).toFixed(6)}`;
}

function App() {
  const [redisUrl, setRedisUrl] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const canAsk = Boolean(sessionId) && prompt.trim().length > 0 && !loading;

  const summary = useMemo(() => {
    if (!stats) {
      return {
        hitRate: "0.00%",
        actualCost: money(0),
        savedCost: money(0),
        latency: "0.00 ms",
        savedTime: "0.00 ms",
      };
    }

    return {
      hitRate: `${(stats.hit_rate * 100).toFixed(2)}%`,
      actualCost: money(stats.total_actual_cost_usd),
      savedCost: money(stats.estimated_total_saved_cost_usd),
      latency: `${Number(stats.total_latency_ms || 0).toFixed(2)} ms`,
      savedTime: `${Number(stats.estimated_total_saved_time_ms || 0).toFixed(2)} ms`,
    };
  }, [stats]);

  async function handleStart() {
    setLoading(true);
    setError("");
    try {
      const data = await startSession(redisUrl.trim());
      setSessionId(data.session_id);
      setHistory([]);
      setStats(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleAsk(customPrompt) {
    const text = (customPrompt ?? prompt).trim();
    if (!text || !sessionId) {
      return;
    }

    setLoading(true);
    setError("");
    try {
      const data = await askQuestion(sessionId, text);
      setHistory(data.history || []);
      setStats(data.stats || null);
      setPrompt("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleReset() {
    if (!sessionId) {
      return;
    }

    setLoading(true);
    setError("");
    try {
      await resetSession(sessionId);
      setHistory([]);
      setStats(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function runDemoScript() {
    if (!sessionId) {
      setError("Start a session first");
      return;
    }

    for (const item of DEMO_PROMPTS) {
      // Sequential calls keep the demo timeline easy to follow.
      // eslint-disable-next-line no-await-in-loop
      await handleAsk(item);
    }
  }

  return (
    <main className="page">
      <h1>Semantic Cache Demo</h1>
      <p className="subtitle">
        Start session, ask, cache, ask similar again, then observe time and cost savings.
      </p>

      <section className="card">
        <h2>1) Start Session</h2>
        <label htmlFor="redis-url">Redis URL (optional override)</label>
        <input
          id="redis-url"
          placeholder="redis://localhost:6379"
          value={redisUrl}
          onChange={(e) => setRedisUrl(e.target.value)}
          disabled={loading}
        />

        <div className="row">
          <button onClick={handleStart} disabled={loading}>
            {sessionId ? "Restart Session" : "Start Session"}
          </button>
          <button onClick={handleReset} disabled={loading || !sessionId}>
            Reset Session
          </button>
        </div>

        <div className="session">Session ID: {sessionId || "(not started)"}</div>
      </section>

      <section className="card">
        <h2>2) Ask Question</h2>
        <textarea
          rows={4}
          placeholder="Type your question"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          disabled={loading || !sessionId}
        />

        <div className="row">
          <button onClick={() => handleAsk()} disabled={!canAsk}>
            Ask
          </button>
          <button onClick={runDemoScript} disabled={loading || !sessionId}>
            Run Demo (2 Questions)
          </button>
        </div>
      </section>

      <section className="card">
        <h2>3) Savings Summary</h2>
        <div className="grid">
          <div className="stat">
            <span>Hit Rate</span>
            <strong>{summary.hitRate}</strong>
          </div>
          <div className="stat">
            <span>Total Actual Cost</span>
            <strong>{summary.actualCost}</strong>
          </div>
          <div className="stat">
            <span>Estimated Saved Cost</span>
            <strong>{summary.savedCost}</strong>
          </div>
          <div className="stat">
            <span>Total Latency</span>
            <strong>{summary.latency}</strong>
          </div>
          <div className="stat">
            <span>Estimated Saved Time</span>
            <strong>{summary.savedTime}</strong>
          </div>
        </div>
      </section>

      <section className="card">
        <h2>4) Conversation Timeline</h2>
        {history.length === 0 ? (
          <p className="subtitle">No turns yet.</p>
        ) : (
          <ul className="timeline">
            {history.map((turn, idx) => (
              <li key={`${turn.timestamp}-${idx}`}>
                <div className="turn-head">
                  <span className={turn.source === "hit_cache" ? "badge hit" : "badge miss"}>
                    {turn.source === "hit_cache" ? "HIT" : "MISS"}
                  </span>
                  <span>{turn.source}</span>
                  <span>{Number(turn.latency_ms).toFixed(2)} ms</span>
                  <span>{money(turn.actual_cost_usd)}</span>
                </div>
                <p>
                  <strong>Q:</strong> {turn.prompt}
                </p>
                <p>
                  <strong>A:</strong> {turn.response}
                </p>
              </li>
            ))}
          </ul>
        )}
      </section>

      {error ? <div className="error">{error}</div> : null}
    </main>
  );
}

export default App;
