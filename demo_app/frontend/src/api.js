const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function parseResponse(response) {
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

export async function startSession(redisUrl) {
  const response = await fetch(`${API_BASE}/session/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ redis_url: redisUrl || null }),
  });
  return parseResponse(response);
}

export async function askQuestion(sessionId, prompt) {
  const response = await fetch(`${API_BASE}/session/${sessionId}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  return parseResponse(response);
}

export async function resetSession(sessionId) {
  const response = await fetch(`${API_BASE}/session/${sessionId}/reset`, {
    method: "POST",
  });
  return parseResponse(response);
}
