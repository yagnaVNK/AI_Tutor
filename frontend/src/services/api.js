import { API_URL } from "../config";
import { getIdToken } from "./firebase";

async function request(path, { method = "GET", body, isForm = false } = {}) {
  const token = await getIdToken();
  const headers = { Authorization: `Bearer ${token}` };
  if (!isForm && body !== undefined) headers["Content-Type"] = "application/json";

  const res = await fetch(`${API_URL}${path}`, {
    method,
    headers,
    body: isForm ? body : body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${method} ${path} failed (${res.status}): ${text}`);
  }
  if (res.status === 204) return null;
  return res.json();
}

export const api = {
  getMe: () => request("/api/me"),
  updateMe: (data) => request("/api/me", { method: "PATCH", body: data }),

  listConversations: () => request("/api/conversations/"),
  getConversation: (id) => request(`/api/conversations/${id}/`),
  createConversation: (data = {}) =>
    request("/api/conversations/", { method: "POST", body: data }),
  deleteConversation: (id) =>
    request(`/api/conversations/${id}/`, { method: "DELETE" }),

  sendChat: ({ message, conversationId, fileIds }) =>
    request("/api/chat", {
      method: "POST",
      body: {
        message,
        conversation_id: conversationId || null,
        file_ids: fileIds || [],
      },
    }),

  listFiles: () => request("/api/files/"),
  uploadFile: (file, conversationId = null) => {
    const form = new FormData();
    form.append("file", file);
    if (conversationId) form.append("conversation", conversationId);
    return request("/api/files/", { method: "POST", body: form, isForm: true });
  },
};
