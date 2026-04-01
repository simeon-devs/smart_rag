export const API = import.meta.env.VITE_API_URL || 'http://localhost:8001';
export const USER_ID = 'demo_user';

export async function apiExtract(message: string) {
  const r = await fetch(`${API}/extract`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: USER_ID, message }),
  });
  return r.json();
}

export async function apiChat(message: string) {
  const r = await fetch(`${API}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: USER_ID, message }),
  });
  return r.json();
}

export async function apiSaveConstraints(field: string, value: unknown) {
  const r = await fetch(`${API}/constraints`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: USER_ID, [field]: value }),
  });
  return r.json();
}

export async function apiGetMemory() {
  const r = await fetch(`${API}/debug/memory/${USER_ID}`);
  return r.json();
}
