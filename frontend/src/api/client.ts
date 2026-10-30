import type { RecoveryPredictionResponse } from '../types/platform';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function detectIncidentMedia(file: File, segmentId: string, fps = 1.5) {
  const form = new FormData();
  form.append('file', file);
  form.append('segment_id', segmentId);
  form.append('frame_sample_fps', String(fps));
  const res = await fetch(`${API_BASE}/incidents/detect`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function predictRecoveryTime(payload: Record<string, unknown>): Promise<RecoveryPredictionResponse> {
  const res = await fetch(`${API_BASE}/recovery/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Prediction request failed' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}
