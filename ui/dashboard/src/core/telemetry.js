import { randomBytes } from 'crypto';

const TRACE_HEADER = 'traceparent';

function randomHex(bytes) {
  return randomBytes(bytes).toString('hex');
}

export function createTraceparent(previous) {
  if (typeof previous === 'string' && previous.trim() !== '') {
    return previous.trim();
  }
  const traceId = randomHex(16);
  const spanId = randomHex(8);
  return `00-${traceId}-${spanId}-01`;
}

export function ensureTraceHeaders(init = {}, traceparent) {
  const headers = { ...(init.headers || {}) };
  const next = createTraceparent(traceparent || headers[TRACE_HEADER]);
  headers[TRACE_HEADER] = next;
  return { ...init, headers };
}

export function extractTraceparent(headers = {}) {
  return headers[TRACE_HEADER] || null;
}

export const TRACEPARENT_HEADER = TRACE_HEADER;
