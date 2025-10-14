import { ensureTraceHeaders, TRACEPARENT_HEADER } from '../../../ui/dashboard/src/core/telemetry.js'

export type MarketBar = {
  timestamp: string
  open: number | null
  high: number
  low: number
  close: number
  volume: number | null
  bidVolume?: number | null
  askVolume?: number | null
  signedVolume?: number | null
}

export type FeatureRequest = {
  symbol: string
  bars: MarketBar[]
}

export type PredictionRequest = FeatureRequest & {
  horizon_seconds?: number
}

export type FeatureResponse = {
  symbol: string
  generated_at: string
  features: Record<string, number>
}

export type PredictionResponse = {
  symbol: string
  generated_at: string
  horizon_seconds: number
  score: number
  signal: Record<string, unknown>
}

export type InferenceOptions = {
  baseUrl?: string
  traceparent?: string | null
  etag?: string | null
}

export type InferenceResult<T> = {
  ok: boolean
  status: number
  data?: T
  error?: string
  traceparent: string | null
  etag: string | null
  notModified: boolean
}

const DEFAULT_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000'

function buildUrl(path: string, baseUrl?: string) {
  const url = baseUrl ?? DEFAULT_BASE_URL
  const normalized = url.endsWith('/') ? url.slice(0, -1) : url
  return `${normalized}${path.startsWith('/') ? '' : '/'}${path}`
}

async function parseError(response: Response): Promise<string> {
  try {
    const payload = await response.json()
    if (payload && typeof payload === 'object' && 'error' in payload) {
      const errorObj = payload as { error?: { message?: string; detail?: string } }
      return (
        errorObj.error?.detail ||
        errorObj.error?.message ||
        `Request failed with status ${response.status}`
      )
    }
  } catch (error) {
    // Ignore parsing errors and fall back to generic message below.
  }
  return `Request failed with status ${response.status}`
}

async function execute<T>(
  path: string,
  body: unknown,
  options: InferenceOptions = {},
): Promise<InferenceResult<T>> {
  const requestInit: RequestInit = ensureTraceHeaders(
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(options.etag ? { 'If-None-Match': options.etag } : {}),
      },
      body: JSON.stringify(body),
    },
    options.traceparent ?? undefined,
  )

  try {
    const response = await fetch(buildUrl(path, options.baseUrl), requestInit)
    const traceparent =
      response.headers.get(TRACEPARENT_HEADER) ||
      ((requestInit.headers as Record<string, string> | undefined)?.[TRACEPARENT_HEADER] ?? null)
    const etag = response.headers.get('etag')

    if (response.status === 304) {
      return {
        ok: true,
        status: response.status,
        traceparent,
        etag,
        notModified: true,
      }
    }

    if (!response.ok) {
      const errorMessage = await parseError(response)
      return {
        ok: false,
        status: response.status,
        error: errorMessage,
        traceparent,
        etag,
        notModified: false,
      }
    }

    const data = (await response.json()) as T
    return {
      ok: true,
      status: response.status,
      data,
      traceparent,
      etag,
      notModified: false,
    }
  } catch (error) {
    const message =
      error instanceof Error ? error.message : 'Unknown error while performing inference request'
    return {
      ok: false,
      status: 0,
      error: message,
      traceparent: null,
      etag: null,
      notModified: false,
    }
  }
}

export async function postFeatures(
  payload: FeatureRequest,
  options?: InferenceOptions,
): Promise<InferenceResult<FeatureResponse>> {
  return execute<FeatureResponse>('/features', payload, options)
}

export async function postPredictions(
  payload: PredictionRequest,
  options?: InferenceOptions,
): Promise<InferenceResult<PredictionResponse>> {
  return execute<PredictionResponse>('/predictions', payload, options)
}
