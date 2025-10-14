'use client'

import type { ChangeEvent, CSSProperties } from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  type FeatureResponse,
  type MarketBar,
  type PredictionResponse,
  postFeatures,
  postPredictions,
} from '../lib/inference'

type ScenarioField = 'initialBalance' | 'riskPerTrade' | 'maxPositions' | 'timeframe'

type ScenarioConfig = {
  initialBalance: number
  riskPerTrade: number
  maxPositions: number
  timeframe: string
}

type ScenarioDraft = Record<ScenarioField, string>

type ScenarioTemplate = {
  id: string
  label: string
  description: string
  defaults: ScenarioConfig
  notes: string[]
}

type PreparedSeries = {
  id: string
  label: string
  description: string
  symbol: string
  bars: MarketBar[]
}

type FieldMeta = {
  label: string
  helper: string
  placeholder: string
  inputMode?: 'decimal' | 'numeric' | 'text'
  type?: 'number' | 'text'
}

const FIELD_META: Record<ScenarioField, FieldMeta> = {
  initialBalance: {
    label: 'Initial balance (USD)',
    helper: 'Recommended: ≥ 1,000 USD to produce stable Monte Carlo paths.',
    placeholder: '10000',
    inputMode: 'decimal',
    type: 'number',
  },
  riskPerTrade: {
    label: 'Risk per trade (%)',
    helper: 'Keep between 0.25% and 2% for resilient drawdown control.',
    placeholder: '1',
    inputMode: 'decimal',
    type: 'number',
  },
  maxPositions: {
    label: 'Max concurrent positions',
    helper: 'Use a small integer (1-5) unless you have portfolio hedging.',
    placeholder: '3',
    inputMode: 'numeric',
    type: 'number',
  },
  timeframe: {
    label: 'Execution timeframe',
    helper: 'Format: <number><unit> with unit in s, m, h, d, w (e.g. 1h).',
    placeholder: '1h',
  },
}

const SCENARIO_TEMPLATES: ScenarioTemplate[] = [
  {
    id: 'momentum-breakout',
    label: 'Momentum Breakout',
    description: 'Targets high volume breakouts with moderate exposure.',
    defaults: {
      initialBalance: 15000,
      riskPerTrade: 1,
      maxPositions: 3,
      timeframe: '1h',
    },
    notes: [
      'Requires fast data refresh (≤ 1 minute).',
      'Pair with trailing stops to lock in momentum exhaustion.',
    ],
  },
  {
    id: 'mean-reversion',
    label: 'Mean Reversion Swing',
    description: 'Aims to fade extended moves with conservative sizing.',
    defaults: {
      initialBalance: 10000,
      riskPerTrade: 0.5,
      maxPositions: 2,
      timeframe: '4h',
    },
    notes: [
      'Ensure data set spans multiple regimes to avoid biased reversion.',
      'Layer with volatility filters to avoid trending environments.',
    ],
  },
  {
    id: 'volatility-breakout',
    label: 'Volatility Expansion',
    description: 'Captures volatility squeezes with disciplined portfolio caps.',
    defaults: {
      initialBalance: 25000,
      riskPerTrade: 0.75,
      maxPositions: 4,
      timeframe: '30m',
    },
    notes: [
      'Backtest with intraday transaction costs and slippage.',
      'Consider volatility-adjusted position sizing for calmer sessions.',
    ],
  },
]

function makeBar(
  timestamp: string,
  open: number,
  high: number,
  low: number,
  close: number,
  volume: number,
  bidVolume?: number,
  askVolume?: number,
  signedVolume?: number,
): MarketBar {
  return {
    timestamp,
    open,
    high,
    low,
    close,
    volume,
    bidVolume: bidVolume ?? null,
    askVolume: askVolume ?? null,
    signedVolume: signedVolume ?? null,
  }
}

const PREPARED_SERIES: PreparedSeries[] = [
  {
    id: 'btc-momentum',
    label: 'BTC momentum burst (5m)',
    description: 'Synthetic 5-minute rally capturing a momentum breakout with rising volume.',
    symbol: 'BTC-USD',
    bars: [
      makeBar('2024-05-01T00:00:00.000Z', 62050.1, 62120.4, 61980.2, 62090.9, 142.5, 70.1, 68.4, 11.2),
      makeBar('2024-05-01T00:05:00.000Z', 62090.9, 62210.8, 62040.6, 62195.5, 158.2, 82.1, 70.5, 19.6),
      makeBar('2024-05-01T00:10:00.000Z', 62195.5, 62340.2, 62120.4, 62288.4, 167.9, 88.9, 73.1, 24.3),
      makeBar('2024-05-01T00:15:00.000Z', 62288.4, 62405.6, 62210.8, 62370.1, 174.6, 90.4, 76.3, 25.8),
      makeBar('2024-05-01T00:20:00.000Z', 62370.1, 62580.4, 62340.2, 62510.7, 192.3, 96.6, 81.4, 32.5),
      makeBar('2024-05-01T00:25:00.000Z', 62510.7, 62640.2, 62490.6, 62610.9, 205.8, 101.2, 86.5, 36.8),
    ],
  },
  {
    id: 'eth-range',
    label: 'ETH range equilibrium (15m)',
    description: 'Neutral 15-minute stretch with balanced order book flows for stress testing mean reversion.',
    symbol: 'ETH-USD',
    bars: [
      makeBar('2024-04-15T08:00:00.000Z', 3060.2, 3068.1, 3054.6, 3062.4, 48.2, 21.4, 21.1, 0.7),
      makeBar('2024-04-15T08:15:00.000Z', 3062.4, 3069.8, 3058.2, 3064.9, 44.1, 20.2, 19.4, 0.3),
      makeBar('2024-04-15T08:30:00.000Z', 3064.9, 3072.1, 3059.4, 3061.5, 46.7, 19.6, 20.1, -0.8),
      makeBar('2024-04-15T08:45:00.000Z', 3061.5, 3065.3, 3056.3, 3058.7, 42.8, 18.7, 19.8, -1.4),
      makeBar('2024-04-15T09:00:00.000Z', 3058.7, 3062.4, 3052.2, 3056.8, 47.5, 20.3, 21.1, -1.1),
      makeBar('2024-04-15T09:15:00.000Z', 3056.8, 3063.2, 3051.6, 3060.5, 43.9, 19.1, 20.4, -0.4),
    ],
  },
]

function timeframeToSeconds(value: string): number | null {
  const trimmed = value.trim().toLowerCase()
  const match = trimmed.match(/^(\d+)([smhdw])$/)
  if (!match) {
    return null
  }
  const quantity = Number(match[1])
  if (!Number.isFinite(quantity) || quantity <= 0) {
    return null
  }
  const unit = match[2]
  const multiplier: Record<string, number> = {
    s: 1,
    m: 60,
    h: 3600,
    d: 86400,
    w: 604800,
  }
  return quantity * multiplier[unit]
}

function parseCsvBars(content: string): MarketBar[] {
  const lines = content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)

  if (lines.length === 0) {
    return []
  }

  const header = lines[0]
  const rawHeaders = header.split(',').map((column) => column.trim())
  const columnMap: Partial<Record<'timestamp' | 'open' | 'high' | 'low' | 'close' | 'volume' | 'bidVolume' | 'askVolume' | 'signedVolume', number>> = {}

  rawHeaders.forEach((column, index) => {
    const normalized = column.toLowerCase().replace(/[^a-z]/g, '')
    switch (normalized) {
      case 'timestamp':
      case 'time':
      case 'datetime':
        columnMap.timestamp = index
        break
      case 'open':
        columnMap.open = index
        break
      case 'high':
        columnMap.high = index
        break
      case 'low':
        columnMap.low = index
        break
      case 'close':
        columnMap.close = index
        break
      case 'volume':
        columnMap.volume = index
        break
      case 'bidvolume':
      case 'bidvol':
        columnMap.bidVolume = index
        break
      case 'askvolume':
      case 'askvol':
        columnMap.askVolume = index
        break
      case 'signedvolume':
      case 'signedvol':
      case 'imbalance':
        columnMap.signedVolume = index
        break
      default:
        break
    }
  })

  if (
    columnMap.timestamp === undefined ||
    columnMap.high === undefined ||
    columnMap.low === undefined ||
    columnMap.close === undefined
  ) {
    throw new Error(
      'CSV must include timestamp, high, low and close columns. Optional columns: open, volume, bidVolume, askVolume, signedVolume.',
    )
  }

  const parseOptionalNumber = (value: string | undefined): number | null => {
    if (value === undefined) {
      return null
    }
    const trimmed = value.trim()
    if (trimmed === '') {
      return null
    }
    const parsed = Number(trimmed)
    return Number.isFinite(parsed) ? parsed : null
  }

  const parseRequiredNumber = (value: string | undefined, label: string): number => {
    const parsed = parseOptionalNumber(value)
    if (parsed === null) {
      throw new Error(`Column "${label}" must contain numeric values.`)
    }
    return parsed
  }

  const bars: MarketBar[] = []

  for (let index = 1; index < lines.length; index += 1) {
    const row = lines[index]
    const cells = row.split(',')
    const timestampRaw = cells[columnMap.timestamp]
    if (!timestampRaw || timestampRaw.trim() === '') {
      continue
    }
    const timestamp = new Date(timestampRaw)
    if (Number.isNaN(timestamp.getTime())) {
      throw new Error(`Unable to parse timestamp value "${timestampRaw}" on row ${index + 1}.`)
    }

    const bar: MarketBar = {
      timestamp: timestamp.toISOString(),
      open:
        columnMap.open !== undefined
          ? parseOptionalNumber(cells[columnMap.open])
          : null,
      high: parseRequiredNumber(cells[columnMap.high], 'high'),
      low: parseRequiredNumber(cells[columnMap.low], 'low'),
      close: parseRequiredNumber(cells[columnMap.close], 'close'),
      volume:
        columnMap.volume !== undefined
          ? parseOptionalNumber(cells[columnMap.volume])
          : null,
    }

    if (columnMap.bidVolume !== undefined) {
      bar.bidVolume = parseOptionalNumber(cells[columnMap.bidVolume])
    }
    if (columnMap.askVolume !== undefined) {
      bar.askVolume = parseOptionalNumber(cells[columnMap.askVolume])
    }
    if (columnMap.signedVolume !== undefined) {
      bar.signedVolume = parseOptionalNumber(cells[columnMap.signedVolume])
    }

    bars.push(bar)
  }

  bars.sort((a, b) => (a.timestamp < b.timestamp ? -1 : a.timestamp > b.timestamp ? 1 : 0))
  return bars
}

type FieldErrors = Record<ScenarioField, string | null>

const helperStyle: CSSProperties = {
  color: '#94a3b8',
  fontSize: '0.85rem',
  marginTop: '0.35rem',
}

const errorStyle: CSSProperties = {
  color: '#f97316',
  fontSize: '0.85rem',
  marginTop: '0.35rem',
  fontWeight: 600,
}

const panelStyle: CSSProperties = {
  backgroundColor: '#1e293b',
  borderRadius: '1rem',
  padding: '1.75rem',
  boxShadow: '0 30px 60px -35px rgba(15, 23, 42, 0.7)',
  border: '1px solid rgba(148, 163, 184, 0.12)',
}

function parseNumber(value: string): number {
  const trimmed = value.replace(/,/g, '').trim()
  if (!trimmed) {
    return Number.NaN
  }
  return Number(trimmed)
}

function toDraft(config: ScenarioConfig): ScenarioDraft {
  return {
    initialBalance: config.initialBalance.toString(),
    riskPerTrade: config.riskPerTrade.toString(),
    maxPositions: config.maxPositions.toString(),
    timeframe: config.timeframe,
  }
}

function parseDraft(draft: ScenarioDraft): ScenarioConfig {
  const initialBalance = parseNumber(draft.initialBalance)
  const riskPerTrade = parseNumber(draft.riskPerTrade)
  const maxPositions = parseNumber(draft.maxPositions)
  return {
    initialBalance,
    riskPerTrade,
    maxPositions: Number.isFinite(maxPositions) ? Math.trunc(maxPositions) : Number.NaN,
    timeframe: draft.timeframe.trim(),
  }
}

function validateDraft(draft: ScenarioDraft): FieldErrors {
  const parsed = parseDraft(draft)
  const errors: FieldErrors = {
    initialBalance: null,
    riskPerTrade: null,
    maxPositions: null,
    timeframe: null,
  }

  if (!Number.isFinite(parsed.initialBalance) || parsed.initialBalance <= 0) {
    errors.initialBalance = 'Enter a positive starting balance. Include only digits (no currency symbols).'
  } else if (parsed.initialBalance < 500) {
    errors.initialBalance = 'Balances under 500 USD often create unstable allocations. Consider at least 500+.'
  }

  if (!Number.isFinite(parsed.riskPerTrade) || parsed.riskPerTrade <= 0) {
    errors.riskPerTrade = 'Risk per trade must be a positive percentage (e.g. 0.5 for 0.5%).'
  } else if (parsed.riskPerTrade > 5) {
    errors.riskPerTrade = 'Risk above 5% is rarely survivable. Reduce exposure or split the position.'
  }

  if (!Number.isFinite(parsed.maxPositions) || parsed.maxPositions <= 0) {
    errors.maxPositions = 'Set how many concurrent positions you allow. Use an integer greater than zero.'
  } else if (parsed.maxPositions > 10) {
    errors.maxPositions = 'Managing more than 10 simultaneous trades is error-prone. Tighten the cap.'
  }

  if (!parsed.timeframe) {
    errors.timeframe = 'Provide a timeframe such as 1m, 30m, 1h or 1d.'
  } else if (!/^\d+(s|m|h|d|w)$/i.test(parsed.timeframe)) {
    errors.timeframe = 'Timeframe must match <number><unit> (units: s, m, h, d, w). Example: 4h.'
  }

  return errors
}

function computeWarnings(config: ScenarioConfig): string[] {
  const warnings: string[] = []
  const { initialBalance, riskPerTrade, maxPositions } = config

  if (Number.isFinite(initialBalance) && Number.isFinite(riskPerTrade) && initialBalance > 0) {
    const riskDollars = (initialBalance * riskPerTrade) / 100
    if (riskDollars > initialBalance * 0.03) {
      warnings.push(
        `Each position risks $${riskDollars.toFixed(2)}, which exceeds 3% of equity. Consider reducing risk per trade.`,
      )
    } else if (riskDollars < initialBalance * 0.001) {
      warnings.push(
        `Each position risks only $${riskDollars.toFixed(2)}. Verify commissions do not dominate P&L.`,
      )
    }

    if (Number.isFinite(maxPositions) && maxPositions > 0) {
      const portfolioAtRisk = riskDollars * maxPositions
      if (portfolioAtRisk > initialBalance * 0.2) {
        warnings.push(
          `Simultaneous risk is $${portfolioAtRisk.toFixed(2)} (~${((portfolioAtRisk / initialBalance) * 100).toFixed(
            1,
          )}% of equity). Add position staggering or tighten limits.`,
        )
      }
    }
  }

  return warnings
}

function buildPreview(config: ScenarioConfig) {
  return {
    initialBalance: Number.isFinite(config.initialBalance) ? Number(config.initialBalance.toFixed(2)) : null,
    riskPerTrade: Number.isFinite(config.riskPerTrade) ? Number(config.riskPerTrade.toFixed(2)) : null,
    maxPositions: Number.isFinite(config.maxPositions) ? config.maxPositions : null,
    timeframe: config.timeframe || null,
  }
}

export default function Home() {
  const [templateId, setTemplateId] = useState<string>(SCENARIO_TEMPLATES[0].id)
  const [draft, setDraft] = useState<ScenarioDraft>(() => toDraft(SCENARIO_TEMPLATES[0].defaults))
  const [dataMode, setDataMode] = useState<'prepared' | 'upload'>('prepared')
  const [selectedSeriesId, setSelectedSeriesId] = useState<string>(PREPARED_SERIES[0].id)
  const [inferenceSymbol, setInferenceSymbol] = useState<string>(PREPARED_SERIES[0].symbol)
  const [bars, setBars] = useState<MarketBar[]>(() => PREPARED_SERIES[0].bars.map((bar) => ({ ...bar })))
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null)
  const [csvError, setCsvError] = useState<string | null>(null)
  const [featureSnapshot, setFeatureSnapshot] = useState<FeatureResponse | null>(null)
  const [predictionSnapshot, setPredictionSnapshot] = useState<PredictionResponse | null>(null)
  const [featureEtag, setFeatureEtag] = useState<string | null>(null)
  const [predictionEtag, setPredictionEtag] = useState<string | null>(null)
  const [traceparent, setTraceparent] = useState<string | null>(null)
  const [inferenceError, setInferenceError] = useState<string | null>(null)
  const [isInferLoading, setIsInferLoading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const selectedTemplate = useMemo(
    () => SCENARIO_TEMPLATES.find((entry) => entry.id === templateId) ?? SCENARIO_TEMPLATES[0],
    [templateId],
  )
  const selectedSeries = useMemo(
    () => PREPARED_SERIES.find((entry) => entry.id === selectedSeriesId) ?? PREPARED_SERIES[0],
    [selectedSeriesId],
  )

  const parsedConfig = useMemo(() => parseDraft(draft), [draft])
  const errors = useMemo(() => validateDraft(draft), [draft])
  const hasErrors = useMemo(() => Object.values(errors).some((item) => item !== null), [errors])
  const warnings = useMemo(() => computeWarnings(parsedConfig), [parsedConfig])
  const preview = useMemo(() => JSON.stringify(buildPreview(parsedConfig), null, 2), [parsedConfig])
  const horizonSeconds = useMemo(() => timeframeToSeconds(parsedConfig.timeframe) ?? 300, [parsedConfig.timeframe])
  const barsCount = bars.length
  const featureGeneratedAt = useMemo(
    () => (featureSnapshot ? new Date(featureSnapshot.generated_at).toLocaleString() : null),
    [featureSnapshot],
  )
  const predictionGeneratedAt = useMemo(
    () =>
      predictionSnapshot ? new Date(predictionSnapshot.generated_at).toLocaleString() : null,
    [predictionSnapshot],
  )
  const featureEntries = useMemo(
    () =>
      featureSnapshot
        ? Object.entries(featureSnapshot.features).sort((a, b) => a[0].localeCompare(b[0]))
        : [],
    [featureSnapshot],
  )
  const signalEntries = useMemo(
    () =>
      predictionSnapshot
        ? Object.entries(predictionSnapshot.signal ?? {}).sort((a, b) => a[0].localeCompare(b[0]))
        : [],
    [predictionSnapshot],
  )

  const riskDollars = useMemo(() => {
    if (!Number.isFinite(parsedConfig.initialBalance) || !Number.isFinite(parsedConfig.riskPerTrade)) {
      return null
    }
    return (parsedConfig.initialBalance * parsedConfig.riskPerTrade) / 100
  }, [parsedConfig])

  const aggregateRisk = useMemo(() => {
    if (riskDollars === null || !Number.isFinite(parsedConfig.maxPositions)) {
      return null
    }
    return riskDollars * parsedConfig.maxPositions
  }, [parsedConfig, riskDollars])

  const handleChange = (field: ScenarioField) => (event: ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value
    setDraft((current) => ({ ...current, [field]: value }))
  }

  const resetTemplate = () => {
    setDraft(toDraft(selectedTemplate.defaults))
  }

  useEffect(() => {
    if (dataMode !== 'prepared') {
      return
    }
    setBars(selectedSeries.bars.map((bar) => ({ ...bar })))
    setInferenceSymbol(selectedSeries.symbol)
    setUploadedFileName(null)
    setCsvError(null)
    setFeatureSnapshot(null)
    setPredictionSnapshot(null)
    setFeatureEtag(null)
    setPredictionEtag(null)
  }, [dataMode, selectedSeries])

  const handleSeriesSelect = useCallback((event: ChangeEvent<HTMLSelectElement>) => {
    setSelectedSeriesId(event.target.value)
    setDataMode('prepared')
  }, [])

  const handleSymbolChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setInferenceSymbol(event.target.value)
  }, [])

  const resetFileInput = useCallback(() => {
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }, [])

  const handleFileUpload = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      if (!file) {
        return
      }
      const reader = new FileReader()
      reader.onload = () => {
        try {
          const text = typeof reader.result === 'string' ? reader.result : ''
          const parsedBars = parseCsvBars(text)
          if (parsedBars.length === 0) {
            setCsvError('CSV does not contain any rows after parsing. Provide at least one bar.')
            setBars([])
          } else {
            setBars(parsedBars)
            setCsvError(null)
          }
          setUploadedFileName(file.name)
          setDataMode('upload')
          setFeatureSnapshot(null)
          setPredictionSnapshot(null)
          setFeatureEtag(null)
          setPredictionEtag(null)
        } catch (error) {
          setCsvError(error instanceof Error ? error.message : 'Failed to parse CSV file.')
          setBars([])
          setUploadedFileName(file.name)
          setDataMode('upload')
          setFeatureSnapshot(null)
          setPredictionSnapshot(null)
          setFeatureEtag(null)
          setPredictionEtag(null)
        }
        resetFileInput()
      }
      reader.readAsText(file)
    },
    [resetFileInput],
  )

  const clearUploadedData = useCallback(() => {
    setCsvError(null)
    setUploadedFileName(null)
    setBars([])
    setFeatureSnapshot(null)
    setPredictionSnapshot(null)
    setFeatureEtag(null)
    setPredictionEtag(null)
    resetFileInput()
  }, [resetFileInput])

  const runInference = useCallback(async () => {
    const trimmedSymbol = inferenceSymbol.trim()
    if (trimmedSymbol === '') {
      setInferenceError('Provide a symbol before requesting online inference.')
      return
    }
    if (bars.length === 0) {
      setInferenceError('Load at least one OHLCV bar via a prepared series or CSV upload.')
      return
    }

    setInferenceError(null)
    setIsInferLoading(true)

    const payloadBars = bars.map((bar) => ({
      ...bar,
      timestamp: new Date(bar.timestamp).toISOString(),
    }))

    const basePayload = { symbol: trimmedSymbol, bars: payloadBars }

    try {
      const featureResult = await postFeatures(basePayload, {
        traceparent,
        etag: featureEtag,
      })

      const nextTraceparent = featureResult.traceparent ?? traceparent ?? null
      setTraceparent(nextTraceparent)

      if (!featureResult.ok) {
        setInferenceError(featureResult.error ?? 'Unable to compute features.')
        setIsInferLoading(false)
        return
      }

      if (!featureResult.notModified && featureResult.data) {
        setFeatureSnapshot(featureResult.data)
      }
      if (featureResult.etag) {
        setFeatureEtag(featureResult.etag)
      }

      const predictionResult = await postPredictions(
        { ...basePayload, horizon_seconds: horizonSeconds },
        {
          traceparent: nextTraceparent,
          etag: predictionEtag,
        },
      )

      if (predictionResult.traceparent) {
        setTraceparent(predictionResult.traceparent)
      }

      if (!predictionResult.ok) {
        setInferenceError(predictionResult.error ?? 'Unable to generate prediction.')
        setIsInferLoading(false)
        return
      }

      if (!predictionResult.notModified && predictionResult.data) {
        setPredictionSnapshot(predictionResult.data)
      }
      if (predictionResult.etag) {
        setPredictionEtag(predictionResult.etag)
      }
    } catch (error) {
      setInferenceError(error instanceof Error ? error.message : 'Unexpected inference error.')
    } finally {
      setIsInferLoading(false)
    }
  }, [bars, featureEtag, horizonSeconds, inferenceSymbol, predictionEtag, traceparent])

  const inferenceDisabled = inferenceSymbol.trim() === '' || bars.length === 0 || isInferLoading

  return (
    <main
      style={{
        minHeight: '100vh',
        background: 'radial-gradient(circle at top, #0f172a 0%, #020617 65%)',
        color: '#e2e8f0',
        padding: '2.5rem 1.5rem',
      }}
    >
      <section style={{ maxWidth: '960px', margin: '0 auto', display: 'grid', gap: '2.5rem' }}>
        <header style={{ textAlign: 'left' }}>
          <h1 style={{ fontSize: '2.5rem', fontWeight: 700, marginBottom: '0.75rem' }}>Scenario Studio</h1>
          <p style={{ color: '#cbd5f5', lineHeight: 1.6 }}>
            Sanity-check strategy inputs before pushing them into execution. Select a template, adjust the levers, and review
            automatic hints about risk concentration and timeframe hygiene.
          </p>
        </header>

        <section style={panelStyle}>
          <div style={{ display: 'grid', gap: '1.5rem' }}>
            <div>
              <label htmlFor="template" style={{ display: 'block', fontWeight: 600, marginBottom: '0.5rem' }}>
                Scenario template
              </label>
              <select
                id="template"
                value={templateId}
                onChange={(event) => {
                  const value = event.target.value
                  setTemplateId(value)
                  const template = SCENARIO_TEMPLATES.find((entry) => entry.id === value)
                  if (template) {
                    setDraft(toDraft(template.defaults))
                  }
                }}
                style={{
                  width: '100%',
                  padding: '0.65rem 0.75rem',
                  backgroundColor: '#0f172a',
                  borderRadius: '0.75rem',
                  border: '1px solid rgba(148, 163, 184, 0.3)',
                  color: '#e2e8f0',
                }}
              >
                {SCENARIO_TEMPLATES.map((template) => (
                  <option key={template.id} value={template.id}>
                    {template.label}
                  </option>
                ))}
              </select>
              <p style={{ ...helperStyle, marginTop: '0.6rem' }}>{selectedTemplate.description}</p>
              <ul style={{ marginTop: '0.75rem', paddingLeft: '1.2rem', color: '#cbd5f5', display: 'grid', gap: '0.35rem' }}>
                {selectedTemplate.notes.map((note) => (
                  <li key={note} style={{ fontSize: '0.9rem', lineHeight: 1.4 }}>
                    {note}
                  </li>
                ))}
              </ul>
            </div>

            <form style={{ display: 'grid', gap: '1.4rem' }}>
              {(Object.keys(FIELD_META) as ScenarioField[]).map((field) => {
                const meta = FIELD_META[field]
                const inputId = `field-${field}`
                return (
                  <div key={field} style={{ display: 'grid', gap: '0.35rem' }}>
                    <label htmlFor={inputId} style={{ fontWeight: 600 }}>
                      {meta.label}
                    </label>
                    <input
                      id={inputId}
                      name={field}
                      value={draft[field]}
                      onChange={handleChange(field)}
                      placeholder={meta.placeholder}
                      inputMode={meta.inputMode}
                      type={meta.type}
                      style={{
                        width: '100%',
                        padding: '0.7rem 0.85rem',
                        borderRadius: '0.75rem',
                        border: '1px solid rgba(148, 163, 184, 0.3)',
                        backgroundColor: '#0f172a',
                        color: '#e2e8f0',
                        fontSize: '1rem',
                      }}
                    />
                    <p style={helperStyle}>{meta.helper}</p>
                    {errors[field] ? <p style={errorStyle}>{errors[field]}</p> : null}
                  </div>
                )
              })}

              <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
                <button
                  type="button"
                  onClick={resetTemplate}
                  style={{
                    backgroundColor: '#0f172a',
                    border: '1px solid rgba(148, 163, 184, 0.4)',
                    color: '#e2e8f0',
                    padding: '0.55rem 1rem',
                    borderRadius: '0.65rem',
                    cursor: 'pointer',
                  }}
                >
                  Reset to template defaults
                </button>
              </div>
            </form>
          </div>
        </section>

        <section style={{ display: 'grid', gap: '1.5rem' }}>
          <article style={panelStyle}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '1rem' }}>Risk snapshot</h2>
            <div style={{ display: 'grid', gap: '1.5rem' }}>
              <div style={{ display: 'grid', gap: '0.85rem' }}>
                <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
                  <div>
                    <span style={{ color: '#94a3b8', fontSize: '0.85rem' }}>Risk per trade</span>
                    <p style={{ fontSize: '1.25rem', fontWeight: 600, marginTop: '0.3rem' }}>
                      {riskDollars === null ? '—' : `$${riskDollars.toFixed(2)}`}
                    </p>
                  </div>
                  <div>
                    <span style={{ color: '#94a3b8', fontSize: '0.85rem' }}>Max portfolio risk</span>
                    <p style={{ fontSize: '1.25rem', fontWeight: 600, marginTop: '0.3rem' }}>
                      {aggregateRisk === null ? '—' : `$${aggregateRisk.toFixed(2)}`}
                    </p>
                  </div>
                  <div>
                    <span style={{ color: '#94a3b8', fontSize: '0.85rem' }}>Timeframe</span>
                    <p style={{ fontSize: '1.25rem', fontWeight: 600, marginTop: '0.3rem' }}>
                      {parsedConfig.timeframe || '—'}
                    </p>
                  </div>
                </div>

                {warnings.length > 0 ? (
                  <ul style={{ marginTop: '0.5rem', paddingLeft: '1.2rem', display: 'grid', gap: '0.4rem' }}>
                    {warnings.map((warning) => (
                      <li key={warning} style={{ color: '#facc15', fontSize: '0.95rem', lineHeight: 1.5 }}>
                        {warning}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p style={{ ...helperStyle, marginTop: '0.5rem' }}>
                    Risk controls look balanced for the selected template. Stress test transaction costs before live execution.
                  </p>
                )}

                {hasErrors ? (
                  <p style={{ ...errorStyle, marginTop: '0.75rem' }}>
                    Resolve the highlighted fields above to unlock export-ready scenario JSON.
                  </p>
                ) : null}
              </div>

              <div
                style={{
                  paddingTop: '1.5rem',
                  borderTop: '1px solid rgba(148, 163, 184, 0.2)',
                  display: 'grid',
                  gap: '1.5rem',
                }}
              >
                <div>
                  <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.35rem' }}>
                    Online inference sandbox
                  </h3>
                  <p style={{ ...helperStyle }}>
                    Upload OHLCV bars or pick a prepared series to request engineered features and signal scores from the
                    inference service. Each request automatically propagates <code>traceparent</code> headers and reuses ETags
                    for cache-friendly polling.
                  </p>
                </div>
                <div
                  style={{
                    display: 'grid',
                    gap: '1.25rem',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                  }}
                >
                  <section style={{ display: 'grid', gap: '0.9rem' }}>
                    <div style={{ display: 'grid', gap: '0.4rem' }}>
                      <label htmlFor="inference-symbol" style={{ fontWeight: 600 }}>
                        Symbol
                      </label>
                      <input
                        id="inference-symbol"
                        value={inferenceSymbol}
                        onChange={handleSymbolChange}
                        placeholder="BTC-USD"
                        style={{
                          width: '100%',
                          padding: '0.6rem 0.75rem',
                          borderRadius: '0.65rem',
                          border: '1px solid rgba(148, 163, 184, 0.35)',
                          backgroundColor: '#0f172a',
                          color: '#e2e8f0',
                        }}
                      />
                    </div>

                    <div style={{ display: 'grid', gap: '0.4rem' }}>
                      <label htmlFor="prepared-series" style={{ fontWeight: 600 }}>
                        Prepared bar sequence
                      </label>
                      <select
                        id="prepared-series"
                        value={selectedSeriesId}
                        onChange={handleSeriesSelect}
                        style={{
                          width: '100%',
                          padding: '0.6rem 0.75rem',
                          borderRadius: '0.65rem',
                          border: '1px solid rgba(148, 163, 184, 0.35)',
                          backgroundColor: '#0f172a',
                          color: '#e2e8f0',
                        }}
                      >
                        {PREPARED_SERIES.map((series) => (
                          <option key={series.id} value={series.id}>
                            {series.label}
                          </option>
                        ))}
                      </select>
                      <p style={helperStyle}>{selectedSeries.description}</p>
                    </div>

                    <div style={{ display: 'grid', gap: '0.4rem' }}>
                      <label htmlFor="csv-upload" style={{ fontWeight: 600 }}>
                        Import CSV
                      </label>
                      <input
                        id="csv-upload"
                        ref={fileInputRef}
                        type="file"
                        accept=".csv,text/csv"
                        onChange={handleFileUpload}
                        style={{ color: '#e2e8f0' }}
                      />
                      {uploadedFileName ? (
                        <p style={{ ...helperStyle, color: '#cbd5f5' }}>
                          Loaded file: <span style={{ fontWeight: 600 }}>{uploadedFileName}</span>
                        </p>
                      ) : null}
                      {csvError ? <p style={errorStyle}>{csvError}</p> : null}
                      {!csvError ? (
                        <p style={helperStyle}>
                          Expected headers: timestamp, open, high, low, close, volume (bid/ask/signed volume optional).
                        </p>
                      ) : null}
                      {uploadedFileName ? (
                        <button
                          type="button"
                          onClick={clearUploadedData}
                          style={{
                            width: 'fit-content',
                            backgroundColor: '#0f172a',
                            border: '1px solid rgba(148, 163, 184, 0.35)',
                            color: '#e2e8f0',
                            padding: '0.35rem 0.75rem',
                            borderRadius: '0.55rem',
                            cursor: 'pointer',
                          }}
                        >
                          Clear uploaded data
                        </button>
                      ) : null}
                    </div>

                    <div style={{ display: 'grid', gap: '0.3rem' }}>
                      <p style={{ color: '#cbd5f5', fontWeight: 600 }}>
                        Active data source: {dataMode === 'prepared' ? selectedSeries.label : uploadedFileName || 'CSV upload'}
                      </p>
                      <p style={helperStyle}>Bars loaded: {barsCount}</p>
                      <p style={helperStyle}>Horizon for predictions: {horizonSeconds} seconds</p>
                    </div>

                    <div style={{ display: 'grid', gap: '0.45rem' }}>
                      <button
                        type="button"
                        onClick={runInference}
                        disabled={inferenceDisabled}
                        style={{
                          background: inferenceDisabled ? '#1e293b' : 'linear-gradient(90deg, #38bdf8, #6366f1)',
                          color: '#0f172a',
                          fontWeight: 700,
                          border: 'none',
                          borderRadius: '0.75rem',
                          padding: '0.75rem 1rem',
                          cursor: inferenceDisabled ? 'not-allowed' : 'pointer',
                          opacity: inferenceDisabled ? 0.65 : 1,
                        }}
                      >
                        {isInferLoading ? 'Running inference…' : 'Compute features & signal'}
                      </button>
                      <p style={{ ...helperStyle, marginTop: '0.1rem' }}>
                        Fetch requests automatically attach <code>traceparent</code> headers and reuse the previous ETag via
                        <code>If-None-Match</code>.
                      </p>
                    </div>
                  </section>

                  <section
                    style={{
                      display: 'grid',
                      gap: '0.9rem',
                      backgroundColor: '#0f172a',
                      borderRadius: '0.85rem',
                      border: '1px solid rgba(148, 163, 184, 0.25)',
                      padding: '1.1rem',
                    }}
                  >
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                      <h4 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Latest results</h4>
                      {inferenceError ? <p style={errorStyle}>{inferenceError}</p> : null}
                      {isInferLoading ? (
                        <p style={{ ...helperStyle, color: '#38bdf8' }}>Waiting for inference response…</p>
                      ) : null}
                      {!featureSnapshot && !predictionSnapshot && !inferenceError ? (
                        <p style={helperStyle}>
                          Run inference to display the computed feature vector and trading signal for the most recent bar.
                        </p>
                      ) : null}
                    </div>

                    {featureSnapshot ? (
                      <div style={{ display: 'grid', gap: '0.5rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                          <h5 style={{ fontSize: '1rem', fontWeight: 700 }}>Feature vector</h5>
                          <span style={{ ...helperStyle, color: '#38bdf8' }}>
                            {featureGeneratedAt ? `Generated at ${featureGeneratedAt}` : null}
                          </span>
                        </div>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                          <tbody>
                            {featureEntries.map(([name, value]) => (
                              <tr key={name}>
                                <td style={{ padding: '0.25rem 0.4rem', color: '#cbd5f5', width: '60%' }}>{name}</td>
                                <td style={{ padding: '0.25rem 0.4rem', textAlign: 'right', color: '#e2e8f0' }}>
                                  {Number.isFinite(value) ? value.toFixed(4) : String(value)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : null}

                    {predictionSnapshot ? (
                      <div style={{ display: 'grid', gap: '0.6rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                          <h5 style={{ fontSize: '1rem', fontWeight: 700 }}>Prediction</h5>
                          <span style={{ ...helperStyle, color: '#38bdf8' }}>
                            {predictionGeneratedAt ? `Generated at ${predictionGeneratedAt}` : null}
                          </span>
                        </div>
                        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                          <div>
                            <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>Score</span>
                            <p style={{ fontWeight: 700 }}>{predictionSnapshot.score.toFixed(4)}</p>
                          </div>
                          <div>
                            <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>Horizon</span>
                            <p style={{ fontWeight: 700 }}>{predictionSnapshot.horizon_seconds} seconds</p>
                          </div>
                          <div>
                            <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>Symbol</span>
                            <p style={{ fontWeight: 700 }}>{predictionSnapshot.symbol}</p>
                          </div>
                        </div>
                        {signalEntries.length > 0 ? (
                          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                            <tbody>
                              {signalEntries.map(([key, value]) => (
                                <tr key={key}>
                                  <td style={{ padding: '0.25rem 0.4rem', color: '#cbd5f5', width: '60%' }}>{key}</td>
                                  <td style={{ padding: '0.25rem 0.4rem', textAlign: 'right', color: '#e2e8f0' }}>
                                    {typeof value === 'number'
                                      ? Number.isFinite(value)
                                        ? value.toFixed(4)
                                        : String(value)
                                      : String(value)}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        ) : (
                          <p style={helperStyle}>Signal metadata not provided by the service.</p>
                        )}
                      </div>
                    ) : null}

                    <div style={{ display: 'grid', gap: '0.35rem' }}>
                      <span style={{ ...helperStyle, color: '#94a3b8' }}>Request metadata</span>
                      <div style={{ display: 'grid', gap: '0.25rem', fontSize: '0.85rem' }}>
                        <div>
                          <span style={{ color: '#94a3b8' }}>traceparent:</span>{' '}
                          <code style={{ color: '#e2e8f0' }}>{traceparent ?? '—'}</code>
                        </div>
                        <div>
                          <span style={{ color: '#94a3b8' }}>Feature ETag:</span>{' '}
                          <code style={{ color: '#e2e8f0' }}>{featureEtag ?? '—'}</code>
                        </div>
                        <div>
                          <span style={{ color: '#94a3b8' }}>Prediction ETag:</span>{' '}
                          <code style={{ color: '#e2e8f0' }}>{predictionEtag ?? '—'}</code>
                        </div>
                      </div>
                    </div>
                  </section>
                </div>
              </div>
            </div>
          </article>

          <article style={panelStyle}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '1rem' }}>Scenario JSON template</h2>
            <p style={{ ...helperStyle, marginBottom: '0.75rem' }}>
              Drop this snippet into <code>docs/scenarios.md</code> or configuration files as a starting point for backtests.
            </p>
            <pre
              style={{
                backgroundColor: '#0f172a',
                padding: '1rem',
                borderRadius: '0.75rem',
                border: '1px solid rgba(148, 163, 184, 0.25)',
                overflowX: 'auto',
                fontSize: '0.9rem',
                lineHeight: 1.5,
              }}
            >
              {preview}
            </pre>
          </article>
        </section>
      </section>
    </main>
  )
}
