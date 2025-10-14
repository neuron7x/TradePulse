'use client'

import type { ChangeEvent, CSSProperties } from 'react'
import { useMemo, useState } from 'react'

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
  const [actionMessage, setActionMessage] = useState<{ kind: 'success' | 'error'; text: string } | null>(null)

  const selectedTemplate = useMemo(
    () => SCENARIO_TEMPLATES.find((entry) => entry.id === templateId) ?? SCENARIO_TEMPLATES[0],
    [templateId],
  )

  const parsedConfig = useMemo(() => parseDraft(draft), [draft])
  const errors = useMemo(() => validateDraft(draft), [draft])
  const hasErrors = useMemo(() => Object.values(errors).some((item) => item !== null), [errors])
  const warnings = useMemo(() => computeWarnings(parsedConfig), [parsedConfig])
  const preview = useMemo(() => JSON.stringify(buildPreview(parsedConfig), null, 2), [parsedConfig])

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
    setActionMessage(null)
    setDraft((current) => ({ ...current, [field]: value }))
  }

  const resetTemplate = () => {
    setDraft(toDraft(selectedTemplate.defaults))
    setActionMessage(null)
  }

  const handleCopy = async () => {
    if (hasErrors) {
      setActionMessage({
        kind: 'error',
        text: 'Resolve form errors before exporting the scenario JSON.',
      })
      return
    }

    try {
      if (!navigator.clipboard || typeof navigator.clipboard.writeText !== 'function') {
        throw new Error('Clipboard API unavailable')
      }
      await navigator.clipboard.writeText(preview)
      setActionMessage({ kind: 'success', text: 'Scenario JSON copied to clipboard.' })
    } catch (error) {
      setActionMessage({
        kind: 'error',
        text: 'Failed to copy the scenario JSON. Please try again.',
      })
    }
  }

  const handleDownload = () => {
    if (hasErrors) {
      setActionMessage({
        kind: 'error',
        text: 'Resolve form errors before exporting the scenario JSON.',
      })
      return
    }

    try {
      const blob = new Blob([preview], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `scenario-${templateId}.json`
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      URL.revokeObjectURL(url)
      setActionMessage({ kind: 'success', text: 'Scenario JSON download started.' })
    } catch (error) {
      setActionMessage({
        kind: 'error',
        text: 'Failed to start the scenario JSON download. Please try again.',
      })
    }
  }

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
                  setActionMessage(null)
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

              <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end', flexWrap: 'wrap' }}>
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
                <button
                  type="button"
                  onClick={handleCopy}
                  disabled={hasErrors}
                  style={{
                    backgroundColor: hasErrors ? 'rgba(15, 23, 42, 0.6)' : '#38bdf8',
                    border: '1px solid rgba(148, 163, 184, 0.4)',
                    color: hasErrors ? '#94a3b8' : '#0f172a',
                    padding: '0.55rem 1rem',
                    borderRadius: '0.65rem',
                    cursor: hasErrors ? 'not-allowed' : 'pointer',
                    fontWeight: 600,
                  }}
                >
                  Copy to clipboard
                </button>
                <button
                  type="button"
                  onClick={handleDownload}
                  disabled={hasErrors}
                  style={{
                    backgroundColor: hasErrors ? 'rgba(15, 23, 42, 0.6)' : '#22d3ee',
                    border: '1px solid rgba(148, 163, 184, 0.4)',
                    color: hasErrors ? '#94a3b8' : '#0f172a',
                    padding: '0.55rem 1rem',
                    borderRadius: '0.65rem',
                    cursor: hasErrors ? 'not-allowed' : 'pointer',
                    fontWeight: 600,
                  }}
                >
                  Download JSON
                </button>
              </div>
              {actionMessage ? (
                <p
                  role="status"
                  style={{
                    marginTop: '0.5rem',
                    color: actionMessage.kind === 'success' ? '#4ade80' : '#f97316',
                    fontSize: '0.95rem',
                    fontWeight: 600,
                  }}
                >
                  {actionMessage.text}
                </p>
              ) : null}
            </form>
          </div>
        </section>

        <section style={{ display: 'grid', gap: '1.5rem' }}>
          <article style={panelStyle}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '1rem' }}>Risk snapshot</h2>
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
