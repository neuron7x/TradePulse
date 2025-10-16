'use client'

import type { ChangeEvent } from 'react'
import { useId, useMemo, useState } from 'react'

import {
  FIELD_META,
  SCENARIO_TEMPLATES,
  type ScenarioDraft,
  type ScenarioField,
  buildPreview,
  buildSparkline,
  buildStatusIndicators,
  buildTimeframeInsights,
  computeWarnings,
  evaluateScenario,
  formatCurrency,
  generatePnlSeries,
  summarisePnl,
  toDraft,
  parseDraft,
  validateDraft,
} from './scenario-data'

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
  const timeframeInsights = useMemo(() => buildTimeframeInsights(parsedConfig.timeframe), [parsedConfig.timeframe])
  const scenarioHealth = useMemo(
    () => evaluateScenario(parsedConfig, warnings, hasErrors),
    [parsedConfig, warnings, hasErrors],
  )
  const pnlSeries = useMemo(() => generatePnlSeries(parsedConfig), [parsedConfig])
  const pnlSummary = useMemo(() => summarisePnl(pnlSeries), [pnlSeries])
  const sparkline = useMemo(() => buildSparkline(pnlSeries), [pnlSeries])
  const statusIndicators = useMemo(
    () => buildStatusIndicators(scenarioHealth, warnings, hasErrors, pnlSummary, parsedConfig.timeframe),
    [scenarioHealth, warnings, hasErrors, pnlSummary, parsedConfig.timeframe],
  )
  const recentPnl = useMemo(() => pnlSeries.slice(-4).reverse(), [pnlSeries])
  const cumulativeClose = useMemo(
    () => (pnlSeries.length > 0 ? pnlSeries[pnlSeries.length - 1].cumulative : 0),
    [pnlSeries],
  )
  const sparklineGradientId = useId()

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

  const riskPercentOfEquity = useMemo(() => {
    if (
      riskDollars === null ||
      !Number.isFinite(parsedConfig.initialBalance) ||
      parsedConfig.initialBalance === 0
    ) {
      return null
    }
    return (riskDollars / parsedConfig.initialBalance) * 100
  }, [parsedConfig, riskDollars])

  const portfolioRiskPercent = useMemo(() => {
    if (
      aggregateRisk === null ||
      !Number.isFinite(parsedConfig.initialBalance) ||
      parsedConfig.initialBalance === 0
    ) {
      return null
    }
    return (aggregateRisk / parsedConfig.initialBalance) * 100
  }, [aggregateRisk, parsedConfig])

  const handleChange = (field: ScenarioField) => (event: ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value
    setActionMessage(null)
    setDraft((current) => ({ ...current, [field]: value }))
  }

  const handleTemplateChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const value = event.target.value
    setTemplateId(value)
    setActionMessage(null)
    const template = SCENARIO_TEMPLATES.find((entry) => entry.id === value)
    if (template) {
      setDraft(toDraft(template.defaults))
    }
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

  const templateHelperId = 'template-description'

  return (
    <main className="scenario-main">
      <section className="scenario-container">
        <header className="scenario-hero">
          <h1>Scenario Studio</h1>
          <p>
            Sanity-check strategy inputs before pushing them into execution. Select a template, adjust the levers, and review
            automatic hints about risk concentration and timeframe hygiene.
          </p>
        </header>

        <div className="scenario-grid">
          <section className="panel panel-form" aria-labelledby="scenario-form-heading">
            <div className="template-select">
              <label htmlFor="template" id="scenario-form-heading">
                Scenario template
              </label>
              <select
                id="template"
                value={templateId}
                onChange={handleTemplateChange}
                className="tp-select"
                aria-describedby={templateHelperId}
              >
                {SCENARIO_TEMPLATES.map((template) => (
                  <option key={template.id} value={template.id}>
                    {template.label}
                  </option>
                ))}
              </select>
              <p id={templateHelperId} className="template-description">
                {selectedTemplate.description}
              </p>
              <ul className="template-notes">
                {selectedTemplate.notes.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            </div>

            <form className="field-grid" noValidate>
              {(Object.keys(FIELD_META) as ScenarioField[]).map((field) => {
                const meta = FIELD_META[field]
                const inputId = `field-${field}`
                const helperId = `${inputId}-helper`
                const errorId = `${inputId}-error`
                const hasError = Boolean(errors[field])
                const describedBy = hasError ? `${helperId} ${errorId}` : helperId
                return (
                  <div key={field} className="field">
                    <label htmlFor={inputId}>{meta.label}</label>
                    <input
                      id={inputId}
                      name={field}
                      value={draft[field]}
                      onChange={handleChange(field)}
                      placeholder={meta.placeholder}
                      inputMode={meta.inputMode}
                      type={meta.type}
                      className="tp-input"
                      aria-invalid={hasError}
                      aria-describedby={describedBy}
                      autoComplete="off"
                      step={meta.type === 'number' ? 'any' : undefined}
                    />
                    <p id={helperId} className="tp-helper">
                      {meta.helper}
                    </p>
                    {hasError ? (
                      <p id={errorId} className="tp-error">
                        {errors[field]}
                      </p>
                    ) : null}
                  </div>
                )
              })}

              <div className="button-row">
                <button type="button" onClick={resetTemplate} className="tp-button tp-button--ghost">
                  Reset to template defaults
                </button>
                <button type="button" onClick={handleCopy} disabled={hasErrors} className="tp-button tp-button--primary">
                  Copy to clipboard
                </button>
                <button type="button" onClick={handleDownload} disabled={hasErrors} className="tp-button tp-button--secondary">
                  Download JSON
                </button>
              </div>
              {actionMessage ? (
                <p
                  role="status"
                  aria-live="polite"
                  className={`tp-status ${actionMessage.kind === 'success' ? 'tp-status--success' : 'tp-status--error'}`}
                >
                  {actionMessage.text}
                </p>
              ) : null}
            </form>
          </section>

          <div className="side-panels">
            <article className="panel health-panel">
              <h2>Scenario health</h2>
              <div
                className={`health-status ${
                  scenarioHealth.status === 'Production-ready'
                    ? 'health-status--ready'
                    : scenarioHealth.status === 'Needs review'
                    ? 'health-status--review'
                    : scenarioHealth.status === 'Resolve errors'
                    ? 'health-status--blocked'
                    : 'health-status--risk'
                }`}
              >
                <span>{scenarioHealth.status}</span>
              </div>
              <p className="health-score">Score: {scenarioHealth.score} / 100</p>
              <div
                className="health-meter"
                role="meter"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={scenarioHealth.score}
                aria-valuetext={`${scenarioHealth.score} out of 100`}
              >
                <span className="health-meter__fill" style={{ width: `${scenarioHealth.score}%` }} />
              </div>
              <p className="health-summary">{scenarioHealth.summary}</p>
              {scenarioHealth.checklist.length > 0 ? (
                <ul className="health-checklist">
                  {scenarioHealth.checklist.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              ) : null}
            </article>

            <article className="panel pnl-panel">
              <div className="pnl-header">
                <h2>P&amp;L trajectory</h2>
                <span className={`trend-badge ${pnlSummary.total >= 0 ? 'trend-badge--positive' : 'trend-badge--negative'}`}>
                  {pnlSummary.total >= 0 ? 'Uptrend' : 'Drawdown'}
                </span>
              </div>
              <p className="tp-helper tp-helper-spaced">
                Visualise rolling performance to detect when profits accelerate or drawdowns deepen.
              </p>
              <div
                className="pnl-chart"
                role="img"
                aria-label={`Cumulative profit and loss sparkline ending at ${formatCurrency(cumulativeClose, 0)}`}
              >
                <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="presentation" aria-hidden="true">
                  <defs>
                    <linearGradient id={sparklineGradientId} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="rgba(56, 189, 248, 0.75)" />
                      <stop offset="100%" stopColor="rgba(14, 116, 144, 0.05)" />
                    </linearGradient>
                  </defs>
                  <polygon fill={`url(#${sparklineGradientId})`} points={sparkline.area} />
                  <polyline
                    points={sparkline.line}
                    fill="none"
                    stroke="rgba(56, 189, 248, 1)"
                    strokeWidth={2.5}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                  <circle
                    cx={sparkline.last.x}
                    cy={sparkline.last.y}
                    r={3.5}
                    fill={pnlSummary.total >= 0 ? '#38bdf8' : '#f97316'}
                    stroke="#0f172a"
                    strokeWidth={1.5}
                  />
                </svg>
              </div>
              <dl className="pnl-metrics">
                <div>
                  <dt>Total</dt>
                  <dd>{formatCurrency(pnlSummary.total, 0)}</dd>
                </div>
                <div>
                  <dt>Positive ratio</dt>
                  <dd>{(pnlSummary.positiveRate * 100).toFixed(0)}%</dd>
                </div>
                <div>
                  <dt>Best session</dt>
                  <dd>{pnlSummary.best ? formatCurrency(pnlSummary.best.value, 0) : '—'}</dd>
                </div>
                <div>
                  <dt>Max drawdown</dt>
                  <dd>{formatCurrency(-Math.abs(pnlSummary.maxDrawdown), 0)}</dd>
                </div>
              </dl>
              <div className="pnl-recent">
                <h3>Recent sessions</h3>
                <ul>
                  {recentPnl.map((point) => (
                    <li key={point.label}>
                      <span>{point.label}</span>
                      <span className={point.value >= 0 ? 'pnl-value pnl-value--positive' : 'pnl-value pnl-value--negative'}>
                        {formatCurrency(point.value, 0)}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </article>

            <article className="panel status-panel">
              <h2>Status overview</h2>
              <p className="tp-helper tp-helper-spaced">
                Track readiness signals across execution, validation, and monitoring workflows.
              </p>
              <div className="status-grid">
                {statusIndicators.map((indicator) => (
                  <div key={indicator.label} className={`status-item status-item--${indicator.level}`}>
                    <span className="status-dot" aria-hidden="true" />
                    <div className="status-copy">
                      <p className="status-label">{indicator.label}</p>
                      <p className="status-detail">{indicator.detail}</p>
                    </div>
                  </div>
                ))}
              </div>
            </article>

            <article className="panel risk-panel">
              <h2>Risk snapshot</h2>
              <div className="metric-grid">
                <div className="metric-tiles">
                  <div className="metric-tile">
                    <span>Risk per trade</span>
                    <p>{riskDollars === null ? '—' : formatCurrency(riskDollars, 2)}</p>
                    {riskPercentOfEquity !== null ? <span>{riskPercentOfEquity.toFixed(2)}% of equity</span> : null}
                  </div>
                  <div className="metric-tile">
                    <span>Max portfolio risk</span>
                    <p>{aggregateRisk === null ? '—' : formatCurrency(aggregateRisk, 2)}</p>
                    {portfolioRiskPercent !== null ? <span>{portfolioRiskPercent.toFixed(2)}% of equity</span> : null}
                  </div>
                  <div className="metric-tile">
                    <span>Timeframe</span>
                    <p>{parsedConfig.timeframe || '—'}</p>
                  </div>
                </div>

                {warnings.length > 0 ? (
                  <ul className="warning-list">
                    {warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="warning-placeholder">
                    Risk controls look balanced for the selected template. Stress test transaction costs before live execution.
                  </p>
                )}

                {hasErrors ? (
                  <p className="tp-error tp-error-inline">
                    Resolve the highlighted fields above to unlock export-ready scenario JSON.
                  </p>
                ) : null}

                {timeframeInsights.length > 0 ? (
                  <div className="timeframe-insights">
                    <h3>Timeframe insights</h3>
                    <ul>
                      {timeframeInsights.map((insight) => (
                        <li key={insight}>{insight}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            </article>

            <article className="panel">
              <h2>Scenario JSON template</h2>
              <p className="tp-helper tp-helper-spaced">
                Drop this snippet into <code>docs/scenarios.md</code> or configuration files as a starting point for backtests.
              </p>
              <pre className="code-preview" aria-label="Scenario JSON preview">
                {preview}
              </pre>
            </article>
          </div>
        </div>
      </section>
    </main>
  )
}
