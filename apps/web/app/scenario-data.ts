export type ScenarioField = 'initialBalance' | 'riskPerTrade' | 'maxPositions' | 'timeframe'

export type ScenarioConfig = {
  initialBalance: number
  riskPerTrade: number
  maxPositions: number
  timeframe: string
}

export type ScenarioDraft = Record<ScenarioField, string>

export type ScenarioTemplate = {
  id: string
  label: string
  description: string
  defaults: ScenarioConfig
  notes: string[]
}

export type FieldMeta = {
  label: string
  helper: string
  placeholder: string
  inputMode?: 'decimal' | 'numeric' | 'text'
  type?: 'number' | 'text'
}

export type ScenarioHealthStatus = 'Production-ready' | 'Needs review' | 'High risk' | 'Resolve errors'

export type ScenarioHealth = {
  status: ScenarioHealthStatus
  score: number
  summary: string
  checklist: string[]
}

export type PnlPoint = {
  label: string
  value: number
  cumulative: number
}

export type PnlSummary = {
  total: number
  best: PnlPoint | null
  worst: PnlPoint | null
  positiveRate: number
  maxDrawdown: number
}

export type StatusLevel = 'ready' | 'watch' | 'action'

export type StatusIndicator = {
  label: string
  level: StatusLevel
  detail: string
}

export const FIELD_META: Record<ScenarioField, FieldMeta> = {
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

export const SCENARIO_TEMPLATES: ScenarioTemplate[] = [
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

function parseNumber(value: string): number {
  const trimmed = value.replace(/,/g, '').trim()
  if (!trimmed) {
    return Number.NaN
  }
  return Number(trimmed)
}

export function toDraft(config: ScenarioConfig): ScenarioDraft {
  return {
    initialBalance: config.initialBalance.toString(),
    riskPerTrade: config.riskPerTrade.toString(),
    maxPositions: config.maxPositions.toString(),
    timeframe: config.timeframe,
  }
}

export function parseDraft(draft: ScenarioDraft): ScenarioConfig {
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

export type FieldErrors = Record<ScenarioField, string | null>

export function validateDraft(draft: ScenarioDraft): FieldErrors {
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

export function computeWarnings(config: ScenarioConfig): string[] {
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

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

export function convertTimeframeToMinutes(timeframe: string): number | null {
  const match = timeframe.match(/^(\d+)([smhdw])$/i)
  if (!match) {
    return null
  }
  const amount = Number(match[1])
  const unit = match[2].toLowerCase()
  if (!Number.isFinite(amount) || amount <= 0) {
    return null
  }
  const multipliers: Record<string, number> = {
    s: 1 / 60,
    m: 1,
    h: 60,
    d: 1440,
    w: 10080,
  }
  const multiplier = multipliers[unit]
  if (multiplier === undefined) {
    return null
  }
  return amount * multiplier
}

export function describeTimeframe(timeframe: string): string | null {
  const match = timeframe.match(/^(\d+)([smhdw])$/i)
  if (!match) {
    return null
  }
  const amount = Number(match[1])
  if (!Number.isFinite(amount)) {
    return null
  }
  const unit = match[2].toLowerCase()
  const labels: Record<string, string> = {
    s: 'second',
    m: 'minute',
    h: 'hour',
    d: 'day',
    w: 'week',
  }
  const label = labels[unit]
  if (!label) {
    return null
  }
  return `${amount} ${amount === 1 ? label : `${label}s`}`
}

export function buildTimeframeInsights(timeframe: string): string[] {
  const minutes = convertTimeframeToMinutes(timeframe)
  if (minutes === null) {
    return []
  }
  const insights: string[] = []
  const description = describeTimeframe(timeframe)
  if (description) {
    insights.push(`Expect data refresh at least every ${description} to keep signals aligned.`)
  }
  if (minutes > 0) {
    const barsPerDay = Math.round((24 * 60) / minutes)
    if (barsPerDay >= 1200) {
      insights.push('Expect well over 1,200 bars per day—ensure streaming analytics and log aggregation are in place.')
    } else if (barsPerDay > 0) {
      insights.push(`Roughly ${barsPerDay.toLocaleString()} bars per day—size Monte Carlo samples accordingly.`)
    }
  }
  if (minutes <= 5) {
    insights.push('Execution cadence is fast; confirm order routing and slippage controls are tuned for low latency.')
  } else if (minutes <= 60) {
    insights.push('Mid-frequency cadence allows session-based monitoring. Prepare intraday review checklists.')
  } else if (minutes >= 720 && minutes < 1440) {
    insights.push('Plan for daily risk syncs—the cadence spans multiple sessions, so overnight gaps matter.')
  } else if (minutes >= 1440) {
    insights.push('Slow cadence—capture macro or fundamental catalysts between bars to avoid stale positioning.')
  }
  return insights
}

function toSafeNumber(value: number | null | undefined, fallback: number): number {
  return Number.isFinite(value) && typeof value === 'number' ? value : fallback
}

export function formatCurrency(value: number, fractionDigits = 2): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  }).format(value)
}

export function generatePnlSeries(config: ScenarioConfig, points = 12): PnlPoint[] {
  const initialBalance = toSafeNumber(config.initialBalance, 15000)
  const riskPerTrade = toSafeNumber(config.riskPerTrade, 1)
  const maxPositions = toSafeNumber(config.maxPositions, 3)
  const timeframeMinutes = convertTimeframeToMinutes(config.timeframe) ?? 60

  if (points <= 0) {
    return []
  }

  const normalizedRisk = Math.max(0.2, Math.min(5, riskPerTrade)) / 100
  const activityFactor = clamp(maxPositions / 6, 0.2, 1.4)
  const cadenceFactor = clamp(timeframeMinutes / 240, 0.35, 1.4)
  const drift = normalizedRisk * 0.65 - (activityFactor - 0.5) * 0.015
  const volatility = normalizedRisk * 2.8 * activityFactor * (1 / Math.sqrt(cadenceFactor))

  let cumulative = 0
  const baseline = initialBalance

  return Array.from({ length: points }).map((_, index) => {
    const t = index + 1
    const wave = Math.sin(t * 0.85 + normalizedRisk * 6.5) * 0.6
    const modulation = Math.cos(t * 0.42 + activityFactor) * 0.35
    const adjustment = Math.sin((t + timeframeMinutes) * 0.12) * 0.2
    const dailyReturn = drift + (wave + modulation + adjustment) * (volatility / 10)
    const pnl = baseline * dailyReturn
    cumulative += pnl
    return {
      label: `Day ${t}`,
      value: Number(pnl.toFixed(2)),
      cumulative: Number(cumulative.toFixed(2)),
    }
  })
}

export function summarisePnl(series: PnlPoint[]): PnlSummary {
  if (series.length === 0) {
    return {
      total: 0,
      best: null,
      worst: null,
      positiveRate: 0,
      maxDrawdown: 0,
    }
  }

  const total = Number(series.reduce((sum, point) => sum + point.value, 0).toFixed(2))
  let best = series[0]
  let worst = series[0]
  let positives = 0
  let runningPeak = 0
  let maxDrawdown = 0

  series.forEach((point) => {
    if (point.value > best.value) {
      best = point
    }
    if (point.value < worst.value) {
      worst = point
    }
    if (point.value >= 0) {
      positives += 1
    }
    runningPeak = Math.max(runningPeak, point.cumulative)
    maxDrawdown = Math.max(maxDrawdown, runningPeak - point.cumulative)
  })

  return {
    total,
    best,
    worst,
    positiveRate: Number((positives / series.length).toFixed(2)),
    maxDrawdown: Number(maxDrawdown.toFixed(2)),
  }
}

export function buildSparkline(series: PnlPoint[]) {
  if (series.length === 0) {
    return {
      line: '0,50 100,50',
      area: '0,100 0,50 100,50 100,100',
      last: { x: 100, y: 50 },
    }
  }

  const values = series.map((point) => point.cumulative)
  const min = Math.min(0, ...values)
  const max = Math.max(0, ...values)
  const range = max - min || 1

  const coordinates = series.map((point, index) => {
    const progress = series.length === 1 ? 0 : index / (series.length - 1)
    const x = progress * 100
    const y = 100 - ((point.cumulative - min) / range) * 100
    return { x, y }
  })

  const line = coordinates.map(({ x, y }) => `${x},${y}`).join(' ')
  const areaPoints = [`0,100`, ...coordinates.map(({ x, y }) => `${x},${y}`), `100,100`]

  return {
    line,
    area: areaPoints.join(' '),
    last: coordinates[coordinates.length - 1] ?? { x: 100, y: 50 },
  }
}

export function buildStatusIndicators(
  scenarioHealth: ScenarioHealth,
  warnings: string[],
  hasErrors: boolean,
  pnlSummary: PnlSummary,
  timeframe: string,
): StatusIndicator[] {
  const indicators: StatusIndicator[] = []

  indicators.push({
    label: 'Execution readiness',
    level:
      scenarioHealth.status === 'Production-ready'
        ? 'ready'
        : scenarioHealth.status === 'Needs review'
        ? 'watch'
        : 'action',
    detail: scenarioHealth.summary,
  })

  const pnlLevel: StatusLevel = pnlSummary.total >= 0
    ? pnlSummary.maxDrawdown <= Math.abs(pnlSummary.total) * 0.6
      ? 'ready'
      : 'watch'
    : 'action'
  const pnlTrend = pnlSummary.total >= 0 ? 'Profitable window' : 'Net drawdown'
  const pnlDetail = `${pnlTrend}: ${formatCurrency(pnlSummary.total, 0)} over the observed period.`
  indicators.push({ label: 'PnL trajectory', level: pnlLevel, detail: pnlDetail })

  indicators.push({
    label: 'Risk posture',
    level: warnings.length === 0 ? 'ready' : warnings.length <= 2 ? 'watch' : 'action',
    detail:
      warnings.length === 0
        ? 'Risk snapshot shows balanced exposure with headroom for automation.'
        : `${warnings.length} warning${warnings.length === 1 ? '' : 's'} active — review the highlighted risk controls.`,
  })

  indicators.push({
    label: 'Form validation',
    level: hasErrors ? 'action' : 'ready',
    detail: hasErrors
      ? 'Resolve validation errors above to unlock exports and scenario sync.'
      : 'Configuration is valid and ready to export or promote.',
  })

  const timeframeDescription = describeTimeframe(timeframe)
  indicators.push({
    label: 'Monitoring cadence',
    level: (() => {
      const minutes = convertTimeframeToMinutes(timeframe)
      if (minutes === null) {
        return 'watch' as StatusLevel
      }
      if (minutes <= 5) {
        return 'watch'
      }
      if (minutes >= 1440) {
        return 'watch'
      }
      return 'ready'
    })(),
    detail:
      timeframeDescription !== null
        ? `Plan monitoring touchpoints roughly every ${timeframeDescription}.`
        : 'Define a timeframe to plan monitoring handoffs.',
  })

  return indicators
}

export function evaluateScenario(
  config: ScenarioConfig,
  warnings: string[],
  hasErrors: boolean,
): ScenarioHealth {
  const checklist: string[] = []

  if (hasErrors) {
    checklist.push('Resolve the highlighted fields above to calculate a deployable scenario.')
    if (warnings.length > 0) {
      checklist.push('Revisit the risk warnings once validation errors are cleared.')
    }
    return {
      status: 'Resolve errors',
      score: 25,
      summary: 'Fix validation errors to unlock export actions and a reliable health score.',
      checklist,
    }
  }

  const { initialBalance, riskPerTrade, maxPositions, timeframe } = config

  if (
    !Number.isFinite(initialBalance) ||
    !Number.isFinite(riskPerTrade) ||
    !Number.isFinite(maxPositions) ||
    !timeframe
  ) {
    checklist.push('Populate every input so health checks can benchmark risk exposure.')
    return {
      status: 'Needs review',
      score: 45,
      summary: 'Complete the remaining fields to benchmark the scenario and surface optimisation ideas.',
      checklist,
    }
  }

  let score = 95

  if (warnings.length > 0) {
    score -= Math.min(45, warnings.length * 12)
    checklist.push('Address the risk snapshot warnings to tighten the scenario envelope.')
  }

  if (initialBalance < 5000) {
    score -= 12
    checklist.push('Increase the initial balance towards ≥ 5k to stabilise Monte Carlo paths.')
  }

  if (riskPerTrade > 2) {
    score -= 10
    checklist.push('Keep risk per trade at or below 2% to stay within resilient drawdown tolerances.')
  } else if (riskPerTrade < 0.25) {
    score -= 6
    checklist.push('Confirm commissions remain negligible when risking under 0.25% per trade.')
  }

  if (maxPositions > 6) {
    score -= 8
    checklist.push('Limit concurrent positions to ≤ 6 unless execution is heavily automated.')
  }

  const riskDollars = (initialBalance * riskPerTrade) / 100
  const portfolioRisk = riskDollars * maxPositions
  if (portfolioRisk > initialBalance * 0.25) {
    score -= 10
    checklist.push('Trim portfolio risk below 25% of equity to avoid cascading losses.')
  }

  const minutes = convertTimeframeToMinutes(timeframe)
  if (minutes !== null) {
    if (minutes <= 5) {
      score -= 6
      checklist.push('Verify data infrastructure supports sub-five-minute execution cadence.')
    } else if (minutes >= 720) {
      checklist.push('Document overnight gap handling for higher timeframe execution.')
    }
  }

  const boundedScore = Math.round(clamp(score, 20, 100))

  let status: ScenarioHealthStatus
  let summary: string
  if (boundedScore >= 80) {
    status = 'Production-ready'
    summary = 'Risk controls look balanced. Document execution assumptions before promotion.'
  } else if (boundedScore >= 55) {
    status = 'Needs review'
    summary = 'Scenario is workable but tighten the highlighted levers before automation.'
  } else {
    status = 'High risk'
    summary = 'Risk envelope is stretched. Reduce concentration before running the strategy in staging.'
  }

  const uniqueChecklist = Array.from(new Set(checklist))

  return {
    status,
    score: boundedScore,
    summary,
    checklist: uniqueChecklist,
  }
}

export function buildPreview(config: ScenarioConfig) {
  return {
    initialBalance: Number.isFinite(config.initialBalance) ? Number(config.initialBalance.toFixed(2)) : null,
    riskPerTrade: Number.isFinite(config.riskPerTrade) ? Number(config.riskPerTrade.toFixed(2)) : null,
    maxPositions: Number.isFinite(config.maxPositions) ? config.maxPositions : null,
    timeframe: config.timeframe || null,
  }
}
