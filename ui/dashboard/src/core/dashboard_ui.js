const DEFAULT_TITLE = 'TradePulse Monitoring Hub';

const DASHBOARD_STYLES = `
  :root {
    color-scheme: dark;
  }

  .tp-dashboard {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 1.5rem;
    padding: 2rem;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: radial-gradient(circle at top left, #0d1b2a, #010409);
    min-height: 100vh;
    color: #f8fafc;
  }

  @media (min-width: 960px) {
    .tp-dashboard {
      grid-template-columns: 320px minmax(0, 1fr);
      align-items: start;
    }
  }

  .tp-dashboard__panel {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 18px;
    box-shadow: 0 24px 48px -32px rgba(15, 23, 42, 0.8);
    padding: 1.75rem;
    backdrop-filter: blur(24px);
  }

  .tp-dashboard__header {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  @media (min-width: 720px) {
    .tp-dashboard__header {
      flex-direction: row;
      align-items: center;
      justify-content: space-between;
    }
  }

  .tp-dashboard__title {
    margin: 0;
    font-size: clamp(1.75rem, 3vw, 2.5rem);
    font-weight: 700;
    letter-spacing: -0.02em;
  }

  .tp-dashboard__meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
  }

  .tp-badge {
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: rgba(14, 165, 233, 0.15);
    color: #38bdf8;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }

  .tp-dashboard__grid {
    display: grid;
    gap: 1.5rem;
  }

  @media (min-width: 720px) {
    .tp-dashboard__grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  .tp-metric-grid {
    display: grid;
    gap: 1rem;
  }

  @media (min-width: 640px) {
    .tp-metric-grid {
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }
  }

  .tp-metric-card {
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.15), rgba(14, 116, 144, 0.15));
    border: 1px solid rgba(148, 163, 184, 0.25);
    padding: 1rem 1.25rem;
    display: grid;
    gap: 0.5rem;
  }

  .tp-metric-card__label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(226, 232, 240, 0.75);
  }

  .tp-metric-card__value {
    font-size: 1.75rem;
    font-weight: 700;
  }

  .tp-metric-card__change {
    font-size: 0.95rem;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }

  .tp-metric-card__change--positive {
    color: #4ade80;
  }

  .tp-metric-card__change--negative {
    color: #f87171;
  }

  .tp-status-list {
    display: grid;
    gap: 1rem;
  }

  .tp-status-item {
    display: grid;
    gap: 0.35rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.25);
  }

  .tp-status-item:last-child {
    border-bottom: none;
    padding-bottom: 0;
  }

  .tp-status-item__heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-weight: 600;
  }

  .tp-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.15);
  }

  .tp-status-pill--ok {
    color: #4ade80;
    background: rgba(74, 222, 128, 0.15);
  }

  .tp-status-pill--warning {
    color: #fbbf24;
    background: rgba(251, 191, 36, 0.15);
  }

  .tp-status-pill--critical {
    color: #f87171;
    background: rgba(248, 113, 113, 0.15);
  }

  .tp-status-item__body {
    color: rgba(226, 232, 240, 0.75);
    font-size: 0.95rem;
  }

  .tp-chart {
    display: grid;
    gap: 0.75rem;
  }

  .tp-chart__bars {
    display: grid;
    gap: 0.5rem;
  }

  .tp-chart__row {
    display: grid;
    grid-template-columns: minmax(80px, 120px) minmax(0, 1fr) 80px;
    gap: 0.75rem;
    align-items: center;
    font-size: 0.95rem;
  }

  .tp-chart__bar {
    height: 10px;
    border-radius: 999px;
    position: relative;
    overflow: hidden;
    background: rgba(100, 116, 139, 0.25);
  }

  .tp-chart__bar-fill {
    position: absolute;
    inset: 0;
    transform-origin: left;
    transition: transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    background: linear-gradient(90deg, #38bdf8, #2563eb);
  }

  .tp-chart__bar-fill--negative {
    background: linear-gradient(90deg, #f87171, #ef4444);
    transform-origin: right;
  }

  .tp-empty {
    padding: 1rem;
    text-align: center;
    font-size: 0.95rem;
    color: rgba(148, 163, 184, 0.8);
    border: 1px dashed rgba(148, 163, 184, 0.25);
    border-radius: 12px;
  }
`;

function escapeHtml(value) {
  if (value === null || value === undefined) {
    return '';
  }
  return String(value).replace(/[&<>"']/g, (char) => {
    switch (char) {
      case '&':
        return '&amp;';
      case '<':
        return '&lt;';
      case '>':
        return '&gt;';
      case '"':
        return '&quot;';
      case '\'':
        return '&#39;';
      default:
        return char;
    }
  });
}

function sanitizeModifier(value, fallback = 'ok') {
  const sanitized = String(value || fallback)
    .toLowerCase()
    .replace(/[^a-z0-9_-]/g, '');
  return sanitized || fallback;
}

function formatCurrency(value, currency = 'USD') {
  if (!Number.isFinite(value)) {
    return '—';
  }
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    maximumFractionDigits: Math.abs(value) >= 1000 ? 0 : 2,
  }).format(value);
}

function formatPercent(value) {
  if (!Number.isFinite(value)) {
    return '—';
  }
  return `${(value * 100).toFixed(Math.abs(value) < 0.1 ? 2 : 1)}%`;
}

function renderChange(value) {
  if (!Number.isFinite(value) || value === 0) {
    return '';
  }
  const direction = value > 0 ? 'positive' : 'negative';
  const sign = value > 0 ? '+' : '−';
  return `<span class="tp-metric-card__change tp-metric-card__change--${direction}">${sign}${formatPercent(Math.abs(value))}</span>`;
}

function buildPnLPanels(entries = [], currency = 'USD') {
  if (!entries.length) {
    return '<div class="tp-empty">PnL data is not available yet.</div>';
  }

  return entries
    .map((entry) => {
      const { label, value, change } = entry;
      const isNegative = Number.isFinite(value) && value < 0;
      const ratio = Math.max(Math.min(Math.abs(value) / (entry.target || Math.max(Math.abs(value), 1)), 1), 0);
      const fillStyle = `transform: scaleX(${ratio.toFixed(3)});`;
      return `
        <div class="tp-chart__row">
          <span>${escapeHtml(label)}</span>
          <div class="tp-chart__bar">
            <div class="tp-chart__bar-fill ${isNegative ? 'tp-chart__bar-fill--negative' : ''}" style="${fillStyle}"></div>
          </div>
          <div style="text-align:right;">
            <div>${formatCurrency(value, currency)}</div>
            ${renderChange(change)}
          </div>
        </div>
      `;
    })
    .join('');
}

function buildStatusList(statusItems = []) {
  if (!statusItems.length) {
    return '<div class="tp-empty">No status alerts. Systems are idle.</div>';
  }

  return statusItems
    .map((item) => {
      const level = item.level || 'ok';
      const pillClass = `tp-status-pill tp-status-pill--${sanitizeModifier(level)}`;
      const description = item.description || 'No additional details provided.';
      return `
        <article class="tp-status-item">
          <header class="tp-status-item__heading">
            <span>${escapeHtml(item.label)}</span>
            <span class="${pillClass}">${escapeHtml(item.state)}</span>
          </header>
          <p class="tp-status-item__body">${escapeHtml(description)}</p>
        </article>
      `;
    })
    .join('');
}

function renderMetricCards(metrics = [], currency = 'USD') {
  if (!metrics.length) {
    return '<div class="tp-empty">Connect a data source to view performance metrics.</div>';
  }

  return metrics
    .map((metric) => {
      const value =
        metric.kind === 'currency'
          ? formatCurrency(metric.value, currency)
          : escapeHtml(metric.value ?? '—');
      const change = renderChange(metric.change ?? NaN);
      return `
        <article class="tp-metric-card">
          <span class="tp-metric-card__label">${escapeHtml(metric.label)}</span>
          <span class="tp-metric-card__value">${value}</span>
          ${change}
        </article>
      `;
    })
    .join('');
}

function renderHeader({ title, subtitle, tags }) {
  const meta = [];
  if (subtitle) {
    meta.push(`<span>${escapeHtml(subtitle)}</span>`);
  }
  (tags || []).forEach((tag) => {
    meta.push(`<span class="tp-badge">${escapeHtml(tag)}</span>`);
  });
  const metaBlock = meta.length
    ? `<div class="tp-dashboard__meta">${meta.join('')}</div>`
    : '';

  return `
    <header class="tp-dashboard__header">
      <div>
        <h1 class="tp-dashboard__title">${escapeHtml(title || DEFAULT_TITLE)}</h1>
        ${metaBlock}
      </div>
      <div class="tp-dashboard__meta">
        <span class="tp-badge">Live</span>
      </div>
    </header>
  `;
}

export function renderDashboard(options = {}) {
  const {
    title = DEFAULT_TITLE,
    subtitle = 'Unified control over execution, risk, and telemetry.',
    tags = ['multi-asset', 'real-time'],
    currency = 'USD',
    metrics = [],
    pnlSeries = [],
    statusItems = [],
  } = options;

  const header = renderHeader({ title, subtitle, tags });
  const metricCards = renderMetricCards(metrics, currency);
  const pnlRows = buildPnLPanels(pnlSeries, currency);
  const statuses = buildStatusList(statusItems);

  const html = `
    <section class="tp-dashboard">
      <div class="tp-dashboard__panel">
        ${header}
        <div class="tp-dashboard__grid" style="margin-top:1.5rem;">
          <section class="tp-dashboard__panel">
            <header style="margin-bottom:1rem;">
              <h2 style="margin:0;font-size:1.25rem;font-weight:600;">Key Metrics</h2>
            </header>
            <div class="tp-metric-grid">${metricCards}</div>
          </section>
          <section class="tp-dashboard__panel">
            <header style="margin-bottom:1rem;">
              <h2 style="margin:0;font-size:1.25rem;font-weight:600;">PnL Overview</h2>
              <p style="margin:0;color:rgba(226,232,240,0.65);font-size:0.95rem;">Track profit and loss across your preferred horizons.</p>
            </header>
            <div class="tp-chart">
              <div class="tp-chart__bars">${pnlRows}</div>
            </div>
          </section>
        </div>
      </div>
      <aside class="tp-dashboard__panel">
        <header style="margin-bottom:1.25rem;">
          <h2 style="margin:0;font-size:1.2rem;font-weight:600;">System Status</h2>
          <p style="margin:0;color:rgba(226,232,240,0.65);font-size:0.95rem;">Monitor exchange connectivity, risk checks, and automation agents.</p>
        </header>
        <div class="tp-status-list">${statuses}</div>
      </aside>
    </section>
  `;

  return { html, styles: DASHBOARD_STYLES };
}

export { DASHBOARD_STYLES, formatCurrency, formatPercent };
