import { renderAreaChart } from '../components/area_chart.js';
import {
  escapeHtml,
  formatCurrency,
  formatNumber,
  formatPercent,
  formatTimestamp,
} from '../core/formatters.js';

/**
 * @typedef {import('../types/events').BarEvent} BarEvent
 * @typedef {import('../types/events').TickEvent} TickEvent
 */

function normalisePnlSeries(pnlPoints = [], currency = 'USD') {
  return pnlPoints.map((point) => ({
    timestamp: point.timestamp,
    value: Number.isFinite(point.value) ? point.value : 0,
    label: `${formatTimestamp(point.timestamp)} • ${formatCurrency(point.value, currency)}`,
  }));
}

function normaliseQuoteSeries(quotes = []) {
  return quotes
    .filter((tick) => Number.isFinite(tick?.last_price) || (Number.isFinite(tick?.bid_price) && Number.isFinite(tick?.ask_price)))
    .map((tick) => {
      const price = Number.isFinite(tick.last_price)
        ? tick.last_price
        : (tick.bid_price + tick.ask_price) / 2;
      return {
        timestamp: tick.timestamp,
        value: price,
        label: `${formatTimestamp(tick.timestamp)} • ${formatNumber(price, { maximumFractionDigits: 4 })}`,
      };
    });
}

function summarisePnl(points = [], currency = 'USD') {
  if (!points.length) {
    return { total: 0, change: 0, runRate: 0 };
  }
  const sorted = points.slice().sort((a, b) => a.timestamp - b.timestamp);
  const first = sorted[0];
  const last = sorted[sorted.length - 1];
  const elapsed = last.timestamp - first.timestamp || 1;
  const change = last.value - first.value;
  const runRate = change / (elapsed / (60 * 60 * 1000));
  return {
    total: last.value,
    change,
    runRate,
    formatted: {
      total: formatCurrency(last.value, currency),
      change: formatCurrency(change, currency),
      runRate: `${formatCurrency(runRate, currency)}/h`,
      changePercent: first.value !== 0 ? formatPercent(change / Math.abs(first.value)) : formatPercent(0),
    },
  };
}

function summariseQuotes(quotes = []) {
  if (!quotes.length) {
    return { last: 0, change: 0 };
  }
  const sorted = quotes.slice().sort((a, b) => a.timestamp - b.timestamp);
  const first = sorted[0].value;
  const last = sorted[sorted.length - 1].value;
  const change = last - first;
  return {
    last,
    change,
    changePercent: first !== 0 ? change / first : 0,
  };
}

export function renderPnlQuotesView({ pnlPoints = [], quotes = [], currency = 'USD' } = {}) {
  const pnlSeries = normalisePnlSeries(pnlPoints, currency);
  const quoteSeries = normaliseQuoteSeries(quotes);
  const pnlChart = renderAreaChart({ id: 'pnl', series: pnlSeries });
  const quoteChart = renderAreaChart({ id: 'quotes', series: quoteSeries });
  const pnlSummary = summarisePnl(pnlSeries, currency);
  const quoteSummary = summariseQuotes(quoteSeries);

  return {
    route: 'pnl',
    title: 'PnL & Quotes',
    html: `
      <section class="tp-view">
        <header class="tp-view__header">
          <h2 class="tp-view__title">PnL & Quotes Intelligence</h2>
          <p class="tp-view__subtitle">Cross-reference live profitability against streaming market data.</p>
        </header>
        <section class="tp-grid tp-grid--two">
          <article class="tp-card">
            <header class="tp-card__header">
              <h3 class="tp-card__title">Net PnL</h3>
              <div class="tp-card__meta">
                <span class="tp-stat">${escapeHtml(pnlSummary.formatted?.total || formatCurrency(0, currency))}</span>
                <span class="tp-stat tp-stat--muted">Δ ${escapeHtml(pnlSummary.formatted?.change || formatCurrency(0, currency))} (${escapeHtml(pnlSummary.formatted?.changePercent || formatPercent(0))})</span>
                <span class="tp-stat tp-stat--muted">Run-rate ${escapeHtml(pnlSummary.formatted?.runRate || `${formatCurrency(0, currency)}/h`)}</span>
              </div>
            </header>
            ${pnlChart.html}
          </article>
          <article class="tp-card">
            <header class="tp-card__header">
              <h3 class="tp-card__title">Quotes</h3>
              <div class="tp-card__meta">
                <span class="tp-stat">${escapeHtml(formatNumber(quoteSummary.last, { maximumFractionDigits: 4 }))}</span>
                <span class="tp-stat tp-stat--muted">Δ ${escapeHtml(formatNumber(quoteSummary.change, { maximumFractionDigits: 4 }))} (${escapeHtml(formatPercent(quoteSummary.changePercent))})</span>
              </div>
            </header>
            ${quoteChart.html}
          </article>
        </section>
      </section>
    `,
    charts: {
      pnl: pnlChart,
      quotes: quoteChart,
    },
  };
}
