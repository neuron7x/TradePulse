import { DashboardState, compareBacktests, exportReport } from './core/index.js';
import { TRACEPARENT_HEADER, createTraceparent, ensureTraceHeaders } from './core/telemetry.js';
import { backtestSamples, performanceSeries, strategyTemplates } from './sample-data.js';

const strategyMeta = new Map(
  strategyTemplates.map(({ name, label, description }) => [name, { label, description }]),
);

const state = new DashboardState({ strategies: strategyTemplates, backtests: backtestSamples });

function generateTraceparent(seed) {
  try {
    return createTraceparent(seed);
  } catch (error) {
    if (typeof window !== 'undefined' && window.crypto?.getRandomValues) {
      const buffer = new Uint8Array(24);
      window.crypto.getRandomValues(buffer);
      const hex = Array.from(buffer, (value) => value.toString(16).padStart(2, '0')).join('');
      const traceId = hex.slice(0, 32);
      const spanId = hex.slice(32, 48);
      return `00-${traceId}-${spanId}-01`;
    }
    console.warn('Traceparent fallback активовано', error); // eslint-disable-line no-console
    return `00-${Date.now().toString(16).padStart(32, '0').slice(-32)}-fallback0000000001-01`;
  }
}

const metricSelect = document.querySelector('[data-metric-select]');
const bestStrategyNode = document.querySelector('[data-best-strategy]');
const bestScoreNode = document.querySelector('[data-best-score]');
const worstStrategyNode = document.querySelector('[data-worst-strategy]');
const worstScoreNode = document.querySelector('[data-worst-score]');
const spreadDeltaNode = document.querySelector('[data-spread-delta]');
const spreadRangeNode = document.querySelector('[data-spread-range]');
const strategyListNode = document.querySelector('[data-strategy-list]');
const backtestTableNode = document.querySelector('[data-backtest-table]');
const backtestForm = document.querySelector('[data-backtest-form]');
const strategyOptionsNode = document.querySelector('[data-strategy-options]');
const strategySeriesNode = document.querySelector('[data-strategy-series]');
const exportButtons = document.querySelectorAll('[data-export]');
const exportOutputNode = document.querySelector('[data-export-output]');
const themeToggle = document.querySelector('[data-theme-toggle]');
const activityFeedNode = document.querySelector('[data-activity-feed]');
const chartCanvas = document.querySelector('[data-equity-chart]');
const traceNode = document.querySelector('[data-trace-id]');

let currentMetric = metricSelect?.value || 'sharpe';
let traceCarrier = ensureTraceHeaders({}, generateTraceparent());
const activityEntries = [];
let lastLeaderSignature = null;
let lastExportFormat = 'json';

function getStrategyLabel(name) {
  return strategyMeta.get(name)?.label || name;
}

function getStrategyDescription(name) {
  return strategyMeta.get(name)?.description || '';
}

function formatNumber(value, options = {}) {
  if (!Number.isFinite(value)) {
    return '—';
  }
  return new Intl.NumberFormat('uk-UA', options).format(value);
}

function formatScore(metric, value) {
  if (!Number.isFinite(value)) {
    return '—';
  }
  if (metric === 'pnl') {
    return `${formatNumber(value, { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })}`;
  }
  if (metric === 'winRate') {
    return `${formatNumber(value, { maximumFractionDigits: 1 })}%`;
  }
  return formatNumber(value, { maximumFractionDigits: 2 });
}

function formatDrawdown(value) {
  if (!Number.isFinite(value)) {
    return '—';
  }
  return `${formatNumber(value, { maximumFractionDigits: 1 })}%`;
}

function formatDate(value) {
  if (!value) {
    return '—';
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat('uk-UA', { day: '2-digit', month: 'short' }).format(date);
}

function logActivity(message, level = 'info') {
  activityEntries.unshift({ message, level, timestamp: new Date() });
  if (activityEntries.length > 12) {
    activityEntries.pop();
  }
  renderActivity();
}

function renderActivity() {
  if (!activityFeedNode) return;
  activityFeedNode.innerHTML = '';
  activityEntries.forEach((entry) => {
    const item = document.createElement('li');
    item.dataset.level = entry.level;
    const message = document.createElement('span');
    message.textContent = entry.message;
    const time = document.createElement('time');
    time.dateTime = entry.timestamp.toISOString();
    time.textContent = entry.timestamp.toLocaleTimeString('uk-UA', {
      hour: '2-digit',
      minute: '2-digit',
    });
    item.append(message, time);
    activityFeedNode.append(item);
  });
}

function renderTrace(reason) {
  const trace = traceCarrier.headers?.[TRACEPARENT_HEADER];
  if (!traceNode || !trace) return;
  const compact = `${trace.slice(0, 8)}…${trace.slice(-6)}`;
  traceNode.textContent = compact;
  if (reason) {
    logActivity(`Trace оновлено: ${reason}`, 'success');
  }
}

function regenerateTrace(reason) {
  traceCarrier = ensureTraceHeaders(traceCarrier, generateTraceparent());
  renderTrace(reason);
}

function renderStrategies() {
  if (!strategyListNode) return;
  const strategies = state.configurator.list();
  strategyListNode.innerHTML = '';

  strategies.forEach((strategy) => {
    const card = document.createElement('article');
    card.className = 'strategy-card';

    const title = document.createElement('h3');
    title.textContent = getStrategyLabel(strategy.name);

    const description = document.createElement('p');
    description.textContent = getStrategyDescription(strategy.name);

    const form = document.createElement('form');
    form.dataset.strategy = strategy.name;

    Object.entries(strategy.params).forEach(([key, value]) => {
      const label = document.createElement('label');
      label.textContent = key;
      const input = document.createElement('input');
      input.name = key;
      const numeric = typeof value === 'number';
      input.type = numeric ? 'number' : 'text';
      input.step = numeric ? '0.01' : undefined;
      input.value = value;
      input.addEventListener('change', (event) => {
        const nextValue = numeric ? Number(event.target.value) : event.target.value;
        const payload = { [key]: numeric && !Number.isNaN(nextValue) ? nextValue : event.target.value };
        const updated = state.updateStrategy(strategy.name, payload);
        logActivity(
          `Оновлено ${getStrategyLabel(strategy.name)}: ${key} → ${updated[key]}`,
          'info',
        );
        regenerateTrace(`параметр ${key}`);
      });
      label.append(input);
      form.append(label);
    });

    card.append(title, description, form);
    strategyListNode.append(card);
  });
}

function deriveStatus(metrics) {
  if (!metrics) return { label: '—', badge: 'badge--green', level: 'info' };
  if (metrics.sharpe >= 1.5 && metrics.maxDrawdown >= -7 && metrics.winRate >= 55) {
    return { label: 'Live-ready', badge: 'badge--blue', level: 'success' };
  }
  if (metrics.sharpe >= 1.0 && metrics.maxDrawdown >= -10) {
    return { label: 'Staging', badge: 'badge--green', level: 'info' };
  }
  return { label: 'Review', badge: 'badge--warning', level: 'warning' };
}

function renderBacktests() {
  if (!backtestTableNode) return;
  const backtests = state.backtests.slice();
  const summary = compareBacktests(backtests, currentMetric);
  backtestTableNode.innerHTML = '';

  summary.ranking.forEach((entry) => {
    const raw = backtests.find((item) => item.metadata?.id === entry.id) || {};
    const metrics = raw.metrics || {};
    const status = deriveStatus(metrics);

    const row = document.createElement('tr');
    row.dataset.status = status.level;
    if (raw.metadata?.note) {
      row.title = raw.metadata.note;
    }

    const idCell = document.createElement('td');
    idCell.textContent = entry.id;

    const strategyCell = document.createElement('td');
    strategyCell.textContent = getStrategyLabel(entry.strategy);

    const sharpeCell = document.createElement('td');
    sharpeCell.textContent = formatNumber(metrics.sharpe, { maximumFractionDigits: 2 });

    const pnlCell = document.createElement('td');
    pnlCell.textContent = formatNumber(metrics.pnl, {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0,
    });

    const ddCell = document.createElement('td');
    ddCell.textContent = formatDrawdown(metrics.maxDrawdown);

    const winRateCell = document.createElement('td');
    winRateCell.textContent = formatNumber(metrics.winRate, { maximumFractionDigits: 1 });

    const statusCell = document.createElement('td');
    const badge = document.createElement('span');
    badge.className = `badge ${status.badge}`;
    badge.textContent = status.label;
    statusCell.append(badge);

    row.append(
      idCell,
      strategyCell,
      sharpeCell,
      pnlCell,
      ddCell,
      winRateCell,
      statusCell,
    );

    const startedAt = raw.metadata?.startedAt;
    const endedAt = raw.metadata?.endedAt;
    if (startedAt || endedAt) {
      const extra = document.createElement('td');
      extra.className = 'is-hidden';
      extra.textContent = `${formatDate(startedAt)} → ${formatDate(endedAt)}`;
      row.append(extra);
    }

    backtestTableNode.append(row);
  });
}

function renderMetrics() {
  const comparison = state.compare(currentMetric);
  const { leaders, spread } = comparison;

  bestStrategyNode.textContent = leaders.best
    ? getStrategyLabel(leaders.best.strategy)
    : '—';
  bestScoreNode.textContent = leaders.best
    ? formatScore(currentMetric, leaders.best.score)
    : '—';

  worstStrategyNode.textContent = leaders.worst
    ? getStrategyLabel(leaders.worst.strategy)
    : '—';
  worstScoreNode.textContent = leaders.worst
    ? formatScore(currentMetric, leaders.worst.score)
    : '—';

  spreadDeltaNode.textContent = formatScore(currentMetric, spread.delta);
  spreadRangeNode.textContent = `${formatScore(currentMetric, spread.min)} – ${formatScore(
    currentMetric,
    spread.max,
  )}`;

  if (leaders.best) {
    const signature = `${leaders.best.strategy}:${leaders.best.score.toFixed(4)}`;
    if (signature !== lastLeaderSignature) {
      lastLeaderSignature = signature;
      logActivity(
        `Лідирує ${getStrategyLabel(leaders.best.strategy)} зі значенням ${formatScore(
          currentMetric,
          leaders.best.score,
        )}`,
        'success',
      );
    }
  }
}

function renderStrategySelectors() {
  const strategies = state.configurator.list();
  if (strategyOptionsNode) {
    strategyOptionsNode.innerHTML = '';
    strategies.forEach((strategy) => {
      const option = document.createElement('option');
      option.value = strategy.name;
      option.textContent = getStrategyLabel(strategy.name);
      strategyOptionsNode.append(option);
    });
  }

  if (strategySeriesNode) {
    strategySeriesNode.innerHTML = '';
    strategies.forEach((strategy) => {
      const option = document.createElement('option');
      option.value = strategy.name;
      option.textContent = getStrategyLabel(strategy.name);
      strategySeriesNode.append(option);
    });
    strategySeriesNode.value = strategies[0]?.name || '';
  }
}

function renderChart(strategyName) {
  if (!chartCanvas) return;
  const ctx = chartCanvas.getContext('2d');
  const series = performanceSeries[strategyName] || [];
  ctx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);

  if (!series.length) {
    ctx.fillStyle = 'rgba(148, 163, 184, 0.6)';
    ctx.font = '16px Inter, sans-serif';
    ctx.fillText('Немає даних для вибраної стратегії', 16, chartCanvas.height / 2);
    return;
  }

  const padding = 24;
  const width = chartCanvas.width - padding * 2;
  const height = chartCanvas.height - padding * 2;
  const values = series.map((point) => point.equity);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const xStep = width / (series.length - 1 || 1);

  const gradient = ctx.createLinearGradient(0, padding, 0, padding + height);
  gradient.addColorStop(0, 'rgba(56, 189, 248, 0.35)');
  gradient.addColorStop(1, 'rgba(14, 116, 144, 0.02)');

  ctx.beginPath();
  series.forEach((point, index) => {
    const x = padding + xStep * index;
    const y = padding + height - ((point.equity - min) / range) * height;
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });

  ctx.lineTo(padding + width, padding + height);
  ctx.lineTo(padding, padding + height);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.beginPath();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.95)';
  series.forEach((point, index) => {
    const x = padding + xStep * index;
    const y = padding + height - ((point.equity - min) / range) * height;
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();

  ctx.fillStyle = 'rgba(226, 232, 240, 0.9)';
  ctx.font = '12px Inter, sans-serif';
  ctx.textAlign = 'center';
  series.forEach((point, index) => {
    const x = padding + xStep * index;
    const y = padding + height - ((point.equity - min) / range) * height;
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fill();
    if (index === series.length - 1) {
      ctx.textAlign = 'right';
      ctx.fillText(formatNumber(point.equity, { maximumFractionDigits: 0 }), x, y - 12);
      ctx.textAlign = 'center';
    }
  });

  ctx.fillStyle = 'rgba(148, 163, 184, 0.8)';
  ctx.textAlign = 'left';
  ctx.fillText(`Початок: ${formatDate(series[0].date)}`, padding, chartCanvas.height - 8);
  ctx.textAlign = 'right';
  ctx.fillText(`Кінець: ${formatDate(series[series.length - 1].date)}`, chartCanvas.width - padding, chartCanvas.height - 8);
}

function updateExport(format = 'json', { silent = false } = {}) {
  const summary = state.compare(currentMetric);
  const payload = {
    generatedAt: new Date().toISOString(),
    metric: summary.metric,
    ranking: summary.ranking,
    leaders: summary.leaders,
    spread: summary.spread,
    strategies: state.configurator.list(),
  };
  const report = exportReport(payload, { format, precision: format === 'csv' ? 2 : 4 });
  if (exportOutputNode) {
    exportOutputNode.value = report;
  }
  lastExportFormat = format;
  if (!silent) {
    logActivity(`Експортовано звіт у форматі ${format.toUpperCase()}`, 'success');
  }
}

function handleBacktestSubmit(event) {
  event.preventDefault();
  const formData = new FormData(backtestForm);
  const id = formData.get('id');
  const strategy = formData.get('strategy');
  const sharpe = Number(formData.get('sharpe'));
  const pnl = Number(formData.get('pnl'));
  const maxDrawdown = Number(formData.get('maxDrawdown'));
  const winRate = Number(formData.get('winRate'));
  const note = formData.get('note');

  if (!id || !strategy) {
    return;
  }

  state.addBacktest({
    metadata: {
      id,
      strategy,
      startedAt: new Date().toISOString(),
      endedAt: new Date().toISOString(),
      note,
    },
    metrics: {
      sharpe,
      pnl,
      maxDrawdown,
      winRate,
    },
  });

  logActivity(`Додано бектест ${id} для ${getStrategyLabel(strategy)}`, 'success');
  regenerateTrace('новий бектест');
  backtestForm.reset();
  renderAll();
}

function renderAll() {
  renderMetrics();
  renderBacktests();
  renderStrategies();
  renderTrace();
  if (strategySeriesNode?.value) {
    renderChart(strategySeriesNode.value);
  }
  updateExport(lastExportFormat, { silent: true });
}

metricSelect?.addEventListener('change', (event) => {
  currentMetric = event.target.value;
  logActivity(`Метрика для порівняння → ${currentMetric}`, 'info');
  renderMetrics();
  renderBacktests();
});

strategySeriesNode?.addEventListener('change', (event) => {
  renderChart(event.target.value);
  logActivity(`Оновлено серію для графіку → ${getStrategyLabel(event.target.value)}`, 'info');
});

backtestForm?.addEventListener('submit', handleBacktestSubmit);

themeToggle?.addEventListener('change', (event) => {
  const isDark = event.target.checked;
  document.documentElement.dataset.theme = isDark ? 'dark' : 'light';
  logActivity(`Тема → ${isDark ? 'Темна' : 'Світла'}`, 'info');
});

exportButtons.forEach((button) => {
  button.addEventListener('click', () => updateExport(button.dataset.export));
});

renderStrategySelectors();
renderAll();
logActivity('Контрольна панель завантажена', 'success');
