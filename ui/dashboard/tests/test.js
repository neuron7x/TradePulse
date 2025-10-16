import assert from 'assert';
import {
  createStrategyConfigurator,
  compareBacktests,
  exportReport,
  DashboardState,
  renderDashboard,
  DASHBOARD_STYLES,
  formatCurrency,
  formatPercent,
} from '../src/core/index.js';
import {
  TRACEPARENT_HEADER,
  createTraceparent,
  ensureTraceHeaders,
  extractTraceparent,
} from '../src/core/telemetry.js';

const configurator = createStrategyConfigurator([
  { name: 'trend', defaults: { lookback: 20, threshold: 0.6 } },
  { name: 'mean_revert', defaults: { lookback: 10, zScore: 1.5 } },
]);

const updated = configurator.update('trend', { threshold: 0.8 });
assert.strictEqual(updated.threshold, 0.8, 'update should override fields');
assert.deepStrictEqual(configurator.get('trend'), updated, 'config should persist updates');
assert.strictEqual(configurator.list().length, 2, 'list should expose all strategies');

const backtests = [
  {
    metadata: { id: 'bt-1', strategy: 'trend' },
    metrics: { sharpe: 1.5, pnl: 1200 },
  },
  {
    metadata: { id: 'bt-2', strategy: 'mean_revert' },
    metrics: { sharpe: 0.9, pnl: 800 },
  },
];

const comparison = compareBacktests(backtests);
assert.strictEqual(comparison.metric, 'sharpe');
assert.strictEqual(comparison.leaders.best.strategy, 'trend');
assert.ok(comparison.spread.delta > 0, 'spread delta should be positive');

const jsonReport = exportReport(comparison, { format: 'json' });
const parsed = JSON.parse(jsonReport);
assert.strictEqual(parsed.metric, 'sharpe');
assert.strictEqual(parsed.leaders.best.strategy, 'trend');

const csvReport = exportReport(comparison, { format: 'csv', precision: 2 });
assert.ok(csvReport.includes('trend'));
assert.ok(csvReport.includes('1.50'));

const injectionSummary = {
  ranking: [
    {
      id: '=HYPERLINK("http://example.com","click")',
      strategy: '=trend|bold*',
      score: 2.5,
    },
  ],
};
const protectedCsv = exportReport(injectionSummary, { format: 'csv', precision: 1 });
assert.ok(
  protectedCsv.includes("'=HYPERLINK\\("),
  'CSV export should neutralise formula injection attempts by prefixing risky values',
);
assert.ok(
  protectedCsv.includes("'=trend\\|bold\\*"),
  'CSV export should escape Markdown meta characters',
);

const protectedMarkdown = exportReport(injectionSummary, {
  format: 'markdown',
  precision: 1,
});
assert.ok(
  protectedMarkdown.includes("'=trend\\|bold\\*"),
  'Markdown export should escape Markdown meta characters while keeping content readable',
);
assert.ok(
  protectedMarkdown.includes('2.5'),
  'Markdown export should include neutralised numeric score',
);

const state = new DashboardState({ strategies: configurator.list(), backtests });
state.updateStrategy('mean_revert', { zScore: 2.0 });
state.addBacktest({ metadata: { id: 'bt-3', strategy: 'volatility' }, metrics: { sharpe: 2.1 } });
const exportedMarkdown = state.export('markdown');
assert.ok(exportedMarkdown.includes('volatility'));

console.log('dashboard tests passed');

const generatedTraceparent = createTraceparent();
assert.ok(generatedTraceparent.startsWith('00-'));
const headers = ensureTraceHeaders({}, generatedTraceparent).headers;
assert.strictEqual(headers[TRACEPARENT_HEADER], generatedTraceparent);
assert.strictEqual(extractTraceparent(headers), generatedTraceparent);
console.log('telemetry tests passed');

const dashboardView = renderDashboard({
  title: 'Execution Control Center',
  subtitle: 'Live oversight across strategies.',
  tags: ['derivatives', 'equities'],
  currency: 'USD',
  metrics: [
    { label: 'Net Exposure', value: 1250000, kind: 'currency', change: 0.012 },
    { label: 'Open Positions', value: 12 },
  ],
  pnlSeries: [
    { label: 'Today', value: 18250, change: 0.024, target: 25000 },
    { label: 'MTD', value: -12250, change: -0.018, target: 30000 },
  ],
  statusItems: [
    { label: 'Exchange Connectivity', state: 'Operational', level: 'ok', description: 'All exchanges responding within latency budget.' },
    { label: 'Risk Engine', state: 'Degraded', level: 'warning', description: 'Position limits approaching 80% threshold.' },
  ],
});

assert.ok(dashboardView.html.includes('PnL Overview'), 'rendered dashboard should include PnL panel heading');
assert.ok(dashboardView.html.includes('System Status'), 'rendered dashboard should include status panel heading');
assert.ok(dashboardView.html.includes('Operational'), 'status label should be present');
assert.ok(dashboardView.html.includes('−1.80%'), 'negative change should use minus symbol and percentage');
assert.ok(dashboardView.styles.includes('.tp-dashboard'), 'styles should expose dashboard root selector');
assert.strictEqual(dashboardView.styles, DASHBOARD_STYLES, 'render should expose shared stylesheet reference');

assert.strictEqual(formatCurrency(10500), '$10,500');
assert.strictEqual(formatPercent(0.256), '25.6%');
assert.strictEqual(formatPercent(0.025), '2.50%');
console.log('dashboard ui rendering tests passed');

const sanitizedView = renderDashboard({
  statusItems: [
    {
      label: '<b>Automation</b>',
      state: '<script>alert(1)</script>',
      level: 'critical',
      description: 'Check <i>agent</i> response.',
    },
  ],
});

assert.ok(!sanitizedView.html.includes('<script>'), 'dashboard should escape script tags');
assert.ok(sanitizedView.html.includes('&lt;b&gt;Automation&lt;/b&gt;'), 'escaped HTML should remain visible as text');
