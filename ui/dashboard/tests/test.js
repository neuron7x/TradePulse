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
  createGlobalFiltersStore,
  DEFAULT_FILTERS,
  DEFAULT_TIMEFRAMES,
  DEFAULT_STRATEGIES,
  serialiseFiltersToQuery,
  deserializeFiltersFromQuery,
  attachFilterControls,
  createRestDataSource,
  createWebSocketDataSource,
  renderFiltersPanel,
  buildQueryFromFilters,
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

const filtersMarkup = renderFiltersPanel(DEFAULT_FILTERS, {
  timeframes: DEFAULT_TIMEFRAMES,
  strategies: DEFAULT_STRATEGIES,
});
assert.ok(filtersMarkup.includes('tp-filters'), 'filters panel should render filter container');
assert.ok(filtersMarkup.includes('data-tp-filter-control="symbols"'), 'filters panel should expose control dataset');

class MemoryStorage {
  constructor() {
    this.store = new Map();
  }

  getItem(key) {
    return this.store.has(key) ? this.store.get(key) : null;
  }

  setItem(key, value) {
    this.store.set(key, value);
  }

  removeItem(key) {
    this.store.delete(key);
  }
}

const storage = new MemoryStorage();
storage.setItem(
  'tradepulse:dashboard:filters',
  JSON.stringify({ symbols: ['SOL-USD'], timeframe: '1d', strategy: 'trend_following' }),
);

const locationStub = {
  pathname: '/dashboard',
  hash: '',
  search:
    '?symbols=BTC-USD,ETH-USD&timeframe=4h&strategy=mean_reversion&from=2024-01-01T00:00:00.000Z&to=2024-01-05T00:00:00.000Z',
};

const historyStub = {
  lastUrl: null,
  replaceState(_state, _title, url) {
    this.lastUrl = url;
  },
};

const eventTarget = new EventTarget();
let lastBroadcast = null;
eventTarget.addEventListener('tradepulse:filters:change', (event) => {
  lastBroadcast = event.detail;
});

const filtersStore = createGlobalFiltersStore({
  allowedSymbols: ['BTC-USD', 'ETH-USD', 'SOL-USD'],
  allowedTimeframes: ['1h', '4h', '1d'],
  allowedStrategies: ['trend_following', 'mean_reversion'],
  storage,
  location: locationStub,
  history: historyStub,
  eventTarget,
  defaults: {
    symbols: ['SOL-USD'],
    timeframe: '1h',
    strategy: 'trend_following',
    dateRange: {
      from: '2023-12-01T00:00:00.000Z',
      to: '2023-12-31T00:00:00.000Z',
    },
  },
});

const snapshots = [];
const unsubscribeFilters = filtersStore.subscribe((snapshot) => {
  snapshots.push(snapshot);
});

assert.ok(snapshots.length >= 1, 'subscription should emit initial filters snapshot');
assert.deepStrictEqual(filtersStore.getState().symbols, ['BTC-USD', 'ETH-USD']);
assert.strictEqual(filtersStore.getState().timeframe, '4h');
assert.strictEqual(filtersStore.getState().strategy, 'mean_reversion');
assert.ok(historyStub.lastUrl === null, 'history should not be mutated before an update');

assert.throws(() => filtersStore.setState({ timeframe: '2h' }), /Unsupported timeframe/);

filtersStore.setState({ timeframe: '1d' });
assert.strictEqual(filtersStore.getState().timeframe, '1d');
assert.ok(historyStub.lastUrl.includes('timeframe=1d'), 'history should reflect filter updates');
assert.strictEqual(lastBroadcast.timeframe, '1d', 'broadcast should include latest filters');
assert.ok(
  storage
    .getItem('tradepulse:dashboard:filters')
    .includes('"timeframe":"1d"'),
  'storage should persist latest timeframe',
);

filtersStore.update((draft) => ({
  dateRange: {
    ...draft.dateRange,
    to: '2024-01-08T00:00:00.000Z',
  },
}));
assert.strictEqual(filtersStore.getState().dateRange.to, '2024-01-08T00:00:00.000Z');

const serialised = filtersStore.serialize();
assert.ok(serialised.includes('timeframe=1d'));
assert.strictEqual(serialised, serialiseFiltersToQuery(filtersStore.getState()));

const parsedFilters = deserializeFiltersFromQuery('symbols=ETH-USD&timeframe=4h&strategy=trend_following', {
  allowedSymbols: ['BTC-USD', 'ETH-USD'],
  allowedTimeframes: ['1h', '4h'],
  allowedStrategies: ['trend_following'],
});
assert.strictEqual(parsedFilters.timeframe, '4h');

const filterControls = [
  {
    value: '',
    dataset: { tpFilterControl: 'symbols', tpFilterEvent: 'input' },
    listeners: new Map(),
    addEventListener(event, handler) {
      this.listeners.set(event, handler);
    },
    removeEventListener(event) {
      this.listeners.delete(event);
    },
    trigger(event) {
      const handler = this.listeners.get(event);
      handler?.();
    },
  },
  {
    value: '',
    dataset: { tpFilterControl: 'timeframe' },
    listeners: new Map(),
    addEventListener(event, handler) {
      this.listeners.set(event, handler);
    },
    removeEventListener(event) {
      this.listeners.delete(event);
    },
    trigger(event) {
      const handler = this.listeners.get(event);
      handler?.();
    },
  },
  {
    value: '',
    dataset: { tpFilterControl: 'strategy' },
    listeners: new Map(),
    addEventListener(event, handler) {
      this.listeners.set(event, handler);
    },
    removeEventListener(event) {
      this.listeners.delete(event);
    },
    trigger(event) {
      const handler = this.listeners.get(event);
      handler?.();
    },
  },
  {
    value: '',
    dataset: { tpFilterControl: 'dateRange.from' },
    listeners: new Map(),
    addEventListener(event, handler) {
      this.listeners.set(event, handler);
    },
    removeEventListener(event) {
      this.listeners.delete(event);
    },
    trigger(event) {
      const handler = this.listeners.get(event);
      handler?.();
    },
  },
  {
    value: '',
    dataset: { tpFilterControl: 'dateRange.to' },
    listeners: new Map(),
    addEventListener(event, handler) {
      this.listeners.set(event, handler);
    },
    removeEventListener(event) {
      this.listeners.delete(event);
    },
    trigger(event) {
      const handler = this.listeners.get(event);
      handler?.();
    },
  },
];

const rootStub = {
  querySelectorAll() {
    return filterControls;
  },
};

const detachControls = attachFilterControls(filtersStore, rootStub);
assert.strictEqual(filterControls[0].value, 'BTC-USD, ETH-USD');

filterControls[0].value = 'SOL-USD';
filterControls[0].trigger('input');
assert.deepStrictEqual(filtersStore.getState().symbols, ['SOL-USD']);

filterControls[1].value = '4h';
filterControls[1].trigger('change');
assert.strictEqual(filtersStore.getState().timeframe, '4h');

filterControls[3].value = '2024-01-02';
filterControls[3].trigger('change');
assert.ok(filtersStore.getState().dateRange.from.includes('2024-01-02'));

detachControls();
assert.strictEqual(filterControls[0].listeners.size, 0, 'detaching controls should remove listeners');

const queryPayload = buildQueryFromFilters(filtersStore.getState());
assert.strictEqual(queryPayload.strategy, filtersStore.getState().strategy);

const requests = [];
async function fakeFetch(url, init) {
  requests.push({ url, init });
  return {
    ok: true,
    status: 200,
    async json() {
      return { url };
    },
  };
}

const restSource = createRestDataSource({
  baseUrl: 'https://api.tradepulse.local/',
  endpoint: '/metrics',
  fetchImpl: fakeFetch,
  filtersStore,
});

let lastRequest = null;
const stopListening = restSource.onRequest((payload) => {
  lastRequest = payload;
});

await restSource.fetch({
  query: { limit: 10 },
  init: { headers: { 'x-test': '1' } },
});

assert.ok(requests[0].url.includes('limit=10'));
assert.ok(requests[0].url.includes('symbols=SOL-USD'));

filtersStore.setState({ strategy: 'trend_following' });
assert.strictEqual(lastRequest.query.strategy, 'trend_following');

stopListening();
restSource.dispose();

const sockets = [];

class StubWebSocket {
  constructor(url) {
    this.url = url;
    this.OPEN = 1;
    this.readyState = 0;
    this.sent = [];
    this.closeCalled = false;
    sockets.push(this);
    setTimeout(() => {
      this.readyState = this.OPEN;
      this.onopen?.({});
    }, 0);
  }

  send(message) {
    this.sent.push(message);
  }

  close() {
    this.closeCalled = true;
  }

  addEventListener(event, handler) {
    if (event === 'open') {
      this.onopen = handler;
    }
  }
}

function delay(ms = 0) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

const wsSource = createWebSocketDataSource({
  url: 'wss://stream.tradepulse.local',
  WebSocketImpl: StubWebSocket,
  filtersStore,
  serializer: (payload) => JSON.stringify(payload),
});

await delay();
const wsSocket = sockets[0];
assert.ok(wsSocket.sent[0].includes('subscribe'), 'websocket should send subscribe payload on open');

filtersStore.setState({ timeframe: '1d' });
await delay();
const updateMessages = wsSocket.sent.filter((entry) => entry.includes('update_filters'));
const latestUpdate = updateMessages[updateMessages.length - 1];
assert.ok(latestUpdate.includes('update_filters'), 'websocket should stream filter update payloads');
assert.ok(latestUpdate.includes('1d'), 'websocket should stream updated filters');

wsSource.dispose();
assert.ok(wsSocket.closeCalled, 'websocket source should close socket on dispose');

filtersStore.reset();
assert.deepStrictEqual(filtersStore.getState().symbols, ['SOL-USD']);
unsubscribeFilters();

console.log('global filters and data source tests passed');
