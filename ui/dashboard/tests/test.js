import assert from 'assert';
import { JSDOM } from 'jsdom';
import {
  createStrategyConfigurator,
  compareBacktests,
  exportReport,
  DashboardState,
  renderDashboard,
  DASHBOARD_STYLES,
  formatCurrency,
  formatPercent,
  RealtimeEventClient,
  deserializeEventEnvelope,
  CHANNEL_STATUS,
  bindClientToStore,
  computeBackoffDelay,
  createDashboardStore,
  DashboardStore,
} from '../src/core/index.js';
import {
  TRACEPARENT_HEADER,
  createTraceparent,
  ensureTraceHeaders,
  extractTraceparent,
} from '../src/core/telemetry.js';

const dom = new JSDOM('<!doctype html><html><body></body></html>');
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.navigator = { userAgent: 'node.js' };
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.CustomEvent = dom.window.CustomEvent;

function resetDom() {
  document.body.innerHTML = '';
}

function renderMarkup(markup) {
  const container = document.createElement('div');
  container.innerHTML = markup;
  document.body.appendChild(container);
  return container;
}

function findByText(text) {
  const elements = Array.from(document.body.querySelectorAll('*'));
  return elements.find((element) => element.textContent && element.textContent.includes(text));
}

function assertText(text, message) {
  const node = findByText(text);
  assert.ok(node, message);
  return node;
}

function assertIncludes(haystack, needle, message) {
  assert.ok(haystack.includes(needle), message);
}

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
assertIncludes(csvReport, 'trend', 'CSV report should include strategy name');
assertIncludes(csvReport, '1.50', 'CSV report should include formatted score');

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
assertIncludes(
  protectedCsv,
  "'=HYPERLINK\\(",
  'CSV export should neutralise formula injection attempts by prefixing risky values',
);
assertIncludes(
  protectedCsv,
  "'=trend\\|bold\\*",
  'CSV export should escape Markdown meta characters',
);

const protectedMarkdown = exportReport(injectionSummary, {
  format: 'markdown',
  precision: 1,
});
assertIncludes(
  protectedMarkdown,
  "'=trend\\|bold\\*",
  'Markdown export should escape Markdown meta characters while keeping content readable',
);
assertIncludes(protectedMarkdown, '2.5', 'Markdown export should include neutralised numeric score');

const state = new DashboardState({ strategies: configurator.list(), backtests });
state.updateStrategy('mean_revert', { zScore: 2.0 });
state.addBacktest({ metadata: { id: 'bt-3', strategy: 'volatility' }, metrics: { sharpe: 2.1 } });
const exportedMarkdown = state.export('markdown');
assertIncludes(exportedMarkdown, 'volatility', 'Export should include new strategy');

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
    {
      label: 'Exchange Connectivity',
      state: 'Operational',
      level: 'ok',
      description: 'All exchanges responding within latency budget.',
    },
    {
      label: 'Risk Engine',
      state: 'Degraded',
      level: 'warning',
      description: 'Position limits approaching 80% threshold.',
    },
  ],
  channel: {
    status: 'open',
    label: 'Connected',
    latencyMs: 42,
  },
});

resetDom();
renderMarkup(dashboardView.html);
assertText('PnL Overview', 'rendered dashboard should include PnL panel heading');
assertText('System Status', 'rendered dashboard should include status panel heading');
assertText('Operational', 'status label should be present');
assertText('Connected', 'channel indicator should display label');
const indicator = document.querySelector('.tp-channel-indicator');
assert.ok(indicator, 'channel indicator should render with expected class');
assert.ok(indicator.className.includes('tp-channel-indicator--open'), 'channel indicator should reflect status');
assertIncludes(dashboardView.html, '−1.80%', 'negative change should use minus symbol and percentage');
assert.ok(dashboardView.styles.includes('.tp-dashboard'), 'styles should expose dashboard root selector');
assert.strictEqual(dashboardView.styles, DASHBOARD_STYLES, 'render should expose shared stylesheet reference');
resetDom();

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
  channel: {
    status: 'error',
    label: 'Error',
    message: 'Auth failure',
  },
});

assert.ok(!sanitizedView.html.includes('<script>'), 'dashboard should escape script tags');
assert.ok(
  sanitizedView.html.includes('&lt;b&gt;Automation&lt;/b&gt;'),
  'escaped HTML should remain visible as text',
);

function createTickEvent(overrides = {}) {
  return {
    event_id: `tick-${Math.random().toString(16).slice(2)}`,
    schema_version: '1',
    symbol: 'ES',
    timestamp: Date.now(),
    bid_price: 4200.25,
    ask_price: 4200.75,
    last_price: 4200.5,
    ...overrides,
  };
}

function createOrderEvent(overrides = {}) {
  return {
    event_id: `ord-${Math.random().toString(16).slice(2)}`,
    schema_version: '1',
    symbol: 'ES',
    timestamp: Date.now(),
    order_id: `order-${Math.random().toString(16).slice(2)}`,
    side: 'BUY',
    order_type: 'LIMIT',
    quantity: 2,
    price: 4200.5,
    time_in_force: 'DAY',
    metadata: { venue: 'CME' },
    ...overrides,
  };
}

function createFillEvent(overrides = {}) {
  return {
    event_id: `fill-${Math.random().toString(16).slice(2)}`,
    schema_version: '1',
    symbol: 'ES',
    timestamp: Date.now(),
    order_id: `order-${Math.random().toString(16).slice(2)}`,
    fill_id: `fill-${Math.random().toString(16).slice(2)}`,
    status: 'FILLED',
    filled_qty: 1,
    fill_price: 4201,
    metadata: { venue: 'CME' },
    ...overrides,
  };
}

const tickEnvelope = JSON.stringify({ type: 'TickEvent', payload: createTickEvent() });
const tickEvent = deserializeEventEnvelope(tickEnvelope);
assert.strictEqual(tickEvent.type, 'tick', 'tick envelope should normalise event type');
assert.strictEqual(tickEvent.payload.symbol, 'ES');

const fillEnvelope = deserializeEventEnvelope({ payload: createFillEvent(), event_type: 'FillEvent' });
assert.strictEqual(fillEnvelope.type, 'fill');

let thrown = false;
try {
  deserializeEventEnvelope('{}');
} catch (error) {
  thrown = true;
  assert.ok(
    /Event payload is empty|Unknown event type/.test(error.message),
    'deserializer should flag invalid messages',
  );
}
assert.ok(thrown, 'invalid payload should throw');

const backoffNoJitter = computeBackoffDelay({
  attempts: 3,
  strategy: { initialDelayMs: 100, maxDelayMs: 1000, multiplier: 2, jitter: 0 },
});
assert.strictEqual(backoffNoJitter, 400, 'backoff should exponentiate with attempts');

const store = createDashboardStore();
let latestState = store.getState();
const storeEvents = [];
const unsubscribeStore = store.subscribe((snapshot) => {
  latestState = snapshot;
  storeEvents.push(snapshot);
});

store.handleEvent({ type: 'tick', payload: createTickEvent({ symbol: 'ES', bid_price: 4200, ask_price: 4201 }) });
store.handleEvent({ type: 'tick', payload: createTickEvent({ symbol: 'NQ', bid_price: 13500, ask_price: 13501 }) });
store.handleEvent({ type: 'order', payload: createOrderEvent({ symbol: 'ES' }) });
store.handleEvent({ type: 'fill', payload: createFillEvent({ symbol: 'ES', filled_qty: 3, fill_price: 4202 }) });

assert.strictEqual(latestState.ticks.length, 2, 'store should capture tick history');
assert.strictEqual(latestState.orders.length, 1, 'store should capture orders');
assert.strictEqual(latestState.fills.length, 1, 'store should capture fills');
assert.ok(latestState.metrics.length >= 2, 'store should derive metrics from ticks');
assert.ok(latestState.pnlSeries.length >= 1, 'store should derive pnl series from fills');
assert.strictEqual(latestState.statusItems[0].label, 'Realtime Feed');
assert.strictEqual(latestState.statusItems[0].state, 'Idle');

store.updateChannel({ status: CHANNEL_STATUS.OPEN, latencyMs: 12 });
const onlineState = store.getState();
assert.strictEqual(onlineState.channel.status, CHANNEL_STATUS.OPEN);
assert.strictEqual(onlineState.statusItems[0].state, 'Streaming');

store.setStatusItems([
  {
    label: 'Risk Limits',
    state: 'Nominal',
    level: 'ok',
    description: 'All limits below threshold.',
  },
]);
const statusState = store.getState();
assert.strictEqual(statusState.statusItems.length, 2, 'store should merge custom status items');
unsubscribeStore();
console.log('dashboard store tests passed');

class FakeScheduler {
  constructor() {
    this.tasks = new Map();
    this.nextId = 1;
    this.lastDelay = null;
  }

  setTimeout(fn, delay) {
    const id = this.nextId++;
    this.tasks.set(id, { fn, delay });
    this.lastDelay = delay;
    return id;
  }

  clearTimeout(id) {
    this.tasks.delete(id);
  }

  flush() {
    const tasks = Array.from(this.tasks.values());
    this.tasks.clear();
    tasks.forEach(({ fn }) => fn());
  }
}

class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  constructor() {
    this.readyState = FakeWebSocket.CONNECTING;
    this.listeners = new Map();
    this.sent = [];
  }

  addEventListener(type, handler) {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, new Set());
    }
    this.listeners.get(type).add(handler);
  }

  removeEventListener(type, handler) {
    this.listeners.get(type)?.delete(handler);
  }

  dispatch(type, event = {}) {
    (this.listeners.get(type) || []).forEach((handler) => handler(event));
  }

  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.dispatch('open', {});
  }

  send(payload) {
    this.sent.push(payload);
  }

  receive(data) {
    this.dispatch('message', { data });
  }

  error(message) {
    this.dispatch('error', { message });
  }

  close(event = { code: 1000, reason: 'closed' }) {
    this.readyState = FakeWebSocket.CLOSED;
    this.dispatch('close', event);
  }
}

function createSocketFactory() {
  const sockets = [];
  const factory = () => {
    const socket = new FakeWebSocket();
    sockets.push(socket);
    return socket;
  };
  factory.sockets = sockets;
  return factory;
}

const scheduler = new FakeScheduler();
const socketFactory = createSocketFactory();
const client = new RealtimeEventClient({
  url: 'ws://test.example',
  webSocketFactory: socketFactory,
  scheduler,
  backoff: { initialDelayMs: 50, maxDelayMs: 200, multiplier: 2, jitter: 0 },
  heartbeatInterval: 10,
  logger: {
    warn() {},
    error() {},
  },
});

const receivedEvents = [];
const statusTransitions = [];
client.subscribe((event) => receivedEvents.push(event));
client.onStatusChange((status) => statusTransitions.push(status.status));

client.connect();
assert.strictEqual(statusTransitions[0], CHANNEL_STATUS.IDLE);
assert.strictEqual(statusTransitions.at(-1), CHANNEL_STATUS.CONNECTING);
const firstSocket = socketFactory.sockets[0];
firstSocket.open();
assert.strictEqual(statusTransitions.at(-1), CHANNEL_STATUS.OPEN, 'client should emit open status on connect');
firstSocket.receive(JSON.stringify({ type: 'TickEvent', payload: createTickEvent() }));
assert.strictEqual(receivedEvents.length, 1, 'client should emit parsed events');
assert.strictEqual(receivedEvents[0].type, 'tick');

firstSocket.error('temporary');
assert.strictEqual(statusTransitions.at(-1), CHANNEL_STATUS.ERROR, 'client should emit error status');
firstSocket.close({ code: 1006, reason: 'abnormal' });
assert.strictEqual(statusTransitions.at(-1), CHANNEL_STATUS.RECONNECTING, 'client should schedule reconnect');
assert.strictEqual(scheduler.lastDelay, 50, 'client should apply backoff strategy');

scheduler.flush();
const secondSocket = socketFactory.sockets[1];
secondSocket.open();
assert.strictEqual(statusTransitions.at(-1), CHANNEL_STATUS.OPEN, 'client should reconnect');
client.close();
assert.strictEqual(client.state.status, CHANNEL_STATUS.CLOSED, 'close should mark state as closed');
console.log('realtime client tests passed');

const integrationStore = new DashboardStore();
const integrationClient = new RealtimeEventClient({
  url: 'ws://integration',
  webSocketFactory: createSocketFactory(),
  scheduler: new FakeScheduler(),
  backoff: { initialDelayMs: 10, maxDelayMs: 20, multiplier: 2, jitter: 0 },
  heartbeatInterval: 0,
  logger: { warn() {}, error() {} },
});

const detach = bindClientToStore(integrationClient, integrationStore);
const integrationSocketFactory = integrationClient.webSocketFactory;
integrationClient.connect();
const socket = integrationSocketFactory.sockets[0];
socket.open();
socket.receive(JSON.stringify({ type: 'TickEvent', payload: createTickEvent({ symbol: 'GC' }) }));
socket.receive(JSON.stringify({ type: 'OrderEvent', payload: createOrderEvent({ symbol: 'GC' }) }));
socket.receive(JSON.stringify({ type: 'FillEvent', payload: createFillEvent({ symbol: 'GC' }) }));
const integrationSnapshot = integrationStore.getState();
assert.strictEqual(integrationSnapshot.ticks[0].symbol, 'GC');
assert.strictEqual(integrationSnapshot.orders.length, 1);
assert.strictEqual(integrationSnapshot.fills.length, 1);
detach();
integrationClient.close();
console.log('store binding smoke test passed');

resetDom();
renderMarkup(
  renderDashboard({
    statusItems: integrationSnapshot.statusItems,
    metrics: integrationSnapshot.metrics,
    pnlSeries: integrationSnapshot.pnlSeries,
    channel: {
      status: integrationSnapshot.channel.status,
      label: integrationSnapshot.statusItems[0].state,
    },
  }).html,
);
assertText('Realtime Feed', 'smoke render should include realtime feed status');
resetDom();

console.log('integration smoke tests passed');
