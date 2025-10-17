import { CHANNEL_STATUS } from '../services/realtime_client.js';

const MAX_TICKS = 200;
const MAX_ORDERS = 100;
const MAX_FILLS = 100;

const CHANNEL_LEVEL = {
  [CHANNEL_STATUS.IDLE]: 'warning',
  [CHANNEL_STATUS.CONNECTING]: 'warning',
  [CHANNEL_STATUS.OPEN]: 'ok',
  [CHANNEL_STATUS.RECONNECTING]: 'warning',
  [CHANNEL_STATUS.CLOSED]: 'warning',
  [CHANNEL_STATUS.ERROR]: 'critical',
};

const DEFAULT_CHANNEL = {
  status: CHANNEL_STATUS.IDLE,
  attempt: 0,
  delay: 0,
  lastError: null,
  updatedAt: Date.now(),
};

function clampHistory(history, limit) {
  if (history.length <= limit) {
    return history;
  }
  return history.slice(history.length - limit);
}

function calculateMidpoint(tick) {
  if (Number.isFinite(tick.last_price)) {
    return tick.last_price;
  }
  if (Number.isFinite(tick.bid_price) && Number.isFinite(tick.ask_price)) {
    return (tick.bid_price + tick.ask_price) / 2;
  }
  return null;
}

function deriveMetrics(latestTicks, previousMidpoints) {
  const metrics = [];
  const nextMidpoints = { ...previousMidpoints };
  Object.entries(latestTicks).forEach(([symbol, tick]) => {
    const midpoint = calculateMidpoint(tick);
    if (Number.isFinite(midpoint)) {
      const previous = nextMidpoints[symbol];
      const change =
        Number.isFinite(previous) && previous !== 0
          ? (midpoint - previous) / Math.abs(previous)
          : 0;
      metrics.push({
        key: `${symbol}-mid`,
        label: `${symbol} Mid`,
        value: midpoint,
        kind: 'currency',
        change,
      });
      nextMidpoints[symbol] = midpoint;
    }
    if (Number.isFinite(tick.bid_price) && Number.isFinite(tick.ask_price)) {
      metrics.push({
        key: `${symbol}-spread`,
        label: `${symbol} Spread`,
        value: tick.ask_price - tick.bid_price,
        kind: 'currency',
      });
    }
  });
  return { metrics, midpoints: nextMidpoints };
}

function derivePnlSeries(pnlBySymbol) {
  return Object.entries(pnlBySymbol).map(([symbol, summary]) => ({
    label: symbol,
    value: Number(summary.notional.toFixed(2)),
    change:
      summary.previousNotional !== 0
        ? (summary.notional - summary.previousNotional) /
          Math.max(Math.abs(summary.previousNotional), 1)
        : 0,
    target: summary.volume ? summary.notional / summary.volume : undefined,
  }));
}

function describeChannel(channel) {
  const status = channel.status ?? CHANNEL_STATUS.IDLE;
  const level = CHANNEL_LEVEL[status] || 'warning';
  const labelMap = {
    [CHANNEL_STATUS.OPEN]: 'Streaming',
    [CHANNEL_STATUS.CONNECTING]: 'Connecting…',
    [CHANNEL_STATUS.RECONNECTING]: 'Reconnecting…',
    [CHANNEL_STATUS.ERROR]: 'Error',
    [CHANNEL_STATUS.CLOSED]: 'Offline',
    [CHANNEL_STATUS.IDLE]: 'Idle',
  };
  const descriptionParts = [];
  if (status === CHANNEL_STATUS.RECONNECTING && channel.delay) {
    descriptionParts.push(`Next attempt in ${channel.delay} ms (try #${channel.attempt}).`);
  }
  if (channel.lastError) {
    descriptionParts.push(`Last error: ${channel.lastError}`);
  }
  if (!descriptionParts.length) {
    descriptionParts.push('Monitoring live execution feed.');
  }
  return {
    label: 'Realtime Feed',
    state: labelMap[status] || 'Unknown',
    level,
    description: descriptionParts.join(' '),
  };
}

export class DashboardStore {
  constructor(initialState = {}) {
    this.listeners = new Set();
    this.channel = { ...DEFAULT_CHANNEL, ...(initialState.channel || {}) };
    this.ticks = [];
    this.orders = [];
    this.fills = [];
    this.latestTicks = {};
    this.pnlBySymbol = {};
    this.midpoints = {};
    this.customStatusItems = initialState.statusItems?.slice() || [];
    this.metrics = [];
    this.pnlSeries = [];
    this.statusItems = [];
    this.lastEvent = null;
    this.recompute();
  }

  subscribe(listener) {
    if (typeof listener !== 'function') {
      throw new Error('Listener must be a function');
    }
    this.listeners.add(listener);
    listener(this.getState());
    return () => {
      this.listeners.delete(listener);
    };
  }

  notify() {
    const snapshot = this.getState();
    this.listeners.forEach((listener) => {
      try {
        listener(snapshot);
      } catch (error) {
        console.error('Dashboard store listener failed', error);
      }
    });
  }

  updateChannel(nextChannel) {
    this.channel = { ...this.channel, ...(nextChannel || {}) };
    this.recompute();
  }

  handleEvent(event) {
    if (!event || typeof event !== 'object') {
      throw new Error('Event is required');
    }
    if (!event.type || !event.payload) {
      throw new Error('Event must include type and payload');
    }
    const { type, payload } = event;
    this.lastEvent = { type, receivedAt: Date.now(), id: payload.event_id };
    if (type === 'tick') {
      this.processTick(payload);
    } else if (type === 'order') {
      this.processOrder(payload);
    } else if (type === 'fill') {
      this.processFill(payload);
    }
    this.recompute();
  }

  processTick(payload) {
    this.ticks = clampHistory([...this.ticks, payload], MAX_TICKS);
    this.latestTicks = { ...this.latestTicks, [payload.symbol]: payload };
  }

  processOrder(payload) {
    this.orders = clampHistory([...this.orders, payload], MAX_ORDERS);
  }

  processFill(payload) {
    this.fills = clampHistory([...this.fills, payload], MAX_FILLS);
    const entry = this.pnlBySymbol[payload.symbol] || {
      notional: 0,
      previousNotional: 0,
      volume: 0,
    };
    entry.previousNotional = entry.notional;
    const fillNotional = Number(payload.fill_price || 0) * Number(payload.filled_qty || 0);
    if (Number.isFinite(fillNotional)) {
      entry.notional += fillNotional;
    }
    if (Number.isFinite(payload.filled_qty)) {
      entry.volume += payload.filled_qty;
    }
    this.pnlBySymbol[payload.symbol] = entry;
  }

  setStatusItems(items) {
    this.customStatusItems = items?.slice() || [];
    this.recompute();
  }

  recompute() {
    const { metrics, midpoints } = deriveMetrics(this.latestTicks, this.midpoints);
    this.midpoints = midpoints;
    this.metrics = metrics;
    this.pnlSeries = derivePnlSeries(this.pnlBySymbol);
    const channelStatus = describeChannel(this.channel);
    this.statusItems = [channelStatus, ...this.customStatusItems];
    this.notify();
  }

  getState() {
    return {
      channel: { ...this.channel },
      metrics: this.metrics.slice(),
      pnlSeries: this.pnlSeries.slice(),
      statusItems: this.statusItems.map((item) => ({ ...item })),
      ticks: this.ticks.slice(),
      orders: this.orders.slice(),
      fills: this.fills.slice(),
      lastEvent: this.lastEvent ? { ...this.lastEvent } : null,
    };
  }
}

export function createDashboardStore(initialState = {}) {
  return new DashboardStore(initialState);
}

export { DEFAULT_CHANNEL, CHANNEL_LEVEL };
