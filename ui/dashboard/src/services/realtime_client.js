const DEFAULT_BACKOFF = {
  initialDelayMs: 1000,
  maxDelayMs: 15000,
  multiplier: 2,
  jitter: 0.1,
};

export const CHANNEL_STATUS = {
  IDLE: 'idle',
  CONNECTING: 'connecting',
  OPEN: 'open',
  RECONNECTING: 'reconnecting',
  CLOSED: 'closed',
  ERROR: 'error',
};

const TYPE_ALIASES = new Map(
  [
    ['tick', 'tick'],
    ['tickevent', 'tick'],
    ['tick_event', 'tick'],
    ['tick.events', 'tick'],
    ['order', 'order'],
    ['orderevent', 'order'],
    ['order_event', 'order'],
    ['fill', 'fill'],
    ['fillevent', 'fill'],
    ['fill_event', 'fill'],
  ].map(([alias, canonical]) => [alias, canonical]),
);

const REQUIRED_FIELDS = {
  tick: ['event_id', 'symbol', 'timestamp', 'bid_price', 'ask_price'],
  order: [
    'event_id',
    'symbol',
    'timestamp',
    'order_id',
    'side',
    'order_type',
    'quantity',
    'metadata',
  ],
  fill: [
    'event_id',
    'symbol',
    'timestamp',
    'order_id',
    'fill_id',
    'status',
    'filled_qty',
    'fill_price',
    'metadata',
  ],
};

function defaultScheduler() {
  const { setTimeout: nativeSetTimeout, clearTimeout: nativeClearTimeout } = globalThis;
  if (typeof nativeSetTimeout !== 'function' || typeof nativeClearTimeout !== 'function') {
    throw new Error('Global scheduler is unavailable');
  }
  return {
    setTimeout: nativeSetTimeout.bind(globalThis),
    clearTimeout: nativeClearTimeout.bind(globalThis),
  };
}

function normalizeType(value) {
  if (!value) {
    return null;
  }
  const raw = String(value).toLowerCase().replace(/[^a-z.]/g, '');
  return TYPE_ALIASES.get(raw) || null;
}

function ensureObject(payload) {
  if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
    return payload;
  }
  throw new Error('Event payload must be an object');
}

function clonePayload(payload) {
  return JSON.parse(JSON.stringify(payload));
}

function ensureFields(type, payload) {
  const required = REQUIRED_FIELDS[type] || [];
  required.forEach((field) => {
    if (!(field in payload)) {
      throw new Error(`${type} event is missing required field: ${field}`);
    }
  });
  return payload;
}

function inferTypeFromPayload(payload) {
  if ('fill_id' in payload) {
    return 'fill';
  }
  if ('order_type' in payload || 'time_in_force' in payload) {
    return 'order';
  }
  if ('bid_price' in payload || 'ask_price' in payload) {
    return 'tick';
  }
  return null;
}

function normaliseEnvelope(message) {
  if (typeof message === 'string') {
    try {
      return JSON.parse(message);
    } catch (error) {
      throw new Error(`Failed to parse message JSON: ${error.message}`);
    }
  }
  if (message instanceof ArrayBuffer) {
    const text = new TextDecoder('utf-8').decode(message);
    return normaliseEnvelope(text);
  }
  if (ArrayBuffer.isView(message)) {
    const text = new TextDecoder('utf-8').decode(message.buffer);
    return normaliseEnvelope(text);
  }
  if (message && typeof message === 'object') {
    return message;
  }
  throw new Error('Unsupported WebSocket message format');
}

export function deserializeEventEnvelope(rawMessage) {
  const envelope = normaliseEnvelope(rawMessage);
  let { type, event_type: altType, kind, topic } = envelope;
  let payload = envelope.payload ?? envelope.data ?? envelope.body ?? null;

  let canonicalType = normalizeType(type || altType || kind || topic);

  if (!payload) {
    const assumedPayload = { ...envelope };
    delete assumedPayload.type;
    delete assumedPayload.event_type;
    delete assumedPayload.kind;
    delete assumedPayload.topic;
    delete assumedPayload.payload;
    delete assumedPayload.data;
    delete assumedPayload.body;
    if (Object.keys(assumedPayload).length) {
      payload = assumedPayload;
    }
  }

  if (!payload) {
    throw new Error('Event payload is empty');
  }

  const payloadObject = ensureObject(payload);

  if (!canonicalType) {
    canonicalType = inferTypeFromPayload(payloadObject);
  }

  if (!canonicalType) {
    throw new Error('Unknown event type');
  }

  const completePayload = ensureFields(canonicalType, payloadObject);
  return Object.freeze({
    type: canonicalType,
    payload: Object.freeze(clonePayload(completePayload)),
  });
}

function computeBackoffDelay({
  attempts,
  strategy,
}) {
  const initial = strategy.initialDelayMs;
  const multiplier = strategy.multiplier;
  const max = strategy.maxDelayMs;
  const jitter = strategy.jitter ?? 0;
  const baseDelay = Math.min(initial * Math.pow(multiplier, Math.max(attempts - 1, 0)), max);
  if (!jitter) {
    return Math.round(baseDelay);
  }
  const spread = baseDelay * jitter;
  const random = Math.random() * spread * 2 - spread;
  return Math.max(0, Math.round(baseDelay + random));
}

export class RealtimeEventClient {
  constructor(options = {}) {
    if (!options.url) {
      throw new Error('WebSocket URL is required');
    }
    this.url = options.url;
    this.protocols = options.protocols;
    this.logger = options.logger || console;
    this.webSocketFactory =
      options.webSocketFactory || ((url, protocols) => new WebSocket(url, protocols));
    this.scheduler = options.scheduler || defaultScheduler();
    this.backoff = {
      ...DEFAULT_BACKOFF,
      ...(options.backoff || {}),
    };
    this.heartbeatInterval = options.heartbeatInterval ?? 30000;
    this.listeners = new Set();
    this.statusListeners = new Set();
    this.socket = null;
    this.reconnectTimer = null;
    this.heartbeatTimer = null;
    this.retryCount = 0;
    this.manualClose = false;
    this.state = {
      status: CHANNEL_STATUS.IDLE,
      attempt: 0,
      lastError: null,
      delay: 0,
    };
  }

  connect() {
    this.manualClose = false;
    this.clearTimers();
    const reconnecting = this.retryCount > 0;
    this.updateStatus(reconnecting ? CHANNEL_STATUS.RECONNECTING : CHANNEL_STATUS.CONNECTING, {
      attempt: this.retryCount,
      delay: reconnecting ? this.state.delay : 0,
    });

    const socket = this.webSocketFactory(this.url, this.protocols);
    this.socket = socket;

    socket.addEventListener('open', this.handleOpen);
    socket.addEventListener('message', this.handleMessage);
    socket.addEventListener('error', this.handleError);
    socket.addEventListener('close', this.handleClose);
    return () => this.close();
  }

  handleOpen = () => {
    this.retryCount = 0;
    this.updateStatus(CHANNEL_STATUS.OPEN, { attempt: 0, delay: 0, lastError: null });
    this.startHeartbeat();
  };

  handleMessage = (event) => {
    try {
      const parsed = deserializeEventEnvelope(event.data);
      this.emitEvent(parsed);
    } catch (error) {
      this.logger?.warn?.('Failed to handle message', error);
    }
  };

  handleError = (event) => {
    const message = event?.message || event?.reason || 'Unknown socket error';
    this.logger?.error?.('WebSocket error', event);
    this.updateStatus(CHANNEL_STATUS.ERROR, {
      lastError: message,
      attempt: this.retryCount,
    });
  };

  handleClose = (event) => {
    this.stopHeartbeat();
    if (this.manualClose) {
      this.updateStatus(CHANNEL_STATUS.CLOSED, {
        attempt: this.retryCount,
        lastError: event?.reason || null,
      });
      return;
    }
    this.scheduleReconnect(event);
  };

  scheduleReconnect(event) {
    this.retryCount += 1;
    const delay = computeBackoffDelay({ attempts: this.retryCount, strategy: this.backoff });
    this.updateStatus(CHANNEL_STATUS.RECONNECTING, {
      attempt: this.retryCount,
      delay,
      lastError: event?.reason || this.state.lastError,
    });
    this.reconnectTimer = this.scheduler.setTimeout(() => {
      this.connect();
    }, delay);
  }

  startHeartbeat() {
    if (!this.heartbeatInterval || this.heartbeatInterval <= 0) {
      return;
    }
    this.stopHeartbeat();
    this.heartbeatTimer = this.scheduler.setTimeout(() => {
      try {
        if (this.socket && this.socket.readyState === 1) {
          this.socket.send?.(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
        }
      } catch (error) {
        this.logger?.warn?.('Failed to send heartbeat', error);
      } finally {
        this.startHeartbeat();
      }
    }, this.heartbeatInterval);
  }

  stopHeartbeat() {
    if (this.heartbeatTimer) {
      this.scheduler.clearTimeout(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  clearTimers() {
    if (this.reconnectTimer) {
      this.scheduler.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.stopHeartbeat();
  }

  emitEvent(event) {
    this.listeners.forEach((listener) => {
      try {
        listener(event);
      } catch (error) {
        this.logger?.error?.('Event listener failed', error);
      }
    });
  }

  updateStatus(status, meta = {}) {
    this.state = {
      ...this.state,
      status,
      ...meta,
      updatedAt: Date.now(),
    };
    this.statusListeners.forEach((listener) => {
      try {
        listener(this.state);
      } catch (error) {
        this.logger?.error?.('Status listener failed', error);
      }
    });
  }

  subscribe(listener) {
    if (typeof listener !== 'function') {
      throw new Error('Listener must be a function');
    }
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  onStatusChange(listener) {
    if (typeof listener !== 'function') {
      throw new Error('Status listener must be a function');
    }
    this.statusListeners.add(listener);
    listener(this.state);
    return () => {
      this.statusListeners.delete(listener);
    };
  }

  close() {
    this.manualClose = true;
    this.clearTimers();
    if (this.socket) {
      try {
        this.socket.removeEventListener('open', this.handleOpen);
        this.socket.removeEventListener('message', this.handleMessage);
        this.socket.removeEventListener('error', this.handleError);
        this.socket.removeEventListener('close', this.handleClose);
        this.socket.close?.();
      } catch (error) {
        this.logger?.warn?.('Error while closing socket', error);
      }
      this.socket = null;
    }
    this.updateStatus(CHANNEL_STATUS.CLOSED, {
      attempt: this.retryCount,
    });
  }
}

export function bindClientToStore(client, store) {
  if (!client) {
    throw new Error('Realtime client is required');
  }
  if (!store) {
    throw new Error('Dashboard store is required');
  }
  const unsubscribeEvent = client.subscribe((event) => {
    store.handleEvent(event);
  });
  const unsubscribeStatus = client.onStatusChange((status) => {
    store.updateChannel(status);
  });
  return () => {
    unsubscribeEvent();
    unsubscribeStatus();
  };
}

export { computeBackoffDelay };
