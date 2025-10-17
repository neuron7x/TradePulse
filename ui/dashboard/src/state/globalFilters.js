const DEFAULT_TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d'];
const DEFAULT_STRATEGIES = ['trend_following', 'mean_reversion', 'volatility_breakout'];
const DEFAULT_SYMBOLS = ['BTC-USD'];

const MILLISECONDS_IN_DAY = 24 * 60 * 60 * 1000;

function toISOStringSafe(value) {
  if (value instanceof Date) {
    return value.toISOString();
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    return new Date(value).toISOString();
  }
  if (typeof value === 'string') {
    const parsed = new Date(value);
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toISOString();
    }
  }
  throw new Error(`Invalid date value: ${value}`);
}

function normaliseSymbols(value, allowedSymbols) {
  if (!value || (Array.isArray(value) && value.length === 0)) {
    if (allowedSymbols?.length) {
      return allowedSymbols.slice(0, 1);
    }
    return DEFAULT_SYMBOLS.slice();
  }

  const input = Array.isArray(value) ? value : String(value).split(/[,\s]+/);
  const cleaned = input
    .map((symbol) => String(symbol || '').trim().toUpperCase())
    .filter(Boolean);

  const unique = Array.from(new Set(cleaned));
  if (!unique.length) {
    throw new Error('At least one symbol is required');
  }

  if (allowedSymbols?.length) {
    const invalid = unique.filter((symbol) => !allowedSymbols.includes(symbol));
    if (invalid.length) {
      throw new Error(`Unsupported symbols selected: ${invalid.join(', ')}`);
    }
  }
  return unique;
}

function normaliseTimeframe(value, allowedTimeframes) {
  const timeframe = String(value || '').trim();
  const candidates = allowedTimeframes?.length ? allowedTimeframes : DEFAULT_TIMEFRAMES;
  if (!timeframe) {
    return candidates[0];
  }
  if (!candidates.includes(timeframe)) {
    throw new Error(`Unsupported timeframe: ${timeframe}`);
  }
  return timeframe;
}

function normaliseStrategy(value, allowedStrategies) {
  const strategy = String(value || '').trim();
  const candidates = allowedStrategies?.length ? allowedStrategies : DEFAULT_STRATEGIES;
  if (!strategy) {
    return candidates[0];
  }
  if (!candidates.includes(strategy)) {
    throw new Error(`Unsupported strategy: ${strategy}`);
  }
  return strategy;
}

function normaliseDateRange(value, defaults) {
  const base = {
    from: defaults?.from,
    to: defaults?.to,
  };

  if (value && typeof value === 'object') {
    if (value.from) {
      base.from = toISOStringSafe(value.from);
    }
    if (value.to) {
      base.to = toISOStringSafe(value.to);
    }
  }

  if (!base.from || !base.to) {
    const end = toISOStringSafe(defaults?.to || Date.now());
    const start = toISOStringSafe(defaults?.from || Date.now() - 7 * MILLISECONDS_IN_DAY);
    base.from = start;
    base.to = end;
  }

  const from = new Date(base.from);
  const to = new Date(base.to);
  if (Number.isNaN(from.getTime()) || Number.isNaN(to.getTime())) {
    throw new Error('Invalid date range provided');
  }
  if (from.getTime() > to.getTime()) {
    throw new Error('Date range start must be before end');
  }

  return {
    from: from.toISOString(),
    to: to.toISOString(),
  };
}

export const DEFAULT_FILTERS = {
  symbols: DEFAULT_SYMBOLS.slice(),
  timeframe: DEFAULT_TIMEFRAMES[3],
  strategy: DEFAULT_STRATEGIES[0],
  dateRange: normaliseDateRange({}, {}),
};

export const FILTER_VALIDATORS = {
  symbols: normaliseSymbols,
  timeframe: normaliseTimeframe,
  strategy: normaliseStrategy,
  dateRange: normaliseDateRange,
};

function cloneState(state) {
  return {
    symbols: state.symbols.slice(),
    timeframe: state.timeframe,
    strategy: state.strategy,
    dateRange: { ...state.dateRange },
  };
}

function parseSearchParams(search, validators, options) {
  if (!search) {
    return null;
  }
  const params = new URLSearchParams(search.startsWith('?') ? search.slice(1) : search);
  const patch = {};

  if (params.has('symbols')) {
    patch.symbols = params.get('symbols').split(',');
  }
  if (params.has('timeframe')) {
    patch.timeframe = params.get('timeframe');
  }
  if (params.has('strategy')) {
    patch.strategy = params.get('strategy');
  }
  const from = params.get('from');
  const to = params.get('to');
  if (from || to) {
    patch.dateRange = { from, to };
  }

  if (!Object.keys(patch).length) {
    return null;
  }

  return applyValidators({ ...options.defaults }, patch, validators, options);
}

function applyValidators(base, patch, validators, options) {
  const next = { ...base };
  const allowedSymbols = options.allowedSymbols;
  const allowedTimeframes = options.allowedTimeframes;
  const allowedStrategies = options.allowedStrategies;

  Object.entries(patch).forEach(([key, value]) => {
    const validator = validators[key];
    if (!validator) {
      throw new Error(`Unknown filter: ${key}`);
    }
    if (key === 'symbols') {
      next.symbols = validator(value, allowedSymbols);
    } else if (key === 'timeframe') {
      next.timeframe = validator(value, allowedTimeframes);
    } else if (key === 'strategy') {
      next.strategy = validator(value, allowedStrategies);
    } else if (key === 'dateRange') {
      next.dateRange = validator(value, next.dateRange);
    }
  });

  return next;
}

function loadFromStorage(storage, storageKey, validators, options) {
  if (!storage || !storageKey) {
    return null;
  }
  try {
    const raw = storage.getItem(storageKey);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw);
    return applyValidators({ ...options.defaults }, parsed, validators, options);
  } catch (error) {
    console.warn('Failed to load filters from storage:', error);
    return null;
  }
}

function persistToStorage(storage, storageKey, state) {
  if (!storage || !storageKey) {
    return;
  }
  try {
    storage.setItem(storageKey, JSON.stringify(state));
  } catch (error) {
    console.warn('Failed to persist filters to storage:', error);
  }
}

function persistToLocation(state, location, history) {
  if (!location) {
    return;
  }
  const params = new URLSearchParams(location.search ? location.search.slice(1) : '');
  params.set('symbols', state.symbols.join(','));
  params.set('timeframe', state.timeframe);
  params.set('strategy', state.strategy);
  params.set('from', state.dateRange.from);
  params.set('to', state.dateRange.to);
  const nextSearch = `?${params.toString()}`;

  if (history?.replaceState) {
    history.replaceState(null, '', `${location.pathname || ''}${nextSearch}${location.hash || ''}`);
  } else {
    location.search = nextSearch;
  }
}

function dispatchFiltersChange(eventTarget, detail) {
  if (!eventTarget?.dispatchEvent) {
    return;
  }
  const eventName = 'tradepulse:filters:change';
  try {
    const CustomEventCtor = typeof CustomEvent === 'function'
      ? CustomEvent
      : class extends Event {
          constructor(type, init = {}) {
            super(type);
            this.detail = init.detail;
          }
        };
    eventTarget.dispatchEvent(new CustomEventCtor(eventName, { detail }));
  } catch (error) {
    console.warn('Failed to dispatch filters change event:', error);
  }
}

export function serialiseFiltersToQuery(state) {
  const params = new URLSearchParams();
  params.set('symbols', state.symbols.join(','));
  params.set('timeframe', state.timeframe);
  params.set('strategy', state.strategy);
  params.set('from', state.dateRange.from);
  params.set('to', state.dateRange.to);
  return params.toString();
}

export function createGlobalFiltersStore(config = {}) {
  const options = {
    allowedSymbols: config.allowedSymbols || null,
    allowedTimeframes: config.allowedTimeframes || null,
    allowedStrategies: config.allowedStrategies || null,
    defaults: {
      ...DEFAULT_FILTERS,
      ...(config.defaults || {}),
    },
  };

  const validators = {
    ...FILTER_VALIDATORS,
    ...(config.validators || {}),
  };

  options.defaults = applyValidators(DEFAULT_FILTERS, options.defaults, validators, options);

  const storageKey = config.storageKey || 'tradepulse:dashboard:filters';
  const storage = config.storage || (typeof window !== 'undefined' ? window.localStorage : null);
  const location = config.location || (typeof window !== 'undefined' ? window.location : null);
  const history = config.history || (typeof window !== 'undefined' ? window.history : null);
  const eventTarget = config.eventTarget || (typeof window !== 'undefined' ? window : null);

  let state = cloneState(options.defaults);

  const fromStorage = loadFromStorage(storage, storageKey, validators, options);
  if (fromStorage) {
    state = cloneState(fromStorage);
  }
  const fromLocation = parseSearchParams(location?.search || '', validators, { ...options, defaults: state });
  if (fromLocation) {
    state = cloneState(fromLocation);
  }

  const subscribers = new Set();

  function getState() {
    return cloneState(state);
  }

  function notify(nextState) {
    const snapshot = cloneState(nextState);
    subscribers.forEach((listener) => {
      try {
        listener(snapshot);
      } catch (error) {
        console.error('Failed to notify subscriber', error);
      }
    });
    dispatchFiltersChange(eventTarget, snapshot);
  }

  function setState(patch) {
    state = applyValidators(state, patch, validators, options);
    const snapshot = cloneState(state);
    persistToStorage(storage, storageKey, snapshot);
    persistToLocation(snapshot, location, history);
    notify(snapshot);
    return snapshot;
  }

  function update(updater) {
    if (typeof updater !== 'function') {
      throw new Error('Updater function is required');
    }
    const draft = updater(cloneState(state));
    if (!draft || typeof draft !== 'object') {
      throw new Error('Updater must return a partial state object');
    }
    return setState(draft);
  }

  function reset() {
    return setState(cloneState(options.defaults));
  }

  function subscribe(listener) {
    if (typeof listener !== 'function') {
      throw new Error('Listener must be a function');
    }
    subscribers.add(listener);
    listener(cloneState(state));
    return () => {
      subscribers.delete(listener);
    };
  }

  const api = {
    getState,
    setState,
    update,
    reset,
    subscribe,
    serialize: () => serialiseFiltersToQuery(state),
  };

  return api;
}

export function deserializeFiltersFromQuery(search, config = {}) {
  const options = {
    allowedSymbols: config.allowedSymbols || null,
    allowedTimeframes: config.allowedTimeframes || null,
    allowedStrategies: config.allowedStrategies || null,
    defaults: {
      ...DEFAULT_FILTERS,
      ...(config.defaults || {}),
    },
  };
  const validators = {
    ...FILTER_VALIDATORS,
    ...(config.validators || {}),
  };
  options.defaults = applyValidators(DEFAULT_FILTERS, options.defaults, validators, options);
  return parseSearchParams(search, validators, options) || cloneState(options.defaults);
}

export {
  DEFAULT_TIMEFRAMES,
  DEFAULT_STRATEGIES,
  normaliseSymbols as normalizeSymbols,
  normaliseTimeframe as normalizeTimeframe,
  normaliseStrategy as normalizeStrategy,
  normaliseDateRange as normalizeDateRange,
};
