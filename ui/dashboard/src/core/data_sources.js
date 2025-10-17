import { serialiseFiltersToQuery } from '../state/globalFilters.js';

function normaliseUrl(baseUrl, endpoint) {
  if (!baseUrl) {
    throw new Error('REST data source requires a baseUrl');
  }
  const url = new URL(endpoint || '/', baseUrl);
  return url;
}

export function buildQueryFromFilters(state) {
  if (!state) {
    throw new Error('Filters state is required');
  }
  return {
    symbols: state.symbols?.join(',') || '',
    timeframe: state.timeframe,
    strategy: state.strategy,
    from: state.dateRange?.from,
    to: state.dateRange?.to,
  };
}

function applyQueryParams(url, query) {
  Object.entries(query).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') {
      return;
    }
    url.searchParams.set(key, Array.isArray(value) ? value.join(',') : value);
  });
}

export function createRestDataSource(options = {}) {
  const {
    baseUrl,
    endpoint = '/analytics',
    fetchImpl = typeof fetch === 'function' ? fetch : null,
    filtersStore,
  } = options;

  if (!filtersStore || typeof filtersStore.getState !== 'function') {
    throw new Error('filtersStore with getState is required for the REST data source');
  }

  if (!fetchImpl) {
    throw new Error('A fetch implementation must be provided');
  }

  const listeners = new Set();

  function buildUrl(extraQuery = {}) {
    const url = normaliseUrl(baseUrl, endpoint);
    const state = filtersStore.getState();
    const query = { ...buildQueryFromFilters(state), ...extraQuery };
    applyQueryParams(url, query);
    return url.toString();
  }

  async function executeRequest({ query = {}, init = {} } = {}) {
    const requestUrl = buildUrl(query);
    const response = await fetchImpl(requestUrl, init);
    if (!response.ok) {
      const error = new Error(`Request failed with status ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return response.json();
  }

  const unsubscribe = filtersStore.subscribe((state) => {
    const query = buildQueryFromFilters(state);
    const url = normaliseUrl(baseUrl, endpoint);
    applyQueryParams(url, query);
    const payload = { query, url: url.toString(), search: serialiseFiltersToQuery(state) };
    listeners.forEach((listener) => {
      try {
        listener(payload);
      } catch (error) {
        console.error('REST listener failed', error);
      }
    });
  });

  function onRequest(listener) {
    if (typeof listener !== 'function') {
      throw new Error('Listener must be a function');
    }
    listeners.add(listener);
    listener({
      query: buildQueryFromFilters(filtersStore.getState()),
      url: buildUrl(),
      search: serialiseFiltersToQuery(filtersStore.getState()),
    });
    return () => listeners.delete(listener);
  }

  return {
    fetch: executeRequest,
    onRequest,
    buildUrl,
    dispose() {
      unsubscribe?.();
      listeners.clear();
    },
  };
}

function createMessagePayload(type, state) {
  return {
    type,
    filters: buildQueryFromFilters(state),
  };
}

function attachOpenListener(socket, handler) {
  if (typeof socket.addEventListener === 'function') {
    socket.addEventListener('open', handler);
    return;
  }
  const previous = socket.onopen;
  socket.onopen = (event) => {
    previous?.(event);
    handler(event);
  };
}

function getReadyState(socket, WebSocketImpl) {
  return typeof socket.readyState === 'number' ? socket.readyState : WebSocketImpl?.readyState;
}

function getOpenState(WebSocketImpl, socket) {
  if (WebSocketImpl?.OPEN !== undefined) {
    return WebSocketImpl.OPEN;
  }
  if (socket?.OPEN !== undefined) {
    return socket.OPEN;
  }
  return 1;
}

export function createWebSocketDataSource(options = {}) {
  const {
    url,
    WebSocketImpl = typeof WebSocket !== 'undefined' ? WebSocket : null,
    filtersStore,
    serializer = JSON.stringify,
  } = options;

  if (!url) {
    throw new Error('WebSocket data source requires a url');
  }
  if (!filtersStore || typeof filtersStore.getState !== 'function') {
    throw new Error('filtersStore with getState is required for the WebSocket data source');
  }
  if (!WebSocketImpl) {
    throw new Error('A WebSocket implementation must be provided');
  }

  const socket = new WebSocketImpl(url);
  const queue = [];
  const openState = getOpenState(WebSocketImpl, socket);

  function flushQueue() {
    while (queue.length) {
      const message = queue.shift();
      socket.send(message);
    }
  }

  attachOpenListener(socket, () => {
    flushQueue();
  });

  function enqueue(message) {
    if (getReadyState(socket, WebSocketImpl) === openState) {
      socket.send(message);
    } else {
      queue.push(message);
    }
  }

  const initialState = filtersStore.getState();
  enqueue(serializer(createMessagePayload('subscribe', initialState)));

  const unsubscribe = filtersStore.subscribe((state) => {
    enqueue(serializer(createMessagePayload('update_filters', state)));
  });

  return {
    socket,
    dispose() {
      unsubscribe?.();
      queue.length = 0;
      socket.close?.();
    },
    getPendingMessages() {
      return queue.slice();
    },
  };
}
