function formatSymbolsInput(symbols = []) {
  return symbols.join(', ');
}

function formatDateInput(value) {
  if (!value) {
    return '';
  }
  return value.slice(0, 10);
}

function getNestedValue(state, path) {
  return path.split('.').reduce((acc, key) => (acc ? acc[key] : undefined), state);
}

function buildPatchFromPath(path, value, state) {
  const segments = path.split('.');
  if (segments.length === 1) {
    return { [path]: value };
  }
  const [head, ...rest] = segments;
  const next = { ...state[head] };
  let cursor = next;
  for (let index = 0; index < rest.length - 1; index += 1) {
    const key = rest[index];
    cursor[key] = { ...(cursor[key] || {}) };
    cursor = cursor[key];
  }
  cursor[rest[rest.length - 1]] = value;
  return { [head]: next };
}

function parseControlValue(control, state) {
  const path = control.dataset?.tpFilterControl;
  if (!path) {
    return null;
  }
  if (path === 'symbols') {
    const symbols = control.value
      .split(/[,\s]+/)
      .map((symbol) => symbol.trim())
      .filter(Boolean);
    return { symbols };
  }
  if (path.startsWith('dateRange.')) {
    const [, key] = path.split('.');
    const next = { ...(state.dateRange || {}) };
    next[key] = control.value;
    return { dateRange: next };
  }
  return buildPatchFromPath(path, control.value, state);
}

function setControlValue(control, state) {
  const path = control.dataset?.tpFilterControl;
  if (!path) {
    return;
  }
  const value = getNestedValue(state, path);
  if (path === 'symbols') {
    control.value = formatSymbolsInput(value || []);
  } else if (path.startsWith('dateRange.')) {
    control.value = formatDateInput(value);
  } else {
    control.value = value ?? '';
  }
}

export function attachFilterControls(store, root = typeof document !== 'undefined' ? document : null) {
  if (!store || typeof store.getState !== 'function') {
    throw new Error('A valid filters store is required');
  }

  if (!root || typeof root.querySelectorAll !== 'function') {
    return () => {};
  }

  const controls = Array.from(root.querySelectorAll('[data-tp-filter-control]'));
  if (!controls.length) {
    return () => {};
  }

  const detachFns = [];

  const state = store.getState();
  controls.forEach((control) => {
    setControlValue(control, state);
    const eventName = control.dataset?.tpFilterEvent || 'change';
    const handler = () => {
      const patch = parseControlValue(control, store.getState());
      if (patch) {
        store.setState(patch);
      }
    };
    control.addEventListener?.(eventName, handler);
    detachFns.push(() => control.removeEventListener?.(eventName, handler));
  });

  const unsubscribe = store.subscribe((nextState) => {
    controls.forEach((control) => {
      setControlValue(control, nextState);
    });
  });

  return () => {
    unsubscribe?.();
    detachFns.forEach((fn) => fn());
  };
}

export { formatSymbolsInput, formatDateInput };
