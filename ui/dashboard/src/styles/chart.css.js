export const CHART_STYLES = `
  .tp-area-chart__container {
    display: grid;
    gap: 0.75rem;
  }

  .tp-area-chart {
    width: 100%;
    height: auto;
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.55);
  }

  .tp-chart-legend {
    list-style: none;
    display: grid;
    gap: 0.5rem;
    margin: 0;
    padding: 0;
  }

  .tp-chart-legend__item {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    color: rgba(226, 232, 240, 0.85);
  }

  .tp-chart-empty {
    padding: 0.75rem;
    font-size: 0.9rem;
    border: 1px dashed rgba(148, 163, 184, 0.35);
    border-radius: 12px;
    text-align: center;
    color: rgba(148, 163, 184, 0.75);
  }
`;
