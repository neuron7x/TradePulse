export const strategyTemplates = [
  {
    name: 'trend_following',
    label: 'Trend Following',
    description: 'Середньострокові прориви з керуванням ризиком через ATR та адаптивним плечем.',
    defaults: {
      lookback: 55,
      atrMultiplier: 2.5,
      maxPositions: 6,
      riskPerTrade: 1.2,
    },
  },
  {
    name: 'mean_reversion',
    label: 'Mean Reversion',
    description: 'Покупка перепроданих активів та продаж перекуплених з фільтрами за обсягом.',
    defaults: {
      lookback: 20,
      entryZScore: -1.8,
      exitZScore: -0.5,
      notionalCap: 750000,
    },
  },
  {
    name: 'vol_breakout',
    label: 'Volatility Breakout',
    description: 'Денний моніторинг волатильності з роботою у вікні Нью-Йорка та жорсткими стопами.',
    defaults: {
      lookback: 15,
      volThreshold: 1.4,
      session: 'NY',
      maxGapRisk: 0.9,
    },
  },
];

export const backtestSamples = [
  {
    metadata: {
      id: 'BT-TR-202404',
      strategy: 'trend_following',
      startedAt: '2024-03-01',
      endedAt: '2024-04-30',
      note: 'Оновлені кореляційні фільтри',
    },
    metrics: {
      sharpe: 1.72,
      pnl: 18450,
      maxDrawdown: -6.2,
      winRate: 58,
    },
  },
  {
    metadata: {
      id: 'BT-MR-202404',
      strategy: 'mean_reversion',
      startedAt: '2024-03-01',
      endedAt: '2024-04-30',
      note: 'Додані обмеження на overnight',
    },
    metrics: {
      sharpe: 1.05,
      pnl: 9400,
      maxDrawdown: -4.8,
      winRate: 64,
    },
  },
  {
    metadata: {
      id: 'BT-VB-202404',
      strategy: 'vol_breakout',
      startedAt: '2024-03-01',
      endedAt: '2024-04-30',
      note: 'Нова логіка сесій',
    },
    metrics: {
      sharpe: 0.82,
      pnl: 5200,
      maxDrawdown: -9.5,
      winRate: 49,
    },
  },
];

export const performanceSeries = {
  trend_following: [
    { date: '2024-03-01', equity: 100000 },
    { date: '2024-03-08', equity: 102400 },
    { date: '2024-03-15', equity: 104100 },
    { date: '2024-03-22', equity: 106550 },
    { date: '2024-03-29', equity: 108250 },
    { date: '2024-04-05', equity: 110100 },
    { date: '2024-04-12', equity: 112300 },
    { date: '2024-04-19', equity: 117800 },
    { date: '2024-04-26', equity: 118450 },
    { date: '2024-04-30', equity: 118900 },
  ],
  mean_reversion: [
    { date: '2024-03-01', equity: 100000 },
    { date: '2024-03-08', equity: 100950 },
    { date: '2024-03-15', equity: 101500 },
    { date: '2024-03-22', equity: 101200 },
    { date: '2024-03-29', equity: 102850 },
    { date: '2024-04-05', equity: 103200 },
    { date: '2024-04-12', equity: 104700 },
    { date: '2024-04-19', equity: 105650 },
    { date: '2024-04-26', equity: 108900 },
    { date: '2024-04-30', equity: 109400 },
  ],
  vol_breakout: [
    { date: '2024-03-01', equity: 100000 },
    { date: '2024-03-08', equity: 100650 },
    { date: '2024-03-15', equity: 101250 },
    { date: '2024-03-22', equity: 100400 },
    { date: '2024-03-29', equity: 101750 },
    { date: '2024-04-05', equity: 101100 },
    { date: '2024-04-12', equity: 102600 },
    { date: '2024-04-19', equity: 103550 },
    { date: '2024-04-26', equity: 104800 },
    { date: '2024-04-30', equity: 105200 },
  ],
};
