export const BASE_STYLES = `
  :root {
    color-scheme: dark;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
  }

  .tp-app {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    min-height: 100vh;
    background: radial-gradient(circle at top left, #0d1b2a, #010409);
    color: #f8fafc;
  }

  @media (min-width: 1080px) {
    .tp-app {
      grid-template-columns: 280px minmax(0, 1fr);
    }
  }

  .tp-shell {
    display: grid;
    grid-template-rows: auto 1fr;
    gap: 1.5rem;
    padding: 2rem;
  }

  .tp-nav {
    padding: 2rem 2rem 0 2rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    background: rgba(15, 23, 42, 0.6);
    border-right: 1px solid rgba(148, 163, 184, 0.15);
    backdrop-filter: blur(24px);
  }

  .tp-nav__title {
    font-size: clamp(1.35rem, 2.5vw, 1.75rem);
    font-weight: 700;
    letter-spacing: -0.01em;
    margin: 0;
  }

  .tp-nav__links {
    display: grid;
    gap: 0.75rem;
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .tp-nav__link {
    display: inline-flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    background: rgba(15, 23, 42, 0.35);
    border: 1px solid transparent;
    color: inherit;
    text-decoration: none;
    transition: background 0.2s ease, border-color 0.2s ease;
  }

  .tp-nav__link:hover {
    background: rgba(56, 189, 248, 0.15);
    border-color: rgba(56, 189, 248, 0.25);
  }

  .tp-nav__link--active {
    background: rgba(56, 189, 248, 0.25);
    border-color: rgba(56, 189, 248, 0.45);
    color: #f0f9ff;
  }

  .tp-nav__badge {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.45);
  }

  .tp-view {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 20px;
    padding: 1.75rem;
    box-shadow: 0 24px 48px -32px rgba(15, 23, 42, 0.8);
  }

  .tp-view__header {
    display: grid;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
  }

  .tp-view__title {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 600;
  }

  .tp-view__subtitle {
    margin: 0;
    font-size: 0.95rem;
    color: rgba(226, 232, 240, 0.7);
  }

  .tp-grid {
    display: grid;
    gap: 1.5rem;
  }

  .tp-grid--two {
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  }

  .tp-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 18px;
    padding: 1.5rem;
    display: grid;
    gap: 1rem;
  }

  .tp-card__header {
    display: grid;
    gap: 0.35rem;
  }

  .tp-card__title {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 600;
  }

  .tp-card__meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    font-size: 0.95rem;
    color: rgba(226, 232, 240, 0.75);
  }

  .tp-stat {
    font-weight: 600;
  }

  .tp-stat--muted {
    color: rgba(148, 163, 184, 0.85);
  }

  .tp-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    background: rgba(148, 163, 184, 0.2);
    color: rgba(226, 232, 240, 0.9);
  }

  .tp-pill--positive {
    background: rgba(74, 222, 128, 0.2);
    color: #4ade80;
  }

  .tp-pill--negative {
    background: rgba(248, 113, 113, 0.2);
    color: #f87171;
  }

  .tp-status {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-size: 0.8rem;
    background: rgba(148, 163, 184, 0.2);
  }

  .tp-status--filled {
    background: rgba(74, 222, 128, 0.2);
    color: #4ade80;
  }

  .tp-status--working {
    background: rgba(251, 191, 36, 0.2);
    color: #fbbf24;
  }

  .tp-status--cancelled {
    background: rgba(248, 113, 113, 0.2);
    color: #f87171;
  }

  .tp-progress {
    position: relative;
    background: rgba(15, 23, 42, 0.6);
    border-radius: 999px;
    overflow: hidden;
    height: 0.75rem;
  }

  .tp-progress__bar {
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, #38bdf8, #2563eb);
  }

  .tp-progress__label {
    display: inline-block;
    margin-left: 0.5rem;
    font-size: 0.85rem;
  }
`;
