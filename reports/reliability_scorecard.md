# Reliability Scorecard

This document captures monthly reliability performance across TradePulse
domains. Update it on the first business day of each month following the SRE
governance review.

## Snapshot – `YYYY-MM`

| Domain | SLO Target | Actual | Error Budget Consumed | Notable Incidents |
|--------|------------|--------|-----------------------|-------------------|
| Client API | 99.9% availability | 99.92% | 20% | `INC-20250105-01` |
| Strategy Runtime | 99.7% success | 99.40% | 85% | `INC-20250111-02` |
| Order Execution | 99.9% timely confirms | 99.88% | 30% | – |
| Market Data | 99.8% freshness | 99.75% | 60% | `INC-20250119-03` |

## Highlights

- **Wins** – Summarise improvements, successful mitigations, or resiliency
  investments that paid off.
- **Risks** – Call out services approaching burn thresholds or recurring issues.
- **Upcoming Work** – Link planned initiatives, RFCs, or experiments aimed at
  improving reliability.

## Action Item Tracker

| ID | Description | Owner | Status | Due Date | Notes |
|----|-------------|-------|--------|----------|-------|
| `REL-245` | Roll out adaptive throttling on Client API edge | `@infra-dev` | In Progress | `YYYY-MM-DD` | Pilot 50% traffic |
| `REL-251` | Harden broker adapter retries | `@exec-eng` | Planned | `YYYY-MM-DD` | Awaiting risk sign-off |

Document progress on SRE-led initiatives here so monthly reviews have a single
source of truth.
