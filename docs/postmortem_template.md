# TradePulse Postmortem Template

> Duplicate this document into the `reliability/postmortems/YYYY/` directory
> within Confluence or the internal git repository when recording an incident.
> Keep the tone blameless and focus on improving systems and processes.

## Incident Metadata

- **Incident ID:** `INC-YYYYMMDD-XX`
- **Severity:** `SEV-1 | SEV-2 | SEV-3`
- **Start Time:** `YYYY-MM-DD HH:MM TZ`
- **End Time:** `YYYY-MM-DD HH:MM TZ`
- **Duration:** `HH:MM`
- **Customer Impact:** Describe the user-facing symptoms, scope of affected
  customers, and any contractual obligations triggered.
- **Services Impacted:** List the TradePulse domains/components.
- **Detected By:** Monitoring alert, customer ticket, synthetic probe, etc.
- **Reported By:** On-call engineer or stakeholder who raised the incident.

## Summary

Provide a concise narrative (3–5 sentences) explaining what happened, why it
occurred, how it was detected, and how it was resolved.

## Timeline

| Time (TZ) | Actor | Event |
|-----------|-------|-------|
| `HH:MM`   | `@user` | `Alert fired ...` |
| `HH:MM`   | `@user` | `Mitigation applied ...` |
| `HH:MM`   | `@user` | `Service restored ...` |

Include all key events from detection through resolution and confirmation of
service health. Reference PagerDuty, Slack, and ticket links where applicable.

## Impact Assessment

- **SLO/SLA Impact:** Document which objectives were breached or threatened and
  the amount of error budget consumed.
- **Financial/Regulatory Impact:** Identify any direct costs, credits, or
  regulatory disclosures required.
- **Customer Communications:** Summarise status page updates, emails, or account
  manager outreach.

## Root Cause Analysis

Analyse the contributing factors using the "Five Whys" or similar technique.
Structure the analysis into:

1. **Trigger** – The immediate event that surfaced the incident.
2. **Contributing Factors** – Preconditions or latent defects.
3. **Systemic Issues** – Organisational or process gaps that enabled recurrence.

## Mitigation & Recovery

Detail the actions taken to restore service, including temporary mitigations,
feature flag flips, or infrastructure changes.

## Follow-up Actions

| ID | Owner | Description | Priority | Due Date | Status |
|----|-------|-------------|----------|----------|--------|
| `REL-###` | `@owner` | `Improve autoscaling policy ...` | `P0` | `YYYY-MM-DD` | `In Progress` |

Ensure every action item has an accountable owner and a tracked ticket. Note the
expected impact on SLIs or error budgets when delivered.

## Learnings

Capture insights that improve detection, response, or prevention. Include links
to updated runbooks, playbooks, or architecture decisions.

## Verification

- **Incident Commander:** `@name` – `YYYY-MM-DD`
- **Domain Lead:** `@name` – `YYYY-MM-DD`
- **SRE Lead:** `@name` – `YYYY-MM-DD`

Confirm that follow-up items are logged and aligned with the Reliability Kanban
before closing the incident review.
