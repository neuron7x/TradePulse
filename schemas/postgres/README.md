# Trading domain schema

This directory contains a PostgreSQL schema for the trading domain used by
TradePulse. Apply the SQL file(s) in this directory before enabling services
that depend on PostgreSQL persistence for orders, executions, and positions.

The schema is designed to be idempotent so it can be re-run safely during
infrastructure provisioning. Each file documents the dependencies it introduces
and should be applied in lexical order.

## Files

- `0001_trading_core.sql` – creates the core trading entities (instruments,
  accounts, orders, executions, positions, and the cash ledger) together with
  supporting types, indexes, and triggers that keep order history, executions,
  and ledger entries synchronised. The script also maintains derived trade
  progress metrics (`orders.filled_quantity` and `orders.avg_fill_price`) and
  enforces product-specific requirements such as limit/stop price validation
  and iceberg visibility constraints.
