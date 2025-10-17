-- Trading domain core schema
-- This script is idempotent and can be re-applied safely.

BEGIN;

-- Enumerated types ----------------------------------------------------------

CREATE TYPE IF NOT EXISTS order_side AS ENUM ('buy', 'sell');
CREATE TYPE IF NOT EXISTS order_type AS ENUM ('market', 'limit', 'stop', 'stop_limit', 'iceberg');
CREATE TYPE IF NOT EXISTS time_in_force AS ENUM ('day', 'gtc', 'ioc', 'fok');
CREATE TYPE IF NOT EXISTS order_status AS ENUM ('new', 'accepted', 'partially_filled', 'filled', 'cancelled', 'rejected', 'expired');

-- Reference data ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS instrument (
    instrument_id      BIGSERIAL PRIMARY KEY,
    symbol             TEXT NOT NULL UNIQUE,
    exchange_code      TEXT NOT NULL,
    asset_type         TEXT NOT NULL CHECK (asset_type IN ('stock','future','option','fx','crypto')),
    currency           TEXT NOT NULL,
    tick_size          NUMERIC(18,8) NOT NULL CHECK (tick_size > 0),
    lot_size           NUMERIC(18,8) NOT NULL CHECK (lot_size > 0),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    archived_at        TIMESTAMPTZ,
    CHECK (archived_at IS NULL OR archived_at >= created_at)
);

CREATE INDEX IF NOT EXISTS instrument_active_idx ON instrument (exchange_code, archived_at) WHERE archived_at IS NULL;

CREATE TABLE IF NOT EXISTS trading_account (
    account_id         BIGSERIAL PRIMARY KEY,
    broker_code        TEXT NOT NULL,
    client_reference   TEXT,
    base_currency      TEXT NOT NULL,
    status             TEXT NOT NULL CHECK (status IN ('active','suspended','closed')),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at          TIMESTAMPTZ,
    CHECK (closed_at IS NULL OR closed_at >= created_at)
);

CREATE UNIQUE INDEX IF NOT EXISTS trading_account_broker_ref_idx
    ON trading_account (broker_code, client_reference) WHERE client_reference IS NOT NULL;

-- Orders --------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS orders (
    order_id           BIGSERIAL PRIMARY KEY,
    client_order_id    TEXT,
    account_id         BIGINT NOT NULL REFERENCES trading_account(account_id) DEFERRABLE INITIALLY IMMEDIATE,
    instrument_id      BIGINT NOT NULL REFERENCES instrument(instrument_id) DEFERRABLE INITIALLY IMMEDIATE,
    parent_order_id    BIGINT REFERENCES orders(order_id) ON DELETE SET NULL DEFERRABLE INITIALLY IMMEDIATE,
    side               order_side NOT NULL,
    order_type         order_type NOT NULL,
    time_in_force      time_in_force NOT NULL,
    quantity           NUMERIC(28,10) NOT NULL CHECK (quantity > 0),
    price              NUMERIC(28,10),
    stop_price         NUMERIC(28,10),
    iceberg_visible    NUMERIC(28,10),
    currency           TEXT NOT NULL,
    source             TEXT NOT NULL,
    placed_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_status        order_status NOT NULL DEFAULT 'new',
    last_status_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    version            BIGINT NOT NULL DEFAULT 1,
    filled_quantity    NUMERIC(28,10) NOT NULL DEFAULT 0,
    avg_fill_price     NUMERIC(28,10),
    CHECK (price IS NULL OR price > 0),
    CHECK (stop_price IS NULL OR stop_price > 0),
    CHECK (iceberg_visible IS NULL OR iceberg_visible > 0),
    CHECK (last_status_at >= placed_at),
    CONSTRAINT orders_quantity_progress_chk CHECK (filled_quantity >= 0 AND filled_quantity <= quantity),
    CONSTRAINT orders_avg_fill_price_chk CHECK (avg_fill_price IS NULL OR avg_fill_price > 0),
    CONSTRAINT orders_price_requirement_chk CHECK (
        (order_type IN ('limit','stop_limit','iceberg') AND price IS NOT NULL)
        OR (order_type IN ('market','stop') AND price IS NULL)
    ),
    CONSTRAINT orders_stop_price_requirement_chk CHECK (
        (order_type IN ('stop','stop_limit') AND stop_price IS NOT NULL)
        OR (order_type NOT IN ('stop','stop_limit') AND stop_price IS NULL)
    ),
    CONSTRAINT orders_iceberg_visible_chk CHECK (
        (order_type = 'iceberg' AND iceberg_visible IS NOT NULL)
        OR (order_type <> 'iceberg' AND iceberg_visible IS NULL)
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS orders_client_id_unique
    ON orders (account_id, client_order_id) WHERE client_order_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS orders_lookup_idx ON orders (instrument_id, side, last_status);
CREATE INDEX IF NOT EXISTS orders_parent_idx ON orders (parent_order_id);
CREATE INDEX IF NOT EXISTS orders_account_placed_idx ON orders (account_id, placed_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS orders_order_account_uidx ON orders (order_id, account_id);
CREATE UNIQUE INDEX IF NOT EXISTS orders_order_instrument_uidx ON orders (order_id, instrument_id);
CREATE INDEX IF NOT EXISTS orders_open_idx
    ON orders (instrument_id, placed_at DESC)
    WHERE last_status IN ('new','accepted','partially_filled');

ALTER TABLE orders
    ADD COLUMN IF NOT EXISTS filled_quantity NUMERIC(28,10) NOT NULL DEFAULT 0;

ALTER TABLE orders
    ADD COLUMN IF NOT EXISTS avg_fill_price NUMERIC(28,10);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'orders_quantity_progress_chk'
    ) THEN
        ALTER TABLE orders
            ADD CONSTRAINT orders_quantity_progress_chk
            CHECK (filled_quantity >= 0 AND filled_quantity <= quantity);
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'orders_avg_fill_price_chk'
    ) THEN
        ALTER TABLE orders
            ADD CONSTRAINT orders_avg_fill_price_chk
            CHECK (avg_fill_price IS NULL OR avg_fill_price > 0);
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'orders_price_requirement_chk'
    ) THEN
        ALTER TABLE orders
            ADD CONSTRAINT orders_price_requirement_chk
            CHECK (
                (order_type IN ('limit','stop_limit','iceberg') AND price IS NOT NULL)
                OR (order_type IN ('market','stop') AND price IS NULL)
            );
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'orders_stop_price_requirement_chk'
    ) THEN
        ALTER TABLE orders
            ADD CONSTRAINT orders_stop_price_requirement_chk
            CHECK (
                (order_type IN ('stop','stop_limit') AND stop_price IS NOT NULL)
                OR (order_type NOT IN ('stop','stop_limit') AND stop_price IS NULL)
            );
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'orders_iceberg_visible_chk'
    ) THEN
        ALTER TABLE orders
            ADD CONSTRAINT orders_iceberg_visible_chk
            CHECK (
                (order_type = 'iceberg' AND iceberg_visible IS NOT NULL)
                OR (order_type <> 'iceberg' AND iceberg_visible IS NULL)
            );
    END IF;
END $$;

-- Order status history ------------------------------------------------------

CREATE TABLE IF NOT EXISTS order_status_history (
    order_id           BIGINT NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE DEFERRABLE INITIALLY IMMEDIATE,
    status             order_status NOT NULL,
    status_reason      TEXT,
    status_payload     JSONB,
    valid_from         TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to           TIMESTAMPTZ,
    version            BIGINT NOT NULL,
    PRIMARY KEY (order_id, version),
    CHECK (valid_from < COALESCE(valid_to, 'infinity'::timestamptz))
);

CREATE INDEX IF NOT EXISTS order_history_active_idx
    ON order_status_history (order_id, valid_to)
    WHERE valid_to IS NULL;

-- Executions ----------------------------------------------------------------

CREATE TABLE IF NOT EXISTS execution (
    execution_id       BIGSERIAL PRIMARY KEY,
    order_id           BIGINT NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE DEFERRABLE INITIALLY IMMEDIATE,
    account_id         BIGINT NOT NULL REFERENCES trading_account(account_id) DEFERRABLE INITIALLY IMMEDIATE,
    instrument_id      BIGINT NOT NULL REFERENCES instrument(instrument_id) DEFERRABLE INITIALLY IMMEDIATE,
    external_exec_id   TEXT,
    execution_time     TIMESTAMPTZ NOT NULL,
    quantity           NUMERIC(28,10) NOT NULL CHECK (quantity > 0),
    price              NUMERIC(28,10) NOT NULL CHECK (price > 0),
    gross_amount       NUMERIC(28,10) GENERATED ALWAYS AS (quantity * price) STORED,
    fee_amount         NUMERIC(28,10) NOT NULL DEFAULT 0,
    fee_currency       TEXT NOT NULL,
    liquidity_flag     TEXT CHECK (liquidity_flag IN ('maker','taker')),
    trade_venue        TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS execution_external_idx
    ON execution (order_id, external_exec_id)
    WHERE external_exec_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS execution_order_time_idx ON execution (order_id, execution_time);
CREATE INDEX IF NOT EXISTS execution_account_time_idx ON execution (account_id, execution_time DESC);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'execution_order_account_fk'
    ) THEN
        ALTER TABLE execution
            ADD CONSTRAINT execution_order_account_fk
            FOREIGN KEY (order_id, account_id)
            REFERENCES orders(order_id, account_id)
            DEFERRABLE INITIALLY IMMEDIATE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'execution_order_instrument_fk'
    ) THEN
        ALTER TABLE execution
            ADD CONSTRAINT execution_order_instrument_fk
            FOREIGN KEY (order_id, instrument_id)
            REFERENCES orders(order_id, instrument_id)
            DEFERRABLE INITIALLY IMMEDIATE;
    END IF;
END $$;

-- Positions -----------------------------------------------------------------

CREATE TABLE IF NOT EXISTS position_lot (
    position_lot_id    BIGSERIAL PRIMARY KEY,
    account_id         BIGINT NOT NULL REFERENCES trading_account(account_id) DEFERRABLE INITIALLY IMMEDIATE,
    instrument_id      BIGINT NOT NULL REFERENCES instrument(instrument_id) DEFERRABLE INITIALLY IMMEDIATE,
    open_quantity      NUMERIC(28,10) NOT NULL,
    open_price         NUMERIC(28,10) NOT NULL CHECK (open_price > 0),
    direction          order_side NOT NULL,
    opened_at          TIMESTAMPTZ NOT NULL,
    closed_at          TIMESTAMPTZ,
    pnl_realized       NUMERIC(28,10) NOT NULL DEFAULT 0,
    CHECK (open_quantity > 0),
    CHECK (closed_at IS NULL OR closed_at >= opened_at)
);

CREATE INDEX IF NOT EXISTS position_active_idx ON position_lot (account_id, instrument_id, closed_at);

CREATE TABLE IF NOT EXISTS position_snapshot (
    snapshot_id        BIGSERIAL PRIMARY KEY,
    account_id         BIGINT NOT NULL REFERENCES trading_account(account_id) DEFERRABLE INITIALLY IMMEDIATE,
    instrument_id      BIGINT NOT NULL REFERENCES instrument(instrument_id) DEFERRABLE INITIALLY IMMEDIATE,
    as_of              TIMESTAMPTZ NOT NULL,
    quantity           NUMERIC(28,10) NOT NULL,
    avg_price          NUMERIC(28,10) NOT NULL,
    unrealized_pnl     NUMERIC(28,10) NOT NULL,
    currency           TEXT NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (account_id, instrument_id, as_of)
);

CREATE INDEX IF NOT EXISTS position_snapshot_asof_idx
    ON position_snapshot (account_id, as_of DESC);

-- Cash ledger ----------------------------------------------------------------

CREATE TABLE IF NOT EXISTS cash_ledger (
    ledger_id          BIGSERIAL PRIMARY KEY,
    account_id         BIGINT NOT NULL REFERENCES trading_account(account_id) DEFERRABLE INITIALLY IMMEDIATE,
    related_order_id   BIGINT REFERENCES orders(order_id) DEFERRABLE INITIALLY IMMEDIATE,
    related_exec_id    BIGINT REFERENCES execution(execution_id) DEFERRABLE INITIALLY IMMEDIATE,
    transaction_type   TEXT NOT NULL CHECK (transaction_type IN ('trade','fee','withdrawal','deposit','adjustment')),
    amount             NUMERIC(28,10) NOT NULL,
    currency           TEXT NOT NULL,
    occurred_at        TIMESTAMPTZ NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS cash_ledger_account_time_idx
    ON cash_ledger (account_id, occurred_at DESC);

-- Trigger helpers -----------------------------------------------------------

CREATE OR REPLACE FUNCTION orders_before_update_version()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.last_status IS DISTINCT FROM OLD.last_status THEN
        NEW.version := OLD.version + 1;
        IF NEW.last_status_at IS NULL OR NEW.last_status_at = OLD.last_status_at THEN
            NEW.last_status_at := now();
        END IF;
    ELSE
        NEW.version := OLD.version;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION orders_status_history_audit()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO order_status_history (order_id, status, status_reason, status_payload, valid_from, version)
        VALUES (NEW.order_id, NEW.last_status, NULL, NULL, NEW.last_status_at, NEW.version);
        RETURN NEW;
    END IF;

    IF NEW.version <> OLD.version THEN
        UPDATE order_status_history
           SET valid_to = NEW.last_status_at
         WHERE order_id = NEW.order_id
           AND valid_to IS NULL;

        INSERT INTO order_status_history (order_id, status, status_reason, status_payload, valid_from, version)
        VALUES (NEW.order_id, NEW.last_status, NULL, NULL, NEW.last_status_at, NEW.version);
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION execution_after_insert_sync_order()
RETURNS TRIGGER AS $$
DECLARE
    order_qty            NUMERIC(28,10);
    filled_qty_before    NUMERIC(28,10);
    new_filled_qty       NUMERIC(28,10);
    avg_price_before     NUMERIC(28,10);
    new_avg_price        NUMERIC(28,10);
    new_status           order_status;
    last_status_before   order_status;
    last_status_ts       TIMESTAMPTZ;
    order_side_val       order_side;
    order_currency       TEXT;
BEGIN
    SELECT o.quantity,
           o.side,
           o.currency,
           o.filled_quantity,
           o.avg_fill_price,
           o.last_status,
           o.last_status_at
      INTO order_qty,
           order_side_val,
           order_currency,
           filled_qty_before,
           avg_price_before,
           last_status_before,
           last_status_ts
      FROM orders o
     WHERE o.order_id = NEW.order_id
     FOR UPDATE;

    IF last_status_before NOT IN ('new','accepted','partially_filled') THEN
        RAISE EXCEPTION 'Execution cannot be recorded for order % in status %',
            NEW.order_id,
            last_status_before
            USING ERRCODE = '55000';
    END IF;

    new_filled_qty := filled_qty_before + NEW.quantity;

    IF new_filled_qty > order_qty THEN
        RAISE EXCEPTION 'Execution would overfill order % (filled % > quantity %)',
            NEW.order_id,
            new_filled_qty,
            order_qty
            USING ERRCODE = '22003';
    END IF;

    IF new_filled_qty = 0 THEN
        new_avg_price := NULL;
    ELSIF filled_qty_before = 0 THEN
        new_avg_price := NEW.price;
    ELSE
        new_avg_price := ((COALESCE(avg_price_before, 0) * filled_qty_before) + (NEW.price * NEW.quantity)) / new_filled_qty;
    END IF;

    IF new_filled_qty >= order_qty THEN
        new_status := 'filled';
    ELSE
        new_status := 'partially_filled';
    END IF;

    IF new_status = last_status_before THEN
        last_status_ts := GREATEST(last_status_ts, NEW.execution_time);
    ELSE
        last_status_ts := NEW.execution_time;
    END IF;

    UPDATE orders
       SET last_status     = new_status,
           last_status_at  = last_status_ts,
           filled_quantity = new_filled_qty,
           avg_fill_price  = new_avg_price
     WHERE order_id = NEW.order_id
       AND last_status IN ('new','accepted','partially_filled')
       AND (new_status <> last_status_before
            OR new_filled_qty <> filled_qty_before
            OR new_avg_price IS DISTINCT FROM avg_price_before);

    INSERT INTO cash_ledger (account_id, related_order_id, related_exec_id, transaction_type, amount, currency, occurred_at)
    VALUES (
        NEW.account_id,
        NEW.order_id,
        NEW.execution_id,
        'trade',
        CASE WHEN order_side_val = 'buy' THEN -NEW.gross_amount ELSE NEW.gross_amount END,
        order_currency,
        NEW.execution_time
    );

    IF NEW.fee_amount <> 0 THEN
        INSERT INTO cash_ledger (account_id, related_order_id, related_exec_id, transaction_type, amount, currency, occurred_at)
        VALUES (
            NEW.account_id,
            NEW.order_id,
            NEW.execution_id,
            'fee',
            -NEW.fee_amount,
            NEW.fee_currency,
            NEW.execution_time
        );
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger wiring ------------------------------------------------------------

DROP TRIGGER IF EXISTS orders_before_update_version_trg ON orders;
CREATE TRIGGER orders_before_update_version_trg
    BEFORE UPDATE ON orders
    FOR EACH ROW
    EXECUTE FUNCTION orders_before_update_version();

DROP TRIGGER IF EXISTS orders_status_history_trg ON orders;
CREATE TRIGGER orders_status_history_trg
    AFTER INSERT OR UPDATE ON orders
    FOR EACH ROW
    EXECUTE FUNCTION orders_status_history_audit();

DROP TRIGGER IF EXISTS execution_after_insert_trg ON execution;
CREATE TRIGGER execution_after_insert_trg
    AFTER INSERT ON execution
    FOR EACH ROW
    EXECUTE FUNCTION execution_after_insert_sync_order();

COMMIT;
