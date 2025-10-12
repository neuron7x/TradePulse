"""Event bus abstractions with Kafka and NATS backends."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Awaitable, Callable, Dict, MutableMapping, Optional

from observability.tracing import activate_trace_headers, inject_trace_context

from .idempotency import EventIdempotencyStore, InMemoryEventIdempotencyStore, current_timestamp


@dataclass(frozen=True)
class TopicMetadata:
    name: str
    partition_key: str
    retry_topic: str
    dlq_topic: str


class EventTopic(Enum):
    """Canonical event bus topics."""

    MARKET_TICKS = TopicMetadata(
        name="tradepulse.market.ticks",
        partition_key="symbol",
        retry_topic="tradepulse.market.ticks.retry",
        dlq_topic="tradepulse.market.ticks.dlq",
    )
    MARKET_BARS = TopicMetadata(
        name="tradepulse.market.bars",
        partition_key="symbol",
        retry_topic="tradepulse.market.bars.retry",
        dlq_topic="tradepulse.market.bars.dlq",
    )
    SIGNALS = TopicMetadata(
        name="tradepulse.signals.generated",
        partition_key="symbol",
        retry_topic="tradepulse.signals.generated.retry",
        dlq_topic="tradepulse.signals.generated.dlq",
    )
    ORDERS = TopicMetadata(
        name="tradepulse.execution.orders",
        partition_key="symbol",
        retry_topic="tradepulse.execution.orders.retry",
        dlq_topic="tradepulse.execution.orders.dlq",
    )
    FILLS = TopicMetadata(
        name="tradepulse.execution.fills",
        partition_key="symbol",
        retry_topic="tradepulse.execution.fills.retry",
        dlq_topic="tradepulse.execution.fills.dlq",
    )

    def __str__(self) -> str:
        return self.value.name

    @property
    def metadata(self) -> TopicMetadata:
        return self.value


@dataclass
class EventEnvelope:
    """Transport-agnostic wrapper for payloads."""

    event_type: str
    partition_key: str
    event_id: str
    payload: bytes
    content_type: str
    schema_version: str
    occurred_at: datetime = field(default_factory=current_timestamp)
    headers: MutableMapping[str, str] = field(default_factory=dict)

    def as_message(self) -> Dict[str, str]:
        base: Dict[str, str] = {
            "event_type": self.event_type,
            "partition_key": self.partition_key,
            "event_id": self.event_id,
            "schema_version": str(self.schema_version),
            "occurred_at": self.occurred_at.isoformat(),
            "content_type": self.content_type,
        }
        for key, value in self.headers.items():
            base[key] = value
        return base


class EventBusBackend(str, Enum):
    KAFKA = "kafka"
    NATS = "nats"


@dataclass
class EventBusConfig:
    backend: EventBusBackend
    bootstrap_servers: Optional[str] = None
    nats_url: Optional[str] = None
    client_id: str = "tradepulse-event-bus"
    consumer_group: str = "tradepulse"
    enable_idempotence: bool = True
    retry_attempts: int = 5
    retry_backoff_ms: int = 250


class BaseEventBus:
    def __init__(self, config: EventBusConfig, idempotency_store: EventIdempotencyStore | None = None) -> None:
        self._config = config
        self._idempotency = idempotency_store or InMemoryEventIdempotencyStore()

    async def publish(self, topic: EventTopic, envelope: EventEnvelope) -> None:
        raise NotImplementedError

    async def subscribe(
        self,
        topic: EventTopic,
        handler: Callable[[EventEnvelope], Awaitable[None]],
        *,
        durable_name: str | None = None,
    ) -> None:
        raise NotImplementedError

    @property
    def idempotency_store(self) -> EventIdempotencyStore:
        return self._idempotency


class KafkaEventBus(BaseEventBus):
    """Kafka-backed asynchronous event bus."""

    def __init__(self, config: EventBusConfig, idempotency_store: EventIdempotencyStore | None = None) -> None:
        if config.backend is not EventBusBackend.KAFKA:
            raise ValueError("KafkaEventBus requires a Kafka backend configuration")
        super().__init__(config, idempotency_store=idempotency_store)
        self._producer = None
        self._consumer_tasks: Dict[str, asyncio.Task[None]] = {}

    async def start(self) -> None:
        from aiokafka import AIOKafkaProducer

        if not self._config.bootstrap_servers:
            raise ValueError("bootstrap_servers must be configured for Kafka backend")
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._config.bootstrap_servers,
            client_id=self._config.client_id,
            enable_idempotence=self._config.enable_idempotence,
            acks="all",
            retry_backoff_ms=self._config.retry_backoff_ms,
            linger_ms=5,
        )
        await self._producer.start()

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()
        for task in self._consumer_tasks.values():
            task.cancel()
        self._consumer_tasks.clear()

    async def publish(self, topic: EventTopic, envelope: EventEnvelope) -> None:
        if self._producer is None:
            raise RuntimeError("KafkaEventBus.start() must be called before publish()")
        inject_trace_context(envelope.headers)
        key = envelope.partition_key.encode("utf-8")
        headers = [(name, value.encode("utf-8")) for name, value in envelope.as_message().items()]
        await self._producer.send_and_wait(topic.metadata.name, envelope.payload, key=key, headers=headers)

    async def subscribe(
        self,
        topic: EventTopic,
        handler: Callable[[EventEnvelope], Awaitable[None]],
        *,
        durable_name: str | None = None,
    ) -> None:
        from aiokafka import AIOKafkaConsumer

        group_id = durable_name or self._config.consumer_group
        consumer = AIOKafkaConsumer(
            topic.metadata.name,
            bootstrap_servers=self._config.bootstrap_servers,
            group_id=group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        )
        await consumer.start()

        async def _consume() -> None:
            try:
                async for msg in consumer:
                    envelope = _envelope_from_kafka_message(msg)
                    if self.idempotency_store.was_processed(envelope.event_id):
                        await consumer.commit()
                        continue
                    try:
                        with activate_trace_headers(envelope.headers):
                            await handler(envelope)
                        self.idempotency_store.mark_processed(envelope.event_id)
                        await consumer.commit()
                    except Exception:
                        await self._publish_retry_or_dlq(topic, envelope)
                        await consumer.commit()
            finally:
                await consumer.stop()

        task = asyncio.create_task(_consume(), name=f"kafka-consumer-{topic.metadata.name}")
        self._consumer_tasks[topic.metadata.name] = task

    async def _publish_retry_or_dlq(self, topic: EventTopic, envelope: EventEnvelope) -> None:
        if self._producer is None:
            return
        attempt = int(envelope.headers.get("retry-attempt", "0")) + 1
        if attempt <= self._config.retry_attempts:
            envelope.headers["retry-attempt"] = str(attempt)
            await self._producer.send_and_wait(
                topic.metadata.retry_topic,
                envelope.payload,
                key=envelope.partition_key.encode("utf-8"),
                headers=[(k, v.encode("utf-8")) for k, v in envelope.as_message().items()],
            )
        else:
            await self._producer.send_and_wait(
                topic.metadata.dlq_topic,
                envelope.payload,
                key=envelope.partition_key.encode("utf-8"),
                headers=[(k, v.encode("utf-8")) for k, v in envelope.as_message().items()],
            )


class NATSEventBus(BaseEventBus):
    """NATS JetStream backed event bus."""

    def __init__(self, config: EventBusConfig, idempotency_store: EventIdempotencyStore | None = None) -> None:
        if config.backend is not EventBusBackend.NATS:
            raise ValueError("NATSEventBus requires a NATS backend configuration")
        super().__init__(config, idempotency_store=idempotency_store)
        self._nc = None
        self._js = None
        self._streams_initialised: Dict[str, asyncio.Lock] = {}

    async def start(self) -> None:
        import nats

        self._nc = await nats.connect(self._config.nats_url or "nats://127.0.0.1:4222", name=self._config.client_id)
        self._js = self._nc.jetstream()

    async def stop(self) -> None:
        if self._nc:
            await self._nc.drain()
            await self._nc.close()

    async def publish(self, topic: EventTopic, envelope: EventEnvelope) -> None:
        if not self._nc or not self._js:
            raise RuntimeError("NATSEventBus.start() must be called before publish()")
        await self._ensure_stream(topic)
        inject_trace_context(envelope.headers)
        headers = {k: v for k, v in envelope.as_message().items()}
        await self._js.publish(
            subject=topic.metadata.name,
            payload=envelope.payload,
            headers=headers,
            timeout=5,
        )

    async def subscribe(
        self,
        topic: EventTopic,
        handler: Callable[[EventEnvelope], Awaitable[None]],
        *,
        durable_name: str | None = None,
    ) -> None:
        if not self._nc or not self._js:
            raise RuntimeError("NATSEventBus.start() must be called before subscribe()")
        await self._ensure_stream(topic)

        async def _callback(msg) -> None:  # type: ignore[no-untyped-def]
            envelope = _envelope_from_nats_message(msg)
            if self.idempotency_store.was_processed(envelope.event_id):
                await msg.ack()
                return
            try:
                with activate_trace_headers(envelope.headers):
                    await handler(envelope)
                self.idempotency_store.mark_processed(envelope.event_id)
                await msg.ack()
            except Exception:
                await self._publish_retry_or_dlq(topic, envelope)
                await msg.ack()

        await self._js.subscribe(
            topic.metadata.name,
            durable=durable_name or self._config.consumer_group,
            cb=_callback,
            manual_ack=True,
            idle_heartbeat=5,
        )

    async def _ensure_stream(self, topic: EventTopic) -> None:
        if not self._nc or not self._js:
            raise RuntimeError("NATS client not initialised")
        lock = self._streams_initialised.setdefault(topic.metadata.name, asyncio.Lock())
        async with lock:
            try:
                await self._js.add_stream(
                    name=topic.metadata.name.replace(".", "_"),
                    subjects=[topic.metadata.name, topic.metadata.retry_topic, topic.metadata.dlq_topic],
                )
            except Exception:
                # Stream likely already exists
                pass

    async def _publish_retry_or_dlq(self, topic: EventTopic, envelope: EventEnvelope) -> None:
        if not self._nc or not self._js:
            return
        attempt = int(envelope.headers.get("retry-attempt", "0")) + 1
        headers = envelope.as_message()
        headers["retry-attempt"] = str(attempt)
        if attempt <= self._config.retry_attempts:
            await self._js.publish(topic.metadata.retry_topic, payload=envelope.payload, headers=headers)
        else:
            await self._js.publish(topic.metadata.dlq_topic, payload=envelope.payload, headers=headers)


def _envelope_from_kafka_message(message) -> EventEnvelope:  # type: ignore[no-untyped-def]
    headers = {key: value.decode("utf-8") for key, value in message.headers}
    occurred_at = datetime.fromisoformat(headers.get("occurred_at")) if "occurred_at" in headers else datetime.utcnow()
    return EventEnvelope(
        event_type=headers.get("event_type", ""),
        partition_key=headers.get("partition_key", message.key.decode("utf-8")),
        event_id=headers.get("event_id", ""),
        payload=message.value,
        content_type=headers.get("content_type", "application/octet-stream"),
        schema_version=headers.get("schema_version", "0.0.0"),
        occurred_at=occurred_at,
        headers=headers,
    )


def _envelope_from_nats_message(message) -> EventEnvelope:  # type: ignore[no-untyped-def]
    headers = dict(message.headers or {})
    occurred_at_raw = headers.get("occurred_at")
    occurred_at = (
        datetime.fromisoformat(occurred_at_raw)
        if isinstance(occurred_at_raw, str)
        else datetime.utcnow()
    )
    return EventEnvelope(
        event_type=headers.get("event_type", ""),
        partition_key=headers.get("partition_key", ""),
        event_id=headers.get("event_id", ""),
        payload=bytes(message.data),
        content_type=headers.get("content_type", "application/octet-stream"),
        schema_version=headers.get("schema_version", "0.0.0"),
        occurred_at=occurred_at,
        headers={k: (v if isinstance(v, str) else json.dumps(v)) for k, v in headers.items()},
    )
