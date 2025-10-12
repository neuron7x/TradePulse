from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pytest

from core.messaging.event_bus import (
    EventBusBackend,
    EventBusConfig,
    EventEnvelope,
    EventTopic,
    KafkaEventBus,
    NATSEventBus,
    _envelope_from_kafka_message,
    _envelope_from_nats_message,
)
from core.messaging.idempotency import InMemoryEventIdempotencyStore


@pytest.fixture(autouse=True)
def stub_aiokafka(monkeypatch):
    """Inject a lightweight aiokafka stub for KafkaEventBus tests."""

    class DummyKafkaMessage:
        def __init__(self, *, key: bytes, value: bytes, headers: list[tuple[str, bytes]]) -> None:
            self.key = key
            self.value = value
            self.headers = headers

    class DummyAIOKafkaProducer:
        instances: List["DummyAIOKafkaProducer"] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.started = False
            self.sent: List[Dict[str, Any]] = []
            DummyAIOKafkaProducer.instances.append(self)

        async def start(self) -> None:  # pragma: no cover - exercised via tests
            self.started = True

        async def stop(self) -> None:  # pragma: no cover - exercised via tests
            self.started = False

        async def send_and_wait(
            self, topic: str, value: bytes, *, key: bytes, headers: list[tuple[str, bytes]]
        ) -> None:
            self.sent.append({"topic": topic, "value": value, "key": key, "headers": headers})

    class DummyAIOKafkaConsumer:
        instances: List["DummyAIOKafkaConsumer"] = []
        queued_messages: List[DummyKafkaMessage] = []

        def __init__(self, *topics: str, **kwargs: Any) -> None:
            self.topics = topics
            self.kwargs = kwargs
            self.started = False
            self._commits: int = 0
            self._stopped = False
            self._messages = list(self.__class__.queued_messages)
            DummyAIOKafkaConsumer.instances.append(self)

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            self._stopped = True

        async def commit(self) -> None:
            self._commits += 1

        def __aiter__(self):  # pragma: no cover - exercised via tests
            return self

        async def __anext__(self):
            if self._messages:
                return self._messages.pop(0)
            raise StopAsyncIteration

    stub_module = type(
        "aiokafka",
        (),
        {
            "AIOKafkaProducer": DummyAIOKafkaProducer,
            "AIOKafkaConsumer": DummyAIOKafkaConsumer,
            "DummyKafkaMessage": DummyKafkaMessage,
        },
    )

    monkeypatch.setitem(sys.modules, "aiokafka", stub_module)
    try:
        yield stub_module
    finally:
        monkeypatch.delitem(sys.modules, "aiokafka", raising=False)


@pytest.fixture(autouse=True)
def stub_nats(monkeypatch):
    """Inject a lightweight NATS stub for NATSEventBus tests."""

    @dataclass
    class DummyNATSMessage:
        headers: Dict[str, Any]
        data: bytes
        ack_count: int = 0

        async def ack(self) -> None:
            self.ack_count += 1

    class DummyJetStream:
        def __init__(self) -> None:
            self.published: List[Dict[str, Any]] = []
            self.subscriptions: List[Dict[str, Any]] = []
            self.add_stream_calls: int = 0

        async def publish(
            self,
            subject: str,
            *,
            payload: bytes,
            headers: Optional[Dict[str, str]] = None,
            timeout: Optional[int] = None,
        ) -> None:
            self.published.append(
                {
                    "subject": subject,
                    "payload": payload,
                    "headers": headers or {},
                    "timeout": timeout,
                }
            )

        async def subscribe(
            self,
            subject: str,
            *,
            durable: str,
            cb: Callable[[DummyNATSMessage], Any],
            manual_ack: bool,
            idle_heartbeat: int,
        ) -> None:
            self.subscriptions.append(
                {
                    "subject": subject,
                    "durable": durable,
                    "cb": cb,
                    "manual_ack": manual_ack,
                    "idle_heartbeat": idle_heartbeat,
                }
            )

        async def add_stream(self, *, name: str, subjects: List[str]) -> None:
            self.add_stream_calls += 1

    class DummyNATSClient:
        def __init__(self) -> None:
            self.jet = DummyJetStream()
            self.drain_called = False
            self.close_called = False

        def jetstream(self) -> DummyJetStream:
            return self.jet

        async def drain(self) -> None:
            self.drain_called = True

        async def close(self) -> None:
            self.close_called = True

    async def connect(url: str, name: str) -> DummyNATSClient:  # pragma: no cover - exercised via tests
        client = DummyNATSClient()
        dummy_module.latest_client = client
        return client

    dummy_module = type("nats", (), {"connect": connect, "DummyNATSMessage": DummyNATSMessage})
    monkeypatch.setitem(sys.modules, "nats", dummy_module)
    try:
        yield dummy_module
    finally:
        monkeypatch.delitem(sys.modules, "nats", raising=False)


@pytest.mark.asyncio
async def test_kafka_start_and_stop_initialises_producer(stub_aiokafka) -> None:
    config = EventBusConfig(backend=EventBusBackend.KAFKA, bootstrap_servers="kafka:9092")
    bus = KafkaEventBus(config)
    await bus.start()
    producer = stub_aiokafka.AIOKafkaProducer.instances[-1]
    assert producer.started is True
    assert producer.kwargs["bootstrap_servers"] == "kafka:9092"

    await bus.stop()
    assert producer.started is False


@pytest.mark.asyncio
async def test_kafka_subscribe_processes_message_and_commits(stub_aiokafka) -> None:
    config = EventBusConfig(backend=EventBusBackend.KAFKA, bootstrap_servers="kafka:9092")
    store = InMemoryEventIdempotencyStore(ttl_seconds=30)
    bus = KafkaEventBus(config, idempotency_store=store)

    await bus.start()

    headers = [
        ("event_type", b"ticks"),
        ("partition_key", b"AAPL"),
        ("event_id", b"evt-123"),
        ("schema_version", b"1.0.0"),
        ("content_type", b"application/avro"),
        ("occurred_at", b"2024-01-01T00:00:00"),
    ]
    message = stub_aiokafka.DummyKafkaMessage(key=b"AAPL", value=b"payload", headers=headers)
    stub_aiokafka.AIOKafkaConsumer.queued_messages = [message]

    received: list[EventEnvelope] = []

    async def handler(envelope: EventEnvelope) -> None:
        received.append(envelope)

    await bus.subscribe(EventTopic.MARKET_TICKS, handler)
    task = bus._consumer_tasks[EventTopic.MARKET_TICKS.metadata.name]
    await asyncio.wait_for(task, timeout=0.1)

    assert len(received) == 1
    assert received[0].partition_key == "AAPL"
    assert store.was_processed("evt-123") is True

    consumer = stub_aiokafka.AIOKafkaConsumer.instances[-1]
    assert consumer._commits == 1
    assert consumer._stopped is True

    await bus.stop()


@pytest.mark.asyncio
async def test_kafka_retry_and_dlq_routing(stub_aiokafka) -> None:
    config = EventBusConfig(backend=EventBusBackend.KAFKA, bootstrap_servers="kafka:9092", retry_attempts=2)
    bus = KafkaEventBus(config)
    producer = stub_aiokafka.AIOKafkaProducer()
    bus._producer = producer

    envelope = EventEnvelope(
        event_type="ticks",
        partition_key="AAPL",
        event_id="evt-1",
        payload=b"payload",
        content_type="application/avro",
        schema_version="1.0.0",
    )

    await bus._publish_retry_or_dlq(EventTopic.MARKET_TICKS, envelope)
    retry_call = producer.sent[-1]
    assert retry_call["topic"] == EventTopic.MARKET_TICKS.metadata.retry_topic

    envelope.headers["retry-attempt"] = "2"
    await bus._publish_retry_or_dlq(EventTopic.MARKET_TICKS, envelope)
    dlq_call = producer.sent[-1]
    assert dlq_call["topic"] == EventTopic.MARKET_TICKS.metadata.dlq_topic


@pytest.mark.asyncio
async def test_nats_publish_subscribe_and_retry(stub_nats) -> None:
    config = EventBusConfig(backend=EventBusBackend.NATS, nats_url="nats://localhost:4222", retry_attempts=1)
    store = InMemoryEventIdempotencyStore(ttl_seconds=30)
    bus = NATSEventBus(config, idempotency_store=store)

    await bus.start()
    client = stub_nats.latest_client
    js = client.jet

    envelope = EventEnvelope(
        event_type="signals",
        partition_key="AAPL",
        event_id="evt-1",
        payload=b"payload",
        content_type="application/json",
        schema_version="1.0.0",
    )

    await bus.publish(EventTopic.SIGNALS, envelope)
    published = js.published[-1]
    assert published["subject"] == EventTopic.SIGNALS.metadata.name

    received: list[EventEnvelope] = []

    async def handler(enveloped: EventEnvelope) -> None:
        received.append(enveloped)

    await bus.subscribe(EventTopic.SIGNALS, handler)
    subscription = js.subscriptions[-1]

    message = stub_nats.DummyNATSMessage(
        headers=envelope.as_message(),
        data=b"payload",
    )
    await subscription["cb"](message)

    assert len(received) == 1
    assert message.ack_count == 1
    assert store.was_processed("evt-1") is True

    async def failing_handler(enveloped: EventEnvelope) -> None:
        raise RuntimeError("boom")

    await bus.subscribe(EventTopic.SIGNALS, failing_handler)
    subscription_failure = js.subscriptions[-1]

    retry_headers = envelope.as_message()
    retry_headers["event_id"] = "evt-2"
    retry_headers["partition_key"] = "MSFT"
    failing_message = stub_nats.DummyNATSMessage(headers=retry_headers, data=b"payload")
    await subscription_failure["cb"](failing_message)

    retry_event = js.published[-1]
    assert retry_event["subject"] == EventTopic.SIGNALS.metadata.retry_topic

    await bus.stop()
    assert client.drain_called is True
    assert client.close_called is True


def test_envelope_serialisation_round_trip(stub_aiokafka) -> None:
    envelope = EventEnvelope(
        event_type="ticks",
        partition_key="AAPL",
        event_id="evt-42",
        payload=b"payload",
        content_type="application/avro",
        schema_version="3.0.0",
        headers={"extra": "value"},
    )
    headers = [(key, value.encode("utf-8")) for key, value in envelope.as_message().items()]
    message = stub_aiokafka.DummyKafkaMessage(key=b"AAPL", value=b"payload", headers=headers)
    reconstructed = _envelope_from_kafka_message(message)
    assert reconstructed.event_id == envelope.event_id
    assert reconstructed.headers["extra"] == "value"


def test_nats_envelope_deserialisation(stub_nats) -> None:
    headers = {
        "event_type": "signals",
        "partition_key": "AAPL",
        "event_id": "evt-100",
        "schema_version": "5.0.0",
        "content_type": "application/json",
        "occurred_at": "2024-01-01T00:00:00",
    }
    message = stub_nats.DummyNATSMessage(headers=headers, data=b"payload")
    reconstructed = _envelope_from_nats_message(message)
    assert reconstructed.schema_version == "5.0.0"
    assert reconstructed.payload == b"payload"

