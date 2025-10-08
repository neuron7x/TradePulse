# TradePulse Architecture & Dataflow

This document provides visual diagrams and descriptions of the TradePulse architecture.

## System Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        CSV[CSV Files]
        WS[WebSocket Streams]
        API[REST APIs]
    end
    
    subgraph "Data Layer"
        Ingest[Data Ingestor]
        Preproc[Preprocessor]
        Ticker[Ticker Stream]
    end
    
    subgraph "Feature Layer"
        Kuramoto[Kuramoto Sync]
        Entropy[Entropy]
        Hurst[Hurst Exponent]
        Ricci[Ricci Curvature]
        Composite[Composite Features]
    end
    
    subgraph "Strategy Layer"
        Agent[Agent System]
        Strategy[Strategy Evaluator]
        Signal[Signal Generator]
    end
    
    subgraph "Execution Layer"
        Risk[Risk Manager]
        Order[Order Executor]
        Position[Position Tracker]
    end
    
    subgraph "Observability"
        Logging[Structured Logging]
        Metrics[Prometheus Metrics]
        Tracing[OpenTelemetry]
    end
    
    CSV --> Ingest
    WS --> Ingest
    API --> Ingest
    
    Ingest --> Preproc
    Preproc --> Ticker
    
    Ticker --> Kuramoto
    Ticker --> Entropy
    Ticker --> Hurst
    Ticker --> Ricci
    
    Kuramoto --> Composite
    Entropy --> Composite
    Hurst --> Composite
    Ricci --> Composite
    
    Composite --> Agent
    Agent --> Strategy
    Strategy --> Signal
    
    Signal --> Risk
    Risk --> Order
    Order --> Position
    
    Ingest -.-> Logging
    Ingest -.-> Metrics
    Ingest -.-> Tracing
    
    Composite -.-> Logging
    Composite -.-> Metrics
    Composite -.-> Tracing
    
    Order -.-> Logging
    Order -.-> Metrics
    Order -.-> Tracing
```

## Data Flow

### 1. Data Ingestion Pipeline

```mermaid
sequenceDiagram
    participant Source
    participant Ingestor
    participant Validator
    participant Preprocessor
    participant Ticker
    
    Source->>Ingestor: Raw Data
    Ingestor->>Validator: Validate Schema
    Validator->>Preprocessor: Clean Data
    Preprocessor->>Ticker: Normalized Tickers
    Ticker->>Ticker: Emit Stream
```

### 2. Feature Computation Pipeline

```mermaid
flowchart LR
    A[Ticker Stream] --> B[Feature Block]
    B --> C[Kuramoto]
    B --> D[Entropy]
    B --> E[Hurst]
    B --> F[Ricci]
    
    C --> G[Composite]
    D --> G
    E --> G
    F --> G
    
    G --> H[Feature Result]
    
    style A fill:#e1f5ff
    style H fill:#ffe1f5
```

### 3. Backtest Execution Flow

```mermaid
graph TD
    A[Historical Data] --> B[Walk-Forward Engine]
    B --> C{Generate Signal}
    C -->|Buy| D[Long Position]
    C -->|Sell| E[Short Position]
    C -->|Hold| F[No Action]
    
    D --> G[Track P&L]
    E --> G
    F --> G
    
    G --> H[Calculate Metrics]
    H --> I[Backtest Result]
    
    I --> J[Max Drawdown]
    I --> K[P&L]
    I --> L[# Trades]
```

### 4. Agent Optimization Loop

```mermaid
graph TB
    A[Initial Population] --> B[Evaluate Fitness]
    B --> C[Select Best]
    C --> D[Crossover]
    D --> E[Mutation]
    E --> F[New Generation]
    F --> B
    
    C -->|Best| G[Deploy Strategy]
```

## Module Dependencies

```mermaid
graph LR
    subgraph "Core"
        Data[data]
        Indicators[indicators]
        Utils[utils]
    end
    
    subgraph "Analytics"
        Agent[agent]
        Phase[phase]
        Strategies[strategies]
    end
    
    subgraph "Trading"
        Backtest[backtest]
        Execution[execution]
    end
    
    subgraph "Observability"
        Logging[logging]
        Metrics[metrics]
        Tracing[tracing]
        Validation[validation]
    end
    
    Data --> Indicators
    Indicators --> Strategies
    Strategies --> Agent
    
    Agent --> Backtest
    Strategies --> Backtest
    Backtest --> Execution
    
    Utils --> Logging
    Utils --> Metrics
    Utils --> Tracing
    Utils --> Validation
    
    Data -.-> Logging
    Indicators -.-> Metrics
    Backtest -.-> Tracing
```

## Observability Stack

```mermaid
graph TB
    subgraph "Application"
        App[TradePulse]
    end
    
    subgraph "Instrumentation"
        Log[Structured Logger]
        Met[Metrics Collector]
        Trace[Tracer]
    end
    
    subgraph "Collection"
        LogAgg[Log Aggregator]
        PromSrv[Prometheus Server]
        TraceSrv[Trace Collector]
    end
    
    subgraph "Storage"
        LogStore[Elasticsearch]
        MetStore[Prometheus TSDB]
        TraceStore[Jaeger/Tempo]
    end
    
    subgraph "Visualization"
        Kibana[Kibana]
        Grafana[Grafana]
        JaegerUI[Jaeger UI]
    end
    
    App --> Log
    App --> Met
    App --> Trace
    
    Log --> LogAgg
    Met --> PromSrv
    Trace --> TraceSrv
    
    LogAgg --> LogStore
    PromSrv --> MetStore
    TraceSrv --> TraceStore
    
    LogStore --> Kibana
    MetStore --> Grafana
    TraceStore --> JaegerUI
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        Web[Web Dashboard]
        CLI[CLI Tool]
        API[API Client]
    end
    
    subgraph "Application Layer"
        Gateway[API Gateway]
        Core[TradePulse Core]
        Worker[Backtest Workers]
    end
    
    subgraph "Data Layer"
        Cache[Redis Cache]
        DB[(PostgreSQL)]
        TimeSeries[(TimescaleDB)]
    end
    
    subgraph "External Services"
        Exchange[Exchange APIs]
        DataFeed[Data Feeds]
    end
    
    Web --> Gateway
    CLI --> Gateway
    API --> Gateway
    
    Gateway --> Core
    Core --> Worker
    
    Core --> Cache
    Core --> DB
    Core --> TimeSeries
    
    Core --> Exchange
    Core --> DataFeed
```

## Security Architecture

```mermaid
graph TB
    subgraph "Security Layers"
        Auth[Authentication]
        Authz[Authorization]
        Valid[Input Validation]
        Encrypt[Encryption]
    end
    
    subgraph "Scanning"
        Secret[Secret Scanning]
        Vuln[Vulnerability Scanning]
        SAST[Static Analysis]
        DAST[Dynamic Analysis]
    end
    
    subgraph "Runtime Protection"
        Rate[Rate Limiting]
        WAF[Web Application Firewall]
        IDS[Intrusion Detection]
    end
    
    subgraph "Monitoring"
        Audit[Audit Logs]
        Alert[Security Alerts]
        SIEM[SIEM Integration]
    end
    
    Auth --> Valid
    Valid --> Authz
    Authz --> Encrypt
    
    Secret --> Audit
    Vuln --> Alert
    SAST --> Alert
    DAST --> Alert
    
    Rate --> IDS
    WAF --> IDS
    IDS --> Alert
    
    Audit --> SIEM
    Alert --> SIEM
```

## Testing Architecture

```mermaid
graph TB
    subgraph "Test Types"
        Unit[Unit Tests]
        Prop[Property Tests]
        Chaos[Chaos Tests]
        Integ[Integration Tests]
        E2E[E2E Tests]
    end
    
    subgraph "Test Infrastructure"
        Runner[Test Runner]
        Coverage[Coverage Tool]
        Mutation[Mutation Testing]
        Mock[Mock Services]
    end
    
    subgraph "CI/CD"
        PR[Pull Request]
        Build[Build]
        Test[Test]
        Deploy[Deploy]
    end
    
    Unit --> Runner
    Prop --> Runner
    Chaos --> Runner
    Integ --> Runner
    E2E --> Runner
    
    Runner --> Coverage
    Runner --> Mutation
    Runner --> Mock
    
    PR --> Build
    Build --> Test
    Test --> Deploy
```

## Performance Optimization

```mermaid
graph LR
    subgraph "Optimization Layers"
        Cache[Caching]
        Batch[Batching]
        Async[Async Processing]
        Parallel[Parallelization]
    end
    
    subgraph "Monitoring"
        Prof[Profiling]
        Bench[Benchmarking]
        Load[Load Testing]
    end
    
    subgraph "Strategies"
        Lazy[Lazy Loading]
        Pool[Connection Pooling]
        Index[Database Indexing]
        CDN[CDN for Static Assets]
    end
    
    Cache --> Prof
    Batch --> Bench
    Async --> Load
    Parallel --> Load
    
    Prof --> Lazy
    Bench --> Pool
    Load --> Index
```

## Key Design Principles

### 1. Separation of Concerns
- **Data Layer**: Handles data ingestion and normalization
- **Feature Layer**: Computes mathematical indicators
- **Strategy Layer**: Generates trading signals
- **Execution Layer**: Manages orders and positions
- **Observability Layer**: Provides monitoring and tracing

### 2. Plug-and-Play Architecture
- All components implement standard interfaces
- Easy to add new indicators, strategies, or data sources
- Modular design enables independent testing and deployment

### 3. Observability First
- All operations are instrumented with metrics
- Structured logging with correlation IDs
- Distributed tracing for end-to-end visibility

### 4. Security by Design
- Input validation at all boundaries
- Secret scanning in CI/CD
- Runtime validation with Pydantic
- Audit trails for all critical operations

### 5. Test-Driven Development
- 314+ comprehensive tests
- Property-based and chaos testing
- Mutation testing for test effectiveness
- 96.91% coverage on critical modules

## Related Documentation

- [Testing Guide](../TESTING.md)
- [Contributing Guidelines](../CONTRIBUTING.md)
- [Advanced Testing & Observability](advanced-testing-observability.md)
- [Security Policy](../SECURITY.md)
