# TradePulse: Kuramoto-Ricci Composite Configuration
# Location: configs/kuramoto_ricci_composite.yaml

# Multi-Scale Kuramoto Configuration
kuramoto:
  # Timeframes for multi-scale analysis (in seconds)
  timeframes:
    - 60      # 1 minute
    - 300     # 5 minutes
    - 900     # 15 minutes
    - 3600    # 1 hour
  
  # Adaptive window settings
  adaptive_window:
    enabled: true
    min_window: 50
    max_window: 500
    base_window: 200  # Used when adaptive is disabled
  
  # Consensus weights (must sum to 1.0)
  consensus_weights:
    M1: 0.1    # 1 minute
    M5: 0.2    # 5 minutes
    M15: 0.3   # 15 minutes
    H1: 0.4    # 1 hour
  
  # Phase extraction
  hilbert:
    detrend: true
    method: "linear"  # linear, constant, or none


# Temporal Ricci Configuration
ricci:
  # Graph construction
  graph:
    n_levels: 20              # Number of price levels for graph nodes
    connection_threshold: 0.1  # Minimum weight for edge creation
  
  # Temporal analysis
  temporal:
    window_size: 100           # Window for each snapshot
    n_snapshots: 10            # Number of snapshots to track
    snapshot_interval: null    # Auto-calculated if null
  
  # Ollivier-Ricci parameters
  ollivier:
    alpha: 0.5                 # Lazy random walk parameter [0, 1]
    timeout: 1                 # Graph edit distance timeout (seconds)
  
  # Topological transition detection
  transition:
    derivative_window: 2       # Window for computing metric changes
    jump_threshold: 0.3        # Threshold for transition detection
    sigmoid_steepness: 10      # Steepness of sigmoid function


# Composite Indicator Configuration
composite:
  # Phase determination thresholds
  thresholds:
    # Kuramoto thresholds
    R_strong_emergent: 0.80     # Strong emergent phase threshold
    R_proto_emergent: 0.40      # Proto emergent phase threshold
    coherence_min: 0.60         # Minimum cross-scale coherence
    
    # Ricci thresholds
    ricci_negative: -0.30       # Negative curvature threshold
    temporal_ricci: -0.20       # Temporal curvature threshold
    
    # Transition threshold
    topological_transition: 0.70  # High transition probability
  
  # Signal generation
  signals:
    min_confidence: 0.50        # Minimum confidence for entry signals
    confidence_boost:
      strong_emergent: 1.5      # Multiplier for strong emergent
      chaotic: 0.5              # Multiplier for chaotic
    
    # Risk multipliers
    risk_multipliers:
      strong_emergent:
        base: 1.0
        confidence_scale: 0.5   # Additional scaling by confidence
      proto_emergent:
        base: 0.7
        confidence_scale: 0.3
      transition: 0.3
      chaotic: 0.3
      post_emergent: 0.2
      
      # Global limits
      min: 0.1
      max: 2.0
  
  # Phase-specific behavior
  phases:
    strong_emergent:
      entry_direction: "temporal_ricci"  # Use temporal_ricci for direction
      exit_urgency: 0.1
      
    proto_emergent:
      entry_scale: 0.5
      exit_urgency: 0.3
      
    post_emergent:
      entry_signal: -0.3         # Mild short bias
      exit_urgency: 0.7
      
    transition:
      entry_signal: 0.0          # No entry during transition
      exit_urgency: "transition_score"  # Use transition score
      
    chaotic:
      entry_signal: 0.0          # Stay out
      exit_urgency: 0.5


# Integration Settings
integration:
  # Data requirements
  data:
    min_history: 1000           # Minimum data points required
    required_columns:
      - close
      - volume
    datetime_index: true
  
  # Performance
  performance:
    parallel_timeframes: true   # Compute timeframes in parallel
    cache_graphs: true          # Cache graph snapshots
    max_cache_size: 100
  
  # Logging
  logging:
    level: "INFO"              # DEBUG, INFO, WARNING, ERROR
    log_signals: true
    log_metrics: true
    save_history: true
    history_path: "outputs/signal_history.csv"


# Backtesting Integration
backtest:
  # Signal-to-trade conversion
  trade_execution:
    entry_signal_threshold: 0.3   # Minimum entry signal to trade
    exit_signal_threshold: 0.6    # Minimum exit signal to close
    
  # Position sizing
  position_sizing:
    base_size: 1.0                # Base position size
    use_risk_multiplier: true     # Use composite risk multiplier
    max_position: 2.0             # Maximum position size
    
  # Risk management
  risk:
    use_confidence_filter: true   # Filter trades by confidence
    min_confidence: 0.5
    stop_loss_multiplier: 1.5     # SL based on risk_multiplier
    take_profit_multiplier: 2.0


# Monitoring & Alerts
monitoring:
  alerts:
    # Phase change alerts
    phase_change:
      enabled: true
      notify_on:
        - STRONG_EMERGENT
        - TRANSITION
    
    # Metric alerts
    metric_thresholds:
      high_transition:
        metric: "topological_transition_score"
        threshold: 0.8
        action: "log"
      
      low_coherence:
        metric: "cross_scale_coherence"
        threshold: 0.3
        action: "log"
      
      extreme_risk:
        metric: "risk_multiplier"
        min: 0.1
        max: 1.8
        action: "alert"
  
  # Dashboard updates
  dashboard:
    update_interval: 60         # Seconds
    metrics_to_display:
      - kuramoto_R
      - consensus_R
      - temporal_ricci
      - topological_transition
      - phase
      - confidence
      - entry_signal
      - exit_signal


# Feature Engineering (for ML integration)
features:
  # Derived features from composite
  engineering:
    kuramoto_features:
      - consensus_R
      - cross_scale_coherence
      - dominant_timeframe_R
      
    ricci_features:
      - static_ricci
      - temporal_ricci
      - topological_transition_score
      - structural_stability
      - edge_persistence
    
    composite_features:
      - phase_encoded          # One-hot encoding of phase
      - confidence
      - entry_signal
      - exit_signal
      - risk_multiplier
  
  # Feature normalization
  normalization:
    method: "robust"           # standard, minmax, robust
    rolling_window: 500


# Experimental Settings
experimental:
  # Advanced Kuramoto
  kuramoto_extensions:
    higher_order_coupling: false     # Triplet interactions
    adaptive_coupling_strength: false # Dynamic coupling
    
  # Advanced Ricci
  ricci_extensions:
    persistent_homology: false       # TDA integration
    ricci_flow: false                # Graph evolution via flow
    
  # Meta-learning
  meta_learning:
    enabled: false
    update_thresholds: false         # Adaptive threshold learning
    strategy_evolution: false        # Genetic programming


# Version & Metadata
metadata:
  version: "1.0.0"
  author: "TradePulse Development"
  description: "Kuramoto-Ricci Composite Indicator Configuration"
  created: "2024-10-06"
  last_modified: "2024-10-06"