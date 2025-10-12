# Feature store retention and validation

The online feature stores back real-time serving with two persistence
backends:

* **Redis** for low-latency access.
* **SQLite** for lightweight, embedded deployments.

Both backends share the same retention primitives implemented in
`core.data.feature_store`:

* `RetentionPolicy.ttl` expires rows that fall outside of the configured
  time-to-live window.
* `RetentionPolicy.max_versions` keeps only the latest *N* rows per
  entity identifier.

The Redis integration now propagates TTL values down to the key-value
store whenever possible. Clients exposing Redis compatible `setex`
semantics inherit automatic key expiry while the in-memory/testing
clients fall back to the existing retention pruning logic.

Offline Delta Lake or Apache Iceberg tables remain the source of truth.
`OfflineStoreValidator` periodically compares the offline snapshot with
the materialised online payloads and raises
`FeatureStoreIntegrityError` whenever mismatches are detected. This
allows production runs to gate deployments on validation checks while
still benefiting from hot storage TTL enforcement.
