# Roadmap

**Author**: Diogo Ribeiro
**ORCID**: [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072)

The goal is a reference implementation of a production ML platform where every
claim in the [README](README.md) is backed by a test. Work is ordered by what
most needs to be true before the next thing can be believed.

Completed work is grouped by the release it went out in, so this document
stays readable as releases accumulate rather than becoming one long list.

## Shipped in [v0.1.0](https://github.com/DiogoRibeiro7/enterprise-ml-platform/releases/tag/v0.1.0)

The first tagged release: 21 commits across fourteen pull requests. Most of
them fixed something that had never worked, and the rest added the tooling
that stops it happening again unnoticed.

- **Pipeline orchestration.** DAG validation, bounded concurrency, retries and
  a circuit breaker. On failure, scheduling stops, in-flight stages are
  cancelled and awaited, and only the stages that actually completed are
  compensated, in reverse completion order.
- **Feature store.** Redis online store keyed by feature set, version and
  entity. Parquet offline store queried through DuckDB, with point-in-time
  retrieval and persistence across restarts. Identifiers are carried through
  feature engineering untransformed rather than target-encoded into the label.
- **Model registry.** MLflow-backed versions with `champion` and `challenger`
  aliases. Promotion and rollback move an alias; versions are immutable.
- **Experiment tracking.** Training records params, metrics and the model
  artifact inside an explicit run, in a configured store, with a configured
  artifact location.
- **Serving.** FastAPI with inference dispatched off the event loop, input
  validation, batch limits, and the serving version in every response.
- **Configuration.** API key, CORS, limits, registry and store locations all
  read from the environment, with a deployment refusing to start if it is
  configured to serve demo models or run unauthenticated.
- **SageMaker deployment.** Model, endpoint configuration and endpoint
  lifecycle, traffic weights, and rollback to a previous endpoint
  configuration. Tested against a stubbed AWS API.
- **A package that installs.** Metadata that passes validation, console scripts
  that run, and a core dependency list cut from roughly 110 packages to 14,
  with each subsystem behind an extra.
- **CI.** ruff, mypy, the suite on Python 3.11–3.13, a packaging smoke test
  that installs the built wheel and runs the console scripts, a Docker build,
  bandit, a credential scan and a dependency audit.

## Shipped in [v0.2.0](https://github.com/DiogoRibeiro7/enterprise-ml-platform/releases/tag/v0.2.0)

- **A worked end-to-end example.** One dataset carried through features,
  point-in-time training set assembly, a tracked training run, registration,
  promotion, serving and rollback, as a runnable script that needs nothing
  external. See
  [examples/fraud_detection/end_to_end.py](examples/fraud_detection/end_to_end.py).
- **Evaluation metrics that survive class imbalance.** Classification reports
  precision, recall, F1 and ROC AUC alongside accuracy, plus the majority class
  rate so accuracy can be read against the baseline it has to beat. Accuracy
  alone cannot rank candidates on an imbalanced problem, which is the judgement
  a registry exists to support.
- **Citation and release metadata that match the README.** `.zenodo.json` and
  `CITATION.cff` described a platform the project had stopped claiming to be.
  A test now holds them, and the package metadata, to each other.

## Since v0.2.0

- **Version-aware serving telemetry.** The prediction API records request
  outcomes, successful rows and latency against the exact model version that
  served them. Prometheus scrapes the real API route, and the Grafana dashboard
  computes per-version throughput, error rate and p95 successful latency.
- **Drift monitoring wired to serving.** Tracked runs store summary-only input
  baselines. Validated serving rows feed bounded windows isolated by model and
  immutable version; the API exposes readiness and scores, Prometheus loads a
  sustained drift alert, and Grafana shows version-scoped drift state.
- **One-command local observability.** Compose starts the API, Redis, MLflow,
  Prometheus and Grafana on loopback-only ports. Grafana provisions the
  Prometheus source plus model and platform-health dashboards, while named
  volumes retain every stateful service across restarts. CI validates the
  Compose model and builds both project-owned images.
- **A strictly typed feature-engineering service.** The full service package is
  checked by mypy instead of hidden behind the legacy exemption. Its shared
  transformer contract now models fluent `fit()` correctly, refitting replaces
  learned categorical state, and shutdown supports Dask's synchronous client.

## Next

In the order each becomes worth doing. Which release they land in is decided
when they land, not here: a version number written down in advance is a claim
that goes stale on its own.

1. **A real SageMaker run.** The deployer is verified against the API contract,
   not against AWS. One end-to-end deployment in a sandbox account would close
   the gap between "the calls are right" and "it deploys".
2. **Type annotations for the legacy services.** `pyproject.toml` lists the
   modules still exempt from strict mypy. That list should only shrink.
3. **Lint the examples.** CI checks `src` and `tests`. The example directory is
   excluded, which is why 27 legacy files there have never been formatted.

## Not planned

- **Multi-cloud deployment.** The GCP and Azure deployers were removed because
  they returned plausible endpoint URLs without calling anything. One provider
  that genuinely works is worth more than three that report success.
- **Model export to ONNX and similar.** Currently raises rather than returning
  a path to a file it never wrote. It will stay that way until there is a
  reason to implement it properly.
- **Kubernetes and Terraform as supported paths.** The manifests and modules
  are in the repository, but nothing applies or tests them, so they are
  illustrative until CI can prove otherwise.
- **Publishing to a package index.** The name has never been reserved and
  publishing is irreversible. Installation is from the tagged source.
