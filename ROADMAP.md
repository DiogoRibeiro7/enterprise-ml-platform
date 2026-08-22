# Roadmap

**Author**: Diogo Ribeiro
**ORCID**: [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072)

The goal is a reference implementation of a production ML platform where every
claim in the [README](README.md) is backed by a test. Work is ordered by what
most needs to be true before the next thing can be believed.

## Done

- **Pipeline orchestration.** DAG validation, bounded concurrency, retries and
  a circuit breaker. On failure, scheduling stops, in-flight stages are
  cancelled and awaited, and only the stages that actually completed are
  compensated, in reverse completion order.
- **Feature store.** Redis online store keyed by feature set, version and
  entity. Parquet offline store queried through DuckDB, with point-in-time
  retrieval and persistence across restarts. Identifiers are carried through
  feature engineering untransformed.
- **Model registry.** MLflow-backed versions with `champion` and `challenger`
  aliases. Promotion and rollback move an alias; versions are immutable.
- **Experiment tracking.** Training records params, metrics and the model
  artifact inside an explicit run, in a configured store.
- **Serving.** FastAPI with inference dispatched off the event loop, input
  validation, batch limits, and the serving version in every response.
- **Configuration.** API key, CORS, limits, registry and store locations all
  read from the environment, with a deployment refusing to start if it is
  configured to serve demo models or run unauthenticated.
- **SageMaker deployment.** Model, endpoint configuration and endpoint
  lifecycle, traffic weights, and rollback to a previous endpoint
  configuration. Tested against a stubbed AWS API.
- **A worked end-to-end example.** One dataset carried through features,
  point-in-time training set assembly, a tracked training run, registration,
  promotion, serving and rollback, as a runnable script that needs nothing
  external.
- **CI.** ruff, mypy, the suite on Python 3.11–3.13, a packaging smoke test
  that installs the built wheel and runs the console scripts, a Docker build,
  bandit and a dependency audit.

## Next

1. **Serving metrics that mean something.** Prediction count, latency
   percentiles and error rate per model version, exported and dashboarded, so
   a promotion can be judged rather than assumed.
2. **Drift monitoring wired to the served model.** Drift detection exists but
   nothing feeds it live serving traffic or acts on its output.
3. **A real SageMaker run.** The deployer is verified against the API contract,
   not against AWS. One end-to-end deployment in a sandbox account would close
   the gap between "the calls are right" and "it deploys".
4. **Type annotations for the legacy services.** `pyproject.toml` lists the
   modules still exempt from strict mypy. That list should only shrink.

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
