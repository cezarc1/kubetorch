# Examples

This fork keeps runnable examples in this repository so they can evolve with the
agent-first batch-run workflow and fork-owned GHCR artifacts.

Current examples include:

- `wetlandbirds_shakedown/`: Visual WetlandBirds shakedown jobs for exercising
  Kubetorch batch-run source, log, note, and artifact capture.
- `qwen3_asr_orin/`: Jetson/Orin Qwen3-ASR profiling and export helpers.
- `tutorials/`: maintained training, inference, distributed, fault-tolerance,
  reinforcement-learning, and orchestration examples recovered from the
  original Runhouse catalog.

The tutorial sources are rendered into the maintained
[Kubetorch documentation](https://cezarc1.github.io/kubetorch/tutorials/).
Their pinned upstream commit and validation status are recorded in
`python_client/kubetorch/docs/_data/catalog.yaml`.
