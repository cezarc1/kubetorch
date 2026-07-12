# Video Context Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all visitor-facing Runhouse video material with concise transcript-derived documentation and remove the page footer credit/copyright block.

**Architecture:** Caption tracks remain temporary research files under `/tmp`. Maintained explanations live in literate example comments or the hand-authored debugging guide, generated tutorial pages are rebuilt from source, and an unrendered YAML record preserves the audit disposition without keeping transcript text or video links.

**Tech Stack:** Python, MyST Markdown, YAML, pytest, Sphinx, yt-dlp

## Global Constraints

- Do not commit raw or cleaned transcript text.
- Do not retain YouTube embeds, links, source markers, Sphinx extension code, or video CSS in published documentation.
- Add only current-fork technical details that are absent from the existing page and code.
- Preserve Runhouse attribution on the homepage, repository README, and maintainer records.
- Preserve the user's unrelated uncommitted `python_client/kubetorch/docs/index.md` edit.
- Run documentation tests with `.venv/bin/python -m pytest --level unit` because the repository defaults to the cluster-oriented `minimal` tier.

---

### Task 1: Define the removal and provenance contracts

**Files:**
- Create: `python_client/tests/docs/test_video_removal.py`
- Modify: `python_client/tests/docs/test_catalog.py`
- Modify: `python_client/tests/docs/test_render_tutorials.py`
- Modify: `python_client/tests/docs/test_sphinx_config.py`
- Test: `python_client/tests/docs/test_video_removal.py`

**Interfaces:**
- Consumes: tutorial catalog entries, `render_literate_source(source, tutorial)`, Sphinx configuration text, maintained docs/example source trees.
- Produces: tests requiring ten audited video records, zero published video references, no renderer-emitted directive, and an empty content footer.

- [ ] **Step 1: Write failing tests**

Add a test that loads `python_client/kubetorch/docs/_data/video_recovery.yaml` and requires exactly ten unique records with this shape:

```yaml
videos:
  - id: CH0mMcR5hZ8
    title: Fault Tolerant Training: Automatically Finding Batch Size for PyTorch Distributed
    source: examples/tutorials/fault_tolerance/batch_size_finding.py
    target: tutorials/fault-tolerance/automatic-batch-size
    disposition: no-new-context
    context: Existing tutorial already explains the progressive batch-size search and OOM recovery loop.
```

Allow only `context-added` and `no-new-context` dispositions. Scan text files under `python_client/kubetorch/docs` and `examples/tutorials` for `youtube.com`, `youtu.be`, `youtube-nocookie.com`, `````{youtube}``, and `::youtube[`; exclude only the unrendered recovery YAML, which contains IDs but no URLs or directives.

Change the renderer test to require that a legacy source marker is discarded and that neither a YouTube directive nor the video ID appears in rendered output. Require every tutorial's `video_id` to be `None`. Require `"footer_content_items": []` in `conf.py`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q --level unit \
  python_client/tests/docs/test_video_removal.py \
  python_client/tests/docs/test_catalog.py \
  python_client/tests/docs/test_render_tutorials.py \
  python_client/tests/docs/test_sphinx_config.py
```

Expected: failures for the missing recovery file, eight catalog video IDs, rendered directives, source URLs/markers, and footer configuration.

---

### Task 2: Recover context and remove video/footer presentation

**Files:**
- Create: `python_client/kubetorch/docs/_data/video_recovery.yaml`
- Modify: `examples/tutorials/reinforcement_learning/verl_training/verl_train.py`
- Modify: `examples/tutorials/transformers_inference/openai_oss_120b.py`
- Modify: `examples/tutorials/vllm_inference/llama.py`
- Modify: `examples/tutorials/batch_inference/simple_deepseek_ocr.py`
- Modify: `python_client/kubetorch/docs/guides/debugging.md`
- Modify: the five other imported example sources containing `::youtube` markers
- Modify: `python_client/kubetorch/docs/_data/catalog.yaml`
- Modify: `scripts/docs/render_tutorials.py`
- Modify: `python_client/kubetorch/docs/conf.py`
- Modify: `python_client/kubetorch/docs/_static/css/kubetorch.css`
- Delete: `python_client/kubetorch/docs/_ext/youtube.py`
- Delete: `python_client/tests/docs/test_youtube.py`

**Interfaces:**
- Consumes: ten temporary English JSON3 caption tracks in `/tmp/kubetorch-video-transcripts` and current fork APIs.
- Produces: source-backed written context, an unrendered audit record, no video rendering machinery, and no author/copyright footer block.

- [ ] **Step 1: Add verified transcript deltas**

Add these maintained explanations:

- VeRL: Kubetorch forms the Ray cluster, synchronizes source, and dispatches `run_grpo`; VeRL still owns the RL algorithm, rollout/training configuration, data preparation, and vLLM integration.
- GPT-OSS: the A100 path uses BF16 with an inferred multi-GPU device map; MXFP4 kernels are an H100/H200 option, and lazy model loading means the first request pays the GPU-load cost after deployment or reload.
- vLLM: the decorated class is a Kubernetes service with an internal HTTP endpoint, while the imported Python proxy can call it from either side of the cluster boundary; independently deployed model services can be reused by composite applications.
- Distributed debugging: use `debug=True` on a call or a trusted `breakpoint()`; limit breakpoints to one rank when one session is desired; copy the complete runtime-printed `kt debug` command because it includes the pod, port, namespace, mode, and sometimes pod IP.
- DeepSeek OCR: `AsyncLLMEngine` continuously batches requests within each replica; fixed min/max replicas suit a finite batch job; total concurrency is per-replica concurrency times replica count; the driver semaphore bounds outstanding local tasks; `kt run` can move the driver into the cluster.

Record the other five videos as `no-new-context` because their existing source already covers the demonstrated batch-size search, membership recovery and weight broadcast, DDP rank wiring/hot sync, independently scaled RAG services, and nested TRL sandbox service.

- [ ] **Step 2: Remove video declarations and rendering**

Remove all eight `video_id` keys from `catalog.yaml`, all nine `::youtube` markers from imported example sources, and the debugging-guide directive. Delete the conditional video block from `render_literate_source`, but retain `YOUTUBE_MARKER` filtering so future recovery imports cannot leak a legacy marker into prose.

Remove `_ext.youtube` from the Sphinx extension list, delete the extension and its dedicated test, and delete the `.kt-video` CSS rules. Add `"footer_content_items": []` to `html_theme_options`.

- [ ] **Step 3: Create the audit record**

Create ten records in `_data/video_recovery.yaml` containing `id`, `title`, `source`, `target`, `disposition`, and one concise `context` sentence. Use `source: null` for the hand-authored debugging guide and `target: null` for the source-only DeepSeek OCR example.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 command. Expected: all focused tests pass.

---

### Task 3: Regenerate and verify the site

**Files:**
- Modify: eight generated tutorial pages formerly containing video directives
- Modify: any additional generated page changed by transcript-derived source prose

**Interfaces:**
- Consumes: catalog and literate source updates from Task 2.
- Produces: checked-in MyST pages matching renderer output and a warning-free local HTML site.

- [ ] **Step 1: Regenerate tutorials**

Run:

```bash
.venv/bin/python scripts/docs/render_tutorials.py --write
.venv/bin/python scripts/docs/render_tutorials.py --check
```

Expected: write mode updates the generated pages and check mode exits successfully with no drift.

- [ ] **Step 2: Run complete validation**

Run:

```bash
.venv/bin/python -m pytest -q --level unit python_client/tests/docs
.venv/bin/python -m compileall -q examples/tutorials
.venv/bin/python -m sphinx -W --keep-going -b html \
  python_client/kubetorch/docs /tmp/kubetorch-docs-oled-preview
```

Expected: all docs tests pass, imported Python compiles, and Sphinx succeeds without warnings.

- [ ] **Step 3: Inspect built HTML**

Search `/tmp/kubetorch-docs-oled-preview` for YouTube domains/directives and the two removed footer strings. Expected: zero matches. Reload the visible browser and sample the debugging, VeRL, vLLM, GPT-OSS, and tutorial index pages.

- [ ] **Step 4: Commit the scoped implementation**

Stage only the recovery record, tests, renderer/config/CSS changes, example-source edits, debugging guide, and regenerated tutorial pages. Do not stage `python_client/kubetorch/docs/index.md`.

```bash
git commit -m "docs: replace videos with recovered context"
```
