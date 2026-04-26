# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## RunPod H100-Specific Troubleshooting

These issues were encountered during the H100 benchmark session (Apr 2026) and should save the next agent 30–60 minutes.

### `OSError: [Errno 122] Disk quota exceeded` during model download

**Root cause:** RunPod network storage (`/workspace`) has a ~50 GB practical write limit per operation, even though `df -h` shows 252 TB. Large model downloads (14B+ with 30+ GB of safetensors shards) hit this limit and fail with `Disk quota exceeded`.

**Fix:**
- Delete previous model caches before downloading the next one to stay under the quota:
  ```bash
  rm -rf /workspace/huggingface_cache/hub/models--<previous-model>
  ```
- For the largest model, use `/dev/shm` (RAM-backed tmpfs, 117 GB on this instance) as the HuggingFace cache:
  ```bash
  mkdir -p /dev/shm/huggingface_cache
  HF_HOME=/dev/shm/huggingface_cache python run_benchmark.py config.yaml
  ```

### `hf_xet` / xet storage download failures

**Symptom:** `RuntimeError: Data processing error: File reconstruction error` or `Internal Writer Error: Background writer channel closed` during `snapshot_download`.

**Fix:** Uninstall the xet backend. SGLang falls back to regular HTTP downloads, which are slower but stable:
```bash
uv pip uninstall hf-xet
```

### `ImportError: libnuma.so.1` / `FileNotFoundError: ninja`

Standard SGLang dependencies; install them before the first run:
```bash
apt-get update && apt-get install -y libnuma1 ninja-build
```

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.