> Owner priority, 19 October 2026: natural cited document chat and bounded paper overviews are now implemented ahead of the remaining Phase 4 work. Explicit context coverage does not close exhaustive-summary or model-quality gates.

> Current generation routing supersedes the earlier free-pool/local runtime choices: see [BYOK integration](openrouter-byok.md). All active generation paths use the shared OpenAI SDK policy; nonconfidential development limits remain.

# RAGBase Local Enterprise — Final Execution Plan

Version: 6.0 — Groq-first grounded document chat

Updated: 18 October 2026

Governing specification: [CONSTITUTION.md](../CONSTITUTION.md), version 6.1.0

Project: [ElMolla10/ragbase](https://github.com/ElMolla10/ragbase)

Status: revised planning baseline. Implementation is paused for this revision. No model weights were downloaded, inference performed, or live deployment changed by this planning task. Earlier implementation work is preserved, including unfinished changes that have not passed integration tests.

> Execution resumed on 18 October 2026 by the project owner. The planning-time status descriptions below are historical; see the [current implementation record](governance/phase3-completion.md) for running features, tests and remaining gates.

> **Historical v4.1 amendment (superseded by current BYOK routing):** Gemma remains first, with five reviewed free fallbacks for endpoint/model failures; this does not bypass the account-wide free quota. The owner selected free Gemma, with JSON mode plus strict application schema/evidence validation and a zero-price routing ceiling. Standard provider privacy is separately authorized only for nonconfidential development/testing; it is not a ZDR/company-confidential approval.  Local generation remains optional and disabled by default in dev-lite. The owner has authorized an alternative OpenRouter mode; [setup](openrouter.md) and [implementation evidence](governance/openrouter-option.md) supersede historical local-only statements for this explicit exception. DeepSeek Harness is a coding-agent application, not the inference gateway used by RAGBase. No harness tools, filesystem or web search are enabled by this option. All seven execution phases remain.

## 1. The correction and its consequences

**Confirmed by the project owner:** the company server has **64 GB of ordinary system RAM, no dedicated GPU, and no dedicated VRAM**. The development laptop has 16 GB system RAM. Server CPU model, physical cores, memory bandwidth, instruction sets, operating system, virtualization limits, and storage are unknown.

The old single-64-GB-GPU assumption is invalid everywhere in this plan. We will use CPU-compatible quantized artifacts, reserve RAM for the whole application, serialize expensive work, and measure practical response time before selecting a production model. Fitting weights in memory does not establish useful inference speed.

The default product runs fully locally: document Q&A, complete-document summaries, comparisons, structured extraction, spreadsheet calculations, charts, and analysis with Arabic, Egyptian questions, mixed-language use, precise citations, and reproducible calculations. An explicitly selected OpenRouter generation mode is now authorized as an alternative to local inference. It uses `google/gemma-4-31b-it:free` through a credential-isolating relay, sends bounded authorized evidence and questions externally, and needs no local generation weights. Other document services remain local. There is no automatic hosted fallback. See [remote setup and boundaries](openrouter.md).

**Provisional decisions:** llama.cpp is the primary server-runtime candidate; Qwen3.5-9B Q4_K_M is the first practical baseline. Compare Qwen3.5-35B-A3B as the main larger challenger and dense Qwen3.5-27B as a quality-first challenger. Keep Qwen3.5-4B for lighter development/fallback. Include the already-downloaded Gemma 4 12B as an additional independent-family comparison through the existing local Ollama installation.

Start with **one active generation**, queued users, and small bounded prompts. There is no committed 64K/128K production-context target and no promised tokens-per-second, latency, or concurrent-generation capacity. Hardware discovery is the first execution step when work resumes; unknown CPU specifications do not prevent planning or independent application development.

## 2. Evidence labels and current repository state

Use these labels throughout execution:

- **Confirmed:** supplied hardware facts and read-only local/repository observations.
- **Documented:** capabilities or artifacts described by primary sources; not necessarily tested by us.
- **Estimate:** a planning allowance or arithmetic calculation, not measured consumption.
- **Provisional recommendation:** a choice to test, which can change after measurement.
- **Pending:** no result exists yet; do not substitute mock scores or mark it passed.

On 18 October 2026, remote `master` still points to `2f0a0204a14f6659b97a061168ccb0bdd8c061f7`, the original reviewed snapshot. The local feature branch contains foundation commit `94fe1c2f939fba4904da14eb540136ca05de04ef`. That foundation added governance, policy assembly, fake/HTTP streaming contracts, candidate metadata, and a lightweight entry point. Its recorded 63 baseline tests and 55 foundation tests are historical evidence for that checkpoint only.

There are also uncommitted database, storage, identity, migration, proxy, and test changes from the interrupted implementation batch. Dependency-download failures prevented that batch from reaching integration acceptance. They must be reviewed and tested before reuse. They are not a completed authorization/storage layer. This revision does not discard them or deploy them.

The legacy demo still has hosted-generation defaults, heavy embedding startup coupling, startup schema creation, public browser assets, and missing enterprise controls. Its existing runnable demo is not the revised enterprise product. Inspection covered configuration, LLM integration, ingestion/retrieval, routes, protection, database startup, Compose, dependencies, and tests.

## 3. Architecture retained and simplified for CPU

```text
Company browser: internally hosted assets, Arabic RTL and source viewers
       |
Internal HTTPS reverse proxy — only user-facing component
       |
Frontend / FastAPI / identity / workspace authorization / policy assembly
       |-- PostgreSQL + pgvector: permissions, versions, evidence, jobs, history
       |-- Durable files: originals, extraction, previews, Parquet, artifacts
       |-- PostgreSQL durable job queue + CPU/RAM admission coordinator
       |       |-- one admitted extraction/OCR or analysis workload
       |       |-- bounded embedding/reranking batches
       |-- Authorized retrieval / typed analysis / coverage workflows
       |                           |
       |                       ModelGateway
       |                           |
       |             Separate CPU generation service, one active slot
       |             GGUF + tokenizer/template + optional image components
       |
Restricted local audit, metrics, backups and administration
```

The model service has no database credentials, original-document mounts, host Docker socket, or general company-filesystem access. Only the application sends authorized evidence and controlled local image inputs. Keep generation, embedding, reranking, OCR, storage, and execution interfaces independently replaceable.

Retain FastAPI, PostgreSQL/pgvector, Alembic, durable storage, and the document-versus-structured-analysis split. Start with a **PostgreSQL-backed durable job queue**, transactional job creation, leases/retries, and reconciliation. A broker is optional later if measured operational needs justify it. This replaces the earlier default Celery/Valkey requirement and reduces idle services; it does not remove durability or job recovery. If an external broker is introduced, use an outbox/reconciler and retain PostgreSQL as job truth.

Use bounded CPU workers and isolated analysis processes. Separate dependency groups/images by API, migration, parsing/OCR, analysis, and inference responsibility; do not keep multiple heavyweight worker copies or inference engines resident merely because images are separate. Start CPU tasks serially and add overlap only after end-to-end measurements.

## 4. Inference runtime decision and compatibility checks

### Provisional primary: llama.cpp CPU server

Reasoning: explicit local GGUF paths, reproducible artifact selection, visible CPU thread/context controls, and a small serving boundary fit this server's constraints. This is an engineering preference, not a benchmark win over Ollama.

Current upstream contains Qwen3.5 dense and MoE implementations, Qwen3.5 tokenizer support, and CPU backend operations for its hybrid architecture. This establishes source-level availability, not successful loading of our precise file on an unknown CPU. Research snapshot: llama.cpp commit `547208dca250ceb3e4c8615598d043f2b602a527`. Select a tested release/build from a recorded revision; do not deploy a moving branch. [Dense architecture](https://github.com/ggml-org/llama.cpp/blob/547208dca250ceb3e4c8615598d043f2b602a527/src/models/qwen35.cpp), [MoE architecture](https://github.com/ggml-org/llama.cpp/blob/547208dca250ceb3e4c8615598d043f2b602a527/src/models/qwen35moe.cpp), [CPU operation support](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md), [tokenizer implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-vocab.cpp).

llama.cpp documents CPU builds and an optional CPU BLAS backend. Build for the actual ISA, or produce a portable baseline on approved staging; never copy an aggressively native laptop build to a different CPU without checking compatibility. Benchmark BLAS for prompt processing rather than assuming it improves generation. No accelerator toolchain is a prerequisite. [Build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

### Ollama: supported alternative and existing laptop path

Ollama's Qwen3.5 catalogue lists the four shortlisted sizes with text/image support, and the explicit `qwen3.5:9b-q4_K_M` tag exists. Its APIs document streaming, tools, structured output and image inputs. These API features still need exact-model/runtime adapter tests. Select the llama.cpp/GGUF variant rather than assuming every catalogue variant uses the same backend. Pin the full manifest/blob digests; a mutable tag alone is insufficient. [Qwen catalogue](https://ollama.com/library/qwen3.5), [specific 9B tag](https://ollama.com/library/qwen3.5:9b-q4_K_M), [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility), [structured outputs](https://docs.ollama.com/capabilities/structured-outputs), [tools](https://docs.ollama.com/capabilities/tool-calling), [vision](https://docs.ollama.com/capabilities/vision).

Ollama is already installed locally at version **0.40.3**, and `gemma4:12b` is downloaded. Reuse that asset for later development experiments; do not replace or reload it during this planning revision. Production should select one measured runtime, not maintain both without a demonstrated need.

If Ollama wins the operational/quality comparison, keep its engine behavior in an adapter. Configure local-only mode, one loaded generation model, one parallel request, a small explicit context and bounded queue. Its documentation warns that parallelism increases memory. Disable cloud features with `OLLAMA_NO_CLOUD=1`, verify logs, and still enforce egress restrictions. Keep model management and pull endpoints inaccessible to ordinary users. [Ollama FAQ](https://docs.ollama.com/faq).

### vLLM: reconsidered, not the default

vLLM has an official CPU backend; it is incorrect to call it GPU-only. Its ISA, dtype, supported architecture, quantization and memory requirements need separate verification for the actual CPU. It is not selected for this pilot because we have no measured advantage for the intended GGUF/single-request workload and would add another serving stack. Reconsider only after hardware discovery and a concrete benchmark need. Do not carry over the previous GPU commands or FP8 configuration. [vLLM CPU documentation](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/).

### Required compatibility matrix for each pinned configuration

| Component | Documented evidence | Required local verification |
|---|---|---|
| Architecture | Qwen3.5 dense/MoE source implementations; Ollama catalogue | Exact artifact loads on the chosen CPU build; all required kernels work |
| Quantization | Publisher lists exact Q4_K_M GGUF files | File checksum, tensor metadata, memory peak and quality at that quantization |
| Tokenizer | Runtime Qwen3.5 support and GGUF metadata path | Match base tokenizer on Arabic/English, combining marks, digits, tool delimiters and full chat-template accounting |
| Images | Matching projector files exist; runtime multimodal APIs are documented | Matching model/projector/template, actual CPU encoding and answer quality; exact build can fail and must be reported |
| Streaming | Server/API support documented | Fragmented UTF-8/SSE, errors, usage, termination, disconnect and cancellation |
| Structured output | Runtime schema-constrained output APIs | Our schemas, refusal/truncation, malformed JSON, bounded retries and independent validation |
| Tool calls | Runtime tool support and Qwen template source | Exact parser/template, correct tool choice/arguments, rejection of unauthorized actions |
| Reasoning | Models/templates expose differing controls | Bound actual token/time cost and keep internal channels out of answers/history |

Tool-call formatting depends on the model template/parser; general runtime support is not proof for every combination. Keep a tested schema-constrained planning path available where appropriate, with application validation. If neither path passes, disable dependent model-planned analysis with a clear explanation instead of executing free-form instructions. [Function-calling guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md), [Qwen template](https://github.com/ggml-org/llama.cpp/blob/master/models/templates/Qwen3.5-4B.jinja), [grammar/schema documentation](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md).

## 5. Model comparison and verified proposed artifacts

The four Qwens are the main CPU comparison set. Gemma is an additional already-available comparator. No model is selected for production. Mistral Small 3.2 is removed from the initial comparison budget, not declared incapable of CPU use.

All Qwen paths below are **Unsloth-published conversions of Qwen checkpoints**, not official Qwen GGUF releases. Publisher metadata identifies the base models and Apache-2.0 license; verify license files, conversion provenance, revision and checksums before acquisition. Do not confuse unrelated modified/uncensored variants with these artifacts. Matching `mmproj-F16.gguf` files are proposed for optional vision; they are not needed for ordinary text-only serving.

| Role / model | Exact proposed artifact | Verified disk payload, decimal GB | Estimated text-process RAM at 4K–8K, one slot | Capabilities and unresolved questions |
|---|---|---|---|---|
| First practical baseline: Qwen3.5-9B | `unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf` | 5.681 weights; 0.918 F16 projector | 9–16 GiB | Text; optional images with matching projector; Arabic/tools/CPU speed untested |
| Main larger challenger: Qwen3.5-35B-A3B | `unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf` | 22.016 weights; 0.899 projector | 28–38 GiB | MoE; full weight footprint matters; fit under our experimental cap and response speed both unproven |
| Quality-first challenger: Qwen3.5-27B | `unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf` | 16.741 weights; 0.928 projector | 22–30 GiB | Dense; possible quality benefit is a hypothesis to test, not an established advantage |
| Lightweight development/fallback: Qwen3.5-4B | `unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf` | 2.741 weights; 0.672 projector | 6–10 GiB | Smaller Arabic/analysis capability must be evaluated; optional laptop use with other heavy services stopped |
| Existing independent comparator: Gemma 4 12B | Local Ollama `gemma4:12b`, GGUF/Q4_K_M, digest recorded below | 8.022 local bundle; not a main-weight-only measurement | 12–20 GiB | Local metadata lists text, vision, tools, thinking and audio; no performance/quality run done; audio is outside scope |

**The RAM column is an engineering allowance, not a measurement, guaranteed fit, or loading-peak bound.** It includes weights plus a provisional cache/buffer allowance; tokenizer/template, runtime version, projector/draft settings and context can change consumption. The 35B upper estimate exceeds some safe application profiles and may be inadmissible. Benchmark loading separately and refuse a configuration whose measured peak exceeds its budget.

Qwen file metadata and image components were inspected without downloading weights. [9B files](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/3885219b6810b007914f3a7950a8d1b469d598a5), [35B files](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/tree/bc014a17be43adabd7066b7a86075ff935c6a4e2), [27B files](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/tree/3221f178a6b842d04f1fb42f1c413534adcc0a6a), [4B files](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/e87f176479d0855a907a41277aca2f8ee7a09523).

### Pin record for later approved acquisition

Sizes are exact bytes from publisher metadata. SHA-256 values are reported LFS hashes, **not independently verified local file hashes**. Acquisition must compute and compare actual bytes. The F16 projector filename is identical across repositories but the files are model-specific and cannot be interchanged.

| Model | Repository revision | Weights bytes / SHA-256 | `mmproj-F16.gguf` bytes / SHA-256 |
|---|---|---|---|
| 9B | `3885219b6810b007914f3a7950a8d1b469d598a5` | 5680522464 / `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` | 918166080 / `f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f` |
| 35B-A3B | `bc014a17be43adabd7066b7a86075ff935c6a4e2` | 22016023168 / `3b46d1066bc91cc2d613e3bc22ce691dd77e6f0d33c9060690d24ce6de494375` | 899283648 / `a516ab92e8240da4734d68352bdfba84c16e830ee40010b8fac80d69c77272ff` |
| 27B | `3221f178a6b842d04f1fb42f1c413534adcc0a6a` | 16740812704 / `84b5f7f112156d63836a01a69dc3f11a6ba63b10a23b8ca7a7efaf52d5a2d806` | 927607040 / `458bc46d8f275866fde5d88c9c554d9d462a6e8e3a028090d9850e17ab6a1217` |
| 4B | `e87f176479d0855a907a41277aca2f8ee7a09523` | 2740937888 / `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` | 672423616 / `cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864` |

### Existing Gemma asset

Read-only local inspection (`ollama list`, `ollama show`, `/api/tags`) found:

- Tag: `gemma4:12b`; full manifest digest `6114515d63c17436a7c0417d82820ac65ad643e2806c5a3c89cb62846436ed0b`.
- GGUF, `gemma4` architecture, 11.9B parameters, Q4_K_M; stored bundle 8,021,618,941 bytes; runner metadata `llamacpp`.
- Advertised context 262,144; minimum runtime metadata 0.30.9; installed Ollama 0.40.3. Version comparison is not an inference test.
- Local metadata includes a 52.38M projector; catalogue additionally identifies an assistant draft model. Existing parameters enable thinking by default and `draft_num_predict=3`.

Count all active components and their caches when benchmarking; do not silently compare a draft-enabled Gemma run with a differently configured Qwen. Use a short text-only baseline and explicitly test thinking/draft settings later. Preserve its current files and configuration now. The catalogue lists Apache-2.0; archive the license and trace the precise conversion lineage before company deployment. A local manifest digest is not the SHA-256 of an individual weight file. [Matching Ollama catalogue entry](https://ollama.com/library/gemma4:12b), [Google model overview](https://ai.google.dev/gemma/docs/core).

The existing Ollama runner is the first route for testing this exact asset. Standalone llama.cpp compatibility of its specific unified 12B bundle, draft and image components remains a separate gate; support for other Gemma 4 sizes is insufficient proof. Re-export/import or a different GGUF must be a separately pinned configuration, not an invisible model change.

### Quantization and model-size tradeoffs

Q4_K_M is a starting point for memory/quality comparison, not proof of acceptable Arabic. Low-bit conversion can change factual recall, negation, table interpretation, structured outputs and refusal. If the 9B baseline has enough headroom but insufficient quality, compare a pinned higher-bit 9B artifact before assuming a larger model is necessary. More bits can increase memory traffic; more parameters can increase compute. Neither direction guarantees a better user experience.

The 35B MoE can reduce activated computation relative to dense models, but all quantized expert weights, shared layers, embeddings and runtime state still count toward RAM and memory traffic. Do not size it as a 3B model or predict its speed from activated parameters. Compare fixed tasks, prompt lengths and generation limits on the actual CPU.

## 6. Whole-server RAM budget

The server is described as 64 GB RAM; actual OS-visible bytes must be measured. If that means 64 billion bytes, it is about 59.6 GiB. Use a conservative **59 GiB planning envelope** until inventory confirms more usable memory. Reduce it further for hypervisor limits or other company workloads. GB in download metadata is decimal; GiB in this budget is binary.

### Initial 9B application profile — estimates and proposed caps

| Component / reserve | GiB | Included workload and control |
|---|---:|---|
| OS and operational reserve | 8 | Kernel, system services, monitoring and ordinary overhead; verify other host workloads separately |
| Generation service | 24 | All charged weight pages, context/recurrent state, runtime buffers, tokenizer and any enabled image components; not 24 GiB of weights |
| PostgreSQL/pgvector | 5 | Shared buffers, connections, query/index work and charged cache; bounded connections and maintenance |
| API, reverse proxy and light services | 2 | Upload buffers, history, source metadata and delivery limits |
| Durable-job coordination / worker supervisor | 1 | PostgreSQL scheduling client and small supervisor; no extra resident broker assumed |
| Embedding/reranking allowance | 3 | One bounded worker; load larger components on demand or evaluate sequentially |
| Shared heavy-job allowance | 6 | At most one admitted extraction/OCR OR spreadsheet/report process, not 6 GiB for each |
| Unallocated peak/uncertainty guard | 10 | Never spend automatically; protects against loading peaks, reclaim pressure and imperfect estimates |
| **Total planning envelope** | **59** | Service allowances total 41 GiB, plus 18 GiB reserve/guard |

The generation allowance deliberately exceeds the expected short-context 9B footprint. It is a ceiling, not a target to fill. Optional image encoding must fit within that generation allowance; it does not receive a hidden additional allocation. During image work, other heavy work is paused, and the 6 GiB heavy-job allowance is not automatically reassigned to a second vision process.

Candidate 27B/35B experiments need a different **explicit operator-selected** profile. Example application experiment: generation 32 + database 5 + API 2 + coordination 1 + embeddings 3 + OS 8 + guard 8 = **59 GiB**, with OCR/analysis disabled. If a 35B configuration cannot stay within 32 GiB, do not load it into that profile. A fixed-evidence standalone experiment may use generation 36, database 5, API 2, coordination 1, OS 8 and guard 7 = 59 GiB with embedding/heavy workers unloaded; it does not qualify full end-to-end service capacity. A model needing still more memory fails this envelope rather than borrowing OS headroom.

### What memory measurements mean

- **Disk size:** bytes in GGUF, projector and other files. This is not total process memory.
- **Mapped address space:** a runtime can map a model file without every page being resident. Virtual size is not RAM consumption.
- **Resident/charged memory:** includes touched model pages and private allocations. Mapped file pages may be reported as both process RSS and filesystem cache; do not double-count the same physical pages. Inspect cgroup charge and PSS where available.
- **Loading peak:** conversion, repacking, initialization, warmup, template/image/draft setup and temporary buffers can exceed steady state. Record cold and warm starts separately.
- **Total application peak:** database/query memory, OCR, Python/Arrow buffers, uploads and analysis can overlap. A quiet standalone model measurement is insufficient.

Do not use a generic dense-transformer KV formula as an exact forecast for Qwen3.5's hybrid recurrent/attention implementation. Record the selected runtime's allocations and actual context tests. A mapped MoE file is not a free substitute for resident working memory; major page faults can destroy responsiveness.

### Enforcement and spill

Use Linux cgroup v2/verified container limits for total process-group memory and swap, plus a host admission policy. Monitor `memory.current`, `memory.peak`, `memory.events`, host available memory, major faults, swap activity, CPU pressure and disk use. A limit can kill a process; it does not guarantee that the request succeeds. Admission checks should prevent unsafe starts, while a last-resort kill protects the host and is recorded as a failed job/configuration. [Kernel cgroup documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html).

For analytics, begin with a 4 GiB DuckDB buffer-manager limit inside a 6 GiB whole-process allowance, one thread, approved scratch storage and a bounded spill quota (for example 20 GiB as an initial policy, subject to available disk). Arrow/Python and some operators can allocate outside DuckDB's buffer manager; never treat its `memory_limit` as a process limit. Use chunked workbook/table ingestion, row/column/output limits, and bounded partitions; not every operation can spill. Stop and report oversized jobs instead of letting them consume the server. [DuckDB memory management](https://duckdb.org/2024/07/09/memory-management), [limits](https://duckdb.org/docs/lts/operations_manual/limits).

Configure PostgreSQL conservatively: a small connection pool, bounded `work_mem`, modest shared buffers and serialized index/maintenance work. Per-operation and per-worker allocations can multiply, so default percentages of total host RAM are inappropriate for this shared server. Freeze exact settings after discovery and workload tests. [PostgreSQL resource settings](https://www.postgresql.org/docs/16/runtime-config-resource.html).

Swap is emergency protection, not planned model capacity or a performance solution. Do not solve weight shortages by paging indefinitely. Disk-spilled analytical intermediates are different from swapping model working memory. Reserve scratch free space and stop admission before disk exhaustion. Size persistent storage from originals + extraction + previews + datasets + indexes + outputs + model/runtime versions + scratch + growth, with separate backups; do not infer storage capacity from RAM.

## 7. CPU scheduling and queue policy

Initial operating policy, subject to benchmark refinement:

1. One active generation service request globally, including visual generation. Users may be simultaneously connected but wait in a bounded application queue.
2. Start with at most one queued interactive request per user and eight queued interactive requests overall. These are admission policies, not throughput promises. Return a clear busy/queued state, support cancellation, and expire abandoned work.
3. Interactive answers take priority over bulk summaries, comparison reductions, vision jobs and ingestion. Break long workflows into durable section/page tasks so an interactive task can run between them.
4. Do not overlap heavy OCR, full indexing, large reranking batches or workbook analysis with latency-sensitive generation initially. One heavy task may run while generation is idle. Light metadata/API/database operations continue.
5. Use one server-wide lease/admission coordinator shared by all workers; separate per-process semaphores are insufficient. A lease records resource class, owner, heartbeat and expiry. Recovery must not accidentally double-admit a still-running process.
6. After discovering physical cores, test generation thread counts that leave capacity for the API/OS. If at least four physical cores exist, initially reserve roughly two; on smaller machines start more conservatively. Tune prompt and generation threads separately and measure memory-bandwidth/NUMA effects. More threads are not automatically faster.
7. Limit OCR, BLAS, OpenMP and analysis thread pools per admitted job to avoid hidden nested parallelism. Do not let every library use all logical CPUs. Exact environment options must match pinned packages.
8. Choose memory caps before launching a model/job. The application never receives a Docker socket or host privilege to enlarge its own limits. Operator profile changes require draining/revalidation.

Cold model loading, index maintenance, backups, restore drills and large experiments are scheduled activities. Queue retries must be idempotent and bounded; do not restart an OOMing large-model job forever. Keep one production generation model resident at a time. Do not add speculative/draft models or a second vision model without measuring their combined cost.

## 8. Context and large-document strategy

Published native context and a configured server limit are different. The Qwen family documents very large native windows, and local Gemma metadata reports 262,144 tokens. Neither establishes an efficient or accurate deployment context on this CPU. [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/README.md).

| Stage | Total context setting | Purpose / promotion condition |
|---|---:|---|
| Smoke | 4K | Tokenizer, template, streaming, one short evidence task, memory observation |
| Baseline | 8K | First interactive benchmark candidate, not an SLA or guaranteed capacity |
| Expansion | 16K then 32K | Only after earlier stages meet quality, memory and agreed responsiveness requirements |
| Optional research | 64K, then 128K | Test only when justified and safely admissible; not pilot release requirements |
| Beyond | Approximately 256K | No initial CPU deployment commitment; advertised capacity alone is not a reason to enable it |

Reserve generated-output capacity within every total window, including supported reasoning tokens. Start benchmark outputs around 256–512 tokens; longer answers/reports use bounded steps. Count system instructions, history, evidence, tool schemas/results and images where applicable using the actual tokenizer/template. Record actual counts. Do not estimate Arabic by English word ratios or silently truncate necessary evidence.

Use retrieval for targeted questions. Use section enumeration, hierarchical summaries, durable coverage records, and reductions for complete summaries/comparisons. Exhaustive extraction must traverse all required sections/versions and disclose failures. Keep tables with headers; return bounded aggregates and relevant rows instead of dumping an entire workbook into a prompt. Progressively retrieve/expand relevant evidence rather than fill the window by default.

Long context must earn its place through position-sensitive Arabic/English tests and complete-workflow timings. Disable or explicitly handle context shifting/truncation; a fluent answer after dropping essential source material is not a successful result.

## 9. Ingestion, OCR, images and source fidelity

Supported product scope remains PDF, DOCX, XLSX, PPTX, CSV, TXT and Markdown, delivered incrementally. Reject macro-enabled/encrypted files initially; add legacy Office conversion and password/recalculation handling only through separately isolated, tested paths.

Upload -> content/type/expansion limits -> quarantine/validation -> hash/version and durable original -> transactional job -> extraction/OCR -> normalized elements/tables -> validation -> indexing -> ready with warnings. Expose queued, extracting, needs_review, indexing, ready and failed states. Retrying must not duplicate evidence; a partially failed version must not publish inconsistent results.

Native text/table extraction is the first path. Use a light PDF parser for good text PDFs; evaluate Docling's CPU layout/table path only on pages/formats that need it. Configure CPU explicitly, bound threads/page batches, and supply offline artifact directories; never allow first-use downloads. [Docling advanced options](https://docling-project.github.io/docling/usage/advanced_options/), [pipeline options](https://docling-project.github.io/docling/reference/pipeline_options/).

For scans, test local Tesseract Arabic/English packs as a CPU baseline, including the speed/accuracy choices among compatible traineddata sets. Do not equate language-pack availability with acceptable Arabic OCR. Evaluate reading order, mixed RTL/LTR, negation, dates, Arabic/Western digits, decimal separators, small text and table alignment. Record uncertain numbers and reconcile before calculations. Retain original page images and locations. [Tesseract traineddata documentation](https://tesseract-ocr.github.io/tessdoc/Data-Files.html).

Word preserves headings, paragraphs, lists, tables and explicit policies for revisions, headers/footers, footnotes/comments. PowerPoint preserves slide/shape locations, notes and previews; chart data is extracted where supported, otherwise clearly labeled visual interpretation. XLSX uses a dedicated adapter for cell types, formulas versus cached values, named/table ranges, units, leading-zero IDs, hidden-region inclusion, dates and provenance. CSV requires explicit encoding/delimiter/type checks. Hidden data is never a permission boundary.

**Image support is retained as a separately qualified capability.** The model/projector pair and runtime must pass CPU image tests before enablement. Use on-demand queued inspection of selected authorized pages/regions, initially one image at a time with a tested pixel/token budget. Crop regions carefully; disclose downsampling/unreadability. Do not send every scanned page through a VLM or claim that image support makes OCR unnecessary.

Vision consumes the same single inference slot and is scheduled around interactive text work. If its quality, RAM or response time is unacceptable, retain OCR/native extraction and visual previews, explicitly report visual interpretation unavailable/pending, and document the unmet capability before pilot scope approval. Do not silently claim chart/diagram understanding. No remote image fetching from document-provided URLs. llama.cpp requires a matching local projector for the multimodal path; explicitly keep it on CPU. [Multimodal documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).

## 10. Retrieval, analysis and provenance

Retain hybrid authorized retrieval with dense and indexed lexical candidates, deduplication, bounded reranking and parent-section expansion. Evaluate Arabic normalization separately from displayed source text and preserve identifiers. Start with the existing E5 baseline; benchmark BGE-M3 and the multilingual reranker under CPU limits rather than automatically replacing them. Reduce candidate batches/reranking depth if latency requires it, then measure recall consequences. No generation-model change requires re-embedding; an embedding change requires its own versioned index and validation.

Begin structure-aware chunk experiments around 400–800 tokens, with table headers and source locations preserved. Candidate counts and retained evidence counts are tuning variables, not copied acceptance requirements. Keep exact supporting-element gold labels and realistic distractors; top-k exceeding tiny corpus size is not a useful ranking evaluation.

Each evidence item includes workspace, document, immutable version/hash, element ID and location: PDF page/box, Word heading/paragraph/cell, slide/shape, or workbook sheet/table/range. Derived datasets and charts map to original inputs and preserve filters, operations, units, exclusions, row counts and execution time.

For analysis, start with schema-validated read-only typed operations implemented by trusted code. DuckDB operates on authorized job-local snapshots, never the application database. Disable external access, extension auto-loading, uncontrolled file/network operations and setting changes; enforce OS limits. Use fixed/decimal monetary precision, check duplicates/joins/subtotals/units/currencies and stale formulas, and ask about material ambiguity. openpyxl preserves formula/cache information but is not a formula evaluator.

The model explains computed results; it does not invent arithmetic. Combine policy retrieval and spreadsheet calculations through their separate evidence paths. Correlation does not establish cause, and forecasts require assumptions. Generated SQL is deferred until a parser/AST allowlist and sandbox tests exist; arbitrary generated Python is outside the initial release.

## 11. Security, constitution, UX and operations

All 55 constitutional rule IDs remain tracked. Constitution version 6.1.0 replaces the invalid hardware/context obligations. Keep one authoritative source with bounded derived runtime core/task instructions, recorded version/hash, independent tool validation and code/deployment enforcement. Documents, hidden cells, slide notes, quoted history and tool outputs remain untrusted evidence. No model can grant permissions or change policy.

Use internal OIDC/SSO and workspace roles. Apply authorization before candidate retrieval and every later lookup, source preview, original download, dataset operation, history display and export. Use PostgreSQL RLS with non-owner, non-superuser, non-BYPASSRLS serving credentials and transaction-local identity. Separate migration/provisioning credentials, worker identity and tested pool reset. Track source dependencies and recheck revocation; begin without cross-user answer caching. [PostgreSQL row security](https://www.postgresql.org/docs/16/ddl-rowsecurity.html).

Bundle browser assets internally, sanitize Markdown and filenames, protect every write/admin/query route, bound history and uploads, use appropriate TLS/cookies/CSRF/browser headers, and coordinate quotas. Keep logs/audit local; restrict full prompt retention, avoid document content in metrics and never log secrets. Define document/derived-data deletion and backup expiry behavior explicitly.

The UI shows selected/used files and versions, job state, extraction warnings, queued generation, cancel/retry, precise source previews and calculation evidence. Arabic RTL and mixed-language tables/citations are required. Initial exports: CSV/XLSX result tables, PNG charts and Markdown reports, with formula-injection protection. Polished DOCX/PDF exports follow source attribution plus visual verification.

Separate process liveness, core readiness and generation readiness. Inference outages leave authorized non-generation functions usable. Keep observability lightweight on the shared CPU. Back up PostgreSQL/files consistently with keys/configuration/manifests to a separate approved destination and test full restoration. Agree RPO/RTO and retention with IT. A single server remains one failure domain; HA is a separate infrastructure decision.

## 12. CPU deployment and configuration examples

These are **future implementation examples**, not deployed configuration or commands executed by this revision. Existing Compose files contain no model service or accelerator reservation to remove; do not mistake them for the finished CPU production stack.

Proposed application settings:

```dotenv
GENERATION_BASE_URL=http://model:8080/v1
GENERATION_MODEL_ALIAS=company-assistant
GENERATION_PROFILE=cpu-qwen35-9b-q4km
GENERATION_TOTAL_CONTEXT_TOKENS=8192
GENERATION_MAX_OUTPUT_TOKENS=512
GENERATION_ACTIVE_REQUESTS=1
GENERATION_QUEUE_LIMIT=8
GENERATION_VISION_ENABLED=false
# Set timeout, thread counts and limits from the approved benchmark profile.
```

The 8K value is an explicit initial experiment. Input accounting must subtract actual instructions/template/history/tool/image cost and output reserve. Never treat these settings as proof that the configuration passed.

Illustrative offline CPU server command after approved artifact preparation:

```bash
/opt/llama/bin/llama-server \
  --model /models/Qwen3.5-9B-Q4_K_M.gguf \
  --alias company-assistant --host 0.0.0.0 --port 8080 \
  --ctx-size 8192 --parallel 1 \
  --threads "${GENERATION_THREADS:?set after CPU discovery}" \
  --threads-batch "${PROMPT_THREADS:?set after CPU discovery}" \
  --n-gpu-layers 0 --no-mmproj-offload \
  --jinja --no-context-shift --cache-ram 0 --no-webui
```

This text-only profile omits the projector and rejects image requests in the application. An enabled vision profile adds the exact matching local `--mmproj` file, stays CPU-only and requires its own measured budget. Do not use remote `-hf` auto-download paths in production. Validate flags against the pinned binary's `--help`. Server API/slot/thread controls are documented, but this command has not been run here. [Server reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

Illustrative model-service fragment for a future Linux Compose deployment:

```yaml
model:
  image: ${CPU_MODEL_IMAGE:?supply tested image pinned by digest}
  # The image owns the reviewed command/profile above; no automatic downloads.
  volumes:
    - ./offline-models/qwen35-9b:/models:ro
  networks: [inference_internal]
  mem_limit: 24g
  memswap_limit: 24g
  cpus: ${MODEL_CPU_QUOTA:?set after hardware discovery}
  pids_limit: 128
  read_only: true
  cap_drop: [ALL]
  security_opt: [no-new-privileges:true]
  # No host port, database secret, company-document mount, or Docker socket.
```

Only the API and model join the inference network. The reverse proxy handles internal HTTPS elsewhere. Enforce offline egress at host/network level; a container flag alone is insufficient. The image must create/use a non-root runtime user and bounded scratch where required. Validate actual limits on the target OS; do not assume resource fields are enforced merely because Compose parses them.

For an optional Ollama profile, verify `OLLAMA_NO_CLOUD=1`, `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`, a small explicit `OLLAMA_CONTEXT_LENGTH`, and a bounded `OLLAMA_MAX_QUEUE`. Select the validated local model digest, inspect CPU placement and disable cloud/web features. Do not apply these settings to the user's existing laptop service during this task.

Use CPU-only dependency builds for embedding/OCR/analysis where applicable; no GPU-specific package indexes, accelerator reservations, engine image variants or toolchains in the default path. Remove unused Streamlit/hosted-inference dependencies from the enterprise API group. Pin transitive packages, parser models, tokenizers, traineddata, fonts, binaries and image digests in the release manifest. A future GPU can replace the inference-service profile after validation without changing application contracts; it is not a pilot prerequisite.

## 13. Hardware discovery checklist — first execution step

Run these read-only commands on the actual server when access is available. Do not assume the laptop or Docker VM reports the server's hardware. Record outputs internally, omitting serial numbers or sensitive host identifiers from public reports.

### Linux server

```bash
uname -m
cat /etc/os-release
lscpu
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
free -b
cat /proc/meminfo
lsblk -d -o NAME,MODEL,SIZE,ROTA,TRAN
df -hT
vmstat 1 5
ps -eo pid,comm,rss,pcpu --sort=-rss
```

Where installed/authorized: `numactl --hardware` for NUMA topology and `sudo dmidecode --type memory` for populated memory slots/configuration. `lscpu` flags distinguish x86 AVX2/AVX-512/VNNI/BF16 or ARM features; no particular instruction set is assumed. Installed DIMM speeds do not measure bandwidth. Later run an approved CPU-only bandwidth benchmark and the pinned runtime benchmark; record virtualization/container CPU and RAM restrictions too.

If Docker is used:

```bash
docker version
docker info --format '{{.NCPU}} CPUs; {{.MemTotal}} bytes visible'
```

### Windows server or laptop discovery

```powershell
Get-CimInstance Win32_Processor |
  Select-Object Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed
Get-CimInstance Win32_ComputerSystem |
  Select-Object TotalPhysicalMemory,HypervisorPresent
Get-CimInstance Win32_OperatingSystem |
  Select-Object Caption,Version,OSArchitecture,TotalVisibleMemorySize,FreePhysicalMemory
Get-CimInstance Win32_PhysicalMemory |
  Select-Object Capacity,Speed,ConfiguredClockSpeed,DeviceLocator,PartNumber
Get-PhysicalDisk | Select-Object FriendlyName,MediaType,Size,HealthStatus
Get-Volume | Select-Object DriveLetter,FileSystem,Size,SizeRemaining
```

Memory values from the OS class use KiB; physical-module capacities are bytes. If ISA detail is needed, use an approved Microsoft Sysinternals Coreinfo binary (`coreinfo64.exe -f`); do not infer full vector support from CPU branding. If WSL is part of the intended path, inspect `wsl --status`, `wsl --list --verbose`, its VM memory limit and CPU visibility.

Linux is the provisional production host recommendation for the full service/isolation stack. If the server runs Windows, first decide with IT between a supported Linux VM and a separately validated native deployment; subtract VM/host overhead. Native Windows model support does not prove that the full Linux Compose/sandbox design runs unchanged on Windows Server.

**Discovery deliverable:** CPU/ISA/topology, measured visible RAM, current utilization, memory-channel/NUMA observations, storage/scratch free space, OS/runtime limits, expected user/corpus profile, internal identity/network/backup requirements. These determine binary selection, thread counts, admissible model/context, worker overlap, response-time goals and whether the initial resource envelope must shrink.

## 14. Laptop and offline workflow

| Profile | Runs where | Resource policy / evidence |
|---|---|---|
| `dev-lite` | 16 GB laptop | Mock generation/embeddings, small PostgreSQL, lightweight worker, synthetic fixtures; aim for roughly 4–6 GiB application memory and leave OS/IDE headroom; measure |
| `dev-existing-gemma` | Laptop, optional after resuming tests | Reuse installed Gemma via Ollama; short 2K–4K text-only test and one request; stop other heavy services first if needed; monitor RAM before loading |
| `dev-small-cpu` | Laptop, optional later | Qwen4B short-context plumbing/quality exploration only if approved artifacts/resources exist; no mandatory download |
| `server-cpu-staging` | Company 64 GB server | One candidate at a time; full CPU/Arabic/image/context/resource measurements |
| `server-cpu-local` | Company server after gates | Pinned offline release, real identity/permissions and measured operating envelope |

The Gemma disk bundle alone is about half the laptop's installed RAM before runtime, cache, OS and development services. Its presence makes future testing convenient, not guaranteed comfortable. Do not load all downloaded models or run an unbounded native context. No laptop run substitutes for server CPU/memory-bandwidth testing. Cloud inference is excluded.

Prepare approved artifacts on a connected staging machine and verify after transfer. The production bundle includes exact model/projector/draft files where enabled, tokenizers/templates, runtime/CPU ISA build, Python/native packages, extraction/OCR artifacts, internal browser assets and licenses/checksums. Production cold start must not fetch anything. A prepared release should support a documented startup command after provisioning; first installation still needs CPU/OS validation, configuration, credentials, transfer and acceptance. It is not instantly production-ready merely because RAM is known.

## 15. Benchmark and acceptance plan

### Procedure

1. Discover hardware and freeze reproducible case definitions before tuning. Review 20–30 sanitized documents and 30 initial questions/calculations for the first slice.
2. Verify the exact runtime/artifact/template/tokenizer without asserting quality. Benchmark text-only 4K first, then 8K; capture cold loading and warmed runs.
3. Compare 9B, 35B-A3B and 27B on the same task content and budgets, with 4B as the lighter comparison and Gemma as the existing independent comparator. Do not force an unsafe large candidate to load. Record rejection/OOM/timeout as an outcome.
4. Use fixed-evidence generation tests to separate model quality from extraction/retrieval quality. Run retrieval-only tests and true end-to-end tasks separately.
5. Expand context to 16K/32K only after previous stages pass. Optional 64K/128K is research, not an inherited release obligation.
6. Test one active request; then two and four connected users under the **single-active queue policy**. Measure queue delay separately. Parallel generation greater than one is a separate optional experiment requiring a new resource decision.
7. Test native extraction, OCR and images independently. Then test realistic overlap/recovery with scheduler limits, company identity and offline controls.

Record runtime/build/ISA, all artifact hashes, quantization, tokenizer/template, constitution/prompt hash, context and actual token counts, batch/thread settings, thinking/draft configuration, active components, case seed/repetition and hardware state. Use the same semantic evidence across models while measuring each tokenizer's actual count; position-controlled context cases may require different filler amounts. Exclude private reasoning from reports/history while measuring its resource cost when present.

### Measurements and release requirements

| Dimension | Measure | Gate |
|---|---|---|
| Arabic | Formal Arabic, Egyptian questions, mixed terms, Arabic-to-English retrieval, negation/conditions, scans and numerical wording | At least 100 reviewed Arabic/mixed cases; freeze required slice thresholds and matched-English quality-gap criteria before held-out evaluation |
| Corpus | 200–300 reviewed cases, realistic distractors, long docs, absent/conflicting answers, versions and exhaustive tasks | Separate tuning/held-out sets; representative company data reviewed internally |
| Extraction/OCR | Text coverage, digit/cell accuracy, table headers, reading order, locations and review flags | Format-specific gold tolerances; uncertain critical figures cannot be silently accepted |
| Retrieval | Supporting-element recall@k, MRR/nDCG, cross-language and permission slices | Proposed pilot objective ≥90% evidence recall@10, confirmed after baseline with defined denominators |
| Answers | Required facts, unsupported claims, calibrated refusal, citation precision/coverage, source/version validity | Proposed ≥95% manually reviewed citation precision; task correctness/refusal/coverage thresholds frozen separately |
| Complete documents | All required sections/versions visited, coverage gaps and citation lineage | No false claims of complete coverage; required extraction/summary/comparison gold cases pass |
| Numerical analysis | Exact results, units/filters/joins, freshness, row counts/exclusions, provenance | 100% correct fixed critical numerical fixtures within defined precision/rounding tolerances |
| Images | Chart/diagram tasks, small text, crops, unsupported-capability behavior | Exact image-enabled configuration passes quality, latency and RAM limits before enablement; otherwise an explicit unresolved capability |
| Context | Beginning/middle/end evidence with distractors at each tested total window | Choose the smallest passing window; no unsupported production context claim |
| Responsiveness | Cold load, prompt/prefill time, time to first user-visible token, decode rate, total completion/workflow time, p50/p95 and queue wait | Business task deadlines set with the owner after hardware baseline and before held-out acceptance; no invented rates |
| Resources | Model and whole-host peak RAM, cgroup events, pressure, major faults/swap, CPU utilization and scratch/disk | No host OOM or sustained swapping in the accepted envelope; every job stays within enforced limits or fails explicitly |
| Access and tools | Cross-workspace IDs, actual serving role, revocation/history/cache/export, injection, sandbox/network attempts | Zero unauthorized content in dedicated tests; no free-form unsafe execution |
| Operations | Cancel, retries, restart, duplicate delivery, dependency outage, offline cold start and full restore | Recovery and offline release drill pass with all enabled workflows |

Timed-out cases are not excluded to improve averages. Test watchdogs must be recorded as test policy rather than advertised latency. If none of the candidates meets the required Arabic quality and acceptable task time simultaneously, the pilot remains unaccepted. Options are narrower explicitly approved scope, queued asynchronous tasks, better CPU/bandwidth hardware, or a later GPU upgrade—not an automatic cloud fallback or fabricated performance result. The separately authorized hosted mode can be evaluated explicitly with the same quality/security gates.

Select models by mandatory gates first, then compare quality, actual wait time, memory and operating complexity. A fast fluent answer with unsupported facts is a failure. No aggregate score compensates for an access leak or a failing required Arabic slice.

## 16. Updated phased backlog and first milestone

The execution has **seven main phases (P0–P6)**. See the [specific phase roadmap and current status](governance/phase-roadmap.md).

The earlier implementation is a starting point to review, not a reason to skip hardware correction or security gates.

| Phase | Work / dependency | Deliverable and exit gate | Indicative remaining engineering effort |
|---|---|---|---|
| P0 — Hardware and policy reconciliation | First execution step; obtain read-only server inventory, user/corpus needs, review prior WIP, remove stale GPU/context assumptions | Discovery report, CPU profile/binary decision, updated rule ledger, initial reviewed cases and explicit blocked decisions | 2–4 days, excluding access wait |
| P1 — Lightweight foundation and scoped storage | Rebase unfinished migrations/configuration/storage on reviewed baseline; separate owner/serving credentials; dev identity only locally; SSO integration path | Practical laptop setup, durable originals, migrations, workspace/RLS/revocation tests using actual serving role; policy tests aligned with v2 | 4–7 days |
| P2 — CPU baseline and authorized PDF slice | P0 CPU information needed for real inference; prepare approved 9B artifact or reuse Gemma only for laptop integration; pin one runtime; add durable jobs, basic extraction and retrieval | Authorized PDF upload -> durable extraction -> evidence -> local streamed answer -> precise source preview; actual CPU/RAM and task-quality record | 5–8 days |
| P3 — CPU OCR and Office fidelity | Native-first PDF, Arabic/English OCR, DOCX/PPTX/XLSX/CSV adapters, offline models, job limits | Format gold cases preserve required text/cells/locations; failures/review states visible; bounded CPU/RAM | 6–10 days |
| P4 — Deterministic analysis | Typed operations, mappings, Parquet/DuckDB, sandbox, spill quotas, provenance, charts/exports | Numerical gold suite and isolation pass; representative workbook job fits policy | 6–10 days |
| P5 — Coverage, optional vision and model comparison | Complete summaries/comparisons, history, RTL UX, queued image tests, 35B/27B/4B/Gemma comparisons and model swapping | Coverage/citation tests; measured enabled capabilities; two validated configurations swap without app/retrieval changes or re-ingestion | 5–9 days |
| P6 — Offline hardening and pilot | Real SSO, safe browser assets, quotas, load/scheduler, backup/restore, clean offline boot | Named release meets all applicable gates, documented service envelope and explicit residual limitations | 5–8 days |

These rough allowances sum to **33–56 remaining engineering days**, about 7–12 five-day working weeks for one developer with AI assistance. They are planning estimates, not calendar commitments; access delays, document review, CPU benchmark run time and compatibility repairs can extend elapsed time. Re-estimate after P0/P1. CPU inference speed can affect product feasibility and evaluation duration; it cannot be inferred from development effort.

Engineering owns implementation, artifact manifests, tests and measured evidence. The project owner owns priorities, business task deadlines and acceptance/policy decisions. Company IT owns server inventory/access, OS/runtime approval, internal identity/TLS/network, retention and recovery requirements. Domain reviewers own Arabic/document/numerical gold judgments. Name these owners at kickoff; one person may hold multiple roles. Progress reports distinguish implemented code, verified behavior, open failures and external dependencies.

**First practical milestone:** after discovery, complete the protected small-PDF workflow with one local CPU model, a small measured context, one active generation, exact source previews, durable retry and a reviewed bilingual test slice. Until server access exists, the same workflow can be built/tested with explicit fixtures and optional short laptop Gemma integration. Mock-only success is a development milestone, not acceptance of CPU model quality.

Track P0–P6 in the backlog. Retain A0–A8/B0–B3 only as historical references. Every change has rule IDs, test evidence, known gaps and server-dependent items. Do not mark all 55 rules verified because a subset of foundation tests passed. No production release until actual serving-role isolation, representative Arabic tasks, memory/scheduler, offline and restore gates pass.

## 17. Artifact reconciliation and model replacement

Planning changes from this revision:

- Replace this existing plan in place; the repository copy mirrors the same version.
- Amend the existing constitution to version 6.1.0 with stable rule IDs and corrected CPU/context/resource obligations.
- Refresh only derived policy **metadata** for the amended constitution, update the rule ledger/backlog, and mark previous deployment-candidate records superseded. New candidate records are planning-only and production-disabled.
- Keep source-code WIP and current containers unchanged. Do not run migrations, restart services, perform inference or download model weights during this planning task.

When implementation resumes, update version/count assumptions in existing tests and replace stale candidate configuration consumers; rerun the relevant checks. Historical tests reference constitution v1 and three candidates, so their old pass counts do not validate the amended plan. Review uncommitted storage/migration work before enabling any route. Archive old source ZIPs as historical snapshots; do not treat their embedded v1 plan as current deployment instructions.

Generation swap procedure: prepare verified CPU artifacts/config; run adapter/tokenizer/text/Arabic/tools/vision-if-enabled and resource tests; drain requests; stop the previous process; start/warm the replacement; check readiness and smoke cases; resume or roll back. Keep application alias `company-assistant`. Engine-specific settings remain in adapters/profiles. No generation-only change requires application-route/frontend/retrieval rewrites or re-ingestion/re-embedding. An embedding-model change is a different versioned-index migration. A runtime change can require adapter code but must remain within that boundary.

Release manifests record source/image/runtime revisions, CPU/ISA build, model/projector/draft/tokenizer/template files and hashes, parser/OCR versions, embedding index/schema versions, constitution/prompt version, limits, licenses and measured acceptance results. Keep the previous compatible release and tested data-recovery path.

## 18. Risks and decisions blocked by unknown CPU information

| Risk / open decision | Provisional response |
|---|---|
| Old or low-bandwidth CPU makes even 9B too slow | Test smaller 4B and existing Gemma; measure task times; preserve asynchronous workflows and explicit feasibility decision |
| ISA/OS/virtualization unknown | P0 determines portable binary, native build, Linux VM overhead and supported isolation path |
| Large Qwen fits disk/RAM but misses task deadlines | Retain as scheduled quality experiment only if useful; do not promote on parameter count |
| MoE runtime/kernel/repacking issues | Exact pinned load test and cold peak; reject failed configuration or choose another tested artifact |
| Smaller quantized model misses Arabic exceptions or calculations | Slice-based evaluation, stronger retrieval, trusted arithmetic and ambiguity handling; higher-bit/sized candidate comparison without assuming success |
| Vision or Arabic OCR has high CPU cost or poor fidelity | Native-first extraction, one queued OCR/image task, previews/review flags; explicit unmet visual scope if gates fail |
| Background tasks consume inference bandwidth/RAM | Global admission leases, one expensive job, thread limits, paused batches and measured overlap before promotion |
| Loading/repacking or workbook peaks exceed estimates | Guard reserve, process/cgroup caps, chunking and spill limits; failed job/configuration, no host-wide overcommit |
| Source revocation leaves stale history/cache/output | Dependency tracking and reauthorization at display/export/tool boundaries |
| Egress or first-use downloads break locality | Prepacked artifacts, local-only runtime configuration, host/network denial and clean offline acceptance |
| Existing Gemma tag changes remotely | Preserve inspected local manifest/blobs; verify hashes and license before transfer; never silently pull over the pinned asset |
| Limited laptop RAM | Default mocks; optional single short model session with other heavy services stopped; no target-server capacity claims |
| Single host or disk fails | Separate consistent backups, tested restore, explicit RPO/RTO; additional hardware for HA is separate |

Blocked choices include exact thread counts, instruction-optimized binary, useful maximum context, latency/concurrency targets, safe overlap, large-model admissibility, image enablement and final model/runtime. Product scope, modular contracts, authorization, provenance, durable jobs and test design can proceed independently once implementation is resumed.

## 19. Obsolete assumptions removed

1. A single GPU with 64 GB VRAM: replaced by 64 GB system RAM and no dedicated GPU.
2. A 44–50 GiB GPU allocation or weights-only memory estimate: replaced by a whole-host RAM envelope including other services and peaks.
3. FP8 Qwen checkpoint as the default deployment artifact: replaced by pinned CPU-oriented quantized GGUF candidates.
4. vLLM as an inherited default: replaced by a provisional llama.cpp primary and documented Ollama alternative; vLLM CPU remains a future measured option.
5. Accelerator reservations, GPU runtime images/toolchains, tensor-parallel settings or GPU kernel checks as prerequisites: removed from the default CPU path.
6. 64K normal and 128K priority production context: replaced by staged 4K/8K/16K/32K tests; larger windows optional only.
7. Two/four active generations as launch settings: replaced by one active request and bounded queued users.
8. Embeddings/reranking can later use spare VRAM: replaced by CPU time/RAM admission and bounded loading.
9. Vision can cheaply share the generation hardware: replaced by separately tested image components and queued workload limits.
10. RAM/disk fit implies acceptable response time, or 3B activated parameters imply a 3B memory footprint: explicitly rejected.
11. Broker infrastructure is mandatory from the first slice: replaced by a PostgreSQL durable queue initially.
12. Downloaded laptop Gemma implies validated company accuracy or target-server speed: explicitly rejected while preserving it as a useful existing asset.
13. The previous 32–51-day estimate or source ZIP is a current CPU release commitment: superseded by this version's phased estimates and acceptance gates.

## 20. Immediate next execution batch after plan review

1. Obtain read-only server discovery outputs and record actual visible RAM/CPU/OS constraints.
2. Review the preserved repository WIP and revised constitutional obligations; align tests/config consumers before claiming a working checkpoint.
3. Finalize the small CPU runtime/artifact acquisition list and resource profile; prepare approved dependencies without first-run downloads.
4. Complete/test scoped identity, migrations, storage, RLS and durable jobs on synthetic data using actual serving credentials.
5. Build the protected PDF slice with mocks; optionally test the already-installed Gemma in a short isolated laptop session after resource checks.
6. On the server, benchmark the 9B baseline at small contexts, then the agreed challengers sequentially under explicit memory budgets.
7. Freeze required task/Arabic/latency thresholds and progress through format, analysis, coverage, optional vision and offline-pilot gates.

This is the execution order, not permission to resume application implementation during the present planning task.

## 21. Primary-source record

Technical recommendations above combine documented facts with explicitly labeled engineering decisions. Sources were checked on 18 October 2026; mutable documentation and catalogue entries must be rechecked against the pinned runtime/artifacts at acquisition. General API support and source code are not substitutes for exact-build execution tests.

- Repository baseline: [RAGBase reviewed snapshot](https://github.com/ElMolla10/ragbase/tree/2f0a0204a14f6659b97a061168ccb0bdd8c061f7).
- Qwen originals/licenses: [4B](https://huggingface.co/Qwen/Qwen3.5-4B), [9B](https://huggingface.co/Qwen/Qwen3.5-9B), [27B](https://huggingface.co/Qwen/Qwen3.5-27B), [35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B).
- Exact converted-file repositories and revisions are linked in Section 5; their reported hashes require later local verification.
- Runtime, model-template, grammar, build, OCR, extraction, database, analytics and memory-control primary references are linked next to the claims they support.
- Gemma local metadata was inspected read-only; [Google's overview](https://ai.google.dev/gemma/docs/core) and [Ollama's matching entry](https://ollama.com/library/gemma4:12b) support its documented identity/capability context, not a performance claim.
