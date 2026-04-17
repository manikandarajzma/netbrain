# AI Primer

A reference for common AI and ML terms used in the Atlas codebase and documentation.

---

## Large Language Model (LLM)

An LLM is a neural network trained on massive amounts of text (books, code, websites) to predict the next token in a sequence. "Large" refers to the number of parameters — modern models range from a few billion to hundreds of billions.

Given an input (called a **prompt**), the model generates a response one token at a time, each token sampled from a probability distribution over the entire vocabulary. The model has no memory between separate conversations — every call is stateless unless you explicitly include prior messages in the prompt.

Examples: LLaMA, GPT, Mistral, Gemma.

Atlas currently uses local Ollama-served chat models, with role-specific model assignment configured in `agents/agent_factory.py`.

---

## Transformer

The transformer is the neural network architecture that all modern LLMs are built on (introduced by Google in the 2017 paper "Attention Is All You Need"). Before transformers, models processed text sequentially — one word at a time — which made it hard to learn relationships between distant words. Transformers process the entire input in parallel and use a mechanism called **attention** to let every token directly relate to every other token in the sequence.

**Attention** — for each token in the input, the model computes a score against every other token to determine which ones are most relevant. For the word "it" in "The server crashed because it ran out of memory", attention lets the model learn that "it" refers to "server", not "memory", regardless of how far apart they are.

**Why it matters for LLMs:**
- Parallelism → can be trained on GPUs efficiently
- Long-range dependencies → understands context across thousands of tokens
- Scales well → more parameters + more data = better results (the "scaling law")

Atlas uses transformer-based chat models. The architecture family is fixed; what differs between models is size (parameter count), training data, and fine-tuning.

---

## Inference

Inference is the act of running a trained model to generate output — as opposed to **training**, which is the process of teaching the model from data.

When Atlas sends a prompt to the LLM and gets a tool call back, that is inference. It is computationally expensive (requires GPU VRAM proportional to model size) but much cheaper than training, which can cost millions of dollars and take weeks.

**Inference server** — a process that loads a model into GPU memory and serves inference requests over HTTP. Ollama and vLLM are both inference servers.

---

## GPU (Graphics Processing Unit)

Originally designed to render graphics, GPUs are now the standard hardware for AI inference and training. The reason: LLMs require billions of simple floating-point multiplications done in parallel — exactly what GPUs are built for. A CPU has tens of cores; a GPU has thousands of smaller ones optimized for parallel math.

For inference, the GPU loads the model weights into its memory (VRAM) and performs matrix multiplications to generate each token. Without a GPU, inference falls back to CPU, which is 10-50x slower depending on model size.

Common GPUs for AI:
- **Consumer**: NVIDIA RTX 3090 (24GB VRAM), RTX 4090 (24GB VRAM)
- **Datacenter**: NVIDIA A100 (80GB VRAM), H100 (80GB VRAM)

Ollama can run on CPU if no GPU is available, at the cost of speed. vLLM requires a GPU.

---

## VRAM (Video RAM)

VRAM is the memory on the GPU — separate from system RAM. The entire model must fit in VRAM for GPU inference to work. If the model is too large, it either falls back to system RAM (slow) or fails to load.

**Rule of thumb for VRAM requirements:**

| Precision | VRAM per billion parameters |
|---|---|
| float32 (full) | ~4 GB |
| bfloat16 (half) | ~2 GB |
| INT8 (quantized) | ~1 GB |
| INT4 / Q4 (quantized) | ~0.5 GB |

So an 8B model at Q4 quantization needs roughly ~5GB VRAM and often fits on a consumer GPU. The same model at bfloat16 needs much more VRAM.

VRAM is a hard limit: unlike system RAM, you cannot swap to disk without killing performance entirely.

---

## Parameters

Parameters are the numerical weights inside a neural network — the values learned during training that encode the model's "knowledge". An 8B model has 8 billion individual floating-point numbers. These are stored on disk (typically in a file like `model.safetensors`) and loaded into GPU VRAM at startup.

More parameters generally means more capability but also more VRAM and slower inference. A 70B model needs ~40GB VRAM; an 8B model needs ~6GB.

**Not to be confused with** function parameters or API parameters — the word is overloaded.

---

## Tokens

The unit LLMs operate on. A tokenizer splits text into chunks (tokens) before feeding it to the model. Tokens are roughly 3-4 characters on average in English — "hello" is one token, "unbelievable" might be two.

- **Context window** — the maximum number of tokens a model can process in a single call (input + output combined). The exact limit depends on the specific model you run.
- **Token limit** — if a conversation history grows beyond the context window, older messages must be truncated.

---

## Temperature

A parameter passed at inference time (not a trained weight) that controls how "creative" or "random" the model's outputs are.

- `0.0` — deterministic, always picks the highest-probability token. Used in Atlas to keep routing, scenario selection, tool use, and structured outputs stable.
- `1.0` — full randomness, outputs vary significantly between runs.
- Values above `1.0` — increasingly chaotic, rarely useful.

Atlas sets `temperature=0.0` everywhere it calls the LLM.

---

## Tool Calling (Function Calling)

A feature where the LLM, instead of generating prose, outputs a structured JSON object indicating which function to call and with what arguments. The application code then executes the function and can feed the result back to the model.

```
Prompt: "Give me the details for INC0010043"

LLM output (tool call):
{
  "name": "get_incident_details",
  "arguments": { "incident_number": "INC0010043" }
}
```

The model does not execute the function — it only outputs the intent. Atlas reads the tool call, executes the actual Atlas tool, and that Atlas tool may then call MCP, Nornir, or another owned backend path.

---

## Embeddings

A numerical representation of text as a vector (a list of floats). Similar meanings produce vectors that are close together in vector space. Used for semantic search, RAG (retrieval-augmented generation), and classification.

Atlas does not currently use embeddings in the live runtime — lane selection, scenario selection, and tool use are done through LLM reasoning plus bounded tool visibility, not vector similarity.

---

## RAG (Retrieval-Augmented Generation)

A pattern where relevant documents are retrieved from a database and injected into the prompt before the LLM responds. Allows the model to answer questions about data it was not trained on without retraining.

Atlas does not use RAG. The LLM selects tools and the tools query live data directly.

---

## Ollama

A local inference server that makes running open-source LLMs simple. It handles model download, VRAM management, quantization, and exposes an HTTP API.

- Default port: `11434`
- API: proprietary (`/api/generate`, `/api/chat`, `/api/tags`)
- Model format: GGUF (quantized weights that run on consumer GPUs or CPU)
- Best for: development, single-user local setups

Atlas currently uses Ollama. The base URL is configured via `OLLAMA_BASE_URL`, and the active model assignments are managed through the role-specific settings in `agents/agent_factory.py`.

> **Planned migration:** Atlas will be migrated from Ollama to vLLM for production use.

---

## vLLM

A high-performance inference server designed for production. Uses **PagedAttention** to manage GPU memory efficiently, enabling much higher throughput than Ollama.

- Default port: `8000`
- API: OpenAI-compatible (`/v1/chat/completions`, `/v1/models`)
- Model format: HuggingFace (full-precision or bfloat16 weights)
- Best for: multi-user production deployments, high request volume

Because vLLM uses the OpenAI API format, switching Atlas from Ollama to vLLM mainly requires changing the model endpoint configuration — the tool-calling architecture itself does not change.

| | Ollama | vLLM |
|---|---|---|
| Target | Local dev | Production |
| API format | Proprietary | OpenAI-compatible |
| Throughput | Low | High |
| Model format | GGUF (quantized) | HuggingFace (full precision) |
| GPU required | No (can use CPU) | Yes |

---

## LangChain

A Python framework for building applications with LLMs. It provides abstractions for:

- **Chat models** — a uniform interface (`ChatOllama`, `ChatOpenAI`, etc.) so you can swap inference backends without changing application logic
- **`bind_tools()`** — attaches tool schemas to a model so it knows which functions are available
- **Messages** — `HumanMessage`, `AIMessage`, `SystemMessage` wrappers that get serialized into the prompt format the model expects
- **Chains** — composable sequences of steps (prompt → model → parser)

Atlas uses LangChain in the agent layer through:
- `agents/agent_factory.py`
- `agents/troubleshoot_agent.py`
- `agents/network_ops_agent.py`
- the agent-facing tool modules under `tools/`

Atlas uses LangChain for model/tool binding and ReAct agent execution. API routing, auth, backend clients, and workflow orchestration live outside LangChain.

---

## LangGraph

A LangChain extension for building **stateful, multi-step agents** as graphs. Each node in the graph is a processing step; edges define control flow (including conditional branches and loops).

Used when a task requires multiple LLM calls in sequence — e.g. "call a tool, observe the result, decide whether to call another tool". LangGraph manages the state between steps.

Atlas uses LangGraph as a small routing and execution boundary:
- `graph/graph_builder.py`
- `graph/graph_nodes.py`
- `graph/graph_state.py`

LangGraph handles coarse routing between troubleshoot, network-ops, and dismiss paths. Heavier orchestration lives in owned services such as `services/troubleshoot_workflow_service.py` and `services/network_ops_workflow_service.py`.

---

## MCP (Model Context Protocol)

An open protocol (developed by Anthropic) for connecting LLMs to external tools and data sources in a standardized way. An MCP server exposes a list of tools with JSON Schema definitions; an MCP client fetches those definitions, passes them to the LLM, and routes tool calls back to the server for execution.

In Atlas:
- `mcp_server.py` is the MCP server — it registers tools (`get_incident_details`, etc.) and handles execution
- `integrations/mcp_client.py` is the MCP client transport

MCP is one backend integration path behind the Atlas tool layer; it is not the direct agent-facing tool surface.

---

## Quantization

A technique for reducing model size by storing weights at lower numerical precision. A full-precision model stores each weight as a 32-bit float (4 bytes). Quantization compresses weights to 8-bit integers (INT8) or lower.

- `Q4_K_M` — a common GGUF quantization level; weights stored at ~4 bits per parameter. An 8B model that would normally need ~16GB VRAM fits in ~5GB.
- Trade-off: smaller VRAM footprint and faster inference at the cost of a small accuracy reduction.

Ollama uses GGUF quantized models. vLLM typically runs full or half precision (bfloat16) on datacenter GPUs.

---

## Context Window

The maximum amount of text (measured in tokens) a model can "see" in a single inference call — both the input prompt and the generated output count toward this limit.

If a conversation grows beyond the context window, the application must truncate older messages. Atlas caps history at the last 10 messages before building graph state in `services/graph_runtime.py`, so hitting the context limit is unlikely in practice.

The exact context window depends on the configured model.

---

## System Prompt

The first message in a prompt, marked with role `system`, that instructs the model how to behave. It is set by the application, not the user.

Atlas uses system prompts in the specialized agent builders:
- `agents/troubleshoot_agent.py`
- `agents/network_ops_agent.py`
