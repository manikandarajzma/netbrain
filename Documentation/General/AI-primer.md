# AI Primer

A reference for common AI and ML terms used in the Atlas codebase and documentation.

---

## Large Language Model (LLM)

An LLM is a neural network trained on massive amounts of text (books, code, websites) to predict the next token in a sequence. "Large" refers to the number of parameters — modern models range from a few billion to hundreds of billions.

Given an input (called a **prompt**), the model generates a response one token at a time, each token sampled from a probability distribution over the entire vocabulary. The model has no memory between separate conversations — every call is stateless unless you explicitly include prior messages in the prompt.

Examples: LLaMA 3.1, GPT-4, Mistral, Gemma.

Atlas currently uses `llama3.1:8b` served via Ollama.

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

So `llama3.1:8b` at Q4 quantization needs ~5GB VRAM — fits on a consumer GPU. The same model at bfloat16 needs ~16GB — requires a high-end consumer or datacenter GPU.

VRAM is a hard limit: unlike system RAM, you cannot swap to disk without killing performance entirely.

---

## Parameters

Parameters are the numerical weights inside a neural network — the values learned during training that encode the model's "knowledge". An 8B model has 8 billion individual floating-point numbers. These are stored on disk (typically in a file like `model.safetensors`) and loaded into GPU VRAM at startup.

More parameters generally means more capability but also more VRAM and slower inference. A 70B model needs ~40GB VRAM; an 8B model needs ~6GB.

**Not to be confused with** function parameters or API parameters — the word is overloaded.

---

## Tokens

The unit LLMs operate on. A tokenizer splits text into chunks (tokens) before feeding it to the model. Tokens are roughly 3-4 characters on average in English — "hello" is one token, "unbelievable" might be two.

- **Context window** — the maximum number of tokens a model can process in a single call (input + output combined). `llama3.1:8b` has a 128k token context window.
- **Token limit** — if a conversation history grows beyond the context window, older messages must be truncated.

---

## Temperature

A parameter passed at inference time (not a trained weight) that controls how "creative" or "random" the model's outputs are.

- `0.0` — deterministic, always picks the highest-probability token. Used in Atlas for tool selection so the model reliably outputs structured JSON rather than varied prose.
- `1.0` — full randomness, outputs vary significantly between runs.
- Values above `1.0` — increasingly chaotic, rarely useful.

Atlas sets `temperature=0.0` everywhere it calls the LLM.

---

## Tool Calling (Function Calling)

A feature where the LLM, instead of generating prose, outputs a structured JSON object indicating which function to call and with what arguments. The application code then executes the function and can feed the result back to the model.

```
Prompt: "What address group is 11.0.0.1 part of?"

LLM output (tool call):
{
  "name": "query_panorama_ip_object_group",
  "arguments": { "ip_address": "11.0.0.1" }
}
```

The model does not execute the function — it only outputs the intent. Atlas reads the JSON, calls the actual MCP tool, and gets the result.

---

## Embeddings

A numerical representation of text as a vector (a list of floats). Similar meanings produce vectors that are close together in vector space. Used for semantic search, RAG (retrieval-augmented generation), and classification.

Atlas does not currently use embeddings — tool selection is done by the LLM directly via tool calling, not vector similarity.

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

Atlas currently uses Ollama. The base URL and model are configured via `OLLAMA_BASE_URL` and `OLLAMA_MODEL` in `.env`.

---

## vLLM

A high-performance inference server designed for production. Uses **PagedAttention** to manage GPU memory efficiently, enabling much higher throughput than Ollama.

- Default port: `8000`
- API: OpenAI-compatible (`/v1/chat/completions`, `/v1/models`)
- Model format: HuggingFace (full-precision or bfloat16 weights)
- Best for: multi-user production deployments, high request volume

Because vLLM uses the OpenAI API format, switching Atlas from Ollama to vLLM requires swapping `ChatOllama` for `ChatOpenAI` in LangChain and pointing it at the vLLM URL — no changes to tool calling logic.

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

Atlas uses LangChain in `chat_service.py` and `mcp_client_tool_selection.py` to call the LLM and parse tool calls. It does not use LangChain for anything else — MCP, API routing, and auth are all handled independently.

---

## LangGraph

A LangChain extension for building **stateful, multi-step agents** as graphs. Each node in the graph is a processing step; edges define control flow (including conditional branches and loops).

Used when a task requires multiple LLM calls in sequence — e.g. "call a tool, observe the result, decide whether to call another tool". LangGraph manages the state between steps.

Atlas does not currently use LangGraph. Tool selection is a single LLM call — the model picks one tool, Atlas executes it, and the result goes directly to a final answer synthesis step without further LLM-driven branching.

---

## MCP (Model Context Protocol)

An open protocol (developed by Anthropic) for connecting LLMs to external tools and data sources in a standardized way. An MCP server exposes a list of tools with JSON Schema definitions; an MCP client fetches those definitions, passes them to the LLM, and routes tool calls back to the server for execution.

In Atlas:
- `mcp_server.py` is the MCP server — it registers tools (`query_panorama_ip_object_group`, etc.) and handles execution
- `mcp_client_tool_selection.py` is the MCP client — it fetches the tool list and feeds it to the LLM

This separation means tools can be added to the MCP server without touching the LLM integration code.

---

## Quantization

A technique for reducing model size by storing weights at lower numerical precision. A full-precision model stores each weight as a 32-bit float (4 bytes). Quantization compresses weights to 8-bit integers (INT8) or lower.

- `Q4_K_M` — a common GGUF quantization level; weights stored at ~4 bits per parameter. An 8B model that would normally need ~16GB VRAM fits in ~5GB.
- Trade-off: smaller VRAM footprint and faster inference at the cost of a small accuracy reduction.

Ollama uses GGUF quantized models. vLLM typically runs full or half precision (bfloat16) on datacenter GPUs.

---

## Context Window

The maximum amount of text (measured in tokens) a model can "see" in a single inference call — both the input prompt and the generated output count toward this limit.

If a conversation grows beyond the context window, the application must truncate older messages. Atlas caps history at the last 10 messages (`conversation_history[-10:]` in `chat_service.py`) before building the prompt, so hitting the context limit is unlikely in practice.

`llama3.1:8b` context window: 128,000 tokens (~96,000 words).

---

## System Prompt

The first message in a prompt, marked with role `system`, that instructs the model how to behave. It is set by the application, not the user.

Atlas uses a system prompt in `chat_service.py` to tell the model to always respond with a tool call rather than prose, and to format arguments correctly.
