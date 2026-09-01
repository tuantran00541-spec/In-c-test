# Local Kimi API

`tools/kvl_api.py` exposes the packed Kimi-VL low-RAM runtime through small Anthropic- and OpenAI-compatible HTTP surfaces. The first revision is intended for local compatibility/smoke testing before the C runtime is converted into a persistent in-process engine.

## Start on Windows

From the repository root, after the Q8 runtime is packed and `kvl_generate.exe` is built:

```powershell
.\.venv\Scripts\python.exe tools\kvl_api.py "packed\kimi-vl-a3b-q8" `
  --host 127.0.0.1 --port 8000 `
  --api-key local-kimi --ram-mib 4096 --cache-mib 512
```

The default model id is:

```text
kimi-vl-a3b-instruct-q8
```

The server serializes inference requests so two clients cannot launch two multi-GiB generator processes at the same time. The existing hard-RAM planner and automatic trunk-cache selection are applied to every generation.

## Endpoints

| Contract | Endpoint |
| --- | --- |
| Health | `GET /healthz` |
| Model discovery | `GET /v1/models` |
| Anthropic Messages | `POST /v1/messages` |
| Anthropic token count | `POST /v1/messages/count_tokens` |
| OpenAI chat completions | `POST /v1/chat/completions` |

Both Anthropic and OpenAI streaming are supported with SSE.

## Claude Code smoke test

Claude Code supports routing through an LLM gateway/custom Anthropic base URL. In PowerShell:

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8000"
$env:ANTHROPIC_API_KEY = "local-kimi"
claude --model kimi-vl-a3b-instruct-q8
```

The local server accepts the key through either `x-api-key` or `Authorization: Bearer`.

This API revision returns **text only**. It accepts tool history blocks as readable conversation context, but it does not emit native Anthropic/OpenAI `tool_use` calls yet. Claude Code can therefore be used to verify model routing, chat history, streaming and latency, but not as a reliable coding agent until native tool-call adaptation is added.

## OpenAI-compatible clients / harnesses

Use:

```text
base_url = http://127.0.0.1:8000/v1
api_key  = local-kimi
model    = kimi-vl-a3b-instruct-q8
```

Example request:

```powershell
$headers = @{ Authorization = "Bearer local-kimi" }
$body = @{
  model = "kimi-vl-a3b-instruct-q8"
  messages = @(@{ role = "user"; content = "Xin chao" })
  max_tokens = 32
  temperature = 0
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v1/chat/completions" `
  -Headers $headers -ContentType "application/json" -Body $body
```

## Current backend boundary

The API process stays alive, but each inference request currently invokes the validated `kvl_generate` CLI once. This deliberately avoids changing model math while the HTTP contract is being validated. The next serving optimization is to split the C runtime into a persistent engine/session API so trunk/global/expert caches survive across HTTP requests without changing these URLs.
