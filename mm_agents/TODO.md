# Refactor agents to use `env_loader.py`

`env_loader.load_mm_agents_env()` loads `GUI-Docker-Env/.env-default` then `.env` (see `env_loader.py`). Call it once at module import (before reading `os.environ` for API URLs) so all agents share the same config.

## Done

- [x] `uitars15_v1.py` — imports and calls `load_mm_agents_env()`

## TODO

- [ ] `uitars_agent.py` — replace hardcoded `DOUBAO_*` / `os.environ[...]` with `load_mm_agents_env()` + `getenv` fallbacks aligned with `OPENAI_*`
- [ ] `uitars15_v2.py` — same for `DOUBAO_API_KEY` / `DOUBAO_API_URL` in HTTP inference paths
- [ ] `agent.py` — replace ad-hoc `load_dotenv()` with `load_mm_agents_env()`; keep `getenv` reads consistent
- [ ] `qwen25vl_agent.py` — call loader at import; keep `DASHSCOPE_*` via env only
- [ ] `qwen3vl_agent.py` — remove hardcoded DashScope URL/key; use env + loader
- [ ] `mano_agent.py` — `MANO_*` / `OPENAI_*` via loader
- [ ] `openai_cua_agent.py` — `OPENAI_API_KEY_CUA` / `OPENAI_API_KEY` after loader
- [ ] `opencua_agent.py` — `OPENCUA_*` after loader
- [ ] `o3_agent.py` — load before module-level `OPENAI_API_KEY` read, or refresh after load
- [ ] `owl_agent.py` — OSS / `api_url` / `api_key` paths; call loader where `os.environ[...]` is required
- [ ] `mobileagent_v3/mobile_agent_modules.py` — OSS + DashScope + OpenAI clients after loader
- [ ] `uipath/llm_client.py`, `uipath/grounder_client.py` — `SERVICE_KEY` etc. after loader
- [ ] `maestro/` — entrypoints (`osworld_run_maestro.py`, `cli_app_maestro.py`, …): call loader once before `maestro/core/engine.py` reads provider env vars
- [ ] `jedi_7b_agent.py` (and related) — replace fixed OpenAI URL with env + loader

## Convention

1. At top of module (after stdlib imports): `from mm_agents.env_loader import load_mm_agents_env` then `load_mm_agents_env()`.
2. Never require bare `os.environ["KEY"]` without a documented default in `.env-default` or a clear error after load.
3. Prefer `OPENAI_BASE_URL` / `OPENAI_API_KEY` for OpenAI-compatible servers; keep provider-specific names only where the protocol differs.
