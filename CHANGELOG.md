# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-01-11

### Added

- Multi-model adversarial debate for spec refinement
- Support for multiple LLM providers via LiteLLM:
  - OpenAI (gpt-4o, gpt-4-turbo, o1)
  - Google (gemini/gemini-2.0-flash, gemini/gemini-pro)
  - xAI (xai/grok-3, xai/grok-beta)
  - Mistral (mistral/mistral-large, mistral/codestral)
  - Groq (groq/llama-3.3-70b-versatile)
  - Deepseek (deepseek/deepseek-chat)
  - Zhipu (zhipu/glm-4, zhipu/glm-4-plus)
- Codex CLI integration for ChatGPT subscription models
- AWS Bedrock integration for enterprise users
- OpenAI-compatible endpoint support for local/self-hosted models
- Document types: PRD (product) and Tech Spec (engineering)
- Interview mode for in-depth requirements gathering
- Claude's active participation in debates alongside opponent models
- Early agreement verification to prevent rubber-stamping
- User review period with change request workflow
- PRD to Tech Spec flow for complete documentation
- Critique focus modes: security, scalability, performance, ux, reliability, cost
- Professional personas: security-engineer, oncall-engineer, junior-developer, qa-engineer, etc.
- Context injection for existing documents
- Session persistence and resume functionality
- Auto-checkpointing for rollback capability
- Preserve intent mode requiring justification for removals
- Cost tracking with per-model breakdown
- Saved profiles for frequently used configurations
- Diff between spec versions
- Export to task list (with JSON output option)
- Telegram integration for async notifications and human-in-the-loop feedback
- Retry with exponential backoff for API resilience
- Response validation warnings for malformed outputs

### Technical

- Modular codebase: debate.py, models.py, providers.py, session.py, prompts.py
- Unit tests with pytest (65 tests)
- CI workflow with linting (ruff), tests, and manifest validation
