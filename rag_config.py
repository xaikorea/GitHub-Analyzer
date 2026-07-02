"""
RAG search tuning config — edit this per target repository.

`search_tutorial_context_v4` in db_store.py uses these to boost matches
for domain concepts. The defaults below are tuned for FastAPI / requests /
pydantic / smolagents tutorials. When you generate tutorials for a
DIFFERENT codebase, add that project's concepts here (or replace the dict)
so keyword search still ranks the right chapters/chunks first.

- GENERIC_QUERY_TERMS: words too generic to be useful as search terms
  (project names, filler words). They are dropped from the query.
- CONCEPT_ALIASES: concept_key -> list of aliases (Korean/English/code
  tokens). A query hitting any alias boosts chunks/chapters containing
  related aliases.

Nothing here is required: if this module is missing or a concept is not
listed, search still works — it just loses the domain-specific boosts and
falls back to plain term matching (and, when embeddings exist, to vector
search which is repo-agnostic).
"""

GENERIC_QUERY_TERMS = {
    "fastapi", "pocketflow", "requests", "pydantic", "smolagents",
    "초보자", "설명", "설명해줘", "알려줘", "무엇", "뭐야", "어떻게",
    "사용법", "개념", "역할", "정리", "예시", "코드", "기반",
    "the", "a", "an", "of", "to", "and", "in", "for", "with",
}

CONCEPT_ALIASES = {
    "dependency_injection": [
        "의존성", "의존성 주입", "주입", "dependency", "dependency injection",
        "depends", "Depends", "DI", "injection",
    ],
    "router": [
        "라우터", "router", "APIRouter", "include_router", "routing", "route",
    ],
    "middleware": [
        "미들웨어", "middleware", "request", "response", "before", "after",
    ],
    "pydantic_model": [
        "pydantic", "pydantic model", "모델", "검증", "validation",
        "BaseModel", "schema", "type hint",
    ],
    "response_model": [
        "응답 모델", "response model", "response", "응답", "serialize", "serialization",
    ],
    "streaming": [
        "스트리밍", "streaming", "StreamingResponse", "EventSourceResponse", "SSE",
    ],
    "path_operation": [
        "경로 작동", "path operation", "endpoint", "엔드포인트",
        "@app.get", "@app.post", "decorator",
    ],
    "application_instance": [
        "애플리케이션", "application", "instance", "인스턴스",
        "FastAPI()", "app = FastAPI",
    ],
    "session": [
        "session", "세션", "cookie", "connection pooling", "persistent",
    ],
    "cookie": [
        "cookie", "cookies", "쿠키", "cookie jar", "RequestsCookieJar",
    ],
    "auth": [
        "auth", "authentication", "인증", "HTTPBasicAuth", "HTTPDigestAuth",
    ],
    "exception": [
        "exception", "예외", "error", "timeout", "ConnectionError", "HTTPError",
    ],
    "transport_adapter": [
        "transport adapter", "adapter", "HTTPAdapter", "retries", "pool",
    ],
    "hook": [
        "hook", "hooks", "response hook", "dispatch_hook",
    ],
    "basemodel": [
        "BaseModel", "베이스모델", "data blueprint", "model_validate", "model_dump",
    ],
    "field": [
        "Field", "FieldInfo", "필드", "alias", "default", "constraint",
    ],
    "config": [
        "ConfigDict", "ConfigWrapper", "configuration", "설정", "model_config",
    ],
    "validator": [
        "validator", "serializer", "field_validator", "model_validator",
        "field_serializer", "Annotated", "custom logic",
    ],
    "core_schema": [
        "core schema", "CoreSchema", "pydantic-core", "validation", "serialization",
    ],
    "type_adapter": [
        "TypeAdapter", "type adapter", "validate_python", "validate_json",
    ],
    "multistep_agent": [
        "MultiStepAgent", "multi step", "ReAct", "think", "act", "observe",
    ],
    "model_interface": [
        "model interface", "LiteLLMModel", "LLM", "ChatMessage",
    ],
    "tool": [
        "tool", "tools", "forward", "도구", "tool call",
    ],
    "agent_memory": [
        "AgentMemory", "memory", "메모리", "ActionStep", "TaskStep",
    ],
    "prompt_templates": [
        "PromptTemplates", "prompt", "template", "system_prompt", "Jinja2",
    ],
    "python_executor": [
        "PythonExecutor", "CodeAgent", "LocalPythonExecutor", "DockerExecutor",
        "sandbox", "execute code",
    ],
    "agent_type": [
        "AgentType", "AgentImage", "AgentAudio", "AgentText", "multimodal",
    ],
    "logger_monitor": [
        "AgentLogger", "Monitor", "LogLevel", "logging", "token", "duration",
    ],
}
