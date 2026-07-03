# GitHub Analyzer — 배포 가이드

이 앱은 **DB(Supabase)** 와 **웹앱(Streamlit Community Cloud)** 을 분리해 배포합니다.
Supabase는 Python/Streamlit 앱을 직접 올리는 곳이 아니라 **Postgres + pgvector 저장소**로 씁니다.

```
[Streamlit Community Cloud]  ──DATABASE_URL──▶  [Supabase Postgres + pgvector]
  app_full_workflow.py                            repositories / tutorials / chapters
  repo 크롤링 · LLM 호출 · RAG UI                   chunks(+embedding) / ontology / traces
```

---

## 1. Supabase (DB / RAG 저장소)

1. **프로젝트 생성** — Region은 가까운 곳(예: `ap-northeast`/`ap-southeast`), Plan은 Free로 시작 가능.
2. **스키마 적용** — 이 repo의 `schema.sql`을 그대로 실행합니다.
   - SQL Editor에 `schema.sql` 내용을 붙여넣고 실행, **또는** 로컬에서:
     ```bash
     psql "$DATABASE_URL" -f schema.sql
     ```
   - `schema.sql` 1행이 `CREATE EXTENSION IF NOT EXISTS vector` 라 **pgvector가 자동 활성화**됩니다.
   - 임베딩 차원은 이미 **`vector(1536)`** (OpenAI `text-embedding-3-small`)로 맞춰져 있어 수정 불필요.
3. **연결 문자열(`DATABASE_URL`)** — Project → Connect에서 복사.
   - Streamlit 같은 **오래 떠 있는 서버**에는 **Session pooler(포트 5432)** 를 권장합니다.
   - 끝에 **`?sslmode=require`** 를 붙이세요:
     ```
     postgresql://postgres.<ref>:<PW>@aws-<region>.pooler.supabase.com:5432/postgres?sslmode=require
     ```
   - 비밀번호에 `@ # % / :` 등 특수문자가 있으면 URL 인코딩이 필요합니다.

---

## 2. Streamlit Community Cloud (웹앱)

share.streamlit.io → **Create app** → "I have an app":

| 항목 | 값 |
|---|---|
| Repository | `xaikorea/GitHub-Analyzer` |
| Branch | `main` |
| **Main file path** | **`app_full_workflow.py`** ← 이 값이 핵심 |
| Advanced settings → **Secrets** | 아래 TOML (실제 값으로 채움) |

```toml
LLM_PROVIDER = "GEMINI"
GEMINI_API_KEY = "..."
GEMINI_MODEL = "gemini-2.5-flash"

OPENAI_API_KEY = "..."
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_BASE_URL = "https://api.openai.com"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

GITHUB_TOKEN = "..."
DATABASE_URL = "postgresql://...:5432/postgres?sslmode=require"

# 팀 전용 로그인 게이트 (둘 다 설정해야 로그인 화면이 뜸)
APP_USERNAME = "..."
APP_PASSWORD = "..."

# 처음엔 0(키워드 검색)으로 시작, 정상 확인 후 1(시맨틱)로
RAG_CREATE_EMBEDDINGS = "0"
MAX_ABSTRACTION_CONTEXT_CHARS = "900000"
```

- **Secrets는 repo에 커밋되지 않습니다.** 실제 키/비번은 여기에만 넣고, `.env`는 절대 push하지 마세요.
- `Deploy`를 누르면 `https://<name>.streamlit.app` URL이 생성됩니다.

### 배포 권한
- 앱을 만들려면 로그인한 GitHub 계정이 **`xaikorea` 조직 접근 권한**을 가져야 합니다.
- 코드 수정 반영은 **브랜치 → PR → main 머지** 방식을 권장합니다(회사 repo).

---

## 3. 배포 후 접근 제한

- **로그인 게이트**(`APP_USERNAME`/`APP_PASSWORD`)가 1차 방어입니다.
- `*.streamlit.app`은 기본적으로 URL을 아는 누구나 접근 가능하므로, 앱 설정에서
  **뷰어 이메일 화이트리스트**를 걸어 2차 방어를 두는 것을 권장합니다.

---

## 4. 업데이트 방법

```bash
git checkout -b feature/xxx
# ...수정...
git commit -m "..."
git push origin feature/xxx      # → PR 생성 → 리뷰 → main 머지
```
`main`이 갱신되면 Streamlit Cloud가 **자동 재배포**합니다.

---

## 5. 주의사항 (이 앱 특성)

- **Fine‑tune 학습/추론은 클라우드에서 동작하지 않습니다.** torch/transformers는 무거워
  `requirements-train.txt`에만 있고 **로컬 GPU 전용**입니다(데이터셋 JSONL 생성까지는 클라우드도 가능).
- **Generate 탭**은 내부에서 `main.py`를 subprocess로 실행해 repo를 크롤링+LLM 호출합니다.
  무료 플랜(RAM 1GB)에선 **큰 repo가 메모리/시간 한도**에 걸릴 수 있으니 작은 repo/하위폴더 위주로.
- 로그인은 **단일 공유 계정**입니다. 개인별 계정·감사로그가 필요하면 별도 인증(SSO 등)을 추가하세요.

---

## 6. 트러블슈팅

| 오류 | 해결 |
|---|---|
| `ModuleNotFoundError` | 누락 패키지를 `requirements.txt`에 추가 후 push. (이 repo는 `psycopg[binary]`(psycopg3)·`google-genai` 사용 — `psycopg2`/`openai` 라이브러리 아님) |
| `connection timeout` / `could not connect` | `DATABASE_URL`을 Direct가 아닌 **pooler** 주소로 사용 |
| SSL 관련 오류 | `DATABASE_URL` 끝에 `?sslmode=require` |
| `invalid api key` / `authentication failed` | Secrets 값에 따옴표·공백·줄바꿈이 섞였는지 확인 |
| 로그인 화면이 없이 열림 | `APP_USERNAME`/`APP_PASSWORD` 둘 다 Secrets에 설정됐는지 확인 |
