# GitHub Analyzer — 팀 사용 가이드

> 팀원이 이 도구를 **바로 쓰고 필요하면 직접 실행**할 수 있도록 정리한 문서입니다.
> 배포/운영 절차는 [`DEPLOYMENT.md`](./DEPLOYMENT.md)를, 기능 상세는 [`README.md`](./README.md)를 참고하세요.

---

## 1. 이게 뭔가요?

어떤 GitHub 저장소든 넣으면 **초보자용 튜토리얼(챕터 + 다이어그램)** 을 자동 생성하고,
그 결과를 **Postgres/pgvector DB**에 저장한 뒤 **검색·대화·시각화**할 수 있는 내부 도구입니다.

- **Generate & Save** — repo → 튜토리얼 생성 + DB 저장
- **RAG Search / Chat** — 저장된 코드베이스 지식으로 질문·대화
- **Multi‑Repo RAG / Agent** — 여러 repo 교차 검색, 에이전트 심층 탐색
- **Ontology** — 개념 그래프 시각화
- **Fine‑tune** — 코드 생성용 학습 데이터셋 생성

---

## 2. 빠르게 접속하기 (배포된 앱)

1. 팀 배포 URL 접속: **`https://<우리팀>.streamlit.app`** *(실제 URL은 팀 채널 공지 참고)*
2. **로그인 화면**에서 팀 계정으로 로그인
   - 아이디 / 비밀번호는 **팀 리드 또는 사내 비밀번호 관리자**에게 요청하세요.
   - ⚠️ 로그인 정보는 이 문서·코드·채팅에 남기지 마세요.
3. 로그인 후 왼쪽 사이드바에서 **저장된 튜토리얼**을 고르고 각 탭을 사용합니다.
4. 사용을 마치면 사이드바 **🚪 로그아웃**.

> 사이드바에는 API 키·DB 접속 정보가 **값 없이 "설정됨" 상태로만** 표시됩니다(보안). 정상입니다.

---

## 3. 로컬에서 직접 실행하기 (개발/디버깅)

### 사전 준비
- Python **3.10 이상**
- 접근 정보 한 세트: LLM 키(**Gemini** 또는 **OpenAI**), **Supabase `DATABASE_URL`**,
  (권장) **GitHub 토큰** — 팀 리드에게 요청

### 설치
```bash
git clone https://github.com/xaikorea/GitHub-Analyzer.git
cd GitHub-Analyzer
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 환경변수 (`.env`)
`.env.sample`을 복사해 `.env`를 만들고 값을 채웁니다. **`.env`는 절대 커밋하지 마세요**(`.gitignore`로 이미 제외됨).
```bash
cp .env.sample .env
```
최소로 채워야 하는 항목:
```env
LLM_PROVIDER=GEMINI
GEMINI_API_KEY=...            # 또는 OPENAI_API_KEY (LLM_PROVIDER=OPENAI)
DATABASE_URL=postgresql://...:5432/postgres?sslmode=require   # Supabase
GITHUB_TOKEN=...             # 권장(레이트리밋/비공개 repo)
APP_USERNAME=...             # 로컬 로그인 게이트(비우면 게이트 꺼짐)
APP_PASSWORD=...
# RAG_CREATE_EMBEDDINGS=1    # 시맨틱(벡터) 검색까지 쓰려면 1 (OPENAI_API_KEY 필요, 비용 발생)
```

### DB 초기화 (최초 1회, 이미 초기화됐으면 생략)
```bash
psql "$DATABASE_URL" -f schema.sql   # 멱등: 다시 돌려도 안전
```

### 실행
```bash
streamlit run app_full_workflow.py
```
브라우저에서 `http://localhost:8501` → 로그인 → 사용.

---

## 4. 탭별 사용법 (요약)

| 탭 | 무엇을 하나 |
|---|---|
| ⚙️ **Setup** | 키·DB·토큰 연결 상태 확인 |
| ➕ **Generate** | repo URL 입력 → 분석 → DB 저장. URL이 전체/브랜치/하위폴더 중 무엇인지 미리 보여줌 |
| 📚 **Library** | 저장된 튜토리얼의 요약·플로우차트·챕터 열람 |
| 🔎 **RAG Search** | 질문 → 관련 청크 검색 + (선택)LLM 답변, 근거·프롬프트 노출 |
| 💬 **Chat** | 선택한 튜토리얼 지식으로 멀티턴 스트리밍 대화 |
| 🔗 **Multi‑Repo RAG** | 여러 repo 교차 검색 + 공통점/차이 합성 |
| 🤖 **Agent** | ReAct / Deep Research 모드로 에이전트 탐색, Judge 검증, 트레이스 |
| 🕸 **Ontology** | 개념 그래프 재구성·검색, 깨진 Mermaid 자동 복구 |
| 🛠 **Fine‑tune** | 코드 생성 데이터셋(JSONL) 생성 + 학습 명령 (학습·추론은 로컬 GPU 전용) |
| 🗑 **Admin** | 튜토리얼/레포 전체 삭제(미리보기 + DELETE 확인) |

---

## 5. 자주 겪는 문제

| 증상 | 원인 / 해결 |
|---|---|
| 로그인 화면이 안 뜨고 "🔓 로그인 미설정" 경고 | `APP_USERNAME`/`APP_PASSWORD` 미설정 → `.env`(로컬) 또는 Secrets(배포)에 설정 |
| Library/RAG 탭에 "저장된 튜토리얼 없음" | `DATABASE_URL` 미설정이거나 아직 아무 repo도 저장 안 함 → Generate에서 저장 |
| 검색 점수가 다 비슷하게 나옴 | 임베딩이 꺼져 **키워드 검색** 중. 시맨틱 원하면 `RAG_CREATE_EMBEDDINGS=1` 후 재저장 |
| Generate가 큰 repo에서 느리거나 멈춤 | include/exclude·max‑size로 범위 축소, 또는 `/tree/<branch>/<하위폴더>` URL 사용 |
| Fine‑tune "코드 생성 실행"이 배포 앱에서 실패 | 정상. torch/transformers는 **로컬 GPU 전용**(`requirements-train.txt`). 데이터셋 생성까지는 됨 |

---

## 6. 보안 수칙 (팀 공통, 꼭 지켜주세요)

- **`.env`·키·비밀번호를 커밋 금지.** 공유는 사내 안전 채널/비밀번호 관리자로만.
- 로컬에서 생성되는 `llm_cache.json`, `logs/`, `output/`, `exports/` 도 커밋하지 않습니다(이미 `.gitignore`).
- 코드 변경은 **브랜치 → PR** 권장(회사 repo).
- 키가 노출된 것 같으면 **즉시 재발급(rotate)** 하고 팀 리드에게 알리세요.

---

## 7. 도움 요청

- 기능/구조 상세: [`README.md`](./README.md)
- 배포·운영: [`DEPLOYMENT.md`](./DEPLOYMENT.md)
- 그 외 문제: 팀 채널에 **재현 방법 + 화면/로그**와 함께 공유
