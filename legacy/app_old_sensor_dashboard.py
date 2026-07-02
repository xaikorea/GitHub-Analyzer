import os
import sys
import time
import subprocess
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent


def parse_patterns(text: str) -> list[str]:
    """
    Streamlit text_area 입력값을 CLI 패턴 리스트로 변환.
    줄바꿈, 쉼표 둘 다 허용.
    """
    if not text:
        return []

    raw = []
    for line in text.replace(",", "\n").splitlines():
        item = line.strip()
        if item:
            raw.append(item)
    return raw


def find_latest_result_dir(output_base: Path) -> Path | None:
    """
    output_base 아래에서 가장 최근 생성/수정된 결과 폴더를 찾는다.
    main.py는 보통 output_base/project_name/index.md 형태로 저장한다.
    """
    if not output_base.exists():
        return None

    candidates = [
        p for p in output_base.iterdir()
        if p.is_dir() and (p / "index.md").exists()
    ]

    if not candidates:
        if (output_base / "index.md").exists():
            return output_base
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_tutorial_generation(
    repo_url: str,
    provider: str,
    api_key: str,
    github_token: str,
    model: str,
    base_url: str,
    include_patterns: list[str],
    exclude_patterns: list[str],
    max_size: int,
    language: str,
    max_abstractions: int,
    output_dir: str,
    no_cache: bool,
):
    """
    기존 main.py를 subprocess로 실행한다.
    기존 프로젝트 코드를 직접 import하지 않기 때문에 충돌 가능성이 낮다.
    """
    env = os.environ.copy()

    # .env 기본값 로드
    load_dotenv(BASE_DIR / ".env", override=True)

    # 현재 프로세스 env도 반영
    env.update(os.environ)

    # Streamlit 입력값으로 provider/env 덮어쓰기
    provider = provider.upper().strip()
    env["LLM_PROVIDER"] = provider

    if provider == "GEMINI":
        if api_key:
            env["GEMINI_API_KEY"] = api_key
        if model:
            env["GEMINI_MODEL"] = model

    elif provider == "OPENAI":
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        if model:
            env["OPENAI_MODEL"] = model
        if base_url:
            env["OPENAI_BASE_URL"] = base_url.rstrip("/")

    if github_token:
        env["GITHUB_TOKEN"] = github_token.strip()

    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "-u",
        "main.py",
        "--repo",
        repo_url,
        "--max-size",
        str(max_size),
        "--language",
        language,
        "--max-abstractions",
        str(max_abstractions),
        "--output",
        output_dir,
    ]

    if include_patterns:
        cmd.append("--include")
        cmd.extend(include_patterns)

    if exclude_patterns:
        cmd.append("--exclude")
        cmd.extend(exclude_patterns)

    if no_cache:
        cmd.append("--no-cache")

    process = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    return process, cmd


st.set_page_config(
    page_title="Codebase Tutorial Generator",
    page_icon="📘",
    layout="wide",
)

st.title("📘 Codebase Tutorial Generator")
st.caption("GitHub 저장소 URL을 입력하면 PocketFlow Tutorial Codebase Knowledge를 실행해서 한국어 튜토리얼을 생성합니다.")

with st.sidebar:
    st.header("LLM 설정")

    provider = st.selectbox(
        "Provider",
        ["GEMINI", "OPENAI"],
        index=0,
    )

    if provider == "GEMINI":
        default_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        model = st.text_input("Gemini Model", value=default_model)
        base_url = ""
        api_key = st.text_input(
            "Gemini API Key",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
        )

    else:
        default_model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
        model = st.text_input("OpenAI Model", value=default_model)
        base_url = st.text_input(
            "OpenAI Base URL",
            value=os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
        )
        api_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )

    github_token = st.text_input(
        "GitHub Token",
        value=os.getenv("GITHUB_TOKEN", ""),
        type="password",
        help="public repo도 rate limit 방지를 위해 권장합니다.",
    )

st.subheader("1. 분석 대상 GitHub 저장소")

repo_url = st.text_input(
    "GitHub Repository URL",
    value="https://github.com/fastapi/fastapi/tree/master/fastapi",
    placeholder="https://github.com/owner/repo 또는 https://github.com/owner/repo/tree/main/src",
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    language = st.text_input("생성 언어", value="Korean")

with col_b:
    max_size = st.number_input(
        "파일 최대 크기(byte)",
        min_value=1000,
        max_value=500000,
        value=60000,
        step=5000,
    )

with col_c:
    max_abstractions = st.number_input(
        "최대 추상화 개수",
        min_value=3,
        max_value=20,
        value=8,
        step=1,
    )

st.subheader("2. 파일 필터")

col1, col2 = st.columns(2)

with col1:
    include_text = st.text_area(
        "Include patterns",
        value="*.py\n*.md\n*.toml\n*.yml\n*.yaml",
        height=160,
        help="한 줄에 하나씩 입력. 예: *.py",
    )

with col2:
    exclude_text = st.text_area(
        "Exclude patterns",
        value="tests/*\ndocs/*\n.github/*\ndocs_src/*\nscripts/*\n__pycache__/*",
        height=160,
        help="한 줄에 하나씩 입력. 예: tests/*",
    )

st.subheader("3. 출력 설정")

col_out1, col_out2 = st.columns(2)

with col_out1:
    output_dir = st.text_input("Output directory", value="streamlit_output")

with col_out2:
    no_cache = st.checkbox("LLM cache 끄기 --no-cache", value=True)

run_button = st.button("🚀 튜토리얼 생성 실행", type="primary")

if run_button:
    if not repo_url.strip():
        st.error("GitHub 저장소 URL을 입력해야 합니다.")
        st.stop()

    if not api_key:
        st.warning("API Key가 비어 있습니다. .env에 키가 있으면 동작할 수 있지만, 없으면 실패합니다.")

    include_patterns = parse_patterns(include_text)
    exclude_patterns = parse_patterns(exclude_text)

    st.write("실행 설정")
    st.json(
        {
            "provider": provider,
            "model": model,
            "repo_url": repo_url,
            "include": include_patterns,
            "exclude": exclude_patterns,
            "max_size": max_size,
            "language": language,
            "max_abstractions": max_abstractions,
            "output_dir": output_dir,
            "no_cache": no_cache,
        }
    )

    log_box = st.empty()
    status_box = st.empty()

    process, cmd = run_tutorial_generation(
        repo_url=repo_url.strip(),
        provider=provider,
        api_key=api_key.strip(),
        github_token=github_token.strip(),
        model=model.strip(),
        base_url=base_url.strip(),
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        max_size=int(max_size),
        language=language.strip(),
        max_abstractions=int(max_abstractions),
        output_dir=output_dir.strip(),
        no_cache=no_cache,
    )

    st.code(" ".join(cmd), language="powershell")

    logs = []
    status_box.info("실행 중입니다. 저장소 크기와 LLM 속도에 따라 시간이 걸립니다.")

    while True:
        line = process.stdout.readline()
        if line:
            logs.append(line.rstrip())
            log_box.code("\n".join(logs[-300:]), language="text")

        if process.poll() is not None:
            remaining = process.stdout.read()
            if remaining:
                logs.extend(remaining.splitlines())
                log_box.code("\n".join(logs[-300:]), language="text")
            break

        time.sleep(0.05)

    return_code = process.returncode

    if return_code == 0:
        status_box.success("튜토리얼 생성 완료")

        result_dir = find_latest_result_dir(BASE_DIR / output_dir.strip())

        if result_dir:
            st.subheader("4. 생성 결과")
            st.write(f"결과 폴더: `{result_dir}`")

            index_path = result_dir / "index.md"
            if index_path.exists():
                st.markdown("### index.md")
                st.markdown(index_path.read_text(encoding="utf-8", errors="replace"))

            chapter_files = sorted(
                p for p in result_dir.glob("*.md")
                if p.name != "index.md"
            )

            if chapter_files:
                selected = st.selectbox(
                    "챕터 파일 선택",
                    [p.name for p in chapter_files],
                )
                selected_path = result_dir / selected
                st.markdown(f"### {selected}")
                st.markdown(selected_path.read_text(encoding="utf-8", errors="replace"))
        else:
            st.warning("실행은 완료됐지만 결과 폴더를 찾지 못했습니다. output 경로를 직접 확인하세요.")

    else:
        status_box.error(f"실행 실패: return code {return_code}")
        st.warning("위 로그의 마지막 Traceback을 확인하세요.")