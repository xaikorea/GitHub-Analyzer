FROM python:3.10-slim

# update packages, install git and remove cache
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the CLI tutorial generator.
#   docker run --rm -e GEMINI_API_KEY=... image --repo https://github.com/...
ENTRYPOINT ["python", "main.py"]

# To run the Streamlit app instead, override the entrypoint, e.g.:
#   docker run --rm -p 8501:8501 -e DATABASE_URL=... --entrypoint streamlit \
#     image run app_full_workflow.py --server.address 0.0.0.0
# (The Streamlit app also needs a reachable Postgres with pgvector; see schema.sql.)
