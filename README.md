AI_QUERY_APP
Simple Flask web app that sends a question to an external LLM (Gemini) and shows the answer. Uses SQLite to log queries.
## Run locally
1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt