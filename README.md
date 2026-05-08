---
title: Agro-Vision
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Agro-Vision

Multi-tenant FastAPI + React platform that wraps a CNN+LSTM crop-variety
classifier. Users sign up, upload a 12-step multispectral GeoTIFF series, and
get back a per-pixel classification map, an AI-assisted health assessment, and
a downloadable multilingual PDF report (English / French / Arabic with RTL).

## Stack

- **Backend**: FastAPI, SQLAlchemy 2.0, Alembic, JWT-cookie auth, SSE for live
  progress + notifications + messaging.
- **Frontend**: React + Vite + i18next + Tailwind, served by FastAPI as a
  built bundle under `app/static/frontend/`.
- **ML**: PyTorch CNN+LSTM in `ml/`, weights at `ml/weights/cnn_lstm.pth`.
- **DB**: SQLite locally; libSQL/Turso on Hugging Face Spaces (set via
  `DATABASE_URL` env var).
- **Hardening**: slowapi rate limits on auth + uploads, security-header
  middleware (CSP, X-Frame-Options, etc.), loguru-based JSON logs in prod.

## Local development

```bash
cp .env.example .env
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
alembic upgrade head

# Dev mode (two ports): backend on 7860, Vite on 5173 with /api proxy
uvicorn app.main:app --reload --host 127.0.0.1 --port 7860 &
cd frontend && npm install && npm run dev
```

Single-port (mirrors HF Spaces / Docker):

```bash
cd frontend && npm run build && cd ..
rm -rf app/static/frontend && mkdir -p app/static/frontend
cp -r frontend/dist/* app/static/frontend/
uvicorn app.main:app --host 127.0.0.1 --port 7860
```

Open http://localhost:7860.

Seed an admin (needed before non-admin users can send messages):

```bash
python scripts/create_admin.py admin@example.com <password> Admin
```

## Environment

See `.env.example` for the full list. Key vars:

| var | required | notes |
| --- | --- | --- |
| `JWT_SECRET` | yes | long random string in prod |
| `COOKIE_SECURE` | yes (prod) | `true` over HTTPS |
| `DATABASE_URL` | yes | `sqlite:///./agrovision.db` locally; libSQL URL in Spaces |
| `TURSO_AUTH_TOKEN` | only if libSQL | from your Turso dashboard |
| `HEALTH_AI_PROVIDER` | no | `heuristic` (default), `gemini`, or `groq` |
| `GEMINI_API_KEY` / `GROQ_API_KEY` | only if used | provider key |
| `DEV_STAGE_DELAY_S` | no | dev-only artificial pause between inference stages |

## Docker / Hugging Face Spaces

The `Dockerfile` is multi-stage (node build → python runtime). HF Spaces
metadata is in this README's front-matter. The container entrypoint runs
`alembic upgrade head` then `uvicorn` on port `7860`.

```bash
docker build -t agrovision .
docker run -p 7860:7860 -e DATABASE_URL=sqlite:///./agrovision.db \
    -e JWT_SECRET=$(openssl rand -hex 32) agrovision
```

Model weights (`ml/weights/cnn_lstm.pth`) ship via Git LFS — `.gitattributes`
pins `*.pth` so a Spaces clone pulls them automatically.

## Phases shipped

1. Auth + inference end-to-end.
2. AI health advisor (Gemini / Groq / heuristic fallback).
3. Reports + farmer notes + PDF (EN/FR/AR with Arabic shaping).
4. Notifications (model + per-user SSE + chime).
5. Messaging (1:1 user ↔ admin with image attachments + chime + nav badge).
6. Public landing page, security headers, rate limits, loguru, HF metadata.

## License

For research / demo use. The CNN+LSTM training data and weights belong to
their respective authors.
