# Crop-Variety



Crop-Variety is a multi-tenant web platform built around a CNN+LSTM classifier. Users upload a 12-step multispectral GeoTIFF time series and receive a per-pixel classification map, an AI-assisted crop health assessment, and a downloadable multilingual PDF report (English, French, Arabic with RTL support).


https://github.com/user-attachments/assets/b1a6fe84-d1a6-4a1f-8fcd-c3e85c8a2959


## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Model and results](#model-and-results)
- [Dataset and acknowledgements](#dataset-and-acknowledgements)
- [Getting started](#getting-started)
- [Development build](#development-build)
- [Docker](#docker)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Security](#security)
- [License](#license)

## Features

- **End-to-end inference pipeline:** upload a GeoTIFF series, track live progress, and view classification and confidence maps.
- **AI health advisor:** crop health assessment with a pluggable provider (Gemini, Groq, or an offline heuristic fallback).
- **Multilingual reports:** downloadable PDF reports in English, French, and Arabic, with correct Arabic text shaping and RTL layout.
- **Notifications:** per-user real-time notifications over Server-Sent Events.
- **Messaging:** one-to-one user and admin chat with image attachments.
- **Admin panel:** user management and broadcast messages.
- **Authentication:** email verification, password reset via OTP, JWT cookie sessions with token versioning.
- **Public landing page** and a fully internationalized UI.

## Architecture

| Layer | Technology |
| --- | --- |
| Backend | FastAPI, SQLAlchemy 2.0, Alembic, SSE for progress, notifications, and messaging |
| Frontend | React, Vite, Tailwind CSS, i18next |
| ML | PyTorch CNN+LSTM (`ml/`), rasterio for GeoTIFF handling |
| Database | SQLite (default) or libSQL/Turso via `DATABASE_URL` |
| Packaging | Multi-stage Docker build (Node build stage, Python runtime stage) |

In production the React app is built to static files and served by FastAPI, so the whole platform runs as a single process on one port.

## Model and results

Evaluation notes and plots are in `results/`. Model weights are stored at `ml/weights/cnn_lstm.pth` and tracked with Git LFS.

## Dataset and acknowledgements

The classifier is trained and evaluated on data from the **MuST-C** dataset (Multi-Sensor and Multi-Temporal Dataset of Multiple Crops for In-Field Phenotyping and Monitoring), published by the [Institute of Geodesy and Geoinformation and the Photogrammetry & Robotics Lab (IPB)](https://www.ipb.uni-bonn.de/data/MuST-C/) at the **University of Bonn**, as part of the PhenoRob Cluster of Excellence. It contains geo-referenced multi-sensor field data (including multispectral UAV imagery) collected over one growing season across six crop species.

- Dataset page: <https://www.ipb.uni-bonn.de/data/MuST-C/>
- Dataset code: <https://github.com/PRBonn/MuST-C>
- Paper: <https://www.nature.com/articles/s41597-025-06462-y>
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The dataset is **not** redistributed in this repository. To retrain or reproduce the results, download it from the link above.

If you use this work, please also cite the dataset:

```bibtex
@article{chong2026sdata,
  author  = {Chong, Yue Linn and Kr\"amer, Julie and Chakhvashvili, Erekle and Marks, Elias and Esser, Felix and Dreier, Ansgar
             and Rosu, Radu Alexandru and Warstat, Kevin and Pude, Ralf and Behnke, Sven and Muller, Onno and Rascher, Uwe
             and Kuhlmann, Heiner and Stachniss, Cyrill and Behley, Jens and Klingbeil, Lasse},
  title   = {{The Multi-Sensor and Multi-Temporal Dataset of Multiple Crops for In-Field Phenotyping and Monitoring}},
  journal = {Scientific Data},
  volume  = {13},
  year    = 2026,
  doi     = {10.1038/s41597-025-06462-y},
  url     = {https://www.nature.com/articles/s41597-025-06462-y}
}
```

The MuST-C authors acknowledge funding from the Deutsche Forschungsgemeinschaft (DFG) under Germany's Excellence Strategy, EXC-2070 - 390732324 (PhenoRob). This project is independent and is not affiliated with or endorsed by the University of Bonn.

## Getting started

The fastest way to run the project is Docker:

```bash
git lfs install
git clone https://github.com/<your-username>/crop-variety-classification.git
cd crop-variety-classification

cp .env.example .env
# Set JWT_SECRET in .env (see Configuration)

docker compose up --build
```

Open <http://localhost:7860>.

Create an admin account (required before regular users can send messages):

```bash
docker compose exec cropvariety python scripts/create_admin.py admin@example.com <password> Admin
```

## Development build

This setup runs the backend and frontend separately with hot reload.

### Prerequisites

- Python 3.12
- Node.js 20 or newer
- Git and Git LFS (for the model weights)
- On Debian/Ubuntu: `libgomp1` and `libexpat1` (needed by PyTorch and the GDAL bundled with rasterio)

### 1. Backend

```bash
git lfs pull                       # fetch model weights
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate

# CPU-only PyTorch keeps the install small
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
pip install -r requirements.txt

cp .env.example .env               # then set JWT_SECRET
alembic upgrade head               # create the database schema

uvicorn app.main:app --reload --host 127.0.0.1 --port 7860
```

The API is now available at <http://127.0.0.1:7860>, with interactive docs at `/docs`.

### 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Vite serves the UI at <http://localhost:5173> and proxies `/api` requests to the backend on port 7860.

### 3. Seed an admin user

```bash
python scripts/create_admin.py admin@example.com <password> Admin
```

### 4. Production-like build (single port)

To test exactly what the Docker image serves:

```bash
cd frontend && npm run build && cd ..
rm -rf app/static/frontend && mkdir -p app/static/frontend
cp -r frontend/dist/* app/static/frontend/
uvicorn app.main:app --host 127.0.0.1 --port 7860
```

### Email in development

Signup verification and password reset send emails. For local development you do not need a mail account: leave `SMTP_USER` and `SMTP_PASS` unset (or remove them) in `.env`, and the app will not attempt real delivery.

To inspect actual messages, you can optionally run [Mailpit](https://github.com/axllent/mailpit), a fake SMTP server with a web inbox, and point the SMTP host and port at it.

### Database migrations

```bash
alembic revision --autogenerate -m "describe_the_change"   # create a migration
alembic upgrade head                                       # apply migrations
alembic downgrade -1                                       # roll back one step
```

Always review autogenerated migrations before committing them.

### Development tips

- Set `DEV_STAGE_DELAY_S` (for example `1.5`) to slow down inference stages so the live progress UI is easy to observe.
- Without a `GEMINI_API_KEY` or `GROQ_API_KEY`, the health advisor falls back to the built-in heuristic provider, so no external accounts are needed to develop.
- Uploaded series and rendered maps are written to `uploads/` and `app/static/maps/`. Both are git-ignored.

## Docker

The `Dockerfile` is multi-stage: it builds the frontend with Node, then assembles a slim Python runtime. On startup the container applies migrations and launches Uvicorn on port 7860.

With Compose (builds from source):

```bash
docker compose up --build
```

With plain Docker:

```bash
docker build -t cropvariety .
docker run -p 7860:7860 --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/uploads:/app/uploads" \
  cropvariety
```

Mounted volumes keep the SQLite database, uploaded series, and rendered maps across container restarts.

## Configuration

All settings are environment variables. See `.env.example` for the complete list.

| Variable | Required | Description |
| --- | --- | --- |
| `JWT_SECRET` | Yes | Long random string. Generate with `openssl rand -hex 32`. |
| `DATABASE_URL` | Yes | `sqlite:///./cropvariety.db` locally, `sqlite:////app/data/cropvariety.db` in Docker, or a libSQL URL. |
| `COOKIE_SECURE` | Production | Set to `true` when served over HTTPS. |
| `TURSO_AUTH_TOKEN` | If using libSQL | Token from your Turso dashboard. |
| `HEALTH_AI_PROVIDER` | No | `heuristic` (default), `gemini`, or `groq`. |
| `GEMINI_API_KEY` / `GROQ_API_KEY` | If used | API key for the selected provider. |
| `DEV_STAGE_DELAY_S` | No | Development only: artificial pause between inference stages. |

Never commit your `.env` file.

## Project structure

```
.
├── app/            FastAPI application (routes, models, services, static assets)
├── frontend/       React + Vite source
├── ml/             CNN+LSTM model code and weights
├── alembic/        Database migrations
├── scripts/        Utility scripts (admin seeding, etc.)
├── metadata/       Dataset and class metadata
├── results/        Evaluation results and plots
├── cli.py          Command-line entry point
├── Dockerfile
└── docker-compose.yml
```

## Security

- Rate limiting on authentication and upload endpoints (slowapi).
- Security headers middleware (CSP, X-Frame-Options, and others).
- HTTP-only JWT cookies with token versioning for session invalidation.
- Structured JSON logging with loguru in production.

## License

The source code is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use is not permitted without a separate written agreement with the author.

The MuST-C dataset is licensed separately under CC BY 4.0 by its authors at the University of Bonn. See [Dataset and acknowledgements](#dataset-and-acknowledgements).
