# Deploying Agro-Vision

This guide assumes you received a prebuilt Docker image tarball
(`agrovision.tar.gz`) plus this repo's `docker-compose.yml` and
`.env.prod.example`. You do **not** need git, Node, or Python on the
server — only Docker.

## 1. Prerequisites

- A Linux server with Docker Engine 24+ and the Compose plugin
  (`docker compose version` should print a v2 string).
- ~3 GB free disk for the image, plus whatever you'll need for
  uploads and rendered maps.
- (Recommended) A public domain pointed at the server and a reverse
  proxy in front (Caddy or nginx) for HTTPS — see §6.

## 2. Load the prebuilt image

Copy `agrovision.tar.gz` to the server, then:

```bash
gunzip -c agrovision.tar.gz | docker load
docker images | grep agrovision   # should list agrovision:latest
```

## 3. Lay out files

On the server, create a directory (e.g. `/opt/agrovision/`) and put
the following inside it:

```
/opt/agrovision/
├── docker-compose.yml      # from the repo
├── .env.prod               # copy of .env.prod.example, edited (see §4)
├── data/                   # auto-created on first run; SQLite DB
├── uploads/                # auto-created; uploaded GeoTIFF series
└── maps/                   # auto-created; rendered classification PNGs
```

## 4. Configure `.env.prod`

```bash
cp .env.prod.example .env.prod
```

Edit `.env.prod` and at minimum set:

| var | what to set |
| --- | --- |
| `JWT_SECRET` | a fresh long secret — `openssl rand -hex 32` |
| `COOKIE_SECURE` | `true` if behind HTTPS, `false` only for LAN/HTTP |
| `SMTP_*` | leave blank to log verification codes to stdout, or fill in your SMTP provider (Brevo / SendGrid / Gmail+App-Password / etc.) |
| `HEALTH_AI_PROVIDER` | `heuristic` (offline default) or `gemini` / `groq` if you want LLM-backed advice — provide the matching `*_API_KEY` |

## 5. First run

```bash
cd /opt/agrovision
docker compose up -d
docker compose logs -f agrovision   # watch alembic + uvicorn boot
```

The container runs `alembic upgrade head` on startup, so DB schema is
created automatically. The app listens on `:7860`.

Seed an initial admin (needed before regular users can send messages):

```bash
docker compose exec agrovision python scripts/create_admin.py \
    admin@yourcompany.com 'a-strong-password' 'Admin Name'
```

Smoke-test: open `http://<server>:7860/`, sign up a test user, run an
inference on the sample GeoTIFFs, download the PDF.

## 6. Putting HTTPS in front (recommended)

Bind the container to localhost only by changing the `ports:` line in
`docker-compose.yml` to `"127.0.0.1:7860:7860"`, then point a Caddy
or nginx vhost at `http://127.0.0.1:7860`. Caddy example:

```caddy
agro.yourcompany.com {
    reverse_proxy 127.0.0.1:7860
    encode gzip
}
```

With HTTPS in place, keep `COOKIE_SECURE=true` in `.env.prod`.

## 7. Updates

When you receive a new image tarball:

```bash
gunzip -c agrovision-NEW.tar.gz | docker load
docker compose up -d        # recreates the container with the new image
```

User data (DB, uploads, maps) lives in the bind-mounted directories
and survives container recreation. Migrations run automatically on
startup.

## 8. Backups

The only stateful directories are `./data`, `./uploads`, and
`./maps`. A nightly tar of `/opt/agrovision/{data,uploads,maps}` to
S3 (or any offsite location) is sufficient.

## Troubleshooting

- **`docker compose up` fails immediately** — check `.env.prod`
  exists and `JWT_SECRET` is set.
- **Browser shows "not secure" or login bounces** — `COOKIE_SECURE=true`
  but you're hitting the site over plain HTTP. Either set up HTTPS
  (§6) or temporarily flip the flag to `false`.
- **Verification emails never arrive** — check `docker compose logs
  agrovision` for the OTP (it falls back to stdout when SMTP is
  unconfigured), or fix your SMTP credentials.
- **Inference is slow** — first request after restart loads the
  PyTorch model into memory; subsequent requests are faster. There is
  no GPU support in this image.
