# Deploying Agro-Vision

This guide assumes you received `docker-compose.yml`, `.env.prod.example`,
and a GitHub Personal Access Token granting read access to the
`ghcr.io/emdcatalyst/agrovision` container image. You do **not** need
git, Node, or Python on the server — only Docker.

## 1. Prerequisites

- A Linux server with Docker Engine 24+ and the Compose plugin
  (`docker compose version` should print a v2 string).
- ~2.5 GB free disk for the image, plus whatever you'll need for
  uploads and rendered maps.
- (Recommended) A public domain pointed at the server and a reverse
  proxy in front (Caddy or nginx) for HTTPS — see §6.

## 2. Authenticate to the image registry

The image is hosted in GitHub Container Registry as a private package.
Log Docker in once using the read-only token you were given:

```bash
echo "<YOUR_TOKEN>" | docker login ghcr.io -u <YOUR_GITHUB_USERNAME> --password-stdin
```

The credentials are stored in `~/.docker/config.json` and persist across
reboots. Pulling the image happens automatically on the first
`docker compose up`.

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

When a new image version is published:

```bash
docker compose pull         # downloads only the changed layers
docker compose up -d        # recreates the container with the new image
```

Layer-diff pulls mean a typical update transfers tens of megabytes,
not the full image. User data (DB, uploads, maps) lives in the
bind-mounted directories and survives container recreation. Migrations
run automatically on startup.

## 8. Backups

The only stateful directories are `./data`, `./uploads`, and
`./maps`. A nightly tar of `/opt/agrovision/{data,uploads,maps}` to
S3 (or any offsite location) is sufficient.

## Troubleshooting

- **`docker compose up` fails with "denied" or "unauthorized" pulling
  the image** — your GHCR login expired or the token doesn't have
  `read:packages` scope. Re-run the `docker login ghcr.io ...` step
  from §2 with a fresh token.
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
