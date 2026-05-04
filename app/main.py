from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import analyses as analyses_api
from .api import auth as auth_api
from .api import sse as sse_api
from .core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title=s.app_name, lifespan=lifespan)

    app.include_router(auth_api.router, prefix=s.api_prefix)
    app.include_router(analyses_api.router, prefix=s.api_prefix)
    app.include_router(sse_api.router, prefix=s.api_prefix)

    s.maps_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static/maps", StaticFiles(directory=str(s.maps_dir)), name="maps")

    frontend_dist = s.static_dir / "frontend"
    if frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str):
            target = frontend_dist / full_path
            if full_path and target.is_file():
                return FileResponse(target)
            index = frontend_dist / "index.html"
            if index.exists():
                return FileResponse(index)
            return {"app": s.app_name, "status": "ok", "frontend": "missing"}
    else:

        @app.get("/")
        async def root():
            return {"app": s.app_name, "status": "ok", "frontend": "not built"}

    return app


app = create_app()
