from fastapi import APIRouter

from . import analyses as analyses_admin
from . import notifications as notifications_admin
from . import stats as stats_admin
from . import users as users_admin

router = APIRouter(prefix="/admin", tags=["admin"])
router.include_router(stats_admin.router)
router.include_router(users_admin.router)
router.include_router(analyses_admin.router)
router.include_router(notifications_admin.router)
