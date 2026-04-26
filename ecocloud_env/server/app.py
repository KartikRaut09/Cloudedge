"""FastAPI application entrypoint for the CloudEdge environment."""

from pathlib import Path

from openenv.core.env_server import create_app
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .environment import EcoCloudEnvironment
from ..models import CloudAction, CloudObservation

app = create_app(EcoCloudEnvironment, CloudAction, CloudObservation)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a simple liveness payload."""
    return {"status": "ok", "env": "cloudedge"}


# --- Serve the visual dashboard ---
DASHBOARD_DIR = Path(__file__).resolve().parent.parent.parent / "dashboard"

if DASHBOARD_DIR.exists():
    @app.get("/")
    def dashboard_index():
        """Serve the dashboard homepage."""
        return FileResponse(DASHBOARD_DIR / "index.html")

    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR)), name="dashboard")
