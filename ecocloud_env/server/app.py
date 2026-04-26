"""FastAPI application entrypoint for the CloudEdge environment."""

from openenv.core.env_server import create_app

from .environment import EcoCloudEnvironment
from ..models import CloudAction, CloudObservation

env = EcoCloudEnvironment()
app = create_app(env, CloudAction, CloudObservation)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a simple liveness payload."""
    return {"status": "ok", "env": "cloudedge"}

