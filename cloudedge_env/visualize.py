"""Backward-compatible entrypoint for the EcoCloud training report."""

from __future__ import annotations

try:
    from training_report import main
except ImportError:
    from ecocloud_env.training_report import main


if __name__ == "__main__":
    main()
