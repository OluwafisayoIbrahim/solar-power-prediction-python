"""Export current research charts to results/plots for website use."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/export_portfolio_plots.py"), run_name="__main__")
