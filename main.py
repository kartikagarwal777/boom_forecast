import warnings
import logging

# Suppress all warnings issued by the warnings module so they don't print to
# stderr during runs. We also explicitly ignore DeprecationWarning which some
# third-party libraries frequently emit.
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Reduce logging verbosity globally and for noisy libraries to avoid warning
# / info logs from third-party packages showing up during runs.
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
for noisy in ("gluonts", "pytorch_lightning", "pytorch_forecasting", "statsforecast", "matplotlib", "urllib3"):
    try:
        logging.getLogger(noisy).setLevel(logging.ERROR)
    except Exception:
        pass

from models.compare import run_compare


if __name__ == "__main__":
    run_compare('data', 'output/compare', datasets_to_run=['ds-2-5T','ds-4-5T','ds-6-5T','ds-0-T','ds-1-T','ds-5-T','ds-3-10S','ds-7-10S','ds-12-10S','ds-1676-30T','ds-1675-30T','ds-1672-30T','ds-1683-H','ds-1678-H','ds-1677-H','ds-1679-D','ds-1671-D','ds-1674-D'])
