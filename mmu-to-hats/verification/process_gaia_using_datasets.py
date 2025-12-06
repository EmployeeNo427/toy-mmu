# NOTE: Updated for datasets 4.x compatibility
# uv run --with-requirements=verification/requirements.in python verification/process_gaia_using_datasets.py
import sys
from pathlib import Path

# Add parent directory to path to import mmu utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mmu.utils import load_dataset_builder_from_path

# Load the dataset descriptions from local copy of the data
gaia = load_dataset_builder_from_path("data/MultimodalUniverse/v1/gaia")
gaia.download_and_prepare()

# Gaia HDF5 files already contain ra, dec, healpix - no catalog join needed (unlike other datasets)
gaia_train = gaia.as_dataset(split="train")
gaia_train.save_to_disk("data/MultimodalUniverse/v1/gaia_with_coordinates")
