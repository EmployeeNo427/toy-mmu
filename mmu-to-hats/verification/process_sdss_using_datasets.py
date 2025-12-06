# NOTE: Updated for datasets 4.x compatibility
# Run this first
# uv pip install -r requirements.txt
# ./download_sdss_hsc.sh
import sys
from pathlib import Path
from datasets import concatenate_datasets

# Add parent directory to path to import mmu utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mmu.utils import get_catalog, load_dataset_builder_from_path

# Load the dataset descriptions from local copy of the data
sdss = load_dataset_builder_from_path("data/MultimodalUniverse/v1/sdss")
sdss.download_and_prepare()

sdss_catalog = get_catalog(sdss)


def match_sdss_catalog_object_ids(example, catalog):
    example_obj_id = example["object_id"].strip("b'")
    catalog_entry = catalog[catalog["object_id"] == example_obj_id]
    assert len(catalog_entry) == 1
    return {
        **example,
        "ra": catalog_entry["ra"][0],
        "dec": catalog_entry["dec"][0],
        "healpix": catalog_entry["healpix"][0],
    }


sdss_train = sdss.as_dataset(split="train")
sdss_mapped = sdss_train.map(
    lambda example: match_sdss_catalog_object_ids(example, sdss_catalog)
)
sdss_mapped.save_to_disk("data/MultimodalUniverse/v1/sdss_with_coordinates")
