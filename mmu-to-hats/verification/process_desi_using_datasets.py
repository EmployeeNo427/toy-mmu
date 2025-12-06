# NOTE: Updated for datasets 4.x compatibility
# Run this first
# uv pip install -r requirements.txt
# ./download_desi_hsc.sh
import sys
from pathlib import Path
from datasets import concatenate_datasets

# Add parent directory to path to import mmu utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mmu.utils import get_catalog, load_dataset_builder_from_path

# Load the dataset descriptions from local copy of the data
desi = load_dataset_builder_from_path("data/MultimodalUniverse/v1/desi")
desi.download_and_prepare()

desi_catalog = get_catalog(desi)


def match_desi_catalog_object_ids(example, catalog):
    example_obj_id = example["object_id"].strip("b'")
    catalog_entry = catalog[catalog["object_id"] == int(example_obj_id)]
    assert len(catalog_entry) == 1
    return {
        **example,
        "ra": catalog_entry["ra"][0],
        "dec": catalog_entry["dec"][0],
        "healpix": catalog_entry["healpix"][0],
    }


desi_train = desi.as_dataset(split="train")
desi_mapped = desi_train.map(
    lambda example: match_desi_catalog_object_ids(example, desi_catalog)
)
desi_mapped.save_to_disk("data/MultimodalUniverse/v1/desi_with_coordinates")
