# NOTE: run with datasets==3.6
# uv run --with-requirements=verification/requirements.in python verification/process_btsbot_using_datasets.py
from datasets import load_dataset_builder
from mmu.utils import get_catalog

# Load the dataset descriptions from local copy of the data
btsbot = load_dataset_builder("data/MultimodalUniverse/v1/btsbot", trust_remote_code=True)
btsbot.download_and_prepare()

btsbot_catalog = get_catalog(btsbot)


def match_btsbot_catalog_object_ids(example, catalog):
    # BTSbot object_id is int64, no string stripping needed
    example_obj_id = example["object_id"]
    catalog_entry = catalog[catalog["object_id"] == example_obj_id]
    assert len(catalog_entry) == 1
    return {
        **example,
        "ra": catalog_entry["ra"][0],
        "dec": catalog_entry["dec"][0],
        "healpix": catalog_entry["healpix"][0],
    }


btsbot_train = btsbot.as_dataset(split="train")
btsbot_mapped = btsbot_train.map(
    lambda example: match_btsbot_catalog_object_ids(example, btsbot_catalog)
)
btsbot_mapped.save_to_disk("data/MultimodalUniverse/v1/btsbot_with_coordinates")
