from datasets import load_dataset

dset = (
    load_dataset(
        "hdf5",
        data_files={
            "train": "scripts/desi/edr_sv3/healpix=*/*.hdf5"
        },
        split="train",
    )
    .select_columns(("spectrum_flux", "ra", "dec"))
)

dset = iter(dset)
