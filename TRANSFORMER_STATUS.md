# Transformer Creation Status

## Completed Transformers (14/22)

### Spectroscopic/Photometric Catalogs
1. ✅ **GAIA** - `gaia_transformer.py` - Gaia DR3 photometry and astrometry
2. ✅ **DESI** - `desi_transformer.py` - DESI EDR SV3 spectra
3. ✅ **VIPERS** - `vipers_transformer.py` - VIPERS spectra and redshifts
4. ✅ **GZ10** - `gz10_transformer.py` - Galaxy Zoo 10 classifications

### Time Series / Lightcurve Catalogs  
5. ✅ **PLAsTiCC** - `plasticc_transformer.py` - LSST lightcurve challenge data
6. ✅ **TESS** - `tess_transformer.py` - TESS lightcurves
7. ✅ **Foundation** - `foundation_transformer.py` - Foundation DR1 SNe Ia
8. ✅ **SNLS** - `snls_transformer.py` - Supernova Legacy Survey
9. ✅ **PS1 SNE Ia** - `ps1_sne_ia_transformer.py` - Pan-STARRS1 SNe Ia
10. ✅ **YSE** - `yse_transformer.py` - Young Supernova Experiment
11. ✅ **Swift SNE Ia** - `swift_sne_ia_transformer.py` - Swift UV/optical SNe
12. ✅ **CFA** - `cfa_transformer.py` - CFA supernova archive
13. ✅ **DES Y3 SNE Ia** - `des_y3_sne_ia_transformer.py` - DES Y3 SNe Ia

### Other
14. ✅ **SDSS** - `sdss_transformer.py` - (already existed)

## Pending Transformers (7/22) - Complex Catalogs

These catalogs require more sophisticated handling:

### Image-based Catalogs
1. ⏳ **HSC** - Complex image sequences with multiple bands
2. ⏳ **JWST** - JWST deep field images (multiple bands)
3. ⏳ **LegacySurvey** - Images + RGB + object masks + catalogs
4. ⏳ **SSL LegacySurvey** - Self-supervised learning image dataset

### IFU / Complex Data
5. ⏳ **MaNGA** - IFU datacubes with spaxels, images, and maps

### Special Cases
6. ⏳ **CSP** - Uses magnitudes instead of flux (different schema)
7. ⏳ **DESI PROVABGS** - Complex with MCMC posterior samples
8. ⏳ **BTSbot** - Many ZTF-specific alert features

### Skipped
- **GUI** - Not a catalog (Streamlit application)

## Notes

All completed transformers follow the pattern established by `sdss_transformer.py`:
- Inherit from `BaseTransformer`
- Define feature groups as class attributes
- Implement `create_schema()` for PyArrow schema
- Implement `dataset_to_table()` for HDF5 to PyArrow conversion

The pending catalogs require additional work to handle:
- Multi-dimensional image arrays
- Multiple image products (flux, mask, PSF, RGB, etc.)
- IFU datacube structures
- Complex nested data structures
