"""Fixed normalisation constants. CHOSEN, not measured.

Fixed rather than computed from data on purpose: per-region statistics would
make a UK input mean something different from an Australian one, and the
cross-region transfer result would be uninterpretable.
"""

H_SCALE = 1.0        # m, depth
DH_SCALE = 0.05      # m, change over one horizon
BED_SCALE = 10.0     # m, after subtracting the crop mean
MANNING_SCALE = 0.1  # fixed, never per-crop


# Esri 10-class Sentinel-2 land cover -> Manning n (FloodCastBench Table 3).
# Verified: code shares match the paper's stated tree (29.66%) and urban (2.75%)
# fractions for Australia. Codes 3 and 6 are retired and absent.
# Codes 9 (snow/ice, 361 cells) and 10 (cloud, 6 cells) are artefacts and take
# the rangeland value - state this in methods.
MANNING_BY_CODE = {
    1: 0.0350,   # water
    2: 0.1200,   # trees
    4: 0.0800,   # flooded vegetation
    5: 0.0350,   # crops
    7: 0.3750,   # built area
    8: 0.0265,   # bare ground
    9: 0.0375,   # artefact -> rangeland
    10: 0.0375,  # artefact -> rangeland
    11: 0.0375,  # rangeland
}
