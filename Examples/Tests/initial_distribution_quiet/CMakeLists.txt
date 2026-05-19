# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_initial_distribution_quiet_picmi  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_initial_distribution_quiet_picmi.py  # inputs
    "analysis.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)
