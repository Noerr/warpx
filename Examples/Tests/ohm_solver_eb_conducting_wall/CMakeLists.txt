# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_ohm_solver_eb_conducting_wall_picmi  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_ohm_solver_eb_conducting_wall_picmi.py"  # inputs
    "analysis.py"  # analysis
    OFF  # checksum
    OFF  # dependency
)
