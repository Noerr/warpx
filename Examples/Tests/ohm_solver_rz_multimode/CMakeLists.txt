# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_rz_ohm_solver_multimode_drift_hall_nodes_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_multimode_drift_picmi.py --hall_divide_at_nodes"  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000004 --rtol 1e-6"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_ohm_solver_multimode_drift_picmi  # name
    RZ  # dims
    2  # nprocs
    "inputs_test_rz_ohm_solver_multimode_drift_picmi.py"  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000004 --rtol 1e-6"  # checksum
    OFF  # dependency
)
