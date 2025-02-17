import os
from typing import Callable

from bioptim import (
    Solver,
)

from constants import MODEL_PATHS, PHASE_TIME, N_SHOOTING
from sommersault_taudot import prepare_ocp as prepare_ocp_ntc
from sommersault_htc_taudot import prepare_ocp as prepare_ocp_with_htc
from sommersault_ktc_taudot import prepare_ocp as prepare_ocp_with_ktc
from src.save_results import save_results_taudot
from src.save_results import save_results_holonomic_taudot

from src.multistart import prepare_multi_start


def main(prepare_ocp: Callable, save_results: Callable, multi_start: bool = False, condition: str = ""):
    # --- Parameters --- #
    movement = "backflip"
    version = "post_submission_v3"

    WITH_MULTI_START = multi_start
    save_folder = f"../results/{str(movement)}_V{version}/{condition}"
    if os.path.isdir(save_folder) is False:
        os.makedirs(save_folder)

    # Solver options
    solver = Solver.IPOPT(show_options=dict(show_bounds=True), _linear_solver="MA57", show_online_optim=False)
    solver.set_maximum_iterations(10000)
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)
    solver.set_tol(1e-6)

    if WITH_MULTI_START:

        end = 20
        if condition == "ntc":
            start = 20
        elif condition == "ktc":
            start = 20
        elif condition == "htc":
            start = 11


        combinatorial_parameters = {
            "bio_model_path": [MODEL_PATHS],
            "phase_time": [PHASE_TIME],
            "n_shooting": [N_SHOOTING],
            "WITH_MULTI_START": [True],
            "seed": list(range(start, end)),
        }

        multi_start = prepare_multi_start(
            prepare_ocp,
            save_results,
            combinatorial_parameters=combinatorial_parameters,
            save_folder=save_folder,
            solver=solver,
            n_pools=1,
        )

        multi_start.solve()
    else:
        ocp = prepare_ocp(MODEL_PATHS, PHASE_TIME, N_SHOOTING, WITH_MULTI_START=False)
        # ocp.add_plot_penalty()

        solver.show_online_optim = False
        sol = ocp.solve(solver)
        sol.print_cost()

        # --- Save results --- #
        # sol.graphs(show_bounds=True, save_name=str(movement) + "_V" + version)
        # sol.animate()

        combinatorial_parameters = [MODEL_PATHS, PHASE_TIME, N_SHOOTING, WITH_MULTI_START, "no_seed"]
        save_results(sol, *combinatorial_parameters, save_folder=save_folder)


if "__main__" == __name__:
    main(prepare_ocp_ntc, save_results_taudot, multi_start=True, condition="ntc")
    main(prepare_ocp_with_ktc, save_results_taudot, multi_start=True, condition="ktc")
    main(prepare_ocp_with_htc, save_results_holonomic_taudot, multi_start=True, condition="htc")
