import os
from typing import Callable

from bioptim import (
    Solver,
)
import matplotlib.pyplot as plt

from constants import MODEL_PATHS, PHASE_TIME, N_SHOOTING
from somersault_taudot import prepare_ocp as prepare_ocp_ntc
from somersault_htc_taudot import prepare_ocp as prepare_ocp_with_htc
from somersault_ktc_taudot import prepare_ocp as prepare_ocp_with_ktc
from src.save_results import save_results_taudot, save_sol_no_ocp, save_results_holonomic_taudot

from src.multistart import prepare_multi_start


def main(
        prepare_ocp: Callable,
        save_results: Callable,
        multi_start: bool = False,
        condition: str = "",
        seed_start: int = 0,
        seed_end: int = 20,
):
    # --- Parameters --- #
    movement = "backflip"
    version = "post_submission_collision_feb25"

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

        combinatorial_parameters = {
            "bio_model_path": [MODEL_PATHS],
            "phase_time": [PHASE_TIME],
            "n_shooting": [N_SHOOTING],
            "WITH_MULTI_START": [True],
            "seed": list(range(seed_start, seed_end)),
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
        combinatorial_parameters = [MODEL_PATHS, PHASE_TIME, N_SHOOTING, False, "no_seed"]
        save_results(sol, *combinatorial_parameters, save_folder=save_folder)
        # sol.graphs(show_bounds=True, save_name=str(movement) + "_V" + version, show_now=False)
        # NOTE: This will save the solution without the ocp, so the graphs cannot be generated after this line
        save_sol_no_ocp(sol, *combinatorial_parameters, save_folder=save_folder)
        # Showing the graphs
        # plt.show()


if "__main__" == __name__:
    conditions= ["htc", "ntc", "ktc"]
    save_funcs = [save_results_holonomic_taudot, save_results_taudot, save_results_taudot]
    prepare_ocps = [prepare_ocp_with_htc, prepare_ocp_ntc, prepare_ocp_with_ktc]

    for save_func, condition, prepare_ocp in zip(save_funcs, conditions, prepare_ocps):
        main(prepare_ocp, save_func, multi_start=False, condition=condition)

    for save_func, condition, prepare_ocp in zip(save_funcs, conditions, prepare_ocps):
        main(prepare_ocp, save_func, multi_start=True, condition=condition, seed_start=0, seed_end=3)

    for save_func, condition, prepare_ocp in zip(save_funcs, conditions, prepare_ocps):
        main(prepare_ocp, save_func, multi_start=True, condition=condition, seed_start=3, seed_end=20)
