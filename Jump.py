"""
The aim of this code is to create a movement a simple jump in 2 phases with a 2D model.
Phase 1:
- one contact (toe)
- objectives functions: maximize velocity of CoM and minimize time of flight

Phase 2:
- zero contact (in the air)
- objectives functions: maximize height of CoM and maximize time of flight


"""
# --- Import package --- #
from matplotlib import pyplot as plt
import numpy as np
import biorbd
from bioptim import (
    BiorbdModel,
    Node,
    Axis,
    OptimalControlProgram,
    ConstraintList,
    Bounds,
    ConstraintFcn,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    Dynamics,
    DynamicsFcn,
    BiMappingList,
    BoundsList,
    InitialGuess,
    QAndQDotBounds,
    InitialGuessList,
    Solver,
)

biorbd_model_path = "/home/lim/Documents/Anais/Robust_standingBack/Model2D"
biorbd_model = BiorbdModel(biorbd_model_path)

# --- Methode 1 ---#
# --- Prepare ocp --- #


def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound):

    # --- Options --- #
    # BioModel path

    bio_model = BiorbdModel(biorbd_model_path)
    tau_min, tau_max, tau_init = -500, 500, 0
    #activation_min, activation_max, activation_init = 0, 1, 0.5
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", [None, None, None, 0, 1, 2, 3], [3, 4, 5, 6])

    # Add objective functions
    objective_functions = ObjectiveList()
    # Phase 1: Maximize velocity CoM + Minimize time (less important)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=-1, phase=0, axes=Axis.Z)  # Changer le signe, rajouter axis
    #objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TIME, weight=0.002, phase=0)

    # Phase 2: Maximize height CoM + Maximize time in air
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=-1, phase=1, axes=Axis.Z)  # Changer le signe, rajouter axis
    #objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TIME, weight=-1.5, phase=1)  # Changer le signe

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL_SHOOTING,
        contact_index=1,
        phase=0)

    # Path constraint
    n_q = bio_model.nb_q
    n_qdot = n_q
    pose_at_first_node = [0.0, 0.0, 0.0, 1.3, 0.7645, -1.4512, 0.264] # BioViz copie

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(bio_model)) # Deux fois pour la deuxième phase ?
    x_bounds[0][:, 0] = pose_at_first_node + [0] * n_qdot
    x_bounds.add(bounds=QAndQDotBounds(bio_model))

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(pose_at_first_node + [0] * n_qdot)
    x_init.add(pose_at_first_node + [0] * n_qdot)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [tau_min] * bio_model.nb_tau,
        [tau_max] * bio_model.nb_tau,
    )
    u_bounds.add(
        [tau_min] * bio_model.nb_tau,
        [tau_max] * bio_model.nb_tau,
    )

    u_init = InitialGuessList()
    u_init.add([tau_init] * bio_model.nb_tau)
    u_init.add([tau_init] * bio_model.nb_tau)


    return OptimalControlProgram(
        bio_model=bio_model,
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=phase_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        variable_mappings=dof_mapping,
        n_threads=3)

# --- Load model --- #


ocp = prepare_ocp(
    biorbd_model_path=biorbd_model_path,
    phase_time=([0.2, 1]),
    n_shooting=([20, 100]),
    min_bound=50,
    max_bound=np.inf)

ocp.add_plot_penalty()
# --- Solve the program --- #
sol = ocp.solve(Solver.IPOPT(show_online_optim=True))
# Show bound= true

# --- Show results --- #

sol.animate()
sol.print()
# sol.graphs()
