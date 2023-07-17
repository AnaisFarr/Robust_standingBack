"""
...
Phase 0: Waiting phase
- zero contact
- objectives functions: minimize torque, time

Phase 1: Salto
- zero contact, holonomics constraints
- objectives functions: minimize torque, time


"""
# --- Import package --- #

import numpy as np
import pickle
# import matplotlib.pyplot as plt
from bioptim import (
    BiorbdModel,
    Node,
    InterpolationType,
    OptimalControlProgram,
    ConstraintList,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    PhaseTransitionList,
    DynamicsFcn,
    BiMappingList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    Solver,
    Axis,
    PenaltyController,
    PhaseTransitionFcn,
)
from casadi import MX, vertcat
from holonomic_research.ocp_example_2 import generate_close_loop_constraint, custom_configure, custom_dynamic
from holonomic_research.biorbd_model_holonomic import BiorbdModelCustomHolonomic
from visualisation import visualisation_closed_loop_4phases_reception


# --- Parameters --- #
movement = "Salto_close_loop_landing"
version = 4
nb_phase = 4
name_folder_model = "/home/mickael/Documents/Anais/Robust_standingBack/Model"


# --- Save results --- #

def save_results(sol, c3d_file_path):
    """
    Solving the ocp
    Parameters
     ----------
     sol: Solution
        The solution to the ocp at the current pool
    c3d_file_path: str
        The path to the c3d file of the task
    """

    data = {}
    q = []
    qdot = []
    states_all = []
    tau = []

    if len(sol.ns) == 1:
        q = sol.states["u"]
        qdot = sol.states["udot"]
        # states_all = sol.states["all"]
        tau = sol.controls["tau"]
    else:
        for i in range(len(sol.states)):
            if i == 1:
                q.append(sol.states[i]["u"])
                qdot.append(sol.states[i]["udot"])
                # states_all.append(sol.states[i]["all"])
                tau.append(sol.controls[i]["tau"])
            else:
                q.append(sol.states[i]["q"])
                qdot.append(sol.states[i]["qdot"])
                # states_all.append(sol.states[i]["all"])
                tau.append(sol.controls[i]["tau"])

    data["q"] = q
    data["qdot"] = qdot
    data["tau"] = tau
    data["cost"] = sol.cost
    data["iterations"] = sol.iterations
    # data["detailed_cost"] = sol.detailed_cost
    data["status"] = sol.status
    data["real_time_to_optimize"] = sol.real_time_to_optimize
    data["phase_time"] = sol.phase_time[1:12]
    data["constraints"] = sol.constraints
    data["controls"] = sol.controls
    data["constraints_scaled"] = sol.controls_scaled
    data["n_shooting"] = sol.ns
    data["time"] = sol.time
    data["lam_g"] = sol.lam_g
    data["lam_p"] = sol.lam_p
    data["lam_x"] = sol.lam_x

    if sol.status == 1:
        data["status"] = "Optimal Control Solution Found"
    else:
        data["status"] = "Restoration Failed !"

    with open(f"{c3d_file_path}", "wb") as file:
        pickle.dump(data, file)


def custom_phase_transition_pre(
        controllers: list[PenaltyController, PenaltyController]) -> MX:
    """
    The constraint of the transition. The values from the end of the phase to the next are multiplied by coef to
    determine the transition. If coef=1, then this function mimics the PhaseTransitionFcn.CONTINUOUS

    coef is a user defined extra variables and can be anything. It is to show how to pass variables from the
    PhaseTransitionList to that function

    Parameters
    ----------
    controllers: list[PenaltyController, PenaltyController]
        The controller for all the nodes in the penalty

    Returns
    -------
    The constraint such that: c(x) = 0
    """

    # Take the values of q of the BioMod without holonomics constraints
    states_pre = controllers[0].states.cx

    nb_independent = controllers[1].model.nb_independent_joints
    u_post = controllers[1].states.cx[:nb_independent]
    udot_post = controllers[1].states.cx[nb_independent:]

    # Take the q of the indepente joint and calculate the q of dependent joint
    v_post = controllers[1].model.compute_v_from_u_explicit_symbolic(u_post)
    q_post = controllers[1].model.q_from_u_and_v(u_post, v_post)

    Bvu = controllers[1].model.coupling_matrix(q_post)
    vdot_post = Bvu @ udot_post
    qdot_post = controllers[1].model.q_from_u_and_v(udot_post, vdot_post)

    states_post = vertcat(q_post, qdot_post)

    return states_pre - states_post


def custom_phase_transition_post(
        controllers: list[PenaltyController, PenaltyController]) -> MX:
    """
    The constraint of the transition. The values from the end of the phase to the next are multiplied by coef to
    determine the transition. If coef=1, then this function mimics the PhaseTransitionFcn.CONTINUOUS

    coef is a user defined extra variables and can be anything. It is to show how to pass variables from the
    PhaseTransitionList to that function

    Parameters
    ----------
    controllers: list[PenaltyController, PenaltyController]
        The controller for all the nodes in the penalty

    Returns
    -------
    The constraint such that: c(x) = 0
    """

    # Take the values of q of the BioMod without holonomics constraints
    nb_independent = controllers[0].model.nb_independent_joints
    u_pre = controllers[0].states.cx[:nb_independent]
    udot_pre = controllers[0].states.cx[nb_independent:]

    # Take the q of the indepente joint and calculate the q of dependent joint
    v_pre = controllers[0].model.compute_v_from_u_explicit_symbolic(u_pre)
    q_pre = controllers[0].model.q_from_u_and_v(u_pre, v_pre)
    Bvu = controllers[0].model.coupling_matrix(q_pre)
    vdot_pre = Bvu @ udot_pre
    qdot_pre = controllers[0].model.q_from_u_and_v(udot_pre, vdot_pre)

    states_pre = vertcat(q_pre, qdot_pre)

    states_post = controllers[1].states.cx

    return states_pre - states_post


# --- Prepare ocp --- #

def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound):
    bio_model = (BiorbdModel(biorbd_model_path[0]),
                 BiorbdModelCustomHolonomic(biorbd_model_path[1]),
                 BiorbdModel(biorbd_model_path[2]),
                 BiorbdModel(biorbd_model_path[3]),
                 )

    # --- Objectives functions ---#
    # Add objective functions
    objective_functions = ObjectiveList()

    # Phase 0 (Waiting phase):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.1, max_bound=0.3, phase=0)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", weight=0.01, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=0)

    # Phase 1 (Salto close loop):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.1, max_bound=0.4, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.01, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.01, phase=1)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="udot", weight=0.01, phase=1)

    # Phase 2 (Second flight):
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=10, min_bound=0.1, max_bound=0.3, phase=2)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", weight=0.01, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=2)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=2)

    # Phase 3 (Landing):
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, node=Node.END, weight=100, axes=Axis.Z, phase=3)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=100, min_bound=0.1, max_bound=0.3, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10, phase=3)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=10, phase=3)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=3)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=10000000, axes=Axis.Y, phase=3)

    # --- Dynamics ---#
    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=0)
    dynamics.add(custom_configure, dynamic_function=custom_dynamic, expand=False, phase=1)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=2)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase=3)

    # Transition de phase
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(custom_phase_transition_pre, phase_pre_idx=0)
    # phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=0)
    # phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=1)
    phase_transitions.add(custom_phase_transition_post, phase_pre_idx=1)
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=2)

    # --- Constraints ---#
    # Constraints
    constraints = ConstraintList()

    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                    node=Node.END,
                    first_marker="BELOW_KNEE",
                    second_marker="CENTER_HAND",
                    phase=0)

    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS_VELOCITY,
                    node=Node.END,
                    first_marker="BELOW_KNEE",
                    second_marker="CENTER_HAND",
                    phase=0)

    # Phase 1 (Salto):
    constraint, constraint_jacobian, constraint_double_derivative = generate_close_loop_constraint(
        bio_model[1],
        "BELOW_KNEE",
        "CENTER_HAND",
        index=slice(1, 3),  # only constraint on x and y
        local_frame_index=11,  # seems better in one local frame than in global frame, the constraint deviates less
    )

    bio_model[1].add_holonomic_constraint(
        constraint=constraint,
        constraint_jacobian=constraint_jacobian,
        constraint_double_derivative=constraint_double_derivative,
    )

    bio_model[1].set_dependencies(independent_joint_index=[0, 1, 2, 5, 6, 7], dependent_joint_index=[3, 4])

    # Phase 4 (Landing):

    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.END,
        normal_component_idx=1,
        tangential_component_idx=0,
        static_friction_coefficient=0.33,
        phase=3,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=1,
        phase=3,
    )

    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.END,
        contact_index=2,
        phase=3,
    )

    # Path constraint
    # pose_takeout_end = [-1.1293, 0.4015, 0.5049, 3.0558, 1.7953, 0.2255, -0.3913, 0.1622]
    # pose_takeout_end = [-1.1293, 0.4015, 0.0, 3.0558, 1.7953, 0.2255, -0.3913, 0.1622]
    pose_takeout_start = [-0.2777, 0.0399, 0.1930, 2.5896, 0.51, 0.5354, -0.8367, 0.1119]
    pose_salto_start = [-0.6369, 1.0356, 1.5062, 0.3411, 1.3528, 2.1667, -1.9179, 0.0393]
    pose_salto_start_CL = [-0.6369, 1.0356, 1.5062, 2.1667, -1.9179, 0.0393]

    pose_salto_end_CL = [-0.6369, 1.0356, 2.7470, 2.1667, -1.9179, 0.0393]
    pose_salto_end = [-0.6369, 1.0356, 2.7470, 0.3411, 1.3528, 2.1667, -1.9179, 0.0393]
    # pose_salto_end_CL = [-0.6369, 1.0356, 2.7470, 1.7447, -1.1335, 0.0097]
    # pose_salto_end = [-0.6369, 1.0356, 2.7470, 0.9906, 0.0252, 1.7447, -1.1335, 0.0097]
    pose_landing_start = [-0.946, 1.7551, 5.8322, 0.52, 0.95, 1.72, -0.81, 0.0]
    pose_landing_end = [-0.946, 0.14, 6.28, 3.1, 0.03, 0.0, 0.0, 0.0]

    tau_min_total = [0, 0, 0, -325.531, -138, -981.1876, -735.3286, -343.9806]
    tau_max_total = [0, 0, 0, 325.531, 138, 981.1876, 735.3286, 343.9806]
    tau_min = [i * 0.9 for i in tau_min_total]
    tau_max = [i * 0.9 for i in tau_max_total]
    tau_init = 0
    mapping = BiMappingList()
    dof_mapping = BiMappingList()
    mapping.add("q", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])
    mapping.add("qdot", to_second=[0, 1, 2, None, None, 3, 4, 5], to_first=[0, 1, 2, 5, 6, 7])
    dof_mapping.add("tau", to_second=[None, None, None, 0, 1, 2, 3, 4], to_first=[3, 4, 5, 6, 7])

    # --- Bounds ---#
    # Initialize x_bounds
    n_q = bio_model[0].nb_q
    n_qdot = n_q
    n_independent = bio_model[1].nb_independent_joints

    # Phase 0: Flight
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds[0]["q"][:, 0] = pose_takeout_start
    x_bounds[0]["q"].min[0, 1:] = -1
    x_bounds[0]["q"].max[0, 1:] = 1
    x_bounds[0]["q"].min[1, 1:] = 0
    x_bounds[0]["q"].max[1, 1:] = 2.5
    x_bounds[0]["q"].min[2, 0] = -np.pi / 4
    x_bounds[0]["q"].max[2, 0] = np.pi / 4
    x_bounds[0]["q"].min[2, 1] = -np.pi / 4
    x_bounds[0]["q"].max[2, 1] = np.pi / 2
    x_bounds[0]["q"].min[2, -1] = np.pi / 2
    x_bounds[0]["q"].max[2, -1] = np.pi

    x_bounds[0]["qdot"].min[0, :] = -5
    x_bounds[0]["qdot"].max[0, :] = 5
    x_bounds[0]["qdot"].min[1, :] = -10
    x_bounds[0]["qdot"].max[1, :] = 10
    x_bounds[0]["qdot"].min[2, :] = -5
    x_bounds[0]["qdot"].max[2, :] = 5
    # x_bounds[0]["q"].max[4, -1] = -1
    # x_bounds[0]["q"].min[4, -1] = -2.3


    # Phase 1: Salto
    x_bounds.add("u", bounds=bio_model[1].bounds_from_ranges("q", mapping=mapping), phase=1)
    x_bounds.add("udot", bounds=bio_model[1].bounds_from_ranges("qdot", mapping=mapping), phase=1)
    x_bounds[1]["u"].min[0, :] = -2
    x_bounds[1]["u"].max[0, :] = 1
    x_bounds[1]["u"].min[1, :] = 0
    x_bounds[1]["u"].max[1, :] = 2.5
    x_bounds[1]["u"].min[2, 0] = np.pi / 2
    x_bounds[1]["u"].max[2, 0] = np.pi
    x_bounds[1]["u"].min[2, 1] = np.pi / 2
    x_bounds[1]["u"].max[2, 1] = 2 * np.pi
    x_bounds[1]["u"].min[2, 2] = np.pi
    x_bounds[1]["u"].max[2, 2] = 2 * np.pi
    x_bounds[1]["udot"].min[0, :] = -5
    x_bounds[1]["udot"].max[0, :] = 5
    x_bounds[1]["udot"].min[1, :] = -2
    x_bounds[1]["udot"].max[1, :] = 10
    x_bounds[1]["u"].max[3, :] = 2.6
    x_bounds[1]["u"].min[3, :] = 1.30
    x_bounds[1]["u"].max[5, :] = 0.1
    x_bounds[1]["u"].min[5, :] = -0.7


    # Phase 2: Second flight
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=2)
    x_bounds[2]["q"].min[0, :] = -2
    x_bounds[2]["q"].max[0, :] = 1
    x_bounds[2]["q"].min[1, :] = 0
    x_bounds[2]["q"].max[1, :] = 2.5
    x_bounds[2]["q"].min[2, 0] = np.pi
    x_bounds[2]["q"].max[2, 0] = 2 * np.pi
    x_bounds[2]["q"].min[2, 1] = np.pi
    x_bounds[2]["q"].max[2, 1] = 2 * np.pi
    x_bounds[2]["q"].min[2, -1] = 2 * np.pi
    x_bounds[2]["q"].max[2, -1] = 2 * np.pi + 1
    x_bounds[2]["qdot"].min[0, :] = -5
    x_bounds[2]["qdot"].max[0, :] = 5
    x_bounds[2]["qdot"].min[1, :] = -10
    x_bounds[2]["qdot"].max[1, :] = 10

    # Phase 3: Landing
    x_bounds.add("q", bounds=bio_model[3].bounds_from_ranges("q"), phase=3)
    x_bounds.add("qdot", bounds=bio_model[3].bounds_from_ranges("qdot"), phase=3)
    x_bounds[3]["q"].min[0, :] = -2
    x_bounds[3]["q"].max[0, :] = 1
    x_bounds[3]["q"].min[1, :] = -1
    x_bounds[3]["q"].max[1, :] = 2.5
    x_bounds[3]["q"].min[2, 0] = 2 * np.pi - 1
    x_bounds[3]["q"].max[2, 0] = 2 * np.pi + 0.5
    x_bounds[3]["q"].min[2, 1:] = 2 * np.pi - 0.5
    x_bounds[3]["q"].max[2, 1:] = 2 * np.pi + 0.5
    # x_bounds[3]["q"].max[7, :] = 0.3
    # x_bounds[3]["q"].min[7, :] = -0.3
    x_bounds[3]["q"][:, -1] = pose_landing_end
    # x_bounds[3]["q"].max[:, -1] = np.array(pose_landing_end) + 0.5
    # x_bounds[3]["q"].min[:, -1] = np.array(pose_landing_end) - 0.5

    # x_bounds[3]["qdot"][:, -1] = [0] * n_qdot

    # Initial guess
    x_init = InitialGuessList()
    x_init.add("q", np.array([pose_takeout_start, pose_salto_start]).T, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("u", np.array([pose_salto_start_CL, pose_salto_end_CL]).T, interpolation=InterpolationType.LINEAR,
               phase=1)
    x_init.add("udot", np.array([[0] * n_independent, [0] * n_independent]).T, interpolation=InterpolationType.LINEAR, phase=1)
    x_init.add("q", np.array([pose_salto_end, pose_landing_start]).T, interpolation=InterpolationType.LINEAR, phase=2)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=2)
    x_init.add("q", np.array([pose_landing_start, pose_landing_end]).T, interpolation=InterpolationType.LINEAR, phase=3)
    x_init.add("qdot", np.array([[0] * n_qdot, [0] * n_qdot]).T, interpolation=InterpolationType.LINEAR, phase=3)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=0)
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=1)
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=2)
    u_bounds.add("tau", min_bound=[tau_min[3], tau_min[4], tau_min[5], tau_min[6], tau_min[7]],
                 max_bound=[tau_max[3], tau_max[4], tau_max[5], tau_max[6], tau_max[7]], phase=3)

    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=0)
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=1)
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=2)
    u_init.add("tau", [tau_init] * (bio_model[0].nb_tau - 3), phase=3)

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
        n_threads=32,
        assume_phase_dynamics=True,
        phase_transitions=phase_transitions,
        variable_mappings=dof_mapping,
    ), bio_model


# --- Load model --- #
def main():
    model_path = str(name_folder_model) + "/" + "Model2D_7Dof_0C_5M_CL_V2.bioMod"
    model_path_contact = str(name_folder_model) + "/" + "Model2D_7Dof_3C_5M_CL_V2.bioMod"
    ocp, bio_model = prepare_ocp(
        biorbd_model_path=(model_path,
                           model_path,
                           model_path,
                           model_path_contact),
        phase_time=(0.2, 0.3, 0.3, 0.3),
        n_shooting=(20, 30, 30, 30),
        min_bound=50,
        max_bound=np.inf,
    )

    # ocp.add_plot_penalty()
    # --- Solve the program --- #
    ocp.print(to_console=True, to_graph=False)
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True), _linear_solver="MA57")
    solver.set_maximum_iterations(1000)

    sol = ocp.solve(solver)
    # sol.print_cost()
    # bio_model[1].compute_external_force_holonomics_constraints(sol.states[1]["u"], sol.states[1]["udot"], sol.controls[1]["tau"])
    sol.graphs()
# --- Show results --- #
    save_results(sol, str(movement) + "_" + "with_pelvis" + "_" + str(nb_phase) + "phases_V" + str(version) + ".pkl")
    visualisation_closed_loop_4phases_reception(bio_model, sol, model_path)


if __name__ == "__main__":
    main()
