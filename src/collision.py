from casadi import solve, MX, vertcat
from bioptim import PenaltyController


def collision_impact(model, q, qdot_minus, e=0):
    """
    Compute the post-collision generalized velocity qdot_plus using CasADi symbolic expressions:

    Arguments:
    ----------
    model :  Your model or system object that exposes:
               - model.massMatrixInverse(q) -> CasADi matrix H_inv
               - constraint_jacobian(q)     -> CasADi matrix G
    q     :  CasADi vector (or DM) of generalized positions
    qdot_minus : CasADi vector (or DM) of pre-collision generalized velocities
    e     :  scalar (coefficient of restitution in [0,1]);
             can be a float or a CasADi symbolic variable

    Returns:
    --------
    qdot_plus : CasADi expression (symbolic or DM) for the post-collision velocity
    """

    # 1) Inverse of the mass matrix M(q).
    M_inv = model.model.massMatrixInverse(q).to_mx()  # shape: (n x n)

    # 2) Constraint Jacobian
    J = model.holonomic_constraints_jacobian(q)  # shape: (m x n)
    J_T = J.T  # shape: (n x m)

    # 3) Delassus matrix S = J * M^{-1} * J^T
    #    shape: (m x m)
    delassus = J @ M_inv @ J_T

    # 4) Compute the impulse: Lambda = -S^{-1} * (e + 1)*J*qdot_minus
    #    (Use casadi.solve(...) rather than S^{-1} for better numeric stability)
    rhs = (e + 1.0) * (J @ qdot_minus)  # shape: (m,)
    Lambda = -solve(delassus, rhs, "symbolicqr")

    # 5) Finally, compute qdot_plus = qdot_minus + M^{-1} * J^T * Lambda
    qdot_plus = qdot_minus + M_inv @ (J_T @ Lambda)

    return qdot_plus


def transition_pre_with_collision(controllers: list[PenaltyController, PenaltyController]) -> MX:
    """
    The constraint of the transition from a holonomic to an model without holonomic constraints.

    Parameters
    ----------
    controllers: list[PenaltyController, PenaltyController]
        The controller for all the nodes in the penalty

    Returns
    -------
    The constraint such that: (q-, qdot-) = (q+, qdot+)
    """

    # Take the values of q of the BioMod without holonomics constraints

    q_pre = controllers[0].states["q"].cx
    qdot_pre = controllers[0].states["qdot"].cx
    qdot_post_estimated = collision_impact(controllers[1].model, q_pre, qdot_pre)

    nb_independent = controllers[1].model.nb_independent_joints
    u_post = controllers[1].states.cx[:nb_independent]
    udot_post = controllers[1].states.cx[nb_independent : nb_independent * 2]

    # Take the q of the independent joint and calculate the q of dependent joint
    v_post = controllers[1].model.compute_v_from_u_explicit_symbolic(u_post)
    q_post = controllers[1].model.state_from_partition(u_post, v_post)

    Bvu = controllers[1].model.coupling_matrix(q_post)
    vdot_post = Bvu @ udot_post
    qdot_post = controllers[1].model.state_from_partition(udot_post, vdot_post)

    tau_pre = controllers[0].states["tau"].cx
    tau_post = controllers[1].states["tau"].cx


    return vertcat(q_pre - q_post, qdot_post_estimated - qdot_post, tau_pre - tau_post)