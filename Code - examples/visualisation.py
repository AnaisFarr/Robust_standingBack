import bioviz

import bioviz
import pickle
from scipy.interpolate import interp1d
import numpy as np

# --- Function visualisation with a pickle file --- #

def get_created_data_from_pickle(file: str):
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break
    datas_q = data_tmp["q"]
    # datas_qdot = data_tmp["qdot"]
    # datas_tau = data_tmp["tau"]
    # data_status = data_tmp["status"]
    # data_mus = data_tmp["controls"]["muscles"]
    # data_time = data_tmp["real_time_to_optimize"]
    # data_it = data_tmp["iterations"]
    # data_cost = data_tmp["detailed_cost"]
    data_time_node = data_tmp["time"]

    return datas_q, data_time_node  # , datas_qdot, datas_tau , data_status, data_mus, data_it, data_time, data_cost


def visualisation_model(name_file_model:str):
    b = bioviz.Viz(name_file_model, show_floor=True, show_meshes=True)
    b.exec()


def visualisation_movement(name_file_movement, name_file_model):
    q, time_node = get_created_data_from_pickle(name_file_movement)
    if len(q) == 1:
            visu = bioviz.Viz(name_file_model, show_floor=True, show_meshes=True)
            visu.load_movement(q)
            visu.exec()

    elif len(q) == 2:
            Q = np.concatenate((q[0], q[1]), axis=1)
            visu = bioviz.Viz(name_file_model, show_floor=True, show_meshes=True)
            visu.load_movement(Q)
            visu.exec()

    elif len(q) == 3:
            Q = np.concatenate((q[0], q[1], q[2]), axis=1)
            visu = bioviz.Viz(name_file_model, show_floor=True, show_meshes=True)
            visu.load_movement(Q)
            visu.exec()

    elif len(q) == 3:
            Q = np.concatenate((q[0], q[1], q[2], q[3]), axis=1)
            visu = bioviz.Viz(name_file_model, show_floor=True, show_meshes=True)
            visu.load_movement(Q)
            visu.exec()

    elif len(q) == 5:
            Q = np.concatenate((q[0], q[1], q[2], q[3], q[4]), axis=1)
            visu = bioviz.Viz(name_file_model, show_floor=True, show_meshes=True)
            visu.load_movement(Q)
            visu.exec()

    elif len(q) == 6:
            Q = np.concatenate((q[0], q[1], q[2], q[3], q[4], q[5]), axis=1)
            visu = bioviz.Viz(name_file_model, show_floor=True, show_meshes=True)
            visu.load_movement(Q)
            visu.exec()

    elif len(q) == 7:
            Q = np.concatenate((q[0], q[1], q[2], q[3], q[4], q[5], q[6]), axis=1)
            visu = bioviz.Viz(name_file_model, show_floor=True, show_meshes=True)
            visu.load_movement(Q)
            visu.exec()
def visualisation_dedoublement_phase(name_file_movement:str, name_file_model:str, name_file_model_2:str):
    q, time_node = get_created_data_from_pickle(name_file_movement)
    if len(q) == 11:
        for i in range(0, len(q)):
            time_node[i] = np.array(time_node[i], dtype=np.float32)
        for i in range(1, len(q)):
            q[i] = q[i][:, 1:]
            time_node[i] = time_node[i][1:]

        # Time simulation salto with errors timing
        Q_1 = np.concatenate((q[0], q[1], q[2], q[3], q[4], q[5], q[6]), axis=1)
        time_Q1 = np.concatenate(
            (time_node[0], time_node[1], time_node[2], time_node[3], time_node[4], time_node[5], time_node[6]),
            axis=0,
        )
        duree_Q1 = [time_Q1[i + 1] - time_Q1[i] for i in range(0, len(time_Q1) - 1)]

        # Change time phase 7 to 11: simulation salto without errors timing
        time_Q7 = time_node[7][0]
        ecart = time_node[1][-1] + time_node[7][1] - time_node[7][0]

        for i in range(7, 11):
            time_node[i] = time_node[i] - (time_Q7 - ecart)
        Q_2 = np.concatenate((q[0], q[1], q[7], q[8], q[9], q[10]), axis=1)
        time_Q2 = np.concatenate(
            (time_node[0], time_node[1], time_node[7], time_node[8], time_node[9], time_node[10])
        )
        duree_Q2 = [time_Q2[i + 1] - time_Q2[i] for i in range(0, len(time_Q2) - 1)]

        # Interpolation simulation salto with errors timing
        Q1_interpolate = np.zeros(shape=(Q_1.shape[0], int((time_Q1[-1] + 0.01) / 0.01)), dtype=float)
        for nb_Dof in range(Q_1.shape[0]):
            interp_func_Q1 = interp1d(time_Q1, Q_1[nb_Dof, :], kind="linear")
            newy = interp_func_Q1(np.arange(time_Q1[0], time_Q1[-1], 0.01))
            Q1_interpolate[nb_Dof] = newy

        # Interpolation simulation salto without errors timing
        Q2_interpolate = np.zeros(shape=(Q_2.shape[0], int((time_Q2[-1] + 0.01) / 0.01)), dtype=float)
        for nb_Dof in range(Q_2.shape[0]):
            interp_func_Q2 = interp1d(time_Q2, Q_2[nb_Dof, :], kind="linear")
            newy = interp_func_Q2(np.arange(time_Q2[0], time_Q2[-1], 0.01))
            Q2_interpolate[nb_Dof] = newy

        if Q1_interpolate.shape[1] < Q2_interpolate.shape[1]:
            Q_add = np.zeros(
                shape=(Q_1.shape[0], int(Q2_interpolate.shape[1] - Q1_interpolate.shape[1])), dtype=float
            )
            for i in range(Q2_interpolate.shape[1] - Q1_interpolate.shape[1]):
                Q_add[:, i] = Q1_interpolate[:, -1]
            Q1_interpolate_new = np.concatenate((Q1_interpolate, Q_add), axis=1)

        if Q2_interpolate.shape[1] < Q1_interpolate.shape[1]:
            Q_add = np.zeros(
                shape=(Q_1.shape[0], int(Q2_interpolate.shape[1] - Q1_interpolate.shape[1])), dtype=float
            )
            for i in range(Q2_interpolate.shape[1] - Q1_interpolate.shape[1]):
                Q_add[:, i] = Q1_interpolate[:, -1]
            Q1_interpolate_new = np.concatenate((Q1_interpolate, Q_add), axis=1)

        # Visualisation simulation salto with errors timing
        visu_1 = bioviz.Viz(name_file_model, show_floor=True, show_meshes=True)
        visu_1.load_movement(Q_1)
        visu_1.exec()

        # Visualisation simulation salto without errors timing
        visu_2 = bioviz.Viz(name_file_model, show_floor=True, show_meshes=True)
        visu_2.load_movement(Q_2)
        visu_2.exec()

        # Visualisation two simulations
        Q_3 = np.concatenate((Q1_interpolate, Q2_interpolate), axis=0)
        visu_3 = bioviz.Viz(name_file_model_2, show_floor=True, show_meshes=True)
        visu_3.load_movement(Q_3)
        visu_3.exec()

def visualisation_closed_loop(bio_model, sol, model_path):
    q = np.zeros((bio_model.nb_q, sol.states["u"].shape[1]))
    for i, ui in enumerate(sol.states["u"].T):
        # vi = bio_model.compute_v_from_u_numeric(ui, v_init=DM(np.zeros(2))).toarray()
        vi = bio_model.compute_v_from_u_explicit_numeric(ui).toarray()
        qi = bio_model.q_from_u_and_v(ui[:, np.newaxis], vi).toarray().squeeze()
        q[:, i] = qi
    visu = bioviz.Viz(model_path)
    visu.load_movement(q)
    visu.exec()

def visualisation_closed_loop_5phases(bio_model, sol, model_path):
    q_0 = sol.states[0]["q"]
    q_1 = sol.states[1]["q"]
    q_2 = sol.states[2]["q"]
    q_holo = np.zeros((bio_model[0].nb_q, sol.states[3]["u"].shape[1]))
    q_4 = sol.states[4]["q"]
    for i, ui in enumerate(sol.states[3]["u"].T):
        # vi = bio_model.compute_v_from_u_numeric(ui, v_init=DM(np.zeros(2))).toarray()
        vi = bio_model[3].compute_v_from_u_explicit_numeric(ui).toarray()
        qi = bio_model[3].q_from_u_and_v(ui[:, np.newaxis], vi).toarray().squeeze()
        q_holo[:, i] = qi
    q = np.concatenate((q_0, q_1, q_2, q_holo, q_4), axis=1)
    visu = bioviz.Viz(model_path)
    visu.load_movement(q)
    visu.exec()

def visualisation_closed_loop_3phases(bio_model, sol, model_path):
    q_0 = sol.states[0]["q"]
    q_holo = np.zeros((bio_model[0].nb_q, sol.states[1]["u"].shape[1]))
    q_2 = sol.states[2]["q"]
    for i, ui in enumerate(sol.states[1]["u"].T):
        vi = bio_model[1].compute_v_from_u_explicit_numeric(ui).toarray()
        qi = bio_model[1].q_from_u_and_v(ui[:, np.newaxis], vi).toarray().squeeze()
        q_holo[:, i] = qi
    q = np.concatenate((q_0, q_holo, q_2), axis=1)
    visu = bioviz.Viz(model_path)
    visu.load_movement(q)
    visu.exec()


# --- Parameters --- #
name_file_model = "/home/mickael/Documents/Anais/Robust_standingBack/Model/Model2D_7Dof_0C_5M_CL_V2.bioMod"
name_file_model_2 = "/home/mickael/Documents/Anais/Robust_standingBack/Model/Model2D_8Dof_2C_5M_2model.bioMod"
name_file_movement = ("/home/mickael/Documents/Anais/Robust_standingBack/Code - examples/Jump-salto/Salto_close_loop_7phases_V1.pkl")

visualisation_model(name_file_model)