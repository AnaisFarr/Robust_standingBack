import os
import numpy as np
import pickle
import rerun as rr

from pyorerun import PhaseRerun, BiorbdModel
from pyorerun.multi_frame_rate_phase_rerun import MultiFrameRatePhaseRerun

folder = "backflip_Vpost_submission_v3/"
# folder_HTC = folder + "HTC"
folder_KTC = folder + "ktc"
folder_FREE = folder + "ntc"
folder_HTC = folder + "htc"
model_path = "../models/Model2D_7Dof_2C_5M_CL_V3.bioMod"

# get matplotlib colors instead
colors = [
    # tab:blue
    (31, 119, 180),
    # tab:orange
    (255, 127, 14),
    # tab:green
    (44, 160, 44),
    # tab:red
    (214, 39, 40),
    # tab:purple
    (148, 103, 189),
    # tab:brown
    (140, 86, 75),
    # tab:pink
    (227, 119, 194),
    # tab:gray
    (127, 127, 127),
    # tab:olive
    (188, 189, 34),
    # tab:cyan
    (23, 190, 207),
]
colors = colors + colors + colors  # duplicate the colors to have enough for all the phases

# Charger les données
phase_reruns = []
# for config, (folder, str_suffix) in enumerate(zip([folder_KTC,  folder_FREE], ["KTC",  "NTC"])):
for config, (folder, str_suffix) in enumerate(zip([folder_HTC, folder_KTC, folder_FREE], ["htc", "ktc", "ntc"])):
# for config, (folder, str_suffix) in enumerate(zip([folder_FREE], ["ntc"])):
    #  get the number of _CVG.pkl files in the folder
    n_files = len([name for name in os.listdir(folder) if name.endswith("G.pkl") and not name.__contains__("no_seed")])
    print(folder)
    print(n_files)
    all_cost = np.array([])
    for i in range(0, n_files):

        suffix = "CVG"
        if os.path.isfile(folder + f"/sol_{i}_CVG.pkl") is False:
            all_cost = np.append(all_cost, np.inf)
            continue
            # suffix = "DVG"

        data = pickle.load(open(folder + f"/sol_{i}_{suffix}.pkl", "rb"))
        time = np.concatenate([np.array(time) for time in data["time"]], axis=0).squeeze()
        q = np.concatenate([np.array(q) for q in data["q"]], axis=1)
        all_cost = np.append(all_cost, np.array(data["cost"]))
        # # offset
        q[0, :] = q[0, :] + (config - 1)

        phase_reruns.append(PhaseRerun(t_span=time, window=f"simulation_{str_suffix}_{i}"))
        m = BiorbdModel(model_path)
        if suffix == "DVG":
            m.options.mesh_color = (0, 0, 0)
        else:
            m.options.mesh_color = colors[i - 1]
        phase_reruns[-1].add_animated_model(m, q)

    where_min = np.argmin(all_cost)
    data = pickle.load(open(folder + f"/sol_{where_min}_{suffix}.pkl", "rb"))

mrr2 = MultiFrameRatePhaseRerun(phase_reruns=phase_reruns)
mrr2.rerun_by_frame("NTC_KTC_HTC")


rr.log(
    "world/cam",
    rr.Pinhole(
        fov_y=0.7853982,
        aspect_ratio=1.7777778,
        camera_xyz=rr.ViewCoordinates.FLU,
        #image_plane_distance=3,
    ),
)
rr.log(
    "world/cam",
    rr.Transform3D(
        translation=[-3, 0, 1.05],
    ),
)
