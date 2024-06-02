import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
import pandas as pd


def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def df_to_dict(df):
    datadict = {}
    for index, row in df.iterrows():
        dataset = row["dataset"]
        scene = row["scene"]
        path = Path("./dataset/" + row["image_path"])

        if dataset not in datadict:
            datadict[dataset] = {}

        if scene not in datadict[scene]:
            datadict[dataset][scene] = []

        datadict[dataset][scene].append(path)
    for dataset in datadict:
        for scene in datadict[dataset]:
            print(f"{dataset} / {scene} -> {len(datadict[dataset][scene])} images")

    return datadict


def arr_to_str(a):
    return ";".join([str(x) for x in a.reshape(-1)])


def dict_to_df(results, data_dict):
    rows = []
    for dataset in data_dict:
        if dataset in results:
            res = results[dataset]
        else:
            res = {}

        for scene in data_dict[dataset]:
            if scene in res:
                scene_res = res[scene]
            else:
                scene_res = {"R": {}, "t": {}}

            for image in data_dict[dataset][scene]:
                if image in scene_res:
                    R = scene_res[image]["R"].reshape(-1)
                    T = scene_res[image]["t"].reshape(-1)
                else:
                    R = np.eye(3).reshape(-1)
                    T = np.zeros((3))
                image_path = str(image.relative_to(Path("./dataset")))
                rows.append(
                    {
                        "image_path": image_path,
                        "dataset": dataset,
                        "scene": scene,
                        "rotation_matrix": arr_to_str(R),
                        "translation_vector": arr_to_str(T),
                    }
                )
    return pd.DataFrame(rows)


def run(
    input_df,
    get_pairs,
    keypoints_matches,
    ransac_and_sparse_reconstruction,
    imc_datasets_path,
    config,
):
    results = {}
    print(config)
    data_dict = df_to_dict(input_df)
    datasets = list(data_dict.keys())

    for dataset in datasets:
        if dataset not in results:
            results[dataset] = {}

        for scene in data_dict[dataset]:
            images_dir = data_dict[dataset][scene][0].parent
            results[dataset][scene] = {}
            image_paths = data_dict[dataset][scene]

            index_pairs = get_pairs(
                image_paths,
                config["device"],
                config["distances_threshold"],
                config["min_pairs"],
                config["tol"],
                config["exhaustive"],
            )
            keypoints_matches(
                image_paths,
                index_pairs,
                config["device"],
                config["max_kp"],
                config["detection_threshold"],
                config["resize_to"],
                config["min_matches"],
            )
            maps = ransac_and_sparse_reconstruction(
                image_paths[0].parent,
                config["min_model_size"],
                config["max_num_models"],
            )

            images_registered = 0
            best_idx = -1
            for idx, rec in maps.items():
                if len(rec.images) > images_registered:
                    images_registered = len(rec.images)
                    best_idx = idx

            if best_idx > -1:
                for k, im in maps[best_idx].images.items():
                    key = Path(imc_datasets_path) / "train" / scene / "images" / im.name
                    results[dataset][scene][key] = {}
                    results[dataset][scene][key]["R"] = deepcopy(im.rotmat())
                    results[dataset][scene][key]["t"] = deepcopy(np.array(im.tvec))

    return dict_to_df(results, data_dict)
