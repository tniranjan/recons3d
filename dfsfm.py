import pycolmap
import sfm.metric
import sfm.utils
import pandas as pd
import os
from copy import deepcopy
import shutil
import yaml
import json
import subprocess
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
import numpy as np
from pathlib import Path

def colmap_to_df(recons_path, input_df):
    recons = pycolmap.Reconstruction()
    recons.read_binary(recons_path)
    results = {}        
    data_dict = sfm.utils.df_to_dict(input_df)
    datasets = list(data_dict.keys())

    for dataset in datasets:
        if dataset not in results:
            results[dataset] = {}

        for scene in data_dict[dataset]:
            images_dir = data_dict[dataset][scene][0].parent
            results[dataset][scene] = {}
            image_paths = data_dict[dataset][scene]

            for k, im in recons.images.items():
                key = Path("./dataset/train") / scene / "images" / im.name
                results[dataset][scene][key] = {}
                results[dataset][scene][key]["R"] = deepcopy(im.rotmat())
                results[dataset][scene][key]["t"] = deepcopy(np.array(im.tvec))

    return sfm.utils.dict_to_df(results, data_dict)

def image_path(row):
    row['image_path'] = "train/" + row['dataset'] + '/images/' + row['image_name']
    return row

def compute_score(recons_path, dataset):
    gt_df =  pd.read_csv("./dataset/train/train_labels.csv")
    gt_df = gt_df[gt_df["dataset"]==dataset]
    gt_df = gt_df.apply(image_path, axis=1).drop_duplicates(subset=['image_path'])
    aliked_df = pd.read_csv("outputs/ALIKED_" + dataset + "_pred.csv")
    gt_subset_df = pd.merge(aliked_df[["image_path"]], gt_df, on = "image_path")
    colmap_df = colmap_to_df(recons_path, gt_subset_df)
    colmap_df.to_csv("outputs/dfsfm_" + dataset + "_pred.csv")    
    return sfm.metric.score(gt_subset_df, colmap_df)

def update_dfsfmyaml(dataset):
    file_path =  "/home/niranjan/DetectorFreeSfM/hydra_configs/demo/dfsfm.yaml"
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(file_path, 'r') as f:
        data = yaml.load(f)
    field = "scene_list"
    val = dataset + "_trimmed"
    keys = field.split('.')
    d = data
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = DoubleQuotedScalarString(val)

    # Write back to the YAML file
    with open(file_path, 'w') as f:
        yaml.dump(data, f)


def make_dfsfm_data(dataset):
    ref_df = pd.read_csv("outputs/ALIKED_" + dataset + "_pred.csv")
    for id, row in ref_df.iterrows():
        src_path = "dataset/" + row["image_path"]
        dst_path = src_path.replace(dataset, dataset + "_trimmed")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)
    update_dfsfmyaml(dataset)

def run_dfsfm():
    datasets = ["multi-temporal-temple-baalshamin", "pond", "transp_obj_glass_cup", "transp_obj_glass_cylinder"]
    # datasets = ["dioscuri","lizard", "multi-temporal-temple-baalshamin", "pond", "transp_obj_glass_cup", "transp_obj_glass_cylinder", "church"]
    scores = {}
    for dataset in datasets:
        make_dfsfm_data(dataset)
        cwd = os.getcwd()
        os.chdir("/home/niranjan/DetectorFreeSfM")
        out = subprocess.run(["python", "eval_dataset.py", "+demo=dfsfm.yaml"], env=os.environ.copy(), capture_output=True, text=True)
        os.chdir(cwd)
        print(out)
        scores[dataset] = compute_score("/home/niranjan/recons3d/dataset/train/" + dataset + "_trimmed/DetectorFreeSfM_loftr_official_coarse_fine__scratch_no_intrin/colmap_refined", dataset)
        json.dump(scores,open("outputs/dfsfm_scores.json", "w"))

if __name__ == "__main__":
    run_dfsfm()