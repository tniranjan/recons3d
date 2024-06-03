from importlib import reload
import sys
sys.path.append("/home/niranjan/DetectorFreeSfM")
import sfm.metric
import sfm.utils
import sfm.methods
import torch
import pandas as pd
import logging
import json
import os
import numpy as np
from pathlib import Path
from copy import deepcopy
import pandas as pd

logging.basicConfig(level=logging.WARNING)

class SFMPipeline():
    def __init__(self, root_dir, n_samples=40):
        self.root_dir = root_dir
        self.n_samples = n_samples
        self.imc_dataset = f"{self.root_dir}/dataset"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = {
            "device": self.device,
            "exhaustive": True,
            "min_pairs": 50,
            "distances_threshold": 0.3,
            "tol": 500,
            "max_kp": 4096,
            "resize_to": 640,
            "detection_threshold": 0.005,
            "min_matches": 100,
            "min_model_size": 5,
            "max_num_models": 3
        }
        logging.basicConfig(level=logging.WARNING)
        self.scores = {}
        sfm.utils.reset_seed(42)

    @staticmethod
    def image_path(row):
        row['image_path'] = "train/" + row['dataset'] + '/images/' + row['image_name']
        return row

    def load_data(self):
        df = pd.read_csv(f'{self.imc_dataset}/train/train_labels.csv')
        self.df = df.apply(self.image_path, axis=1).drop_duplicates(subset=['image_path'])

    def run(self, input_df,kp_method):
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

                index_pairs = sfm.utils.get_pairs(image_paths, self.config)
                if (kp_method=="LOFTR"):
                    sfm.methods.match_loftr(image_paths, index_pairs, self.config)
                else:
                    sfm.methods.keypoints_matches(image_paths, index_pairs, self.config, kp_method)
                maps = sfm.methods.ransac_and_sparse_reconstruction(image_paths[0].parent, self.config)

                images_registered = 0
                best_idx = -1
                for idx, rec in maps.items():
                    if len(rec.images) > images_registered:
                        images_registered = len(rec.images)
                        best_idx = idx

                if best_idx > -1:
                    for k, im in maps[best_idx].images.items():
                        key = Path(self.imc_dataset) / "train" / scene / "images" / im.name
                        results[dataset][scene][key] = {}
                        results[dataset][scene][key]["R"] = deepcopy(im.rotmat())
                        results[dataset][scene][key]["t"] = deepcopy(np.array(im.tvec))

        return sfm.utils.dict_to_df(results, data_dict)

    def evaluate(self, kp_method):
        print("Evaluating for " + kp_method)
        self.load_data()
        G = self.df.groupby(['dataset', 'scene'])['image_path']
        
        for g in G:
            image_paths = []
            dataset_name = g[0][0]            
            n = self.n_samples
            n = n if n < len(g[1]) else len(g[1])
            g = g[0], g[1].sample(n, random_state=42).reset_index(drop=True)
            
            for image_path in g[1]:
                image_paths.append(image_path)
            
            gt_df = self.df[self.df.image_path.isin(image_paths)].reset_index(drop=True)
            empty_df = gt_df.copy().drop(columns=['rotation_matrix', 'translation_vector'])
            
            pred_df = self.run(empty_df,kp_method)
            
            pred_df.to_csv(f"{self.root_dir}/outputs/{kp_method}_{dataset_name}_pred.csv")
            mAA = round(sfm.metric.score(gt_df, pred_df), 4)
            
            print('*** Total mean Average Accuracy ***')
            print(f"mAA: {mAA}")
            
            self.scores[dataset_name] = mAA
            
            vals = [self.scores, {key: value for key, value in self.config.items() if key != "device"}]
            json.dump(vals, open(f"{self.root_dir}/outputs/{kp_method}_scores.json", "w"))