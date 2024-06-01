from importlib import reload
import sys
sys.path.append("/home/niranjan/DetectorFreeSfM")
import sfm.metric
import sfm.utils
import sfm.pipeline
import torch
import pandas as pd
import logging
import json
import os

logging.basicConfig(level=logging.WARNING)
def image_path(row):
  row['image_path'] = "train/" + row['dataset'] + '/images/' + row['image_name']
  return row

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = { "device" : device, "exhaustive": True, "min_pairs" : 50, "distances_threshold" : 0.3, "tol" :500, "max_kp" : 4096,"resize_to" : 640,"detection_threshold" : 0.005,
"min_matches" : 100, "min_model_size" : 5, "max_num_models" : 3}
n_samples = 40

root_dir = "."

imc_dataset = f"{root_dir}/dataset"
train_df = pd.read_csv(f'{imc_dataset}/train/train_labels.csv')
train_df = train_df.apply(image_path,axis=1).drop_duplicates(subset=['image_path'])
G = train_df.groupby(['dataset','scene'])['image_path']

scores={}
for g in G:
  image_paths = []
  dataset_name = g[0][0]
  n = n_samples
  n = n if n < len(g[1]) else len(g[1])
  g = g[0],g[1].sample(n,random_state=42).reset_index(drop=True)
  for image_path in g[1]:
    image_paths.append(image_path)
  gt_df = train_df[train_df.image_path.isin(image_paths)].reset_index(drop=True)
  empty_df = gt_df.copy().drop(columns = ['rotation_matrix','translation_vector'])
  pred_df = sfm.pipeline.run(empty_df, sfm.utils.get_pairs, sfm.utils.keypoints_matches, sfm.utils.ransac_and_sparse_reconstruction, imc_dataset, config)
  pred_df.to_csv(root_dir+"/outputs/" + dataset_name + "_pred.csv")
  mAA = round(sfm.metric.score(gt_df, pred_df),4)
  print('*** Total mean Average Accuracy ***')
  print(f"mAA: {mAA}")
  scores[dataset_name] = mAA
  vals = [scores, {key:value for key,value in config.items() if key!="device"}]
  json.dump(vals, open(root_dir + "/outputs/scores.json","w"))