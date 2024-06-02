from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations
import torch
import torch.nn.functional as F
import kornia as K
from transformers import AutoImageProcessor, AutoModel


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

def get_pairs(images_list, config):
    if config["exhaustive"]:
        return list(combinations(range(len(images_list)), 2))
    
    
    processor = AutoImageProcessor.from_pretrained(
        "/content/drive/MyDrive/cs231n/image-matching-challenge-2024/dinov2/pytorch/base/1/"
    )
    model = (
        AutoModel.from_pretrained(
            "/content/drive/MyDrive/cs231n/image-matching-challenge-2024/dinov2/pytorch/base/1/"
        )
        .eval()
        .to(config["device"])
    )
    embeddings = []

    for img_path in images_list:
        image = K.io.load_image(img_path, K.io.ImageLoadType.RGB32, device=device)[
            None, ...
        ]
        with torch.inference_mode():
            inputs = processor(
                images=image,
                return_tensors="pt",
                do_rescale=False,
                do_resize=True,
                do_center_crop=True,
                size=224,
            ).to(config["device"])
            outputs = model(**inputs)
            embedding = F.normalize(outputs.last_hidden_state.max(dim=1)[0])
        embeddings.append(embedding)

    embeddings = torch.cat(embeddings, dim=0)
    distances = torch.cdist(embeddings, embeddings).cpu()
    distances_ = (distances <= ["distances_threshold"]).numpy()
    np.fill_diagonal(distances_, False)
    z = distances_.sum(axis=1)
    idxs0 = np.where(z == 0)[0]
    for idx0 in idxs0:
        t = np.argsort(distances[idx0])[1:config["min_pairs"]]
        distances_[idx0, t] = True

    s = np.where(distances >= config["tol"])
    distances_[s] = False

    idxs = []
    for i in range(len(images_list)):
        for j in range(len(images_list)):
            if distances_[i][j]:
                idxs += [(i, j)] if i < j else [(j, i)]

    idxs = list(set(idxs))
    return idxs