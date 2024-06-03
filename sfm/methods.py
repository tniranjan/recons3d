import kornia as K
import kornia.feature as KF
from lightglue.utils import load_image
import lightglue
import h5py
import pycolmap
import torch
import datetime, sys
from collections import defaultdict
from PIL import Image
import numpy as np
from copy import deepcopy
sys.path.append("colmap-db-import")
from database import *
from h5_to_db import *
import sfm.d2net
import cv2

def keypoints_matches(images_list, pairs, config, kp_method="ALIKED"):
    if kp_method == "ALIKED":
        extractor = lightglue.ALIKED(
                max_num_keypoints=config["max_kp"],
                detection_threshold=config["detection_threshold"],
                resize=config["resize_to"],
            ).eval().to(config["device"])
    elif kp_method == "SuperPoint":
        extractor = lightglue.SuperPoint(
                max_num_keypoints=config["max_kp"],
                detection_threshold=config["detection_threshold"],
                resize=config["resize_to"],
            ).eval().to(config["device"])
    elif kp_method == "DISK":
        extractor = lightglue.DISK(
                max_num_keypoints=config["max_kp"],
                detection_threshold=config["detection_threshold"],
                resize=config["resize_to"],
            ).eval().to(config["device"])
    elif kp_method == "SIFT":
        extractor = lightglue.SIFT(
                max_num_keypoints=config["max_kp"],
                detection_threshold=config["detection_threshold"],
                resize=config["resize_to"],
            ).eval().to(config["device"])
    elif kp_method == "DoGHardNet":
        extractor = lightglue.DoGHardNet(
                max_num_keypoints=config["max_kp"],
                detection_threshold=config["detection_threshold"],
                resize=config["resize_to"],
            ).eval().to(config["device"])
    elif kp_method == "D2Net":
        extractor = sfm.d2net.D2NetExtractor(max_kp=config["max_kp"], detection_threshold= config["detection_threshold"], resize_to=config["resize_to"]) 
    else:
        raise RuntimeError("Unknown method")
    
    matcher = cv2.BFMatcher(cv2.NORM_L2, True) if kp_method == "D2Net" else KF.LightGlueMatcher(kp_method.lower(),{"width_confidence": -1, "depth_confidence": -1, "mp": True if "cuda" in str(config["device"]) else False, },).eval().to(config["device"])
    
    # rotation = create_model("swsl_resnext50_32x4d").eval().to(device)
    keypoints_fname = "scratch/keypoints.h5"
    desc_fname = "scratch/descriptors.h5"
    matches_fname = "scratch/matches.h5"
    with h5py.File(keypoints_fname, mode="w") as f_kp, h5py.File(
        desc_fname, mode="w"
    ) as f_desc:
        for image_path in images_list:
            with torch.inference_mode():
                if kp_method != "D2Net":
                    image = load_image(image_path).to(config["device"])
                    feats = extractor.extract(image)
                else:
                    feats = extractor.extract(image_path)

                f_kp[image_path.name] = (
                    feats["keypoints"].reshape(-1, 2).detach().cpu().numpy()
                )
                f_desc[image_path.name] = (
                    feats["descriptors"]
                    .reshape(len(f_kp[image_path.name]), -1)
                    .detach()
                    .cpu()
                    .numpy()
                )

    with h5py.File(keypoints_fname, mode="r") as f_kp, h5py.File(
        desc_fname, mode="r"
    ) as f_desc, h5py.File(matches_fname, mode="w") as f_matches:
        for pair in pairs:
            key1, key2 = images_list[pair[0]].name, images_list[pair[1]].name
            if kp_method!="D2Net":
                kp1 = torch.from_numpy(f_kp[key1][...]).to(config["device"])
                kp2 = torch.from_numpy(f_kp[key2][...]).to(config["device"])
                desc1 = torch.from_numpy(f_desc[key1][...]).to(config["device"])
                desc2 = torch.from_numpy(f_desc[key2][...]).to(config["device"])
                with torch.inference_mode():
                    _, idxs = matcher(
                        desc1,
                        desc2,
                        KF.laf_from_center_scale_ori(kp1[None]),
                        KF.laf_from_center_scale_ori(kp2[None]),
                    )
                    idxs = idxs.detach().cpu().numpy().reshape(-1, 2)
            else:
                kp1 = f_kp[key1]
                kp2 = f_kp[key2]
                desc1 = f_desc[key1]
                desc2 = f_desc[key2]
                matches = matcher.match(desc1, desc2 )
                matches = (sorted(matches, key=lambda x: x.distance))
                idxs = np.array([[match.queryIdx, match.trainIdx] for match in matches])

                # marches = matcher(
                #         desc1,
                #         desc2,
                #         KF.laf_from_center_scale_ori(kp1[None]),
                #         KF.laf_from_center_scale_ori(kp2[None]),
                #     )
            if len(idxs):
                group = f_matches.require_group(key1)
            if len(idxs) >= config["min_matches"]:
                group.create_dataset(
                    key2, data=idxs
                )


def ransac_and_sparse_reconstruction(images_path, config):
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    db_name = "scratch/" + f"colmap_{time_str}.db"
    db = COLMAPDatabase.connect(db_name)
    db.create_tables()
    fname_to_id = add_keypoints(
        db, "scratch/", images_path, "", "simple-pinhole", False
    )
    add_matches(db, "scratch/", fname_to_id)
    db.commit()

    pycolmap.match_exhaustive(db_name, sift_options={"num_threads": 8})
    
    maps = pycolmap.incremental_mapping(
        database_path=db_name,
        image_path=images_path,
        output_path=".",
        options=pycolmap.IncrementalMapperOptions(
            {
                "min_model_size": config["min_model_size"],
                "max_num_models": config["max_num_models"],
                "num_threads": 8,
            }
        ),
    )
    return maps


def get_unique_idxs(A, dim=0):
    unique, idx, counts = torch.unique(
        A, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    return first_indices


def match_loftr(images_list, pairs, config):
    matcher = KF.LoFTR(pretrained="outdoor")
    matcher = matcher.to(config["device"]).eval()
    matches_fname = "scratch/matches.h5"
    features_fname = "scratch/keypoints.h5"
    # First we do pairwise matching, and then extract "keypoints" from loftr matches.
    with h5py.File(matches_fname, mode="w") as f_match:
        for pair_idx in pairs:
            idx1, idx2 = pair_idx
            fname1, fname2 = images_list[idx1], images_list[idx2]
            key1, key2 = fname1.name, fname2.name
            # Load img1
            timg1 = K.color.rgb_to_grayscale(
                K.io.load_image(fname1, K.io.ImageLoadType.RGB32)[None, ...]
            )
            H1, W1 = timg1.shape[2:]
            timg_resized1 = K.geometry.resize(
                timg1, config["resize_to"], side="long", antialias=True
            )
            h1, w1 = timg_resized1.shape[2:]

            # Load img2
            timg2 = K.color.rgb_to_grayscale(
                K.io.load_image(fname2, K.io.ImageLoadType.RGB32)[None, ...]
            )
            H2, W2 = timg2.shape[2:]

            timg_resized2 = K.geometry.resize(
                timg2, config["resize_to"], side="long", antialias=True
            )
            h2, w2 = timg_resized2.shape[2:]
            with torch.inference_mode():
                input_dict = {
                    "image0": timg_resized1.to(config["device"]),
                    "image1": timg_resized2.to(config["device"]),
                }
                correspondences = matcher(input_dict)
            mkpts0 = correspondences["keypoints0"].cpu().numpy()
            mkpts1 = correspondences["keypoints1"].cpu().numpy()

            mkpts0[:, 0] *= float(W1) / float(w1)
            mkpts0[:, 1] *= float(H1) / float(h1)

            mkpts1[:, 0] *= float(W2) / float(w2)
            mkpts1[:, 1] *= float(H2) / float(h2)

            n_matches = len(mkpts1)
            group = f_match.require_group(key1)
            if n_matches >= config["min_matches"]:
                group.create_dataset(
                    key2, data=np.concatenate([mkpts0, mkpts1], axis=1)
                )

    # Let's find unique loftr pixels and group them together.
    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts = defaultdict(int)
    with h5py.File(matches_fname, mode="r") as f_match:
        for k1 in f_match.keys():
            group = f_match[k1]
            for k2 in group.keys():
                matches = group[k2][...]
                total_kpts[k1]
                kpts[k1].append(matches[:, :2])
                kpts[k2].append(matches[:, 2:])
                current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0] += total_kpts[k1]
                current_match[:, 1] += total_kpts[k2]
                total_kpts[k1] += len(matches)
                total_kpts[k2] += len(matches)
                match_indexes[k1][k2] = current_match

    for k in kpts.keys():
        kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(
            torch.from_numpy(kpts[k]), dim=0, return_inverse=True
        )
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()
    for k1, group in match_indexes.items():
        for k2, m in group.items():
            m2 = deepcopy(m)
            m2[:, 0] = unique_match_idxs[k1][m2[:, 0]]
            m2[:, 1] = unique_match_idxs[k2][m2[:, 1]]
            mkpts = np.concatenate(
                [
                    unique_kpts[k1][m2[:, 0]],
                    unique_kpts[k2][m2[:, 1]],
                ],
                axis=1,
            )
            unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts), dim=0)
            m2_semiclean = m2[unique_idxs_current]
            unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
            m2_semiclean = m2_semiclean[unique_idxs_current1]
            unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
            m2_semiclean2 = m2_semiclean[unique_idxs_current2]
            out_match[k1][k2] = m2_semiclean2.numpy()
    with h5py.File(features_fname, mode="w") as f_kp:
        for k, kpts1 in unique_kpts.items():
            f_kp[k] = kpts1

    with h5py.File(matches_fname, mode="w") as f_match:
        for k1, gr in out_match.items():
            group = f_match.require_group(k1)
            for k2, match in gr.items():
                group[k2] = match
