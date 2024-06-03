import numpy as np
import torch
import imageio.v2 as imageio
import cv2
import sys

# Importing D2Net related modules from third-party directory
sys.path.append("third_party/d2net/")
from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale


class D2NetExtractor:
    def __init__(self, max_kp, detection_threshold, resize_to=640, model_file="/home/niranjan/recons3d/third_party/d2net/models/d2_tf_no_phototourism.pth"):
        self.max_kp = max_kp
        self.detection_threshold = detection_threshold
        self.resize_to = resize_to

        # CUDA
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")

        # Creating CNN model
        self.model = D2Net(
            model_file=model_file,
            use_relu=True,
            use_cuda=self.use_cuda
        )

    def sort_and_filter(self, A, B, scores, max_rows, score_threshold):
        # Convert to numpy arrays if they aren't already
        A = np.array(A)
        B = np.array(B)
        scores = np.array(scores)

        # Sort by scores in descending order
        sorted_indices = np.argsort(scores)[::-1]
        A = A[sorted_indices]
        B = B[sorted_indices]
        scores = scores[sorted_indices]

        # Filter based on score threshold
        valid_indices = scores >= score_threshold
        A = A[valid_indices]
        B = B[valid_indices]
        scores = scores[valid_indices]

        # Limit the number of rows to max_rows
        if len(scores) > max_rows:
            A = A[:max_rows]
            B = B[:max_rows]
            scores = scores[:max_rows]

        return A, B, scores

    def extract(self, im_path):
        image = imageio.imread(im_path)
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        # Resize image if needed
        resized_image = image
        if max(resized_image.shape) > self.resize_to:
            resized_image = cv2.resize(
                resized_image, None, None,
                self.resize_to / max(resized_image.shape), self.resize_to / max(resized_image.shape)
            ).astype('float')
        if sum(resized_image.shape[:2]) > 2800:
            resized_image = cv2.resize(
                resized_image, None, None,
                2800 / sum(resized_image.shape[:2]), 2800 / sum(resized_image.shape[:2])
            ).astype('float')

        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        input_image = preprocess_image(
            resized_image,
            preprocessing="caffe"
        )
        with torch.no_grad():
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=self.device
                ),
                self.model,
                scales=[1]
            )

        # Input image coordinates
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j
        # i, j -> u, v
        keypoints = keypoints[:, [1, 0, 2]]
        keypoints, descriptors, scores = self.sort_and_filter(keypoints, descriptors, scores / scores.max(), self.max_kp, self.detection_threshold)

        return {"keypoints": keypoints, "descriptors": descriptors, "scores": scores}

