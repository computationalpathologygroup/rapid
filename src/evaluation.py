import numpy as np
import pathlib
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
import copy
from typing import List, Any
from scipy.ndimage import zoom
from scipy.spatial.distance import directed_hausdorff, cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

from keypoints import get_keypoints


def compute_dice(mask1: np.ndarray, mask2: np.ndarray, normalized: bool = False) -> float:
    """
    Function to compute the dice score between two 2D masks.
    """

    assert len(mask1.shape) == 2, "mask1 must be 2 dimensional"
    assert len(mask2.shape) == 2, "mask2 must be 2 dimensional"
    assert len(np.unique(mask1)) == 2, "mask1 must be binary"
    assert len(np.unique(mask2)) == 2, "mask2 must be binary"

    # Normalize
    mask1 = (mask1 / np.max(mask1)).astype("uint8")
    mask2 = (mask2 / np.max(mask2)).astype("uint8")

    # Compute dice
    eps = 1e-6
    intersection = np.sum(mask1 * mask2)
    dice = (2. * intersection) / (np.sum(mask1) + np.sum(mask2) + eps)

    # Normalize by max achievable dice
    if normalized:
        dice = (2. * intersection) / (2*np.min([np.sum(mask1), np.sum(mask2)]) + eps)

    return dice


def compute_reconstruction_dice(masks: List, normalized: bool = False) -> float:
    """
    Function to compute the dice score between all masks in a list.
    """

    # Compute dice between all masks
    dice_scores = []

    for i in range(len(masks)-1):
        dice_scores.append(compute_dice(mask1 = masks[i], mask2 = masks[i+1], normalized=normalized))

    return [np.round(i, 3) for i in dice_scores]


def compute_orientation_accuracy(gt_orientations: dict, pred_orientations: dict, ref_idx: int) -> dict:
    """
    Function to compute the orientation accuracy between two lists of orientations.
    """ 
    
    # Get values sorted by key
    gt_orientations_ = np.array([gt_orientations[k] for k in sorted(gt_orientations.keys())])
    pred_orientations_ = np.array([pred_orientations[k] for k in sorted(pred_orientations.keys())])

    # Move everything to positive range, since -90 == +270 for rotation
    gt_orientations_ = [i+360 if i < 0 else i for i in gt_orientations_]
    pred_orientations_ = [i+360 if i < 0 else i for i in pred_orientations_]
    
    # Scale back to [0, 360] range as there may be some wrap-around
    gt_orientations_ = [i%360 for i in gt_orientations_]
    pred_orientations_ = [i%360 for i in pred_orientations_]

    # Compute delta with ref slide and scale to [0, 360] range again
    gt_deltas = gt_orientations_ - gt_orientations_[ref_idx]
    gt_deltas = [i+360 if i < 0 else i for i in gt_deltas]
    pred_deltas = pred_orientations_ - pred_orientations_[ref_idx]
    pred_deltas = [i+360 if i < 0 else i for i in pred_deltas]

    # Compute absolute difference and handle edge cases where deltas are close to 360
    deltas = np.abs(np.array(gt_deltas) - np.array(pred_deltas)).tolist()
    deltas = [np.min([i, 360-i]) for i in deltas]
    
    orientation_thres = 15
    orientation_correct = np.all([i<orientation_thres for i in deltas])
    orientation_accuracy = [np.round(i, 1) for i in deltas]
    
    return orientation_correct, orientation_accuracy, gt_deltas, pred_deltas


def compute_tre_keypoints(images: List, masks: List, detector: Any, matcher: Any, detector_name: str, savedir: pathlib.Path, spacing: float, thres: float = 0.9) -> float:
    """
    Function to compute the target registration error between two sets of keypoints
    """

    from visualization import plot_tre_per_pair

    tre_per_pair = []

    # Detect keypoints, match and compute TRE
    for c in range(len(images)-1):

        # Get keypoints
        ref_points, moving_points, scores = get_keypoints(
            detector = detector, 
            matcher = matcher,
            detector_name = detector_name,
            ref_image = images[c], 
            moving_image = images[c+1],
            ref_mask = masks[c],
            moving_mask = masks[c+1],
            patch_method = "regular",
            thres = thres
        )

        # Compute median TRE
        tre = np.median(np.linalg.norm(ref_points - moving_points, axis=-1))

        # Scale w.r.t. pixel spacing
        scaled_tre = tre * spacing

        if np.isnan(scaled_tre):
            print(f"Warning: nan value for TRE between images {c} and {c+1}, found {len(ref_points)} keypoints.")

        tre_per_pair.append(scaled_tre)

        savepath = savedir.joinpath("evaluation", f"tre_{c}_{c+1}.png")
        plot_tre_per_pair(
            ref_image = images[c], 
            moving_image = images[c+1], 
            ref_points = ref_points, 
            moving_points = moving_points, 
            scores = scores,
            tre = scaled_tre,
            savepath = savepath
        )

    return [np.round(i, 1) if not np.isnan(i) else np.nan for i in tre_per_pair]


def compute_tre_landmarks(images: List, landmarks: List, savedir: pathlib.Path) -> float:
    """
    Function to compute the target registration error between two sets of landmarks.
    """
    
    from visualization import plot_tre_landmarks
    
    tre_per_pair = []
    
    for i in range(len(images)-1):
        
        tre = np.mean(np.linalg.norm(landmarks[i] - landmarks[i+1], axis=-1))
        tre_per_pair.append(tre)
        
        save_path = savedir.joinpath("evaluation", f"landmarks_{i}_{i+1}.png")
        plot_tre_landmarks(
            images = images[i:i+2],
            landmarks = landmarks[i:i+2],
            tre = tre,
            save_path = save_path
        )
    
    return [np.round(i, 1) for i in tre_per_pair]

def compute_reconstruction_hausdorff(masks: List, level: int, spacing: float) -> float:
    """
    Function to compute the Hausdorff distance between adjacent contours.
    """

    hausdorff_per_pair = []

    # Compute hausdorff between all masks
    for i in range(len(masks)-1):

        # Get contours
        contour_a, _ = cv2.findContours(masks[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_a = np.squeeze(max(contour_a, key=cv2.contourArea))
        contour_b, _ = cv2.findContours(masks[i+1], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_b = np.squeeze(max(contour_b, key=cv2.contourArea))

        # Compute hausdorff
        hausdorff = np.max([directed_hausdorff(contour_a, contour_b)[0], directed_hausdorff(contour_a, contour_b)[0]])

        # Scale w.r.t. pixel spacing
        scale_factor = spacing * 2**level
        scaled_hausdorff = hausdorff * scale_factor

        hausdorff_per_pair.append(scaled_hausdorff)

    return int(np.mean(hausdorff_per_pair))


def compute_contour_distance(masks: List, level: int, spacing: float) -> float:
    """
    Function to compute the median distance between adjacent contours.
    """

    dist_per_pair = []

    # Compute median contour distance between all masks
    for i in range(len(masks)-1):

        # Get contours
        contour_a, _ = cv2.findContours(masks[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_a = np.squeeze(max(contour_a, key=cv2.contourArea))
        contour_b, _ = cv2.findContours(masks[i+1], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_b = np.squeeze(max(contour_b, key=cv2.contourArea))

        # Compute hausdorff
        distance = np.median(np.min(cdist(contour_a, contour_b), axis=1))

        # Scale w.r.t. pixel spacing
        scale_factor = spacing * 2**level
        scaled_distance = distance * scale_factor

        dist_per_pair.append(scaled_distance)

    return int(np.mean(dist_per_pair))

def compute_cosine_similarity(f1: np.ndarray, f2: np.ndarray, f1_mask: np.ndarray, f2_mask: np.ndarray, masked: bool = True) -> float:
    """
    Function to compute the cosine similarity between two sets of features.
    """
    
    def mask_bg_features(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Function to mask background features.
        """
               
        # Mask bg features
        mask = cv2.resize(mask, (int(np.sqrt(features.shape[0])), int(np.sqrt(features.shape[0])))).reshape(-1)
        fg_features = copy.deepcopy(features)
        fg_features[mask==0] = 0
        
        return fg_features
    
    # Reshape features to 2D and convert to torch tensors
    f1_flat = f1.reshape(-1, f1.shape[-1])
    f2_flat = f2.reshape(-1, f2.shape[-1])

    # Compute cosine similarity for foreground features only
    if masked:
        f1_flat = mask_bg_features(f1_flat, f1_mask)
        f2_flat = mask_bg_features(f2_flat, f2_mask)
    
    # Compute cosine similarity
    f1_tensor = torch.from_numpy(f1_flat).to("cuda")
    f2_tensor = torch.from_numpy(f2_flat).to("cuda")
    mean_cosine_similarity = F.cosine_similarity(f1_tensor, f2_tensor, dim=1).mean().item()

    return mean_cosine_similarity
    
