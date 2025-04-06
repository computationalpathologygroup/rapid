import numpy as np
import torch
import cv2
import kornia as K
import warnings
import copy
from typing import List, Any, Callable
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
warnings.filterwarnings("ignore", category=UserWarning, module='torch.*')

from lightglue.utils import rbd


def divide_image_in_patches(image: np.ndarray) -> tuple[List[np.ndarray], List[tuple[int, int]]]:
    """
    Divide image in a 2x2 grid with 4 patches.
    """
    
    # Divide into patches
    ul = image[:image.shape[0]//2, :image.shape[1]//2]
    ur = image[:image.shape[0]//2, image.shape[1]//2:]
    ll = image[image.shape[0]//2:, :image.shape[1]//2]
    lr = image[image.shape[0]//2:, image.shape[1]//2:]
    patches = [ul, ur, ll, lr]
    offsets = [(0, 0), (image.shape[0]//2, 0), (0, image.shape[1]//2), (image.shape[0]//2, image.shape[1]//2)]

    return patches, offsets


def get_keypoints(detector: Any, matcher: Any, detector_name: str, ref_image: np.ndarray, moving_image: np.ndarray, ref_mask: np.ndarray, moving_mask: np.ndarray, patch_method: str = "regular", thres: float = 0.9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper function to get keypoints from any of the supported detectors.
    
    Methods:
     - regular: keypoints are detected and matched in the original image.
     - patch: image is divided in 2x2 patches, keypoints are detected at patch-level and matched at image-level.
     - parcellated: image is divided in 2x2 patches, keypoints are detected and matched at patch-level.
    """

    assert patch_method in ["regular", "patch", "parcellated"]

    if detector_name == "superpoint":
        ref_points, moving_points, scores = get_lightglue_keypoints(ref_image, moving_image, detector, matcher, patch_method)
    elif detector_name == "sift":
        ref_points, moving_points, scores = get_sift_keypoints(ref_image, moving_image, detector, matcher, patch_method)
    elif detector_name == "roma":
        ref_points, moving_points, scores = get_roma_keypoints(ref_image, moving_image, matcher, patch_method, thres)
    elif detector_name == "dedode":
        ref_points, moving_points, scores = get_dedode_keypoints(ref_image, moving_image, detector, matcher, patch_method)
    elif detector_name == "omniglue":
        ref_points, moving_points, scores = get_omniglue_keypoints(ref_image, moving_image, detector, matcher, patch_method)

    # Filter keypoints detected erroneously outside the masked part of the image
    ref_points_filtered, moving_points_filtered, scores_filtered = [], [], []
    for ref_point, moving_point, score in zip(ref_points, moving_points, scores):
        if ref_mask[int(ref_point[0]), int(ref_point[1])]>0 and moving_mask[int(moving_point[0]), int(moving_point[1])]>0:
            ref_points_filtered.append(ref_point)
            moving_points_filtered.append(moving_point)
            scores_filtered.append(score)

    ref_points_filtered, moving_points_filtered, scores_filtered = np.array(ref_points_filtered), np.array(moving_points_filtered), np.array(scores_filtered)

    return ref_points_filtered, moving_points_filtered, scores_filtered


def get_keypoints_parcellated(ref_image: np.ndarray, moving_image: np.ndarray, detect_fn: Callable, match_fn: Callable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic function to get matching keypoints in parcellated mode.
    Finds keypoints locally (per quadrant) and matches locally (per quadrant). 
    """
    
    # Divide image in 4 quadrants
    xmid = ref_image.shape[1]//2
    ymid = ref_image.shape[0]//2
    xstart = [xmid, 0, xmid, 0]
    xend = [-1, xmid, -1, xmid]
    ystart = [0, 0, ymid, ymid]
    yend = [ymid, ymid, -1, -1]
    
    all_ref_points, all_moving_points, all_scores = [], [], []
    
    # Restrict keypoint matching to each quadrant
    for i, (x1, x2, y1, y2) in enumerate(zip(xstart, xend, ystart, yend)):
        
        # Create quadrant by setting non-quadrant pixels to white
        ref_image_quadrant = copy.copy(ref_image)
        ref_image_quadrant[x1:x2, :] = np.max(ref_image)
        ref_image_quadrant[:, y1:y2] = np.max(ref_image)
        
        moving_image_quadrant = copy.copy(moving_image)
        moving_image_quadrant[x1:x2, :] = np.max(moving_image)
        moving_image_quadrant[:, y1:y2] = np.max(moving_image)
        
        # Use the provided keypoint function
        features = detect_fn(ref_image_quadrant, moving_image_quadrant)
        ref_points, moving_points, scores = match_fn(*features)
        
        if ref_points.shape[0] > 0: 
            all_ref_points.append(ref_points)
            all_moving_points.append(moving_points)
            all_scores.append(scores)
    
    all_ref_points = np.concatenate(all_ref_points)
    all_moving_points = np.concatenate(all_moving_points)
    all_scores = np.concatenate(all_scores)
    
    return all_ref_points, all_moving_points, all_scores

def get_lightglue_keypoints(ref_image: np.ndarray, moving_image: np.ndarray, detector: Any, matcher: Any, patch_method: str = "regular") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints with LightGlue
    """
    
    def detect(ref_img: np.ndarray, moving_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Convert images to tensor
        ref_tensor = torch.tensor(ref_img.transpose((2, 0, 1)) / 255., dtype=torch.float).cuda()
        moving_tensor = torch.tensor(moving_img.transpose((2, 0, 1)) / 255., dtype=torch.float).cuda()

        # Extract features and match
        with torch.inference_mode():
            ref_features = detector.extract(ref_tensor)
            moving_features = detector.extract(moving_tensor)
            
        return ref_features, moving_features
            
    def match(ref_features: torch.Tensor, moving_features: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.inference_mode():
            matches01 = matcher({'image0': ref_features, 'image1': moving_features})

        # Extract matches
        ref_features, moving_features, matches01 = [rbd(x) for x in [ref_features, moving_features, matches01]] 
        matches = matches01['matches']

        # Get matching keypoints
        ref_points = np.float32([i.astype("int") for i in ref_features['keypoints'][matches[..., 0]].detach().cpu().numpy()])
        moving_points = np.float32([i.astype("int") for i in moving_features['keypoints'][matches[..., 1]].detach().cpu().numpy()])
        scores = matches01["scores"].detach().cpu().numpy()
        
        return ref_points, moving_points, scores

    if patch_method == "parcellated":
        return get_keypoints_parcellated(ref_image, moving_image, detect, match)
    elif patch_method == "patch":
        return get_lightglue_keypoints_patch_based(ref_image, moving_image, detect, match)
    else:
        return match(*detect(ref_image, moving_image))


def get_lightglue_keypoints_patch_based(ref_image: np.ndarray, moving_image: np.ndarray, detect_fn: Callable, match_fn: Callable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints in patch-based mode.
    Finds keypoints locally, but matches globally. 
    """
    
    # Divide image in nxn equal patches
    ref_patches, ref_offsets = divide_image_in_patches(ref_image)
    moving_patches, moving_offsets = divide_image_in_patches(moving_image)
    
    # Detect keypoints in each patch individually
    all_features = [detect_fn(i, j) for i, j in zip(ref_patches, moving_patches)]
    ref_features = [i[0] for i in all_features]
    moving_features = [i[1] for i in all_features]
    
    # Add offset to keypoints to account for patch offsets
    device = ref_features[0]["keypoints"].device
    ref_offsets = [torch.tensor(offset, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) for offset in ref_offsets]
    moving_offsets = [torch.tensor(offset, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) for offset in moving_offsets]
    
    for rf, rf_offset, mf, mf_offset in zip(ref_features, ref_offsets, moving_features, moving_offsets):
        rf["keypoints"] = rf["keypoints"] + rf_offset
        mf["keypoints"] = mf["keypoints"] + mf_offset
    
    # Concatenate features from all patches
    keys = ref_features[0].keys()
    ref_features_all = {j: torch.cat([i[j] for i in ref_features], dim=1) for j in keys}
    moving_features_all = {j: torch.cat([i[j] for i in moving_features], dim=1) for j in keys}
    
    # Update image size
    ref_features_all["image_size"] = torch.tensor([ref_image.shape[0], ref_image.shape[1]], dtype=torch.float32, device=device)
    moving_features_all["image_size"] = torch.tensor([moving_image.shape[0], moving_image.shape[1]], dtype=torch.float32, device=device)
    
    # Match keypoints globally
    ref_points, moving_points, scores = match_fn(ref_features_all, moving_features_all)
    
    return ref_points, moving_points, scores
    

def get_sift_keypoints(ref_image: np.ndarray, moving_image: np.ndarray, detector: Any, matcher: Any, patch_method: str = "regular") -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints with classical SIFT.
    """

    def detect(ref_image: np.ndarray, moving_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Convert to grayscale
        ref_image_gray = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY)
        moving_image_gray = cv2.cvtColor(moving_image, cv2.COLOR_RGB2GRAY)

        # Detect keypoints
        ref_points, ref_features = detector.detectAndCompute(ref_image_gray, None)
        moving_points, moving_features = detector.detectAndCompute(moving_image_gray, None)
        
        return ref_points, moving_points, ref_features, moving_features

    def match(ref_points: np.ndarray, moving_points: np.ndarray, ref_features: np.ndarray, moving_features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Match keypoints
        matches = matcher.knnMatch(ref_features, moving_features, k=2)

        # Apply Lowes ratio test
        matches_filtered = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                matches_filtered.append(m)

        ref_points = np.float32([ref_points[m.queryIdx].pt for m in matches_filtered])
        moving_points = np.float32([moving_points[m.trainIdx].pt for m in matches_filtered])

        max_distance = np.max([m.distance for m in matches_filtered])
        scores = np.array([1 - m.distance/max_distance for m in matches_filtered])

        return ref_points, moving_points, scores

    if patch_method == "regular":
        return match(*detect(ref_image, moving_image))
    elif patch_method == "parcellated":
        return get_keypoints_parcellated(ref_image, moving_image, detect, match)
    elif patch_method == "patch":
        return get_sift_keypoints_patch_based(ref_image, moving_image, detect, match)


def get_sift_keypoints_patch_based(ref_image: np.ndarray, moving_image: np.ndarray, detect_fn: Callable, match_fn: Callable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints in patch-based mode.
    Finds keypoints locally, but matches globally. 
    """
    
    # Divide image in nxn equal patches
    ref_patches, ref_offsets = divide_image_in_patches(ref_image)
    moving_patches, moving_offsets = divide_image_in_patches(moving_image)
    
    # Get features from each patch
    all_features = [detect_fn(i, j) for i, j in zip(ref_patches, moving_patches)]
    
    # Separate features
    ref_points_list = [f[0] for f in all_features]
    moving_points_list = [f[1] for f in all_features]
    ref_features_list = [f[2] for f in all_features]
    moving_features_list = [f[3] for f in all_features]
    
    # Add offsets to keypoint coordinates
    for i, (ref_kps, moving_kps, offset_ref, offset_moving) in enumerate(zip(ref_points_list, moving_points_list, ref_offsets, moving_offsets)):
        for kp in ref_kps:
            pt = list(kp.pt)
            pt[0] += offset_ref[0]
            pt[1] += offset_ref[1]
            kp.pt = tuple(pt)
            
        for kp in moving_kps:
            pt = list(kp.pt)
            pt[0] += offset_moving[0] 
            pt[1] += offset_moving[1]
            kp.pt = tuple(pt)
    
    # Concatenate all keypoints and features
    ref_points = np.hstack(ref_points_list)
    moving_points = np.hstack(moving_points_list)
    ref_features = np.vstack(ref_features_list)
    moving_features = np.vstack(moving_features_list)
    
    # Match all keypoints globally
    ref_points, moving_points, scores = match_fn(ref_points, moving_points, ref_features, moving_features)
    
    return ref_points, moving_points, scores


def get_roma_keypoints(ref_image: np.ndarray, moving_image: np.ndarray, matcher: Any, patch_method: str = "regular", thres: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints with ROMA
    """
    
    def detect_and_match(ref_image: np.ndarray, moving_image: np.ndarray, thres: float) -> tuple[np.ndarray, np.ndarray]:
        
        # Resize to ROMA scale
        ROMA_SIZE = 560
        ref_image_save = cv2.resize(ref_image, (ROMA_SIZE, ROMA_SIZE))
        moving_image_save = cv2.resize(moving_image, (ROMA_SIZE, ROMA_SIZE))

        with torch.inference_mode():
            # Match images directly
            warp, certainty = matcher.match(Image.fromarray(ref_image_save), Image.fromarray(moving_image_save))

        # Convert to pixel coordinates
        matches, certainty = matcher.sample(warp, certainty)
        ref_points, moving_points = matcher.to_pixel_coordinates(matches, ROMA_SIZE, ROMA_SIZE, ROMA_SIZE, ROMA_SIZE)

        # Scale to original pixel size
        upscale_x = ref_image.shape[1] / ROMA_SIZE
        upscale_y = ref_image.shape[0] / ROMA_SIZE
        ref_points = torch.stack([ref_points[:, 0] * upscale_x, ref_points[:, 1] * upscale_y], dim=1).cpu().numpy()
        moving_points = torch.stack([moving_points[:, 0] * upscale_x, moving_points[:, 1] * upscale_y], dim=1).cpu().numpy()

        # Only keep matches with high certainty
        ref_points = ref_points[certainty.cpu().numpy() > thres]
        moving_points = moving_points[certainty.cpu().numpy() > thres]
        scores = certainty.cpu().numpy()[certainty.cpu().numpy() > thres]
        
        del matches, certainty, warp
        torch.cuda.empty_cache()
        
        return ref_points, moving_points, scores

    if patch_method == "regular":
        return detect_and_match(ref_image, moving_image, thres)
    elif patch_method == "patch":
        return get_roma_keypoints_patch_based(ref_image, moving_image, matcher, detect_and_match, thres)
    elif patch_method == "parcellated":
        return get_roma_keypoints_parcellated(ref_image, moving_image, matcher, detect_and_match, thres)

def get_roma_keypoints_patch_based(ref_image: np.ndarray, moving_image: np.ndarray, matcher: Any, detect_match_fn: Callable, thres: float) -> tuple[np.ndarray, np.ndarray]:  
    """
    Function to get matching keypoints in patch-based mode.
    Note that ROMA performs detection and matching at the same 
    time and, in contrast to other detectors, cannot perform
    local detection and global matching. Hence, this function
    is similar to ROMA's parcellated function. 
    """
    
    # Divide image in nxn equal patches
    ref_patches, ref_offsets = divide_image_in_patches(ref_image)
    moving_patches, moving_offsets = divide_image_in_patches(moving_image)
    
    # Get features from each patch
    all_features = [detect_match_fn(i, j, thres) for i, j in zip(ref_patches, moving_patches)]
    
    # Separate features
    ref_points_list = [f[0] for f in all_features]
    moving_points_list = [f[1] for f in all_features]
    scores_list = [f[2] for f in all_features]
    
    # Add offsets to keypoints
    for i, (ref_pts, mov_pts, offset_A, offset_B) in enumerate(zip(ref_points_list, moving_points_list, ref_offsets, moving_offsets)):
        ref_points_list[i] = ref_pts + np.array(offset_A)
        moving_points_list[i] = mov_pts + np.array(offset_B)
    
    # Concatenate all features
    ref_points = np.concatenate(ref_points_list, axis=0)
    moving_points = np.concatenate(moving_points_list, axis=0)
    scores = np.concatenate(scores_list, axis=0)
    
    return ref_points, moving_points, scores


def get_roma_keypoints_parcellated(ref_image: np.ndarray, moving_image: np.ndarray, matcher: Any, detect_match_fn: Callable, thres: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints in parcellated mode.
    Finds keypoints locally (per quadrant) and matches locally (per quadrant). 
    """
    
    # Divide image in 4 quadrants
    xmid = ref_image.shape[1]//2
    ymid = ref_image.shape[0]//2
    xstart = [xmid, 0, xmid, 0]
    xend = [-1, xmid, -1, xmid]
    ystart = [0, 0, ymid, ymid]
    yend = [ymid, ymid, -1, -1]
    
    all_ref_points, all_moving_points, all_scores = [], [], []
    
    # Restrict keypoint matching to each quadrant
    for i, (x1, x2, y1, y2) in enumerate(zip(xstart, xend, ystart, yend)):
        
        # Create quadrant by setting non-quadrant pixels to white
        ref_image_quadrant = copy.copy(ref_image)
        ref_image_quadrant[x1:x2, :] = np.max(ref_image)
        ref_image_quadrant[:, y1:y2] = np.max(ref_image)
        
        moving_image_quadrant = copy.copy(moving_image)
        moving_image_quadrant[x1:x2, :] = np.max(moving_image)
        moving_image_quadrant[:, y1:y2] = np.max(moving_image)
        
        # Use the provided keypoint function
        ref_points, moving_points, scores = detect_match_fn(ref_image_quadrant, moving_image_quadrant, thres)
        
        if ref_points.shape[0] > 0: 
            all_ref_points.append(ref_points)
            all_moving_points.append(moving_points)
            all_scores.append(scores)
    
    all_ref_points = np.concatenate(all_ref_points)
    all_moving_points = np.concatenate(all_moving_points)
    all_scores = np.concatenate(all_scores)
    
    return all_ref_points, all_moving_points, all_scores

def get_dedode_keypoints(ref_image: np.ndarray, moving_image: np.ndarray, detector: Any, matcher: Any, patch_method: str = "regular") -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints with DeDoDe
    """

    detector, descriptor = detector

    def detect(ref_image: np.ndarray, moving_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # DeDoDe requires image paths instead of images
        ref_path = "/tmp/ref.png"
        moving_path = "/tmp/moving.png"
        cv2.imwrite(ref_path, ref_image)
        cv2.imwrite(moving_path, moving_image)
        
        with torch.inference_mode():
            # Fetch keypoints
            detections_A = detector.detect_from_path(ref_path, num_keypoints = 10_000)
            keypoints_A, P_A = detections_A["keypoints"], detections_A["confidence"]

            detections_B = detector.detect_from_path(moving_path, num_keypoints = 10_000)
            keypoints_B, P_B = detections_B["keypoints"], detections_B["confidence"]

            # Use decoupled descriptor to describe keypoints
            description_A = descriptor.describe_keypoints_from_path(ref_path, keypoints_A)["descriptions"]
            description_B = descriptor.describe_keypoints_from_path(moving_path, keypoints_B)["descriptions"]

        return keypoints_A, description_A, keypoints_B, description_B, P_A, P_B
        
    def match(keypoints_A: np.ndarray, description_A: np.ndarray, keypoints_B: np.ndarray, description_B: np.ndarray, P_A: np.ndarray, P_B: np.ndarray, W_A: int, H_A: int, W_B: int, H_B: int) -> tuple[np.ndarray, np.ndarray]:
        # Match and convert to pixel coordinates
        matches_A, matches_B, _ = matcher.match(
            keypoints_A, 
            description_A,
            keypoints_B, 
            description_B,
            P_A = P_A,
            P_B = P_B,
            normalize = True, 
            inv_temp=20, 
            threshold = 0.03
        )
        matches_A, matches_B = matcher.to_pixel_coords(
            matches_A, 
            matches_B, 
            H_A, 
            W_A, 
            H_B, 
            W_B
        )

        ref_points = matches_A.detach().cpu().numpy()
        moving_points = matches_B.detach().cpu().numpy()
        scores = np.ones(ref_points.shape[0])
        
        return ref_points, moving_points, scores

    if patch_method == "regular":
        H_A, W_A = ref_image.shape[:2]
        H_B, W_B = moving_image.shape[:2]
        return match(*detect(ref_image, moving_image), H_A, W_A, H_B, W_B)
    elif patch_method == "patch":
        H_A, W_A = ref_image.shape[0] // 2, ref_image.shape[1] // 2
        H_B, W_B = moving_image.shape[0] // 2, moving_image.shape[1] // 2
        return get_dedode_keypoints_patch_based(ref_image, moving_image, detect, match, H_A, W_A, H_B, W_B)
    elif patch_method == "parcellated":
        H_A, W_A = ref_image.shape[:2]
        H_B, W_B = moving_image.shape[:2]
        return get_dedode_keypoints_parcellated(ref_image, moving_image, detect, match, H_A, W_A, H_B, W_B)


def get_dedode_keypoints_patch_based(ref_image: np.ndarray, moving_image: np.ndarray, detect_fn: Callable, match_fn: Callable, H_A: int, W_A: int, H_B: int, W_B: int) -> tuple[np.ndarray, np.ndarray]:  
    """
    Function to get matching keypoints in patch-based mode.
    Finds keypoints locally, but matches globally. 
    """
    
    # Divide image in nxn equal patches
    ref_patches, ref_offsets = divide_image_in_patches(ref_image)
    moving_patches, moving_offsets = divide_image_in_patches(moving_image)
    
    # Get features from each patch
    all_features = [detect_fn(i, j) for i, j in zip(ref_patches, moving_patches)]
    
    # Separate features
    keypoints_A_list = [f[0] for f in all_features]
    description_A_list = [f[1] for f in all_features]
    keypoints_B_list = [f[2] for f in all_features]
    description_B_list = [f[3] for f in all_features]
    P_A_list = [f[4] for f in all_features]
    P_B_list = [f[5] for f in all_features]
    
    # Add offsets to keypoints
    for i, (kA, kB, offset_A, offset_B) in enumerate(zip(keypoints_A_list, keypoints_B_list, ref_offsets, moving_offsets)):
        keypoints_A_list[i] = kA + torch.tensor(offset_A, device=kA.device, dtype=kA.dtype)
        keypoints_B_list[i] = kB + torch.tensor(offset_B, device=kB.device, dtype=kB.dtype)
    
    # Concatenate all features
    keypoints_A = torch.cat(keypoints_A_list, dim=0)
    description_A = torch.cat(description_A_list, dim=0)
    keypoints_B = torch.cat(keypoints_B_list, dim=0)
    description_B = torch.cat(description_B_list, dim=0)
    P_A = torch.cat(P_A_list, dim=0)
    P_B = torch.cat(P_B_list, dim=0)
    
    # Match all keypoints globally
    ref_points, moving_points, scores = match_fn(
        keypoints_A, 
        description_A,
        keypoints_B, 
        description_B,
        P_A,
        P_B,
        H_A,
        W_A, 
        H_B,
        W_B
    )
    
    return ref_points, moving_points, scores


def get_dedode_keypoints_parcellated(ref_image: np.ndarray, moving_image: np.ndarray, detect_fn: Callable, match_fn: Callable, H_A: int, W_A: int, H_B: int, W_B: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic function to get matching keypoints in parcellated mode.
    Finds keypoints locally (per quadrant) and matches locally (per quadrant). 
    """
    
    # Divide image in 4 quadrants
    xmid = ref_image.shape[1]//2
    ymid = ref_image.shape[0]//2
    xstart = [xmid, 0, xmid, 0]
    xend = [-1, xmid, -1, xmid]
    ystart = [0, 0, ymid, ymid]
    yend = [ymid, ymid, -1, -1]
    
    all_ref_points, all_moving_points, all_scores = [], [], []
    
    # Restrict keypoint matching to each quadrant
    for i, (x1, x2, y1, y2) in enumerate(zip(xstart, xend, ystart, yend)):
        
        # Create quadrant by setting non-quadrant pixels to white
        ref_image_quadrant = copy.copy(ref_image)
        ref_image_quadrant[x1:x2, :] = np.max(ref_image)
        ref_image_quadrant[:, y1:y2] = np.max(ref_image)
        
        moving_image_quadrant = copy.copy(moving_image)
        moving_image_quadrant[x1:x2, :] = np.max(moving_image)
        moving_image_quadrant[:, y1:y2] = np.max(moving_image)
        
        # Use the provided keypoint function
        keypoints_A, description_A, keypoints_B, description_B, P_A, P_B = detect_fn(ref_image_quadrant, moving_image_quadrant)
        ref_points, moving_points, scores = match_fn(keypoints_A, description_A, keypoints_B, description_B, P_A, P_B, H_A, W_A, H_B, W_B)
        
        if ref_points.shape[0] > 0: 
            all_ref_points.append(ref_points)
            all_moving_points.append(moving_points)
            all_scores.append(scores)
    
    all_ref_points = np.concatenate(all_ref_points)
    all_moving_points = np.concatenate(all_moving_points)
    all_scores = np.concatenate(all_scores)
    
    return all_ref_points, all_moving_points, all_scores

def get_omniglue_keypoints(ref_image: np.ndarray, moving_image: np.ndarray, detector: Any, matcher: Any, patch_method: str = "regular") -> tuple[np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints with omniglue.
    """
    
    def detect_and_match(ref_img: np.ndarray, moving_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Nice and easy
        ref_points, moving_points, scores = matcher.FindMatches(ref_img, moving_img)
        return ref_points, moving_points, scores
    
    def match(ref_points: np.ndarray, moving_points: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Wrapper to make parcellated function work.
        """
        return ref_points, moving_points, scores
    
    if patch_method == "parcellated":
        return get_keypoints_parcellated(ref_image, moving_image, detect_and_match, match)
    elif patch_method == "patch":
        return get_omniglue_keypoints_patch_based(ref_image, moving_image, detect_and_match)
    else:
        return detect_and_match(ref_image, moving_image)
    
def get_omniglue_keypoints_patch_based(ref_image: np.ndarray, moving_image: np.ndarray, detect_fn: Callable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to get matching keypoints in patch-based mode.
    Note that omniglue performs detection and matching at the same 
    time and, in contrast to other detectors, cannot perform
    local detection and global matching. Hence, this function
    is similar to omniglue's parcellated function. 
    """
    # Divide images into patches
    ref_patches, ref_offsets = divide_image_in_patches(ref_image)
    moving_patches, moving_offsets = divide_image_in_patches(moving_image)
    
    # Get matches from each patch pair
    all_ref_points, all_moving_points, all_scores = [], [], []
    
    for ref_patch, moving_patch, ref_offset, moving_offset in zip(ref_patches, moving_patches, ref_offsets, moving_offsets):
        # Get matches for this patch pair
        ref_points, moving_points, scores = detect_fn(ref_patch, moving_patch)
        
        # Add offsets to convert to original image coordinates
        if ref_points.shape[0] > 0:
            ref_points = ref_points + np.array(ref_offset)
            moving_points = moving_points + np.array(moving_offset)
            
            all_ref_points.append(ref_points)
            all_moving_points.append(moving_points)
            all_scores.append(scores)
    
    all_ref_points = np.concatenate(all_ref_points)
    all_moving_points = np.concatenate(all_moving_points)
    all_scores = np.concatenate(all_scores)

    return all_ref_points, all_moving_points, all_scores
