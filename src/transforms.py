import cv2
import numpy as np
import pyvips
import torch
import sys

from skimage import transform
from skimage.transform import EuclideanTransform
from skimage.measure import ransac
from typing import List, Any

import pymagsac
sys.path.append("/detectors/DALF_CVPR_2023")
from modules.tps import RANSAC
from modules.tps import pytorch as tps_pth
from modules.tps import numpy as tps_np


def apply_affine_magsac(moving_points: np.ndarray, ref_points: np.ndarray, scores: np.ndarray, image: np.ndarray) -> tuple([np.ndarray, np.ndarray, List, EuclideanTransform]):
    """Apply MAGSAC to filter plausible matches for affine transform.

    Args:
        moving_points: Numpy array of shape (N,2) containing points from moving image
        ref_points: Numpy array of shape (N,2) containing points from reference image
        scores: Numpy array of shape (N,) containing confidence scores for each match
        image: Numpy array of shape (H,W,3) containing the moving image

    Returns:
        tuple containing:
            - ref_points: Filtered reference points (N_inliers, 2)
            - moving_points: Filtered moving points (N_inliers, 2) 
            - inliers: Boolean array indicating inlier matches
            - mat: EuclideanTransform object containing transformation matrix
    """

    # Convert to 3D points
    ref_points_3d = np.hstack((ref_points, np.ones((len(ref_points), 1))))    
    moving_points_3d = np.hstack((moving_points, np.ones((len(moving_points), 1))))
    matches = np.ascontiguousarray(np.hstack((ref_points_3d, moving_points_3d)))

    # Magsac is a bit more sensitive to the sigma_th parameter, 
    # so we try a range of values. Number of inliers w.r.t. sigma_th
    # follows a U curve, so we don't end up with very high/low sigma_th.
    all_sigmas = []
    all_num_inliers = []

    for sigma_th in np.arange(1, 0.1*image.shape[0], 10):
        mat, inliers = pymagsac.findRigidTransformation(
            matches, 
            probabilities = scores,
            use_magsac_plus_plus = True,
            sigma_th = sigma_th
        )
        all_sigmas.append(sigma_th)
        all_num_inliers.append(np.sum(inliers)) 
        
    # Select the best sigma_th based on the number of inliers
    sigma_th = all_sigmas[np.argmax(all_num_inliers)]
    mat, inliers = pymagsac.findRigidTransformation(
        matches, 
        probabilities = scores,
        use_magsac_plus_plus = True,
        sigma_th = sigma_th
    )

    if np.sum(inliers) > np.max([0.01*len(ref_points), 10]):
        # Convert to proper 3x3 matrix
        matrix = np.ones((3, 3))
        matrix[:2, :2] = -mat.T[:2, :2]
        matrix[:2, 2] = -mat.T[:2, 3]
        mat = EuclideanTransform(matrix = matrix)
    else:
        # Compute regular transform if magsac fails
        print(f"  -- unable to fit magsac++")
        mat = transform.estimate_transform(
            "euclidean", 
            moving_points, 
            ref_points
        )
        inliers = np.array([True] * len(moving_points))

    # Filter based on inliers
    ref_points = ref_points[inliers]
    moving_points = moving_points[inliers]

    return ref_points, moving_points, inliers, mat


def apply_affine_ransac(moving_points: np.ndarray, ref_points: np.ndarray, image: np.ndarray, ransac_thres: float) -> tuple([np.ndarray, np.ndarray, Any, EuclideanTransform]):
    """Apply RANSAC to filter plausible matches for affine transform.

    Args:
        moving_points: Numpy array of shape (N,2) containing points from moving image
        ref_points: Numpy array of shape (N,2) containing points from reference image
        image: Numpy array of shape (H,W,3) containing the moving image
        ransac_thres: Float threshold for RANSAC inlier detection

    Returns:
        tuple containing:
            - ref_points: Filtered reference points (N_inliers, 2)
            - moving_points: Filtered moving points (N_inliers, 2)
            - inliers: Boolean array indicating inlier matches
            - model: EuclideanTransform object containing transformation matrix
    """

    min_matches = 10
    inliers = np.array([False] * len(moving_points))
    res_thres = int(image.shape[0] * ransac_thres)

    # Default to identity matrix 
    model = EuclideanTransform(rotation = 0, translation = 0)

    # Apply ransac to further filter plausible matches
    try:
        model, inliers = ransac(
            (moving_points, ref_points),
            EuclideanTransform, 
            min_samples=min_matches,
            residual_threshold=res_thres,
            max_trials=1000
        )

        # Case where convergence fails
        if not isinstance(inliers, np.ndarray):
            inliers = np.array([False] * len(moving_points))
        # Case where ransac found too few inliers
        elif isinstance(inliers, np.ndarray) and np.sum(inliers) < np.max([0.1*len(ref_points), 10]):
            print(f"  -- unable to fit ransac")
            model = transform.estimate_transform(
                "euclidean", 
                moving_points, 
                ref_points
            )
            inliers = np.array([True] * len(moving_points))
        # Regular case
        else:
            ref_points = np.float32([p for p, i in zip(ref_points, inliers) if i])
            moving_points = np.float32([p for p, i in zip(moving_points, inliers) if i])
    except:
        print(f"  -- unable to fit ransac")
        model = transform.estimate_transform(
            "euclidean", 
            moving_points, 
            ref_points
        )
        inliers = np.array([True] * len(moving_points))
        
    return ref_points, moving_points, inliers, model


def estimate_affine_transform(moving_points: np.ndarray, ref_points: np.ndarray, scores: np.ndarray, image: np.ndarray, filter_method: str, ransac_thres: float) -> tuple([np.ndarray, List]):
    """Estimate an affine transform between two sets of points.

    Args:
        moving_points: Numpy array of shape (N,2) containing points from moving image
        ref_points: Numpy array of shape (N,2) containing points from reference image
        scores: Numpy array of shape (N,) containing confidence scores for each match
        image: Numpy array of shape (H,W,3) containing the moving image
        filter_method: String indicating filtering method ('ransac', 'magsac', or 'none')
        ransac_thres: Float threshold for RANSAC inlier detection

    Returns:
        tuple containing:
            - matrix: EuclideanTransform object containing transformation matrix
            - num_inliers: Number of inlier matches used for transform estimation
    """

    if len(moving_points) > 0:
        if filter_method == "ransac":
            ref_points, moving_points, inliers, matrix = apply_affine_ransac(
                moving_points = moving_points, 
                ref_points = ref_points, 
                image = image, 
                ransac_thres = ransac_thres
            )
        elif filter_method == "magsac":
            ref_points, moving_points, inliers, matrix = apply_affine_magsac(
                moving_points = moving_points, 
                ref_points = ref_points,
                scores = scores,
                image = image,
            )
        elif filter_method == "none":
            inliers = np.array([True] * len(moving_points))
            matrix = transform.estimate_transform(
                "euclidean", 
                moving_points, 
                ref_points
            )
    else:
        # Identity transform when there are no matches
        matrix = EuclideanTransform(rotation = 0, translation = 0)
        inliers = np.array([False] * len(moving_points))
           
    return matrix, np.sum(inliers)


def apply_affine_transform(image: np.ndarray, mask: np.ndarray, tform: np.ndarray, landmarks: np.ndarray = None) -> tuple([np.ndarray, Any]):
    """Apply an affine transform to an image and mask.

    Args:
        image: Numpy array of shape (H,W,3) containing input image
        mask: Numpy array of shape (H,W) containing binary mask, or None
        tform: 2x3 affine transformation matrix
        landmarks: Optional numpy array of shape (N,2) containing landmark coordinates

    Returns:
        tuple containing:
            - image_warped: Transformed image array (H,W,3)
            - mask_warped: Transformed mask array (H,W) or None
            - landmarks_warped: Transformed landmarks array (N,2) or None
    """

    assert len(image.shape) == 3, "image must be 3 dimensional"

    # Warp the main image
    rows, cols, _ = image.shape
    image_warped = cv2.warpAffine(image, tform, (cols, rows), borderValue=(255, 255, 255))

    # Warp mask if available 
    if type(mask) == np.ndarray:
        mask_warped = cv2.warpAffine(mask, tform, (cols, rows), borderValue=(0, 0, 0))
        mask_warped = ((mask_warped > 128)*255).astype("uint8")
    else:
        mask_warped = None

    # Warp landmarks if available
    if landmarks is not None:
        landmarks_warped = cv2.transform(landmarks.reshape(-1, 1, 2), tform).reshape(-1, 2)
    else:
        landmarks_warped = None
        
    return image_warped, mask_warped, landmarks_warped


def apply_affine_transform_fullres(image: pyvips.Image, mask: pyvips.Image, rotation: float, translation: float, center: tuple, scaling: float) -> tuple([pyvips.Image, Any]):
    """Apply an affine transform to full resolution images.

    Args:
        image: Pyvips Image object containing input image
        mask: Pyvips Image object containing binary mask, or None
        rotation: Float rotation angle in degrees
        translation: Tuple (tx,ty) containing translation amounts
        center: Tuple (cx,cy) containing rotation center coordinates
        scaling: Float scaling factor for coordinate conversion

    Returns:
        tuple containing:
            - image_warped: Transformed pyvips Image
            - mask_warped: Transformed pyvips mask Image or None
    """

    # Get upscaled transformation matrix
    center = (float(center[0] * scaling), float(center[1] * scaling))
    translation = (translation[0] * scaling, translation[1] * scaling)
    tform = cv2.getRotationMatrix2D(center, rotation, 1)
    tform[:, 2] += translation

    # Warp the main image
    image_warped = image.affine(
        (tform[0, 0], tform[0, 1], tform[1, 0], tform[1, 1]),
        interpolate = pyvips.Interpolate.new("bicubic"),
        odx = tform[0, 2],
        ody = tform[1, 2],
        oarea = (0, 0, image.width, image.height),
        background = 255
    )

    # Warp mask if available 
    if type(mask) == pyvips.Image:
        mask_warped = mask.affine(
            (tform[0, 0], tform[0, 1], tform[1, 0], tform[1, 1]),
            interpolate = pyvips.Interpolate.new("nearest"),
            odx = tform[0, 2],
            ody = tform[1, 2],
            oarea = (0, 0, mask.width, mask.height),
            background = 0
        )
    else:
        mask_warped = None

    return image_warped, mask_warped


def apply_deformable_ransac(moving_points: np.ndarray, ref_points: np.ndarray, device: Any, ransac_thres_deformable: float = 0.05) -> tuple([np.ndarray, np.ndarray, np.ndarray]):
    """Apply RANSAC to filter plausible matches for deformable transform.

    Args:
        moving_points: Numpy array of shape (N,2) containing points from moving image
        ref_points: Numpy array of shape (N,2) containing points from reference image
        device: PyTorch device to use for computations
        ransac_thres_deformable: Float threshold for RANSAC inlier detection

    Returns:
        tuple containing:
            - ref_points: Filtered reference points (N_inliers, 2)
            - moving_points: Filtered moving points (N_inliers, 2)
    """

    # Apply ransac to further filter plausible matches
    inliers = RANSAC.nr_RANSAC(ref_points, moving_points, device, thr = ransac_thres_deformable)

    ref_points = np.float32([p for p, i in zip(ref_points, inliers) if i])
    moving_points = np.float32([p for p, i in zip(moving_points, inliers) if i])

    return ref_points, moving_points


def estimate_deformable_transform(moving_image: np.ndarray, ref_image: np.ndarray, moving_points: np.ndarray, ref_points: np.ndarray, deformable_level: int, keypoint_level: int, device: Any, lambda_param: float = 0.1) -> tuple([pyvips.Image, Any]):
    """Estimate parameters for deformable transform using thin plate splines.

    Args:
        moving_image: Numpy array of shape (H,W,3) containing moving image
        ref_image: Numpy array of shape (H,W,3) containing reference image
        moving_points: Numpy array of shape (N,2) containing points from moving image
        ref_points: Numpy array of shape (N,2) containing points from reference image
        deformable_level: Integer indicating deformation grid resolution
        keypoint_level: Integer indicating keypoint detection resolution
        device: PyTorch device to use for computations
        lambda_param: Float regularization parameter for TPS

    Returns:
        tuple containing:
            - index_map: Pyvips Image containing backward mapping coordinates
            - grid: PyTorch tensor containing deformation grid
    """

    # Get image shapes
    h1, w1 = ref_image.shape[:2]
    h2, w2 = moving_image.shape[:2]
    
    # Normalize coordinates
    c_ref = np.float32(ref_points) / np.float32([w1,h1])
    c_moving = np.float32(moving_points) / np.float32([w2,h2])

    # Downsample image to prevent OOM in deformable grid. Also induces
    # some additional regularization.
    downsample = 16
    moving_image_ds = cv2.resize(moving_image, (w2//downsample, h2//downsample), interpolation=cv2.INTER_AREA)

    # Compute theta from coordinates
    moving_image_ds = torch.tensor(moving_image_ds).to(device).permute(2,0,1)[None, ...].float()
    theta = tps_np.tps_theta_from_points(c_ref, c_moving, reduced=True, lambd=lambda_param)
    theta = torch.tensor(theta).to(device)[None, ...]

    # Create downsampled grid to sample from
    grid = tps_pth.tps_grid(theta, torch.tensor(c_moving, device=device), moving_image_ds.shape)
    del moving_image_ds, theta
    torch.cuda.empty_cache()
    
    # Upsample grid to accomodate original image
    dx = grid.cpu().numpy()[0, :, :, 0]
    dx = ((dx + 1) / 2) * (w2 - 1)

    dy = grid.cpu().numpy()[0, :, :, 1] 
    dy = ((dy + 1) / 2) * (h2 - 1)

    # Upsample using affine rather than resize to account for shape rounding errors
    dx = pyvips.Image.new_from_array(dx).resize(downsample)
    dy = pyvips.Image.new_from_array(dy).resize(downsample)

    # Ensure deformation field is exactly as large as the image. Discepancies can
    # occur due to rounding errors in the shape of the image.
    if (dx.width != w2) or (dx.height != h2):
        dx = dx.gravity("centre", w2, h2)
        dy = dy.gravity("centre", w2, h2)

    index_map = dx.bandjoin([dy])

    return index_map, grid


def apply_deformable_transform(moving_image: np.ndarray, moving_mask: np.ndarray, index_map: pyvips.Image, landmarks: np.ndarray = None) -> tuple([np.ndarray, np.ndarray, Any]):
    """Apply deformable transform using backward mapping.

    Args:
        moving_image: Numpy array of shape (H,W,3) containing moving image
        moving_mask: Numpy array of shape (H,W) containing binary mask
        index_map: Pyvips Image containing backward mapping coordinates
        landmarks: Optional numpy array of shape (N,2) containing landmark coordinates

    Returns:
        tuple containing:
            - moving_image_warped: Transformed image array (H,W,3)
            - moving_mask_warped: Transformed mask array (H,W)
            - landmarks_warped: Transformed landmarks array (N,2) or None
    """

    # Apply transform
    moving_image = pyvips.Image.new_from_array(moving_image)
    moving_image_warped = moving_image.mapim(
        index_map, 
        interpolate=pyvips.Interpolate.new('bicubic'), 
        background=[255, 255, 255]
    ).numpy().astype(np.uint8)

    moving_mask = pyvips.Image.new_from_array(moving_mask)
    moving_mask_warped = moving_mask.mapim(
        index_map,
        interpolate=pyvips.Interpolate.new('nearest'),
        background=[0, 0, 0]
    ).numpy().astype(np.uint8)

    # Multiply image by mask to get rid of black borders
    moving_image_warped[moving_mask_warped < np.max(moving_mask_warped)] = 255

    # Warp landmarks if available
    if landmarks is not None:
        landmarks_warped = warp_landmarks_deformable(landmarks, index_map)
    else:
        landmarks_warped = None

    return moving_image_warped, moving_mask_warped, landmarks_warped


def apply_deformable_transform_fullres(image: pyvips.Image, mask: pyvips.Image, grid: Any, scaling: int) -> tuple([pyvips.Image, pyvips.Image]):
    """Apply thin plate splines transform to full resolution images.

    Args:
        image: Pyvips Image object containing input image
        mask: Pyvips Image object containing binary mask
        grid: PyTorch tensor containing deformation grid
        scaling: Integer scaling factor for coordinate conversion

    Returns:
        tuple containing:
            - image_warped: Transformed pyvips Image
            - mask_warped: Transformed pyvips mask Image
    """

    # Convert torch grid to pyivps grid
    dx = grid.cpu().numpy()[0, :, :, 0]
    dx = ((dx + 1) / 2) * (image.width - 1)

    dy = grid.cpu().numpy()[0, :, :, 1] 
    dy = ((dy + 1) / 2) * (image.height - 1)

    # Scale to full resolution
    dx = pyvips.Image.new_from_array(dx).resize(scaling)
    dy = pyvips.Image.new_from_array(dy).resize(scaling)

    # Ensure deformation field is exactly as large as the image
    width, height = image.width, image.height
    if (dx.width != width) or (dy.height != height):
        dx = dx.gravity("centre", width, height)
        dy = dy.gravity("centre", width, height)

    index_map = dx.bandjoin([dy])

    # Apply to image
    image_warped = image.mapim(
        index_map, 
        interpolate=pyvips.Interpolate.new('bicubic'), 
        background=[255, 255, 255]
    )
    mask_warped = mask.mapim(
        index_map,
        interpolate=pyvips.Interpolate.new('nearest'),
        background=[0, 0, 0]
    )

    return image_warped, mask_warped

def find_landmark_position_in_backwarp(landmark: np.ndarray, index_map: np.ndarray, search_radius: int = 200) -> np.ndarray:
    """Find where a landmark ends up by searching the backward mapping.
    
    Args:
        landmark: Numpy array [x,y] containing original landmark position
        index_map: Numpy array (H,W,2) containing backward mapping coordinates
        search_radius: Integer radius around original position to search
        
    Returns:
        warped_position: Numpy array [x,y] containing warped landmark position
    """
    x, y = landmark.astype(int)
    h, w = index_map.shape[:2]
    
    # Define search area (bounded by image size)
    y_min = max(0, y - search_radius)
    y_max = min(h, y + search_radius)
    x_min = max(0, x - search_radius)
    x_max = min(w, x + search_radius)
    
    # Extract the search region from index map
    search_region = index_map[y_min:y_max, x_min:x_max]
    
    # Calculate distance between backward mapped positions and the landmark
    diff_x = search_region[..., 0] - x
    diff_y = search_region[..., 1] - y
    distances = np.sqrt(diff_x**2 + diff_y**2)
    
    # Find the minimum distance
    min_dist_idx = np.argmin(distances)
    min_dist = distances.flat[min_dist_idx]
    
    # If we found a reasonable match
    if min_dist < search_radius:
        # Convert flat index back to 2D coordinates
        y_idx, x_idx = np.unravel_index(min_dist_idx, distances.shape)
        # Return the position in the output image (x_min/y_min offset added back)
        return np.array([x_min + x_idx, y_min + y_idx])
    
    # If no good match found, return original position
    return landmark

def warp_landmarks_deformable(landmarks: np.ndarray, index_map: pyvips.Image) -> np.ndarray:
    """Warp landmarks using backward mapping through search.
    
    Args:
        landmarks: Numpy array of shape (N,2) containing (x,y) coordinates
        index_map: Pyvips Image with shape (H,W,2) containing backward mapping
        
    Returns:
        warped_landmarks: Numpy array of shape (N,2) containing warped coordinates
    """
    # Convert index_map to numpy for easier handling
    index_map_np = np.stack([index_map.numpy()[..., 0], 
                           index_map.numpy()[..., 1]], axis=-1)
    search_radius = int(0.1 * index_map_np.shape[0])
    
    # Process each landmark
    warped_landmarks = []
    for landmark in landmarks:
        warped_landmark = find_landmark_position_in_backwarp(
            landmark = landmark, 
            index_map = index_map_np, 
            search_radius = search_radius
        )
        warped_landmarks.append(warped_landmark)
    
    return np.array(warped_landmarks)