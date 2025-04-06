import pyvips
import numpy as np
import cv2
import math
import torch 
import copy 
import subprocess
import shutil
import torchstain
import kornia as K
import warnings
import pandas as pd
import sys
sys.path.append("/detectors")
import os
os.environ["VIPS_CONCURRENCY"] = "8"

from pathlib import Path
from typing import List
from skimage.color import rgb2hed
from torchvision import transforms
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.*')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.*')
warnings.filterwarnings("ignore", category=UserWarning, module='pyvips.*')
import logging
logging.getLogger('pyvips').setLevel(logging.WARNING)

from visualization import *
from utils import *
from config import Config
from transforms import *
from keypoints import *
from evaluation import *
from dino import *


class Rapid:

    def __init__(self, case_df: pd.DataFrame, mode: str, detector: str, patch_method: str, save_dir: Path, affine_ransac_thres: float, keypoint_thres: float, keypoint_filter: str) -> None:
        
        self.config = Config()

        # Process case dataframe
        self.case_df = case_df.sort_values(by="imagepath")
        self.case_id = case_df["case"].values[0]
        self.image_paths = sorted(self.case_df["imagepath"].values.tolist())
        self.image_paths = [Path(i) for i in self.image_paths]
        self.image_ids = [i.stem for i in self.image_paths]
        
        # Check for optional columns that may need to be processed
        if "maskpath" in self.case_df.columns.values.tolist():
            self.mask_paths = sorted(self.case_df["maskpath"].values.tolist())
            self.mask_paths = [Path(i) for i in self.mask_paths]
            assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks do not match."
            self.masks_available = True
        else:
            self.mask_paths = []
            self.masks_available = False
        self.pred_orientations = dict([str(i), 0] for i in self.image_ids)
        if "gt_orientation" in self.case_df.columns.values.tolist():
            self.gt_orientations = dict([str(i), j] for i, j in zip(self.image_ids, self.case_df["gt_orientation"].values.tolist()))
        else:
            self.gt_orientations = None
            self.orientation_accuracy = dict([[str(i), "n.a."] for i in self.image_ids])
        if "landmarks_xy" in self.case_df.columns.values.tolist():
            # Convert string representation of coordinates to numpy arrays
            self.landmarks_xy = [
                np.array(eval(coords))
                for coords in self.case_df["landmarks_xy"].values.tolist()
            ]
        else:
            self.landmarks_xy = False
        
        self.mode = mode
        self.detector_name = detector
        self.patch_method = patch_method
        self.keypoint_thres = keypoint_thres
        self.save_dir = save_dir.joinpath(f"{self.case_id}")

        # Create some dirs for saving
        self.local_save_dir = Path(f"/tmp/rapid/{self.case_id}")
        self.local_save_dir.mkdir(parents=True, exist_ok=True)
        for dir in ["keypoints", "warps", "evaluation", "debug"]:
            self.local_save_dir.joinpath(dir).mkdir(parents=True, exist_ok=True)
        
        # Process and load config params
        self.scramble = self.config.scramble if not self.mode == "valis" else False
        self.supported_detectors = ["dalf", "sift", "superpoint", "roma", "dedode", "omniglue"]
        assert self.detector_name in self.supported_detectors, f"Only the following detectors are implemented {self.supported_detectors}."
        assert len(self.image_paths) >= self.config.min_images_for_reconstruction, f"Need at least {self.config.min_images_for_reconstruction} images to perform a reasonable reconstruction."

        # Set level at which to load the image
        self.optimal_image_size = self.config.optimal_image_size
        self.full_resolution_spacing = self.config.full_resolution_spacing
        
        # Set device for GPU-based keypoint detection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gpu_detectors = ["superpoint", "omniglue", "roma"]
        cpu_detectors = [i for i in self.supported_detectors if i not in gpu_detectors]
        if all([self.device.type == "cpu", self.detector_name in gpu_detectors]):
            raise ValueError(f"Detector {self.detector_name} requires GPU but no GPU available. Consider choosing a cpu-based detector such as {cpu_detectors}.")
        self.detector, self.matcher = self.init_detector(name = self.detector_name)

        # Set some RANSAC parameters
        self.keypoint_filter = keypoint_filter
        self.ransac_thres_affine = affine_ransac_thres

        return
    
    def init_detector(self, name: str) -> None:
        """
        Method to initialize the matching algorithm. 
        """

        # Initialize the keypoint detector and matcher to prevent repeating this dozen of times
        if name == "superpoint":
            from lightglue import LightGlue, SuperPoint
            detector = SuperPoint(max_num_keypoints=None).eval().cuda()
            matcher = LightGlue(features=self.detector_name).eval().cuda() 

        elif name == "sift":
            detector = cv2.SIFT_create()
            matcher = cv2.BFMatcher()

        elif name == "roma": 
            sys.path.append("/detectors/RoMa")
            from romatch import roma_outdoor
            detector = None
            matcher = roma_outdoor(
                device=self.device, 
                dino_type = "meta",
            )

        elif name == "dedode":
            from DeDoDe import dedode_detector_L, dedode_descriptor_G
            from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher

            # Weights are pulled automatically from corresponding repo's
            dedode_detector = dedode_detector_L(weights=None)
            dedode_descriptor = dedode_descriptor_G(weights=None, dinov2_weights=None)
            detector = (dedode_detector, dedode_descriptor)
            matcher = DualSoftMaxMatcher()

        elif name == "omniglue":
            
            sys.path.append("/detectors/omniglue")
            from omniglue.src.omniglue.omniglue_extract import OmniGlue
            detector = None
            matcher = OmniGlue(
                og_export='/detectors/omniglue/models/og_export',
                sp_export='/detectors/omniglue/models/superpoint_v6_from_tf.pth',
                dino_export='/detectors/omniglue/models/dinov2_vitb14_pretrain.pth',
            )

        return detector, matcher

    def load_images(self) -> None:
        """
        Method to load images using pyvips.
        """

        print(f" - loading images")
        self.raw_images = []
        self.optimal_image_levels = []
        
        for im_path in self.image_paths:

            # Load image
            if im_path.suffix == ".mrxs":
                # Load image that is closest to optimal image size
                image = pyvips.Image.new_from_file(str(im_path))
                image_levels = int(image.get("openslide.level-count"))
                image_sizes = [[int(image.width/(2**i)), int(image.height/(2**i))] for i in range(image_levels)]
                optimal_image_level = np.argmin([abs(np.mean(i)-self.optimal_image_size) for i in image_sizes])
                image = pyvips.Image.new_from_file(str(im_path), level=optimal_image_level)

            elif im_path.suffix == ".tif":
                # Load image that is closest to optimal image size
                image = pyvips.Image.new_from_file(str(im_path))
                image_levels = int(image.get("n-pages"))
                image_sizes = [[int(image.width/(2**i)), int(image.height/(2**i))] for i in range(image_levels)]
                optimal_image_level = np.argmin([abs(np.mean(i)-self.optimal_image_size) for i in image_sizes])
                image = pyvips.Image.new_from_file(str(im_path), page=optimal_image_level)
            else:
                raise ValueError("Sorry, only .tifs and .mrxs are supported.")

            # Dispose of alpha band if present
            if image.bands == 4:
                image_np = image.flatten().numpy().astype(np.uint8)
            elif image.bands == 3: 
                image_np = image.numpy().astype(np.uint8)

            # Save images
            self.raw_images.append(image_np)
            self.optimal_image_levels.append(optimal_image_level)

        self.pixel_spacing_image = (1000 / image.get("xres")) * 2**self.optimal_image_levels[0]
        
        # Plot initial reconstruction
        plot_initial_reconstruction(
            images=self.raw_images, 
            save_dir=self.local_save_dir
        )

        self.load_images_fullres()
        
        if self.landmarks_xy:
            # Scale landmarks to match the resolution of the images
            self.landmarks_xy = [i / (2 ** j) for i, j in zip(self.landmarks_xy, self.optimal_image_levels)]

        return 

    def load_images_fullres(self) -> None:
        """
        Method to load the full res images.
        """

        self.raw_fullres_images = [] 
        self.fullres_image_levels = []
        all_spacings = []

        for c, im_path in enumerate(self.image_paths):
                       
            # Load image
            if im_path.suffix == ".mrxs":
                # Determine optimal level based on pixel spacing
                image = pyvips.Image.new_from_file(str(im_path))
                spacings = [1000 / image.get("xres")*2**i for i in range(int(image.get("openslide.level-count")))]
                optimal_level = np.argmin([abs(i-self.full_resolution_spacing) for i in spacings])
                
                # Ensure that full res image is at least as large as np images
                if optimal_level >= self.optimal_image_levels[c]:
                    optimal_level = self.optimal_image_levels[c]
                    
                # Load image at that spacing
                image_fullres = pyvips.Image.new_from_file(str(im_path), level=optimal_level)
            elif im_path.suffix == ".tif":
                # Determine optimal level based on pixel spacing
                image = pyvips.Image.new_from_file(str(im_path))
                spacings = [1000 / image.get("xres")*2**i for i in range(image.get("n-pages"))]
                optimal_level = np.argmin([abs(i-self.full_resolution_spacing) for i in spacings])
                
                # Ensure that full res image is at least as large as np images
                if optimal_level >= self.optimal_image_levels[c]:
                    optimal_level = self.optimal_image_levels[c]
                    
                # Load image at that spacing
                image_fullres = pyvips.Image.new_from_file(str(im_path), page=optimal_level)
            else:
                raise ValueError("Sorry, only .tifs and .mrxs are supported.")

            # Dispose of alpha band if present
            if image_fullres.bands == 4:
                image_fullres = image_fullres.flatten().cast("uchar")

            # Get pixel spacing and level
            pixel_spacing = spacings[optimal_level]
            all_spacings.append(pixel_spacing)
            self.fullres_image_levels.append(optimal_level)

            # Save images
            self.raw_fullres_images.append(image_fullres)

        # Pixel spacing and level consistency are soft requirements - may give unexpected results so warning required
        if len(np.unique(self.fullres_image_levels)) != 1:
            warnings.warn(f"Number of levels is not consistent between images. Found {np.unique(self.fullres_image_levels)}.\nThis may give unexpected results.")
        if len(np.unique(all_spacings)) > 1:
            warnings.warn(f"Pixel spacing is not consistent between images. Found {np.unique(all_spacings)}.\nThis may give unexpected results.")

        self.fullres_scaling = int(self.raw_fullres_images[0].width / self.raw_images[0].shape[1])

        return

    def load_masks(self) -> None:
        """
        Method to load the masks. These are based on the tissue segmentation algorithm described in:
        Bándi P, Balkenhol M, van Ginneken B, van der Laak J, Litjens G. 2019. Resolution-agnostic tissue segmentation in
        whole-slide histopathology images with convolutional neural networks. PeerJ 7:e8242 DOI 10.7717/peerj.8242.
        """

        self.raw_masks = []
        self.optimal_mask_levels = []

        if self.masks_available:
            print(f" - loading corresponding masks")
            for c, mask_path in enumerate(self.mask_paths):

                # Load mask that is closest to optimal image size
                mask = pyvips.Image.new_from_file(str(mask_path))
                mask_levels = int(mask.get("n-pages")) if mask_path.suffix == ".tif" else int(mask.get("openslide.level-count"))
                mask_sizes = [[int(mask.width/(2**i)), int(mask.height/(2**i))] for i in range(mask_levels)]
                optimal_mask_level = np.argmin([abs(np.mean(i)-self.optimal_image_size) for i in mask_sizes])
                mask_np = pyvips.Image.new_from_file(str(mask_path), page=optimal_mask_level).numpy()

                # Ensure size match between mask and image
                im_shape = self.raw_images[c].shape[:2]
                if im_shape != mask_np.shape[:2]:
                    mask_np = cv2.resize(mask_np, (im_shape[1], im_shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_np = ((mask_np > np.min(mask_np))*255).astype(np.uint8)
                
                # Clean up the mask by performing morphological closing and flood filling
                pad = int(0.1 * mask_np.shape[0])
                mask_pad = np.pad(mask_np, [[pad, pad], [pad, pad]], mode="constant", constant_values=0)

                ksize = int(0.01 * np.max(mask_np.shape))
                kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(ksize, ksize))
                mask_closed = cv2.morphologyEx(src=mask_pad, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=2)
            
                # Flood fill to get rid of holes in mask
                mask_ff = np.zeros((mask_closed.shape[0] + 2, mask_closed.shape[1] + 2)).astype("uint8")
                _, _, mask_final, _ = cv2.floodFill(mask_closed, mask_ff, (0, 0), 255)
                mask_final = 1 - mask_final[1+pad:-1-pad, 1+pad:-1-pad]
                
                self.raw_masks.append(mask_final)
                self.optimal_mask_levels.append(optimal_mask_level)

            self.load_masks_fullres()
        else:
            self.generate_masks()

        return

    def load_masks_fullres(self) -> None:
        """
        Method to load the full res masks.
        """

        self.raw_fullres_masks = [] 
        self.fullres_mask_levels = []
        all_spacings = []

        for c, mask_path in enumerate(self.mask_paths):
            
            # Investigate optimal mask level
            mask = pyvips.Image.new_from_file(str(mask_path))
            mask_levels = int(mask.get("n-pages")) if mask_path.suffix == ".tif" else int(mask.get("openslide.level-count"))
            spacings = [1000 / mask.get("xres")*2**i for i in range(mask_levels)]
            optimal_level = np.argmin([abs(i-self.full_resolution_spacing) for i in spacings])
            
            # Load mask 
            mask_fullres = pyvips.Image.new_from_file(str(mask_path), page=optimal_level)
            fullres_scaling_mask = mask_fullres.width / self.raw_fullres_images[c].width
            if fullres_scaling_mask != 1:
                mask_fullres = mask_fullres.resize(1/fullres_scaling_mask)

            # Ensure size congruency with image
            if not all([mask_fullres.width == self.raw_fullres_images[c].width, mask_fullres.height == self.raw_fullres_images[c].height]):
                mask_fullres = mask_fullres.gravity("centre", self.raw_fullres_images[c].width, self.raw_fullres_images[c].height)

            # Cast to uint8
            mask_fullres = ((mask_fullres > mask_fullres.min())*255).cast("uchar")
            self.raw_fullres_masks.append(mask_fullres)
            
            self.fullres_mask_levels.append(optimal_level)
            all_spacings.append(spacings[optimal_level])

        # Similar n-levels is soft requirement as long as pixel spacing is consistent
        if len(np.unique(self.fullres_mask_levels)) != 1:
            warnings.warn(f"Number of levels is not consistent between masks. Found {np.unique(self.fullres_mask_levels)}.")
        assert np.std(all_spacings) < np.mean(all_spacings)*0.01, f"Pixel spacing is not consistent between masks. Found {np.unique(all_spacings)}."

        return

    def generate_masks(self) -> None:
        """
        Method to generate masks using a simple thresholding if masks have not been precomputed.
        """
        
        print(f" - no masks found, generating masks")
        
        for c, image in enumerate(self.raw_images):
            
            # Convert image to HED
            image_hed = rgb2hed(image)
            
            # Apply thresholding on eosine channel
            thres = np.percentile(np.unique(image_hed[:, :, 1]), 1)
            mask = ((image_hed[:, :, 1] > thres)*1).astype("uint8")
            
            # Keep largest cc
            _, labeled_im, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            largest_cc_label = np.argmax(stats[1:, -1]) + 1
            mask = ((labeled_im == largest_cc_label) * 255).astype("uint8")
            
            # Pad mask to perform morphological closing
            pad = int(0.1 * mask.shape[0])
            mask_pad = np.pad(mask, [[pad, pad], [pad, pad]], mode="constant", constant_values=0)

            ksize = int(0.01 * np.max(mask.shape))
            kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(ksize, ksize))
            mask_closed = cv2.morphologyEx(src=mask_pad, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=2)
        
            # Flood fill to get rid of holes in mask
            mask_ff = np.zeros((mask_closed.shape[0] + 2, mask_closed.shape[1] + 2)).astype("uint8")
            _, _, mask_final, _ = cv2.floodFill(mask_closed, mask_ff, (0, 0), 255)
            mask_final = 1 - mask_final[1+pad:-1-pad, 1+pad:-1-pad]
                       
            self.raw_masks.append(mask_final)
            
        assert all([i.shape[:2]==j.shape for i, j in zip(self.raw_images, self.raw_masks)]), "Image and mask shapes do not match." 
        assert all([np.sum(i)>0 for i in self.raw_masks]), "Error in generating mask."
        
        self.generate_masks_fullres()
        
        return

    def generate_masks_fullres(self) -> None:
        """
        Method to resize the generated masks to match the resolution of the fullres images.
        """

        self.raw_fullres_masks = [] 

        for mask, image in zip(self.raw_masks, self.raw_fullres_images):
            
            # Resize mask to full resolution
            scaling = image.width / mask.shape[1]
            fullres_mask = pyvips.Image.new_from_array(mask, 1).resize(scaling)
            
            # Fix any potential scaling errors
            fullres_mask = fullres_mask.gravity("centre", image.width, image.height).cast("uchar")
            
            self.raw_fullres_masks.append(fullres_mask)

        return

    def crop_images(self) -> None:
        """
        Method to crop all whitespace from the images and masks.
        """

        self.cropped_images = []
        self.cropped_masks = []
        self.crop_bboxes = []
        
        for c, (image, mask) in enumerate(zip(self.raw_images, self.raw_masks)):

            # Get bounding box coords
            bbox = cv2.boundingRect(mask)

            # Crop image and mask
            cropped_image = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            cropped_mask = mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

            self.cropped_images.append(cropped_image)
            self.cropped_masks.append(cropped_mask)
            self.crop_bboxes.append(bbox)
            
            # Crop landmarks if present
            if self.landmarks_xy:
                self.landmarks_xy[c] = self.landmarks_xy[c] - np.array(bbox[:2])

        self.crop_images_fullres()


        return

    def crop_images_fullres(self) -> None:
        """
        Method to crop the full resolution images.
        """

        self.cropped_fullres_images = []
        self.cropped_fullres_masks = []

        for image_fullres, mask_fullres, bbox in zip(self.raw_fullres_images, self.raw_fullres_masks, self.crop_bboxes):
            
            # Also crop full resolution versions
            cropped_image_fullres = image_fullres.crop(bbox[0]*self.fullres_scaling, bbox[1]*self.fullres_scaling, bbox[2]*self.fullres_scaling, bbox[3]*self.fullres_scaling)
            cropped_mask_fullres = mask_fullres.crop(bbox[0]*self.fullres_scaling, bbox[1]*self.fullres_scaling, bbox[2]*self.fullres_scaling, bbox[3]*self.fullres_scaling)
            
            self.cropped_fullres_images.append(cropped_image_fullres)
            self.cropped_fullres_masks.append(cropped_mask_fullres)

        return

    def apply_masks(self) -> None:
        """
        Method to mask the images based on the convex hull of the contour. This allows
        for some information to be retained outside of the prostate. Images are padded
        to a square shape.
        """

        self.images = []
        self.masks = []

        print(f" - applying masks")

        # Get common square size by taking max dimension and adding padding
        factor = 1.2
        max_dim = int(np.max([max(i.shape[0], i.shape[1]) for i in self.cropped_images + self.cropped_masks]) * factor)

        for c, (image, mask) in enumerate(zip(self.cropped_images, self.cropped_masks)):
            
            # Pad numpy image to common square size
            h1 = int(np.ceil((max_dim - image.shape[0]) / 2))
            h2 = int(np.floor((max_dim - image.shape[0]) / 2))
            w1 = int(np.ceil((max_dim - image.shape[1]) / 2))
            w2 = int(np.floor((max_dim - image.shape[1]) / 2))
            image = np.pad(image, ((h1, h2), (w1, w2), (0, 0)), mode="constant", constant_values=255)
            
            h1_mask = int(np.ceil((max_dim - mask.shape[0]) / 2))
            h2_mask = int(np.floor((max_dim - mask.shape[0]) / 2))
            w1_mask = int(np.ceil((max_dim - mask.shape[1]) / 2))
            w2_mask = int(np.floor((max_dim - mask.shape[1]) / 2))
            mask = np.pad(mask, ((h1_mask, h2_mask), (w1_mask, w2_mask)), mode="constant", constant_values=0)
            
            mask = ((mask > 0)*255).astype("uint8")

            # Apply mask to image
            image[mask == 0] = 255

            # Save masked image 
            self.images.append(image)
            self.masks.append(mask)

            self.apply_masks_fullres(c, h1, w1, h1_mask, w1_mask, max_dim)
            
            if self.landmarks_xy:
                self.landmarks_xy[c] = self.landmarks_xy[c] + np.array([w1, h1])

        if self.scramble:
            self.images, self.masks, self.fullres_images, self.fullres_masks = self.scramble_images(
                images = self.images,
                masks = self.masks,
                fullres_images = self.fullres_images,
                fullres_masks = self.fullres_masks
            )

        return

    def apply_masks_fullres(self, c: int, h1: int, w1: int, h1_mask: int, w1_mask: int, max_dim: int) -> None:
        """
        Method to apply the full resolution masks.
        """

        if not hasattr(self, "fullres_images"):
            self.fullres_images = []
            self.fullres_masks = []

        fullres_image = self.cropped_fullres_images[c]
        fullres_mask = self.cropped_fullres_masks[c]

        # Pad pyvips images to common size
        fullres_image = fullres_image.embed(
            w1 * self.fullres_scaling, 
            h1 * self.fullres_scaling, 
            max_dim * self.fullres_scaling, 
            max_dim * self.fullres_scaling, 
            extend="white"
        )
        fullres_mask = fullres_mask.embed(
            w1_mask * self.fullres_scaling, 
            h1_mask * self.fullres_scaling, 
            max_dim * self.fullres_scaling, 
            max_dim * self.fullres_scaling, 
            extend="black"
        )
        inverse_fullres_mask = (fullres_mask.max() - fullres_mask)*(255/fullres_mask.max())

        # Mask image by adding white to image and then casting to uint8
        masked_image_fullres = (fullres_image + inverse_fullres_mask).cast("uchar")
        self.fullres_images.append(masked_image_fullres)
        self.fullres_masks.append(fullres_mask)

        return

    def pad_images(self) -> None:
        """
        Method to pad the images to a square shape. This should only be applied
        in case of validation of VALIS through RAPID's internal validation
        mechanism, since all feature extractors expect square images.
        """

        self.images = []
        self.masks = []
        
        # Get common square size by taking max dimension and adding padding
        factor = 1.2
        max_dim = int(np.max([max(i.shape[0], i.shape[1]) for i in self.raw_images + self.raw_masks]) * factor)

        for c, (image, mask) in enumerate(zip(self.raw_images, self.raw_masks)):    
            
            # Pad numpy image to common square size
            h1 = int(np.ceil((max_dim - image.shape[0]) / 2))
            h2 = int(np.floor((max_dim - image.shape[0]) / 2))
            w1 = int(np.ceil((max_dim - image.shape[1]) / 2))
            w2 = int(np.floor((max_dim - image.shape[1]) / 2))
            image = np.pad(image, ((h1, h2), (w1, w2), (0, 0)), mode="constant", constant_values=255)
            
            h1_mask = int(np.ceil((max_dim - mask.shape[0]) / 2))
            h2_mask = int(np.floor((max_dim - mask.shape[0]) / 2))
            w1_mask = int(np.ceil((max_dim - mask.shape[1]) / 2))
            w2_mask = int(np.floor((max_dim - mask.shape[1]) / 2))
            mask = np.pad(mask, ((h1_mask, h2_mask), (w1_mask, w2_mask)), mode="constant", constant_values=0)
            
            mask = ((mask > 0)*255).astype("uint8")

            # Apply mask to image
            image[mask == 0] = 255

            # Save masked image 
            self.images.append(image)
            self.masks.append(mask)

        self.pad_images_fullres()
            
        return

    def pad_images_fullres(self) -> None:
        """
        Method to pad the full resolution images to a square shape.
        """
        
        self.fullres_images = []
        self.fullres_masks = []

        # Get common square size by taking max dimension and adding padding
        factor = 1.2
        max_dim = int(np.max([max(i.width, i.height) for i in self.raw_fullres_images + self.raw_fullres_masks]) * factor)

        for image, mask in zip(self.raw_fullres_images, self.raw_fullres_masks):

            # Pad pyvips images to common size
            fullres_image = image.embed(
                0, 
                0, 
                max_dim, 
                max_dim, 
                extend="white"
            )   
            fullres_mask = mask.embed(
                0, 
                0, 
                max_dim, 
                max_dim, 
                extend="black"
            )
            self.fullres_images.append(fullres_image)
            self.fullres_masks.append(fullres_mask)

        return

    def scramble_images(self, images: List, masks: List, fullres_images: List, fullres_masks: List) -> tuple([List, List, List, List]): 
        """
        Method to apply a random rotation + translation to the images to 
        increase reconstruction difficulty. 
        """
        
        images_scrambled = []
        masks_scrambled = []
        fullres_images_scrambled = []
        fullres_masks_scrambled = []
        
        for c, (im, mask, im_f, mask_f, im_id) in enumerate(zip(images, masks, fullres_images, fullres_masks, self.image_ids)):

            # Get random tform matrix
            random_rot = np.random.randint(0, 360)
            random_trans = np.random.randint(-int(im.shape[0]*0.1), int(im.shape[0]*0.1), 2)
            tform = cv2.getRotationMatrix2D((im.shape[1]//2, im.shape[0]//2), random_rot, 1)
            tform[:, 2] += random_trans
            self.pred_orientations[im_id] += random_rot

            # Apply to regular image and mask (and landmarks if present)
            im_scrambled, mask_scrambled, landmarks = apply_affine_transform(
                image = im,
                tform = tform,
                mask = mask,
                landmarks = self.landmarks_xy[c] if self.landmarks_xy else None
            )
            images_scrambled.append(im_scrambled)
            masks_scrambled.append(mask_scrambled)
            if self.landmarks_xy:
                self.landmarks_xy[c] = landmarks

            # Apply to full res
            im_f_scrambled, mask_f_scrambled = apply_affine_transform_fullres(
                image = im_f,
                mask = mask_f,
                rotation = random_rot,
                translation = random_trans,
                center = (im.shape[1]//2, im.shape[0]//2),
                scaling = self.fullres_scaling
            )

            fullres_images_scrambled.append(im_f_scrambled)
            fullres_masks_scrambled.append(mask_f_scrambled)

        plot_scrambled_images(images_scrambled, self.local_save_dir)

        return images_scrambled, masks_scrambled, fullres_images_scrambled, fullres_masks_scrambled

    def normalize_stains(self) -> None:
        """
        Method to perform stain normalization to aid in keypoint detection. 
        """

        print(f" - finding optimal stain normalization")

        # Run initialization with every single image to find optimal ref image        
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x*255)
        ])
        self.stain_normalizer = torchstain.normalizers.ReinhardNormalizer(backend="torch")

        target_means = []
        for i in self.images:
            self.stain_normalizer.fit(T(i))
            target_means.append(self.stain_normalizer.target_means)
            
        # Find the image that is closest to the median stain in LAB colourspace
        target_means_np = np.stack([tm.numpy() for tm in target_means])
        median_lab = np.median(target_means_np, axis=0)
        distances = np.linalg.norm(target_means_np - median_lab, axis=1)
        self.stain_ref_image_idx = np.argmin(distances)
        ref_image = self.images[self.stain_ref_image_idx]

        normalized_images = []
        self.stain_normalizer = torchstain.normalizers.ReinhardNormalizer(backend="torch")
        self.stain_normalizer.fit(T(ref_image))

        print(f" - applying stain normalization")

        # Apply stain normalization to all images
        for im, mask in zip(self.images, self.masks):

            # Get stain normalized image, remove background artefacts and save
            norm_im = self.stain_normalizer.normalize(T(im))
            norm_im = norm_im.numpy().astype("uint8")
            norm_im[mask == 0] = 255
            normalized_images.append(norm_im)

        # Plot resulting normalization
        plot_stain_normalization(
            images = self.images,
            normalized_images = normalized_images,
            save_dir = self.local_save_dir
        )

        self.images = copy.copy(normalized_images)

        self.normalize_stains_fullres()

        return

    def normalize_stains_fullres(self) -> None:
        """
        Method to perform full resolution stain normalization directly on each slide.
        """
        
        # Fit reference image
        ref_image = self.fullres_images[self.stain_ref_image_idx]

        normalizer = Reinhard_normalizer()
        normalizer.fit(ref_image)

        normalized_images = []

        for im, mask in zip(self.fullres_images, self.fullres_masks):

            # Normalize image
            norm_im = normalizer.transform(im)

            # Correct for potential background artefacts
            inv_mask = (mask.max() - mask)*(255/mask.max())
            norm_im = (norm_im + inv_mask).cast("uchar")
            normalized_images.append(norm_im)

        self.fullres_images = copy.copy(normalized_images)

        return

    def align_center(self, images: List[np.ndarray], masks: List[np.ndarray], fullres_images: List[pyvips.Image], fullres_masks: List[pyvips.Image]) -> None:
        """
        Baseline method of just aligning all images with respect to their center point.
        """

        center_points = []

        # Find contour for all masks
        for mask in masks:
            
            # Get contour from mask
            contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour = np.squeeze(max(contour, key=cv2.contourArea))

            # Get center of contour
            M = cv2.moments(contour)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            center_points.append(center)
        
        common_center = np.mean(center_points, axis=0).astype("int")

        # Apply translation to all images
        final_images = []
        final_masks = []
        final_images_fullres = []
        final_masks_fullres = []

        for c, (image, mask, image_fullres, mask_fullres, center) in enumerate(zip(images, masks, fullres_images, fullres_masks, center_points)):

            # Apply translation
            translation = common_center - center
            translation_matrix = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
            image, mask, landmarks = apply_affine_transform(
                image = image,
                tform = translation_matrix,
                mask = mask,
                landmarks = self.landmarks_xy[c] if self.landmarks_xy else None
            )

            final_images.append(image)
            final_masks.append(mask)
            if self.landmarks_xy:
                self.landmarks_xy[c] = landmarks

            # Apply translation in full resolution
            image_fullres, mask_fullres = apply_affine_transform_fullres(
                image = image_fullres,
                mask = mask_fullres,
                rotation = 0,
                translation = translation,
                center = list(common_center),
                scaling = self.fullres_scaling
            )

            final_images_fullres.append(image_fullres)
            final_masks_fullres.append(mask_fullres)

        plot_align_center(
            images = final_images,
            center = common_center,
            savepath = self.local_save_dir.joinpath("03_align_center.png")
        )

        return final_images, final_masks, final_images_fullres, final_masks_fullres

    def find_rotations(self) -> None:
        """
        Method to get the rotation of the prostate based on an
        ellipsoid approximating the fit of the prostate.
        """

        print(f" - performing prealignment")
        self.ellipses = []

        # Find ellipse for all images
        for mask in self.masks:
            
            # Get contour from mask
            contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour = np.squeeze(max(contour, key=cv2.contourArea))

            # Fit ellipse based on contour 
            ellipse = cv2.fitEllipse(contour)
            self.ellipses.append(ellipse)

        # Middle of stack as reference
        self.ref_idx = int(len(self.images) // 2)

        # Find common centerpoint of all ellipses to orient towards
        self.common_center = np.mean([i[0] for i in self.ellipses], axis=0).astype("int")

        # Determine the dorsal side of the prostate 
        self.dorsal_rotation = find_dorsal_rotation(
            mask = self.masks[self.ref_idx], 
            ellipse = self.ellipses[self.ref_idx],
            center = self.common_center,
            savepath = self.local_save_dir.joinpath("debug", "dorsal_rotation.png")
        )

        # Plot resulting ellipse
        plot_ellipses(
            images=self.images, 
            ellipses=self.ellipses,
            ref_idx = self.ref_idx,
            save_dir=self.local_save_dir
        )

        return

    def prealignment(self, images: List[np.ndarray], masks: List[np.ndarray], images_fullres: List[pyvips.Image], masks_fullres: List[pyvips.Image]) -> tuple([List, List, List, List]):
        """
        Step to align the images based on an ellipse fitted through the prostate.
        """

        # As part of the prealignment we need to find the rotation for each fragment.
        self.find_rotations()

        final_images = []
        final_masks = []
        final_images_fullres = []
        final_masks_fullres = []

        for c, (image, mask, image_fullres, mask_fullres, ellipse, im_id) in enumerate(zip(images, masks, images_fullres, masks_fullres, self.ellipses, self.image_ids)):

            # Ensure horizontal slices 
            center, axis, rotation = ellipse
            if axis[1] > axis[0]:
                rotation += 90
            rotation += self.dorsal_rotation
            self.pred_orientations[im_id] += np.round(rotation, 1)

            # Adjust rotation 
            rotation_matrix = cv2.getRotationMatrix2D(tuple(center), rotation, 1)
            rotated_image, rotated_mask, landmarks = apply_affine_transform(
                image = image,
                tform = rotation_matrix,
                mask = mask,
                landmarks = self.landmarks_xy[c] if self.landmarks_xy else None
            )

            # Adjust translation 
            translation = self.common_center - center
            translation_matrix = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
            rotated_image, rotated_mask, landmarks = apply_affine_transform(
                image = rotated_image,
                tform = translation_matrix,
                mask = rotated_mask,
                landmarks = landmarks
            )

            final_images.append(rotated_image)
            final_masks.append(rotated_mask)
            if self.landmarks_xy:
                self.landmarks_xy[c] = landmarks

            # Apply rotation in full resolution
            rotated_image_fullres, rotated_mask_fullres = apply_affine_transform_fullres(
                image = image_fullres,
                mask = mask_fullres,
                rotation = rotation,
                translation = [0, 0],
                center = center,
                scaling = self.fullres_scaling
            )

            # Apply translation in full resolution
            translation = self.common_center - center
            rotated_image_fullres, rotated_mask_fullres = apply_affine_transform_fullres(
                image = rotated_image_fullres,
                mask = rotated_mask_fullres,
                rotation = 0,
                translation = translation,
                center = list(self.common_center),
                scaling = self.fullres_scaling
            )

            final_images_fullres.append(rotated_image_fullres)
            final_masks_fullres.append(rotated_mask_fullres)

        # Plot resulting prealignment
        plot_prealignment(
            images=final_images, 
            save_dir=self.local_save_dir
        )

        return final_images, final_masks, final_images_fullres, final_masks_fullres

    def initial_affine_registration(self, images: List[np.ndarray], masks: List[np.ndarray], images_fullres: List[pyvips.Image], masks_fullres: List[pyvips.Image]) -> tuple([List, List, List, List]):
        """
        Method to perform the affine registration between adjacent slides. This
        step consists of:
            1) Computing keypoints and matches between a pair of images.
            2) Finding the optimal orientation of the moving image.
            3) Using the keypoint to find an affine transform.
            4) Extrapolating this transform to the full res images.

        The affine transform is a limited transform that only includes 
        rotation and translation. 
        """

        print(f" - initializing affine reconstruction")

        final_images = [None] * len(images)
        final_images[self.ref_idx] = images[self.ref_idx]
        final_masks = [None] * len(images)
        final_masks[self.ref_idx] = masks[self.ref_idx]
        final_images_fullres = [None] * len(images)
        final_images_fullres[self.ref_idx] = images_fullres[self.ref_idx]
        final_masks_fullres = [None] * len(images)
        final_masks_fullres[self.ref_idx] = masks_fullres[self.ref_idx]
        if self.landmarks_xy:
            final_landmarks_xy = [None] * len(images)
            final_landmarks_xy[self.ref_idx] = self.landmarks_xy[self.ref_idx]

        moving_indices = list(np.arange(0, self.ref_idx)[::-1]) + list(np.arange(self.ref_idx+1, len(images)))
        moving_indices = list(map(int, moving_indices))
        ref_indices = list(np.arange(0, self.ref_idx)[::-1] + 1) + list(np.arange(self.ref_idx+1, len(images)) - 1)
        ref_indices = list(map(int, ref_indices))

        self.affine_rotations = dict([str(i), 0] for i in self.image_ids)

        # Init dino
        dino_extractor = Dino_extractor(
            method = "roma",
            cpt_path = None
        )

        for mov, ref in zip(moving_indices, ref_indices):

            best_dice = 0
            best_dino_cosine = -1

            # Try a few rotations based on the shape of the moving image
            ellipse_axis = self.ellipses[mov][1]
            rotations = np.arange(0, 181, 180) if np.max(ellipse_axis) > 1.25*np.min(ellipse_axis) else np.arange(0, 360, 90)

            for rot in rotations:

                # Compute flipped version of image
                rotation_matrix = cv2.getRotationMatrix2D(tuple(self.common_center.astype("float")), rot, 1)
                moving_image, moving_mask = images[mov], masks[mov]
                moving_image, moving_mask, landmarks = apply_affine_transform(
                    image = moving_image,
                    tform = rotation_matrix,
                    mask = moving_mask,
                    landmarks = self.landmarks_xy[mov] if self.landmarks_xy else None
                )
                ref_image, ref_mask = final_images[ref], final_masks[ref]

                # Extract keypoints
                ref_points, moving_points, scores = get_keypoints(
                    detector = self.detector, 
                    matcher = self.matcher,
                    detector_name = self.detector_name,
                    ref_image = ref_image, 
                    moving_image = moving_image,
                    ref_mask = ref_mask,
                    moving_mask = moving_mask,
                    patch_method = self.patch_method,
                    thres = self.keypoint_thres
                )
                plot_keypoint_pairs(
                    ref_image = ref_image,
                    moving_image = moving_image,
                    ref_points = ref_points,
                    moving_points = moving_points,
                    scores = scores,
                    tform = "affine",
                    ransac_thres = self.ransac_thres_affine,
                    savepath = self.local_save_dir.joinpath("keypoints", f"keypoints_affine1_{mov}_to_{ref}_rot_{rot}.png"),
                    filter_method = self.keypoint_filter
                )

                affine_matrix, num_matches = estimate_affine_transform(
                    moving_points = moving_points, 
                    ref_points = ref_points, 
                    scores = scores,
                    image = moving_image, 
                    filter_method = self.keypoint_filter, 
                    ransac_thres = self.ransac_thres_affine
                )
                moving_image_warped, moving_mask_warped, landmarks = apply_affine_transform(
                    image = moving_image,
                    tform = affine_matrix.params[:-1, :],
                    mask = moving_mask,
                    landmarks = landmarks
                )

                # Plot resulting warp
                plot_warped_images(
                    ref_image = ref_image,
                    ref_mask = final_masks[ref],
                    moving_image = moving_image,
                    moving_image_warped = moving_image_warped,
                    moving_mask_warped = moving_mask_warped,
                    savepath = self.local_save_dir.joinpath("warps", f"warps_affine1_{mov}_to_{ref}_rot_{rot}.png"),
                )
                dice_overlap = compute_dice(ref_mask, moving_mask_warped, normalized=True)
                
                #### EXPERIMENTAL  ####
                ref_dino_features = dino_extractor.extract(ref_image, batched=False)
                moving_dino_features = dino_extractor.extract(moving_image_warped, batched=False)
                
                # Compute cosine similarity
                cosine_similarity = compute_cosine_similarity(
                    f1 = ref_dino_features, 
                    f2 = moving_dino_features, 
                    f1_mask = ref_mask, 
                    f2_mask = moving_mask_warped, 
                    masked = True
                )
                plot_dino_features(
                    ref_image = ref_image,
                    moving_image = moving_image_warped,
                    ref_mask = ref_mask,
                    moving_mask = moving_mask_warped,
                    ref_dino_features = ref_dino_features,
                    moving_dino_features = moving_dino_features,
                    cosine_similarity = cosine_similarity,
                    savepath = self.local_save_dir.joinpath("keypoints", f"dino_feats_affine1_{mov}_to_{ref}_rot_{rot}.png")
                )
                #### \\\ EXPERIMENTAL ####
                
                if cosine_similarity > best_dino_cosine:
                    best_dice = dice_overlap
                    best_dino_cosine = cosine_similarity

                    # Save final image
                    final_images[mov] = moving_image_warped.astype("uint8")
                    final_masks[mov] = moving_mask_warped.astype("uint8")
                    best_rotation = -math.degrees(affine_matrix.rotation) + rot
                    self.affine_rotations[str(self.image_ids[mov])] = best_rotation

                    # Save landmarks
                    if self.landmarks_xy:
                        final_landmarks_xy[mov] = landmarks
                    
                    # Perform full resolution reconstruction
                    moving_image_fullres_warped, moving_mask_fullres_warped = apply_affine_transform_fullres(
                        image = images_fullres[mov],
                        mask = masks_fullres[mov],
                        rotation = rot,
                        translation = [0, 0],
                        center = self.common_center,
                        scaling = self.fullres_scaling
                    )
                    moving_image_fullres_warped, moving_mask_fullres_warped = apply_affine_transform_fullres(
                        image = moving_image_fullres_warped,
                        mask = moving_mask_fullres_warped,
                        rotation = -math.degrees(affine_matrix.rotation),
                        translation = affine_matrix.translation,
                        center = [0, 0],
                        scaling = self.fullres_scaling
                    )
                    final_images_fullres[mov] = moving_image_fullres_warped
                    final_masks_fullres[mov] = moving_mask_fullres_warped

        # Update final landmarks
        if self.landmarks_xy:
            self.landmarks_xy = final_landmarks_xy

        # Update final rotations
        for im_id in self.image_ids:
            self.pred_orientations[im_id] += np.round(self.affine_rotations[str(im_id)], 1)

        return final_images, final_masks, final_images_fullres, final_masks_fullres

    def finetune_affine_registration(self, images: List[np.ndarray], masks: List[np.ndarray], images_fullres: List[pyvips.Image], masks_fullres: List[pyvips.Image]) -> tuple([List, List, List, List]):
        """
        Method to finetune the affine registration.
        """ 
        
        print(f" - finetuning affine reconstruction")

        final_images = [None] * len(images)
        final_images[self.ref_idx] = images[self.ref_idx]
        final_masks = [None] * len(images)
        final_masks[self.ref_idx] = masks[self.ref_idx]
        final_images_fullres = [None] * len(images)
        final_images_fullres[self.ref_idx] = images_fullres[self.ref_idx]
        final_masks_fullres = [None] * len(images)
        final_masks_fullres[self.ref_idx] = masks_fullres[self.ref_idx]

        moving_indices = list(np.arange(0, self.ref_idx)[::-1]) + list(np.arange(self.ref_idx+1, len(images)))
        moving_indices = list(map(int, moving_indices))
        ref_indices = list(np.arange(0, self.ref_idx)[::-1] + 1) + list(np.arange(self.ref_idx+1, len(images)) - 1)
        ref_indices = list(map(int, ref_indices))

        self.affine_rotations_finetune = dict([str(i), 0] for i in self.image_ids)
        
        dino_extractor = Dino_extractor(
            method = "roma",
            cpt_path = None
        )
        
        for mov, ref in zip(moving_indices, ref_indices):

            # Compute flipped version of image
            moving_image, moving_mask = images[mov], masks[mov]
            ref_image, ref_mask = final_images[ref], final_masks[ref]

            # Extract keypoints
            ref_points, moving_points, scores = get_keypoints(
                detector = self.detector, 
                matcher = self.matcher,
                detector_name = self.detector_name,
                ref_image = ref_image, 
                moving_image = moving_image,
                ref_mask = ref_mask,
                moving_mask = moving_mask,
                patch_method = self.patch_method,
                thres = self.keypoint_thres
            )
            plot_keypoint_pairs(
                ref_image = ref_image,
                moving_image = moving_image,
                ref_points = ref_points,
                moving_points = moving_points,
                scores = scores,
                tform = "affine",
                ransac_thres = self.ransac_thres_affine,
                savepath = self.local_save_dir.joinpath("keypoints", f"keypoints_affine2_{mov}_to_{ref}.png"), 
                filter_method = self.keypoint_filter
            )

            affine_matrix, _ = estimate_affine_transform(
                moving_points = moving_points, 
                ref_points = ref_points, 
                scores = scores,
                image = moving_image, 
                filter_method = self.keypoint_filter, 
                ransac_thres = self.ransac_thres_affine
            )
            moving_image_warped, moving_mask_warped, landmarks = apply_affine_transform(
                image = moving_image,
                tform = affine_matrix.params[:-1, :],
                mask = moving_mask,
                landmarks = self.landmarks_xy[mov] if self.landmarks_xy else None
            )

            # Plot resulting warp
            plot_warped_images(
                ref_image = ref_image,
                ref_mask = final_masks[ref],
                moving_image = moving_image,
                moving_image_warped = moving_image_warped,
                moving_mask_warped = moving_mask_warped,
                savepath = self.local_save_dir.joinpath("warps", f"warps_affine2_{mov}_to_{ref}.png"),
            )

            # Only apply final transformation if it improves overlap
            ref_dino_features = dino_extractor.extract(ref_image, batched=False)
            moving_dino_features = dino_extractor.extract(moving_image, batched=False)
            moving_warped_dino_features = dino_extractor.extract(moving_image_warped, batched=False)
            
            # Compute cosine similarity
            initial_cosine_similarity = compute_cosine_similarity(  
                f1 = ref_dino_features, 
                f2 = moving_dino_features, 
                f1_mask = ref_mask, 
                f2_mask = moving_mask, 
                masked = True
            )
            finetuned_cosine_similarity = compute_cosine_similarity(
                f1 = ref_dino_features, 
                f2 = moving_warped_dino_features, 
                f1_mask = ref_mask, 
                f2_mask = moving_mask_warped, 
                masked = True
            )
            
            if finetuned_cosine_similarity > initial_cosine_similarity:
                # Save final image
                final_images[mov] = moving_image_warped.astype("uint8")
                final_masks[mov] = moving_mask_warped.astype("uint8")
                self.affine_rotations_finetune[str(self.image_ids[mov])] = -math.degrees(affine_matrix.rotation)

                # Save landmarks if present
                if self.landmarks_xy:
                    self.landmarks_xy[mov] = landmarks

                # Perform full resolution reconstruction
                moving_image_fullres_warped, moving_mask_fullres_warped = apply_affine_transform_fullres(
                    image = images_fullres[mov],
                    mask = masks_fullres[mov],
                    rotation = -math.degrees(affine_matrix.rotation),
                    translation = affine_matrix.translation,
                    center = [0, 0],
                    scaling = self.fullres_scaling
                )
                final_images_fullres[mov] = moving_image_fullres_warped
                final_masks_fullres[mov] = moving_mask_fullres_warped
            else:
                final_images[mov] = moving_image
                final_masks[mov] = moving_mask
                final_images_fullres[mov] = images_fullres[mov]
                final_masks_fullres[mov] = masks_fullres[mov]

        # Update final rotations
        for im_id in self.image_ids:
            self.pred_orientations[im_id] += np.round(self.affine_rotations_finetune[str(im_id)], 1)
        
        return final_images, final_masks, final_images_fullres, final_masks_fullres

    def deformable_registration(self, images: List[np.ndarray], masks: List[np.ndarray], images_fullres: List[pyvips.Image], masks_fullres: List[pyvips.Image]) -> tuple([List, List, List, List]):
        """
        Method to perform a deformable registration between adjacent slides. This
        step consists of:
            1) Computing keypoints and matches between a pair of images.
            2) Using the keypoint to find a thin plate splines transform.
            3) Extrapolating this transform to the full res images.

        In principle we can use any deformable registration here as long as the
        deformation can be represented as a grid to warp the image.
        """

        print(f" - performing deformable reconstruction")

        # We use the mid slice as reference point and move all images toward this slice.
        final_images = [None] * len(images)
        final_images[self.ref_idx] = images[self.ref_idx]
        final_masks = [None] * len(images)
        final_masks[self.ref_idx] = masks[self.ref_idx]
        final_images_fullres = [None] * len(images)
        final_images_fullres[self.ref_idx] = images_fullres[self.ref_idx]
        final_masks_fullres = [None] * len(images)
        final_masks_fullres[self.ref_idx] = masks_fullres[self.ref_idx]

        moving_indices = list(np.arange(0, self.ref_idx)[::-1]) + list(np.arange(self.ref_idx+1, len(images)))
        moving_indices = list(map(int, moving_indices))
        ref_indices = list(np.arange(0, self.ref_idx)[::-1] + 1) + list(np.arange(self.ref_idx+1, len(images)) - 1)
        ref_indices = list(map(int, ref_indices))

        # Lower threshold to get more keypoints
        self.matcher.sample_thresh = 0.01

        for mov, ref in zip(moving_indices, ref_indices):

            # Compute flipped version of image
            moving_image, moving_mask = images[mov], masks[mov]
            ref_image, ref_mask = images[ref], masks[ref]

            # Extract keypoints
            ref_points, moving_points, scores = get_keypoints(
                detector = self.detector,
                matcher = self.matcher, 
                detector_name = self.detector_name,
                ref_image = ref_image, 
                moving_image = moving_image, 
                ref_mask = ref_mask,
                moving_mask = moving_mask,
                patch_method = self.patch_method, 
                thres = self.keypoint_thres
            )
            plot_keypoint_pairs(
                ref_image = ref_image,
                moving_image = moving_image,
                ref_points = ref_points,
                moving_points = moving_points,
                scores = scores,
                tform = "deformable",
                ransac_thres = self.ransac_thres_affine,
                savepath = self.local_save_dir.joinpath("keypoints", f"keypoints_deformable_{mov}_to_{ref}.png"),
                filter_method = self.keypoint_filter
            )

            # Apply transforms
            index_map, grid = estimate_deformable_transform(
                moving_image = moving_image,
                ref_image = ref_image,
                moving_points = moving_points, 
                ref_points = ref_points, 
                deformable_level = self.optimal_image_levels[0],
                keypoint_level = self.optimal_image_levels[0],
                device = self.device,
                lambda_param = 1000
            )
            moving_image_warped, moving_mask_warped, landmarks_warped = apply_deformable_transform(
                moving_image = moving_image,
                moving_mask = moving_mask,
                index_map = index_map,
                landmarks = self.landmarks_xy[mov] if self.landmarks_xy else None
            )
            plot_warped_deformable_images(
                ref_image = ref_image,
                ref_mask = ref_mask,
                moving_image = moving_image,
                moving_mask = moving_mask,
                moving_image_warped = moving_image_warped,
                moving_mask_warped = moving_mask_warped,
                moving_landmarks = self.landmarks_xy[mov] if self.landmarks_xy else None,
                moving_landmarks_warped = landmarks_warped,
                grid = grid,
                savepath = self.local_save_dir.joinpath("warps", f"warps_deformable_{mov}_to_{ref}.png")
            )

            # Save final image
            final_images[mov] = moving_image_warped.astype("uint8")
            final_masks[mov] = moving_mask_warped.astype("uint8")
            if self.landmarks_xy:
                self.landmarks_xy[mov] = landmarks_warped

            # Perform full resolution reconstruction
            moving_image_fullres_warped, moving_mask_fullres_warped = apply_deformable_transform_fullres(
                image = images_fullres[mov],
                mask = masks_fullres[mov],
                grid = grid,
                scaling = self.fullres_scaling
            )
            final_images_fullres[mov] = moving_image_fullres_warped
            final_masks_fullres[mov] = moving_mask_fullres_warped

        return final_images, final_masks, final_images_fullres, final_masks_fullres


    def registration(self) -> None:
        """
        Method to perform all steps of the registration process.
        """

        # Skip everything if we're just evaluating the VALIS baseline
        if self.mode == "valis":
            self.final_images = self.images
            self.final_masks = self.masks
            self.final_images_fullres = self.fullres_images
            self.final_masks_fullres = self.fullres_masks
            return

        if self.mode == "baseline":
            images, masks, fullres_images, fullres_masks = self.align_center(
                images = self.images,
                masks = self.masks,
                fullres_images = self.fullres_images,
                fullres_masks = self.fullres_masks
            )

            self.final_images = images
            self.final_masks = masks
            self.final_images_fullres = fullres_images
            self.final_masks_fullres = fullres_masks

            return

        # Pre-alignment as initial step
        images, masks, fullres_images, fullres_masks = self.prealignment(
            images = self.images,
            masks = self.masks,
            images_fullres = self.fullres_images,
            masks_fullres = self.fullres_masks
        )

        # Initial affine registration to improve prealignment
        images, masks, fullres_images, fullres_masks = self.initial_affine_registration(
            images = images,
            masks = masks,
            images_fullres = fullres_images,
            masks_fullres = fullres_masks,
        )
        plot_final_reconstruction(
            final_images = images, 
            save_dir = self.local_save_dir, 
            tform = "affine1"
        )
        
        # Finetune affine registration
        images, masks, fullres_images, fullres_masks = self.finetune_affine_registration(
            images = images,
            masks = masks,
            images_fullres = fullres_images,
            masks_fullres = fullres_masks,
        )
        plot_final_reconstruction(
            final_images = images, 
            save_dir = self.local_save_dir, 
            tform = "affine2"
        )

        # Deformable registration as final step
        if self.mode == "deformable":
            images, masks, fullres_images, fullres_masks = self.deformable_registration(
                images = images,
                masks = masks,
                images_fullres = fullres_images,
                masks_fullres = fullres_masks,
            )
            plot_final_reconstruction(
                final_images = images, 
                save_dir = self.local_save_dir, 
                tform = "deformable"
            )

        # Save to use for 3D reconstruction
        self.final_images = images
        self.final_masks = masks
        self.final_images_fullres = fullres_images
        self.final_masks_fullres = fullres_masks
        
        return

    def reconstruct_3d_volume(self) -> None:
        """
        Method to create a 3D representation of the stacked slices. We leverage sectioning
        variables such as slice thickness, slice distance and x-y-z downsampling levels
        to create an anatomical true to size 3D volume.
        """

        print(f" - creating 3D volume")

        # Prostate specific variables
        slice_thickness = self.config.slice_thickness
        slice_distance = self.config.slice_distance

        # Downsample images for computational efficiency in shape analysis 
        partial_xy_downsample = 2 ** (self.optimal_image_levels[0]+1 - self.optimal_image_levels[0])
        total_xy_downsample = 2 ** (self.optimal_image_levels[0]+1)
        new_size = tuple(int(i / partial_xy_downsample) for i in self.final_images[0].shape[:2][::-1])
        
        self.final_images_ds = [cv2.resize(i, new_size, interpolation=cv2.INTER_AREA) for i in self.final_images]
        self.final_masks_ds = [cv2.resize(i, new_size, interpolation=cv2.INTER_NEAREST) for i in self.final_masks]
        
        # Fetch fresh contours
        self.final_contours_ds = []
        for mask in self.final_masks_ds:
            contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour = np.squeeze(max(contour, key=cv2.contourArea))
            self.final_contours_ds.append(contour)

        # Block size is the number of empty slices we have to insert between
        # actual slices for a true to size 3D model.
        z_downsample = slice_distance / total_xy_downsample
        self.block_size = int(np.round(z_downsample * slice_thickness)-1)

        # Pre-allocate 3D volumes 
        self.final_reconstruction_3d = np.zeros(
            (self.final_images_ds[0].shape[0], 
             self.final_images_ds[0].shape[1], 
             self.block_size*(len(self.final_images_ds)+1),
             self.final_images_ds[0].shape[2]),
             dtype="uint8"
        )
        self.final_reconstruction_3d_mask = np.zeros(
            self.final_reconstruction_3d.shape[:-1]  ,             
            dtype="uint8"
        )

        # Populate actual slices
        for c, (im, mask) in enumerate(zip(self.final_images_ds, self.final_masks_ds)):
            self.final_reconstruction_3d[:, :, self.block_size*(c+1), :] = im
            self.final_reconstruction_3d_mask[:, :, self.block_size*(c+1)] = mask
            
        self.final_reconstruction_3d[self.final_reconstruction_3d == 255] = 0

        # Interpolate the volume
        self.interpolate_3d_volume()

        return
    
    def interpolate_3d_volume(self) -> None:
        """
        Method to interpolate the 2D slices to a binary 3D volume.
        """

        self.filled_slices = [self.block_size*(i+1) for i in range(len(self.final_images_ds))]

        # Loop over slices for interpolation
        for i in range(len(self.filled_slices)-1):

            # Get two adjacent slices
            slice_a = self.final_reconstruction_3d_mask[:, :, self.filled_slices[i]]
            slice_b = self.final_reconstruction_3d_mask[:, :, self.filled_slices[i+1]]

            # Get contours, simplify and resample
            num_points = 360
            contour_a = self.final_contours_ds[i]
            contour_a = simplify_contour(contour_a)
            contour_a = resample_contour_radial(contour_a, num_points)

            contour_b = self.final_contours_ds[i+1]
            contour_b = simplify_contour(contour_b)
            contour_b = resample_contour_radial(contour_b, num_points)

            for j in range(self.block_size-1):

                # Compute weighted average of contour a and b
                fraction = j / (self.block_size-1)
                contour = (1-fraction) * contour_a + fraction * contour_b

                # Fill contour to make a mask
                mask = np.zeros_like(slice_a)
                cv2.drawContours(mask, [contour.astype("int")], -1, (255),thickness=cv2.FILLED)

                # savepath = self.debug_dir.joinpath(f"contour_{self.filled_slices[i]+j+1}.png")
                # plot_interpolated_contour(slice_a, contour_a, mask, contour, slice_b, contour_b, savepath)

                self.final_reconstruction_3d_mask[:, :, self.filled_slices[i]+j+1] = mask

        # Plot snapshot of result
        image_indices = sorted([i.name.split("_")[0] for i in self.local_save_dir.glob("*.png")])
        idx = int(image_indices[-1]) + 1
        plot_3d_volume(
            volume = self.final_reconstruction_3d_mask, 
            savepath = self.local_save_dir.joinpath(f"{str(idx).zfill(2)}_reconstruction_3d.png")
        )

        return

    def evaluate_reconstruction(self):
        """
        Method to compute the metrics to evaluate the reconstruction quality.
        """

        print(" - evaluating reconstruction")

        # Compute average registration error between keypoints of adjacent images
        self.tre = compute_tre_keypoints(
            images = self.final_images,
            masks = self.final_masks,
            detector = self.detector,
            matcher = self.matcher,
            detector_name = self.detector_name,
            savedir = self.local_save_dir,
            spacing = self.pixel_spacing_image,
            thres = self.keypoint_thres
        )

        # Compute dice score of all adjacent masks
        self.reconstruction_dice = compute_reconstruction_dice(masks = self.final_masks, normalized = True)

        # Compute median contour distance of all adjacent masks
        self.contour_distance = compute_contour_distance(
            masks = self.final_masks,
            level = self.optimal_image_levels[0],
            spacing = self.pixel_spacing_image
        )

        # Compute orientation accuracy
        if self.gt_orientations is not None:
            self.orientation_correct, self.orientation_accuracy, self.gt_deltas, self.pred_deltas = compute_orientation_accuracy(
                gt_orientations = self.gt_orientations,
                pred_orientations = self.pred_orientations,
                ref_idx = self.ref_idx
            )
            plot_orientation_accuracy(
                final_images = self.final_images,
                deltas = self.orientation_accuracy,
                save_dir = self.local_save_dir
            )

        # Compute TRE between landmarks
        if self.landmarks_xy:
            self.tre_landmarks = compute_tre_landmarks(
                images = self.final_images,
                landmarks = self.landmarks_xy,
                savedir = self.local_save_dir
            )

        return

    def save_results(self):
        """
        Copy all created figures from Docker to external storage.
        """

        print(f" - saving results")

        # Save warped images for later inspection
        self.local_save_dir.joinpath("fullres_images").mkdir(parents=True, exist_ok=True)
        for name, im, mask in zip(self.image_ids, self.final_images_fullres, self.final_masks_fullres):
            im.write_to_file(
                str(self.local_save_dir.joinpath("fullres_images", f"{name}.tif")),
                tile=True,
                compression="jpeg",
                bigtiff=True,
                pyramid=True,
                Q=80
            )
            mask.write_to_file(
                str(self.local_save_dir.joinpath("fullres_images", f"{name}_mask.tif")),
                tile=True,
                compression="lzw",
                bigtiff=True,
                pyramid=True,
            )

        # Upload local results to external storage 
        self.save_dir.mkdir(parents=True, exist_ok=True)
        subprocess.call(f"rsync -r {self.local_save_dir} {self.save_dir.parent}", shell=True)
        shutil.rmtree(self.local_save_dir)

        # Clean up some torch memory to prevent OOM
        torch.cuda.empty_cache()

        return
    
    def run(self):
        """
        Method to run the full pipeline.
        """

        self.load_images()
        self.load_masks()
        if self.mode == "valis":
            self.pad_images()
        else:
            self.crop_images()
            self.apply_masks()
            self.normalize_stains()
        self.registration()
        # self.reconstruct_3d_volume()
        self.evaluate_reconstruction()
        self.save_results()

        return

