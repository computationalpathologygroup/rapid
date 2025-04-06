import argparse
import pandas as pd
import json
import numpy as np
from pathlib import Path
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

from rapid import Rapid


def collect_arguments():
    """
    Function to collect all arguments.
    """

    # Parse arguments
    parser = argparse.ArgumentParser(description='Parse arguments for 3D reconstruction.')
    parser.add_argument(
        "--joboverview",
        required=True,
        type=Path,
        help="Path to a csv/excel file containing the data to be reconstructed. This should be a file with the columns " + \
              "'imagepath' with the full path to the image (i.e. /data/image01.tif), 'maskpath' with the full path to " + \
              "the corresponding tissue mask (i.e. /data/image01_mask.tif), 'savepath' with the full path to save " + \
              "the reconstructed image (i.e. /data/image01_registered.tif), 'case' with the group identifier of " + \
              "a list of images (i.e. prostate01) and optionally a column 'gt_orientation' specifying the ground " + \
              "truth orientation in degrees. This last column will be used solely for evaluating reconstruction accuracy."
    )
    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        default="affine",
        help="Mode to run RAPID, options are ['prealignment', 'affine', 'deformable', 'valis', 'baseline']."
    )
    parser.add_argument(
        "--detector",
        required=True,
        type=str,
        default="roma",
        help="Detector to use, options are ['superpoint', 'sift', 'roma', 'dedode', 'omniglue']."
    )
    parser.add_argument(
        "--patch_method",
        type=str,
        default="regular",
        help="Method to use for keypoint detection, options are ['regular', 'patch', 'parcellated']."
    )
    parser.add_argument(
        "--keypoint_filter",
        type=str,
        default="ransac",
        help="Method to filter keypoints, options are ['none', 'ransac', 'magsac']."
    )
    parser.add_argument(
        "--affine_ransac_thres",
        type=float,
        default=0.02,
        help="Threshold for RANSAC sampling in the affine transform."
    )
    parser.add_argument(
        "--keypoint_thres",
        type=float,
        default=0.9,
        help="Threshold for keypoint detection."
    )
    parser.add_argument(
        "--savedir",
        required=True,
        type=Path,
        help="Path to a directory where to save the results."
    )
    args = parser.parse_args()

    joboverview = args.joboverview
    mode = args.mode.lower()
    detector = args.detector.lower()
    patch_method = args.patch_method.lower()
    affine_ransac_thres = args.affine_ransac_thres
    keypoint_thres = args.keypoint_thres
    keypoint_filter = args.keypoint_filter.lower()
    save_dir = args.savedir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    assert joboverview.exists(), "Data file does not exist."
    assert joboverview.suffix in [".csv", ".xlsx"], "Job overview must be a csv or excel file."
    assert mode in ["prealignment", "affine", "deformable", "valis", "baseline"], "Mode not recognized, must be any of ['prealignment', 'affine', 'deformable', 'valis', 'baseline']."
    assert detector in ["superpoint", "sift", "roma", "dedode", "omniglue"], "Detector not recognized, must be any of ['superpoint', 'sift', 'roma', 'dedode', 'omniglue']."
    assert patch_method in ["regular", "patch", "parcellated"], "Patch method not recognized, must be any of ['regular', 'patch', 'parcellated']."
    assert keypoint_filter in ["none", "ransac", "magsac"], "Keypoint filter not recognized, must be any of ['none', 'ransac', 'magsac']."

    return joboverview, mode, detector, patch_method, affine_ransac_thres, keypoint_thres, keypoint_filter, save_dir


def main(): 
    """
    Main function.
    """

    np.random.seed(42)

    # Get args
    joboverview, mode, detector, patch_method, affine_ransac_thres, keypoint_thres, keypoint_filter, save_dir = collect_arguments()
    
    # Get overview of all cases to be processed
    overview_df = pd.read_csv(joboverview) if joboverview.suffix == ".csv" else pd.read_excel(joboverview)
    required_columns = ["imagepath", "case"]
    overview_df_columns = overview_df.columns.values.tolist()
    assert all([i in overview_df_columns for i in required_columns]), "Joboverview df must contain the columns 'imagepath' and 'case'. Please run 'python main.py --help' for more information."
    
    cases = np.unique(list(overview_df["case"].values))

    print(f"\nRunning job with following parameters:" \
          f"\n - num cases: {len(cases)}" \
          f"\n - detector: {detector}" \
          f"\n - mode: {mode}" \
    )
    if save_dir.joinpath("aggregate_metrics.xlsx").exists():
        metrics_df = pd.read_excel(save_dir.joinpath("aggregate_metrics.xlsx"))
    else:
        metrics_df = pd.DataFrame()

    for case in cases:

        if not save_dir.joinpath(str(case), "fullres_images").is_dir():
            case_df = overview_df[overview_df["case"] == case]
            print(f"\nProcessing case {case} (n={len(case_df)} slides)")

            try:
                rapid = Rapid(
                    case_df = case_df,
                    mode = mode,
                    detector = detector,
                    patch_method = patch_method,
                    save_dir = save_dir,
                    affine_ransac_thres = affine_ransac_thres,
                    keypoint_thres = keypoint_thres,
                    keypoint_filter = keypoint_filter
                )
                rapid.run()

                # Save results in dataframe
                _metrics_df = pd.DataFrame({
                    "case": [str(case)],
                    "partition": [overview_df.loc[overview_df["case"] == case, "partition"].values[0]],
                    "dice": [rapid.reconstruction_dice] if hasattr(rapid, "reconstruction_dice") else [np.nan],
                    "tre_estimated": [rapid.tre] if hasattr(rapid, "tre") else [np.nan],
                    "median_contour_dist": [rapid.contour_distance] if hasattr(rapid, "contour_distance") else [np.nan],
                    "mode": [mode],
                    "orientation_accuracy": [rapid.orientation_accuracy] if hasattr(rapid, "orientation_accuracy") else [np.nan],
                    "orientation_correct": [rapid.orientation_correct] if hasattr(rapid, "orientation_correct") else [np.nan],
                    "tre_landmarks": [rapid.tre_landmarks] if hasattr(rapid, "tre_landmarks") else [np.nan]
                })
            except Exception as e:
                print(f"\nError processing case {case}: {e}")
                _metrics_df = pd.DataFrame({
                    "case": [case],
                    "partition": [overview_df.loc[overview_df["case"] == case, "partition"].values[0]],
                    "dice": [np.nan],
                    "tre": [np.nan],
                    "median_contour_dist": [np.nan],
                    "mode": [mode],
                    "orientation_accuracy": [np.nan],
                    "orientation_correct": [np.nan],
                })

            metrics_df = pd.concat([metrics_df, _metrics_df], ignore_index=True)
            metrics_df.to_excel(save_dir.joinpath("aggregate_metrics.xlsx"), index=False)

            # Save most important config details          
            config = {
                "mode": mode,
                "detector": detector,
                "patch_method": patch_method,
                "full_resolution_spacing": rapid.full_resolution_spacing,
                "ransac_thresholds": rapid.ransac_thres_affine,
                "keypoint_thres": keypoint_thres,
                "keypoint_filter": keypoint_filter
            } 
            with open(save_dir.joinpath("reconstruction_config.json"), "w") as f:
                json.dump(config, f)
        else:
            print(f"\nCase {case} already processed, skipping.")

    return


if __name__ == "__main__":    
    main()