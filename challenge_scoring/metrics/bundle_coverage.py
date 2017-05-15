#!/usr/bin/env python

from __future__ import division

import numpy as np


from challenge_scoring.tractanalysis.robust_streamlines_metrics \
    import compute_robust_tract_counts_map


def _compute_f1_score(overlap, overreach):
    # https://en.wikipedia.org/wiki/F1_score
    recall = overlap
    precision = 1 - overreach
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def _compute_overlap(gt_data, candidate_data):
    basic_non_zero = np.count_nonzero(gt_data)
    overlap = np.logical_and(gt_data, candidate_data)
    overlap_count = np.float32(np.count_nonzero(overlap))
    return overlap_count / basic_non_zero


def _compute_overreach(gt_data, candidate_data):
    diff = candidate_data - gt_data
    diff[diff < 0] = 0
    overreach_count = np.count_nonzero(diff)
    if np.count_nonzero(candidate_data) == 0:
        return 0

    return overreach_count / np.count_nonzero(candidate_data)


def _compute_overreach_normalize_gt(gt_data, candidate_data):
    diff = candidate_data - gt_data
    diff[diff < 0] = 0
    overreach_count = np.count_nonzero(diff)
    return overreach_count / np.count_nonzero(gt_data)


def _create_binary_map(tractogram, ref_img):
    tractogram.to_world().apply_affine(np.linalg.inv(ref_img.affine))  # Send to voxel space.
    translation = np.eye(4)
    translation[:-1,-1] = 0.5
    tractogram.apply_affine(translation) # Shift of half a voxel.

    sl_map = compute_robust_tract_counts_map(tractogram.streamlines,
                                             ref_img.shape)
    return (sl_map > 0).astype(np.int16)


def compute_bundle_coverage_scores(tractogram, ground_truth_mask):
    """ Computes scores related to bundle coverage.

    This function computes, for a given bundle, the bundle overlap (OL),
    bundle overreach (OR) bundle overreach normalized (ORn) and the
    f1-score (F1).

    Parameters
    ----------
    tractogram : Tractogram object
        Streamlines to score.
    ground_truth_mask : `:class:Nifti1Image` object
        Mask of the ground truth bundle.
    """
    gt_data = ground_truth_mask.get_data()
    candidate_data = _create_binary_map(tractogram, ground_truth_mask)
    overlap = _compute_overlap(gt_data, candidate_data)
    overreach = _compute_overreach(gt_data, candidate_data)
    overreach_norm = _compute_overreach_normalize_gt(gt_data, candidate_data)
    f1_score = _compute_f1_score(overlap, overreach)

    return {'OL': overlap,
            'OR': overreach,
            'ORn': overreach_norm,
            'F1': f1_score}
