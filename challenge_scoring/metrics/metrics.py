#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import logging
import os

import nibabel as nib
import numpy as np

from dipy.tracking.streamline import set_number_of_points
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.metrics import length as slength

from tractconverter.formats.tck import TCK

# TODO check names
from challenge_scoring import NB_POINTS_RESAMPLE
from challenge_scoring.io.streamlines import get_tracts_voxel_space_for_dipy, \
                                       save_tracts_tck_from_dipy_voxel_space, \
                                       save_valid_connections, \
                                       save_invalid_connections
from challenge_scoring.metrics.invalid_connections import group_and_assign_ibs
from challenge_scoring.metrics.valid_connections import auto_extract_VCs


def score_from_files(filename, masks_dir, bundles_dir,
                     tracts_attribs, basic_bundles_attribs,
                     save_full_vc=False,
                     save_full_ic=False,
                     save_IBs=False,
                     save_VBs=False,
                     segmented_out_dir='', segmented_base_name='',
                     verbose=False):
    """
    Computes all metrics in order to score a tractogram.

    Given a ``tck`` file of streamlines and a folder containing masks,
    compute the percent of: Valid Connections (VC), Invalid Connections (IC),
    Valid Connections but Wrong Path (VCWP), No Connections (NC),
    Average Bundle Coverage (ABC), Average ROIs Coverage (ARC),
    coverage per bundles and coverage per ROIs. It also provides the number of:
    Valid Bundles (VB), Invalid Bundles (IB) and streamlines per bundles.


    Parameters
    ------------
    filename : str
       name of a tracts file
    masks_dir : str
       name of the directory containing the masks
    save_segmented : bool
        if true, saves the segmented VC, IC, VCWP and NC

    Returns
    ---------
    scores : dict
        dictionnary containing a score for each metric
    indices : dict
        dictionnary containing the indices of streamlines composing VC, IC,
        VCWP and NC

    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    rois_dir = os.path.join(masks_dir, "rois")
    bundles_masks_dir = os.path.join(masks_dir, "bundles")
    wm_file = os.path.join(masks_dir, "wm.nii.gz")

    wm = nib.load(wm_file)
    streamlines_gen = get_tracts_voxel_space_for_dipy(filename, wm_file, tracts_attribs)

    ROIs = [nib.load(os.path.join(rois_dir, f)) for f in sorted(os.listdir(rois_dir))]
    bundles_masks = [nib.load(os.path.join(bundles_masks_dir, f)) for f in sorted(os.listdir(bundles_masks_dir))]
    ref_bundles = []

    # Ref bundles will contain {'name': 'name_of_the_bundle', 'threshold': thres_value,
    #                           'streamlines': list_of_streamlines}
    dummy_attribs = {'orientation': 'LPS'}
    qb = QuickBundles(20, metric=AveragePointwiseEuclideanMetric())

    for bundle_idx, bundle_f in enumerate(sorted(os.listdir(bundles_dir))):
        bundle_attribs = basic_bundles_attribs.get(os.path.basename(bundle_f))
        if bundle_attribs is None:
            raise ValueError("Missing basic bundle attribs for {0}".format(bundle_f))

        # Already resample to avoid doing it for each iteration of chunking
        orig_strl = [s for s in get_tracts_voxel_space_for_dipy(
                                os.path.join(bundles_dir, bundle_f),
                                wm_file, dummy_attribs)]
        resamp_bundle = set_number_of_points(orig_strl, NB_POINTS_RESAMPLE)
        resamp_bundle = [s.astype('f4') for s in resamp_bundle]

        bundle_cluster_map = qb.cluster(resamp_bundle)
        bundle_cluster_map.refdata = resamp_bundle

        bundle_mask_inv = nib.Nifti1Image((1 - bundles_masks[bundle_idx].get_data()) * wm.get_data(),
                                          bundles_masks[bundle_idx].get_affine())

        bundle_name, _ = os.path.splitext(os.path.basename(bundle_f))
        ref_bundles.append({'name': bundle_name,
                            'threshold': bundle_attribs['cluster_threshold'],
                            'cluster_map': bundle_cluster_map,
                            'mask': bundles_masks[bundle_idx],
                            'mask_inv': bundle_mask_inv})

    score_func = score_auto_extract_auto_IBs

    return score_func(streamlines_gen, bundles_masks, ref_bundles, ROIs, wm,
                      save_full_vc=save_full_vc,
                      save_full_ic=save_full_ic,
                      save_IBs=save_IBs,
                      save_VBs=save_VBs,
                      out_segmented_strl_dir=segmented_out_dir,
                      base_out_segmented_strl=segmented_base_name,
                      ref_anat_fname=wm_file)





def score_auto_extract_auto_IBs(streamlines, bundles_masks, ref_bundles, ROIs, wm,
                                save_full_vc=False,
                                save_full_ic=False,
                                save_IBs=False,
                                save_VBs=False,
                                out_segmented_strl_dir='',
                                base_out_segmented_strl='',
                                ref_anat_fname=''):
    """
    TODO document
    
     # New algorithm
    # Step 1: remove streamlines shorter than threshold (currently 35)
    # Step 2: apply Quickbundle with a distance threshold of 20
    # Step 3: remove singletons
    # Step 4: assign to closest ROIs pair


    Parameters
    ------------
    streamlines : sequence
        sequence of T streamlines. One streamline is an ndarray of shape (N, 3),
        where N is the number of points in that streamline, and
        ``streamlines[t][n]`` is the n-th point in the t-th streamline. Points
        are of form x, y, z in *voxel* coordinates.
    bundles_masks : sequence
        list of nibabel objects corresponding to mask of bundles
    ROIs : sequence
        list of nibabel objects corresponding to mask of ROIs
    wm : nibabel object
        mask of the white matter
    save_segmented : bool
        if true, returns indices of streamlines composing VC, IC, VCWP and NC

    Returns
    ---------
    scores : dict
        dictionnary containing a score for each metric
    indices : dict
        dictionnary containing the indices of streamlines composing VC, IC,
        VCWP and NC

    """

    # Load all streamlines, since streamlines is a generator.
    full_strl = [s for s in streamlines]

    # Extract VCs and VBs
    VC_indices, found_vbs_info = auto_extract_VCs(full_strl, ref_bundles)
    VC = len(VC_indices)

    if save_VBs or save_full_vc:
        save_valid_connections(found_vbs_info, full_strl, out_segmented_strl_dir,
                               base_out_segmented_strl, ref_anat_fname,
                               save_vbs=save_VBs, save_full_vc=save_full_vc)


    logging.debug("Starting IC, IB scoring")

    total_strl_count = len(full_strl)
    candidate_ic_strl_indices = sorted(set(range(total_strl_count)) - VC_indices)

    candidate_ic_streamlines = []
    rejected_streamlines = []

    # Chosen from GT dataset
    length_thres = 35.

    # Filter streamlines that are too short, consider them as NC
    for idx in candidate_ic_strl_indices:
        if slength(full_strl[idx]) >= length_thres:
            candidate_ic_streamlines.append(full_strl[idx].astype('f4'))
        else:
            rejected_streamlines.append(full_strl[idx].astype('f4'))

    logging.debug('Found {} candidate IC'.format(len(candidate_ic_streamlines)))
    logging.debug('Found {} streamlines that were too short'.format(len(rejected_streamlines)))

    ic_counts = 0
    nb_ib = 0

    if len(candidate_ic_streamlines):
        additional_rejected, ic_counts, nb_ib = group_and_assign_ibs(
                                                   candidate_ic_streamlines,
                                                   ROIs, save_IBs, save_full_ic,
                                                   out_segmented_strl_dir,
                                                   base_out_segmented_strl,
                                                   ref_anat_fname)

        rejected_streamlines.extend(additional_rejected)

    # TODO add argument
    if len(rejected_streamlines) > 0:
        out_nc_fname = os.path.join(out_segmented_strl_dir,
                                    '{}_NC.tck'.format(base_out_segmented_strl))
        out_file = TCK.create(out_nc_fname)
        save_tracts_tck_from_dipy_voxel_space(out_file, ref_anat_fname,
                                              rejected_streamlines)

    if ic_counts != len(candidate_ic_strl_indices) - len(rejected_streamlines):
        raise ValueError("Some streamlines were not correctly assigned to NC")

    VC /= total_strl_count
    IC = (len(candidate_ic_strl_indices) - len(rejected_streamlines)) / total_strl_count
    NC = len(rejected_streamlines) / total_strl_count
    VCWP = 0

    nb_VB_found = [v['nb_streamlines'] > 0 for k, v in found_vbs_info.iteritems()].count(True)
    streamlines_per_bundle = {k: v['nb_streamlines'] for k, v in found_vbs_info.iteritems() if v['nb_streamlines'] > 0}

    scores = {}
    scores['version'] = 2
    scores['algo_version'] = 5
    scores['VC'] = VC
    scores['IC'] = IC
    scores['VCWP'] = VCWP
    scores['NC'] = NC
    scores['VB'] = nb_VB_found
    scores['IB'] = nb_ib
    scores['streamlines_per_bundle'] = streamlines_per_bundle
    scores['total_streamlines_count'] = total_strl_count

    # Get bundle overlap, overreach and f1-score for each bundle.
    scores['overlap_per_bundle'] = {k: v["overlap"] for k, v in found_vbs_info.items()}
    scores['overreach_per_bundle'] = {k: v["overreach"] for k, v in found_vbs_info.items()}
    scores['overreach_norm_gt_per_bundle'] = {k: v["overreach_norm"] for k, v in found_vbs_info.items()}
    scores['f1_score_per_bundle'] = {k: v["f1_score"] for k, v in found_vbs_info.items()}

    # Compute average bundle overlap, overreach and f1-score.
    scores['mean_OL'] = np.mean(list(scores['overlap_per_bundle'].values()))
    scores['mean_OR'] = np.mean(list(scores['overreach_per_bundle'].values()))
    scores['mean_ORn'] = np.mean(list(scores['overreach_norm_gt_per_bundle'].values()))
    scores['mean_F1'] = np.mean(list(scores['f1_score_per_bundle'].values()))

    return scores
