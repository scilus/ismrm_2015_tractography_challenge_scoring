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

from challenge_scoring import NB_POINTS_RESAMPLE
from challenge_scoring.io.streamlines import get_tracts_voxel_space_for_dipy, \
    save_tracts_tck_from_dipy_voxel_space, \
    save_valid_connections
from challenge_scoring.metrics.invalid_connections import group_and_assign_ibs
from challenge_scoring.metrics.valid_connections import auto_extract_VCs


def _prepare_gt_bundles_info(bundles_dir, bundles_masks_dir,
                             gt_bundles_attribs, ref_anat_fname):
    # Ref bundles will contain {'name': 'name_of_the_bundle',
    #                           'threshold': thres_value,
    #                           'streamlines': list_of_streamlines}

    dummy_attribs = {'orientation': 'LPS'}
    qb = QuickBundles(20, metric=AveragePointwiseEuclideanMetric())

    ref_bundles = []

    for bundle_idx, bundle_f in enumerate(sorted(os.listdir(bundles_dir))):
        bundle_name = os.path.splitext(os.path.basename(bundle_f))[0]

        bundle_attribs = gt_bundles_attribs.get(os.path.basename(bundle_f))
        if bundle_attribs is None:
            raise ValueError(
                "Missing basic bundle attribs for {0}".format(bundle_f))

        # Already resample to avoid doing it for each iteration of chunking
        orig_strl = [s for s in get_tracts_voxel_space_for_dipy(
            os.path.join(bundles_dir, bundle_f),
            ref_anat_fname, dummy_attribs)]

        resamp_bundle = set_number_of_points(orig_strl, NB_POINTS_RESAMPLE)
        resamp_bundle = [s.astype('f4') for s in resamp_bundle]

        bundle_cluster_map = qb.cluster(resamp_bundle)
        bundle_cluster_map.refdata = resamp_bundle

        bundle_mask = nib.load(os.path.join(bundles_masks_dir,
                                            bundle_name + '.nii.gz'))

        ref_bundles.append({'name': bundle_name,
                            'threshold': bundle_attribs['cluster_threshold'],
                            'cluster_map': bundle_cluster_map,
                            'mask': bundle_mask})

    return ref_bundles


def score_submission(streamlines_fname,
                     tracts_attribs,
                     base_data_dir,
                     basic_bundles_attribs,
                     save_full_vc=False,
                     save_full_ic=False,
                     save_full_nc=False,
                     save_IBs=False,
                     save_VBs=False,
                     segmented_out_dir='',
                     segmented_base_name='',
                     verbose=False):
    """
    Score a submission, using the following algorithm:
        1: extract all streamlines that are valid, which are classified as
           Valid Connections (VC) making up Valid Bundles (VB).
        2: remove streamlines shorter than an threshold based on the GT dataset
        3: cluster the remaining streamlines
        4: remove singletons
        5: assign each cluster to the closest ROIs pair. Those make up the
           Invalid Connections (IC), grouped as Invalid Bundles (IB).
        6: streamlines that are neither in VC nor IC are classified as
           No Connection (NC).


    Parameters
    ------------
    streamlines_fname : string
        path to the file containing the streamlines.
    tracts_attribs : dictionary
        contains the attributes of the submission. Must contain the
        'orientation' attribute for .vtk files.
    base_data_dir : string
        path to the direction containing the scoring data.
    basic_bundles_attribs : dictionary
        contains the attributes of the basic bundles
        (name, list of streamlines, segmentation threshold)
    save_full_vc : bool
        indicates if the full set of VC will be saved in an individual file.
    save_full_ic : bool
        indicates if the full set of IC will be saved in an individual file.
    save_full_nc : bool
        indicates if the full set of NC will be saved in an individual file.
    save_IBs : bool
        indicates if the invalid bundles will be saved in individual file for
        each IB.
    save_VBs : bool
        indicates if the valid bundles will be saved in individual file for
        each VB.
    segmented_out_dir : string
        the path to the directory where segmented files will be saved.
    segmented_base_name : string
        the base name to use for saving segmented files.
    verbose : bool
        indicates if the algorithm needs to be verbose when logging messages.

    Returns
    ---------
    scores : dict
        dictionnary containing a score for each metric
    """

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Prepare needed scoring data
    logging.debug('Preparing GT data')
    masks_dir = os.path.join(base_data_dir, "masks")
    rois_dir = os.path.join(masks_dir, "rois")
    bundles_dir = os.path.join(base_data_dir, "bundles")
    bundles_masks_dir = os.path.join(masks_dir, "bundles")
    ref_anat_fname = os.path.join(masks_dir, "wm.nii.gz")

    ROIs = [nib.load(os.path.join(rois_dir, f))
            for f in sorted(os.listdir(rois_dir))]

    ref_bundles = _prepare_gt_bundles_info(bundles_dir,
                                           bundles_masks_dir,
                                           basic_bundles_attribs,
                                           ref_anat_fname)

    streamlines_gen = get_tracts_voxel_space_for_dipy(streamlines_fname,
                                                      ref_anat_fname,
                                                      tracts_attribs)

    # Load all streamlines, since streamlines is a generator.
    full_strl = [s for s in streamlines_gen]

    # Extract VCs and VBs
    VC_indices, found_vbs_info = auto_extract_VCs(full_strl, ref_bundles)
    VC = len(VC_indices)

    if save_VBs or save_full_vc:
        save_valid_connections(found_vbs_info, full_strl, segmented_out_dir,
                               segmented_base_name, ref_anat_fname,
                               save_vbs=save_VBs, save_full_vc=save_full_vc)

    logging.debug("Starting IC, IB scoring")

    total_strl_count = len(full_strl)
    candidate_ic_strl_indices = sorted(
        set(range(total_strl_count)) - VC_indices)

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

    logging.debug('Found {} candidate IC'.format(
        len(candidate_ic_streamlines)))
    logging.debug('Found {} streamlines that were too short'.format(
        len(rejected_streamlines)))

    ic_counts = 0
    nb_ib = 0

    if len(candidate_ic_streamlines):
        additional_rejected, ic_counts, nb_ib = group_and_assign_ibs(
            candidate_ic_streamlines,
            ROIs, save_IBs, save_full_ic,
            segmented_out_dir,
            segmented_base_name,
            ref_anat_fname)

        rejected_streamlines.extend(additional_rejected)

    if ic_counts != len(candidate_ic_strl_indices) - len(rejected_streamlines):
        raise ValueError("Some streamlines were not correctly assigned to NC")

    if len(rejected_streamlines) > 0 and save_full_nc:
        out_nc_fname = os.path.join(segmented_out_dir,
                                    '{}_NC.tck'.format(segmented_base_name))
        save_tracts_tck_from_dipy_voxel_space(out_nc_fname, ref_anat_fname,
                                              rejected_streamlines)

    VC /= total_strl_count
    IC = (len(candidate_ic_strl_indices) -
          len(rejected_streamlines)) / total_strl_count
    NC = len(rejected_streamlines) / total_strl_count
    VCWP = 0

    nb_VB_found = [v['nb_streamlines'] > 0 for k,
                   v in found_vbs_info.items()].count(True)
    streamlines_per_bundle = {
        k: v['nb_streamlines']
        for k, v in found_vbs_info.items() if v['nb_streamlines'] > 0}

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
    scores['overlap_per_bundle'] = {k: v["overlap"]
                                    for k, v in found_vbs_info.items()}
    scores['overreach_per_bundle'] = {k: v["overreach"]
                                      for k, v in found_vbs_info.items()}
    scores['overreach_norm_gt_per_bundle'] = {
        k: v["overreach_norm"] for k, v in found_vbs_info.items()}
    scores['f1_score_per_bundle'] = {k: v["f1_score"]
                                     for k, v in found_vbs_info.items()}

    # Compute average bundle overlap, overreach and f1-score.
    scores['mean_OL'] = np.mean(list(scores['overlap_per_bundle'].values()))
    scores['mean_OR'] = np.mean(list(scores['overreach_per_bundle'].values()))
    scores['mean_ORn'] = np.mean(
        list(scores['overreach_norm_gt_per_bundle'].values()))
    scores['mean_F1'] = np.mean(list(scores['f1_score_per_bundle'].values()))

    return scores
