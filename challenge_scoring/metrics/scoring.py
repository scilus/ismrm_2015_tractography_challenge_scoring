#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import logging
import os

import nibabel as nib
import numpy as np

from dipy.io.stateful_tractogram import set_sft_logger_level
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.streamline import set_number_of_points
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.metrics import length as slength

from challenge_scoring import NB_POINTS_RESAMPLE
from challenge_scoring.io.streamlines import save_valid_connections
from challenge_scoring.metrics.invalid_connections import group_and_assign_ibs
from challenge_scoring.metrics.valid_connections import auto_extract_VCs


def _prepare_gt_bundles_info(bundles_dir, bundles_masks_dir,
                             gt_bundles_attribs, ref_anat_fname):
    """
    Returns
    -------
    ref_bundles: list[dict]
        Each dict will contain {'name': 'name_of_the_bundle',
                                'threshold': thres_value,
                                'streamlines': list_of_streamlines},
                                'cluster_map': the qb cluster map,
                                'mask': the loaded bundle mask (nifti).}
    """
    qb = QuickBundles(20, metric=AveragePointwiseEuclideanMetric())

    ref_bundles = []

    for bundle_idx, bundle_f in enumerate(sorted(os.listdir(bundles_dir))):
        bundle_name = os.path.splitext(os.path.basename(bundle_f))[0]

        bundle_attribs = gt_bundles_attribs.get(os.path.basename(bundle_f))
        if bundle_attribs is None:
            raise ValueError(
                "Missing basic bundle attribs for {0}".format(bundle_f))

        orig_sft = load_tractogram(
            os.path.join(bundles_dir, bundle_f), ref_anat_fname,
            bbox_valid_check=False, trk_header_check=False)
        orig_sft.to_vox()
        orig_sft.to_center()

        # Already resample to avoid doing it for each iteration of chunking
        orig_strl = orig_sft.streamlines

        resamp_bundle = set_number_of_points(orig_strl, NB_POINTS_RESAMPLE)
        resamp_bundle = [s.astype(np.float32) for s in resamp_bundle]

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
                     base_data_dir,
                     basic_bundles_attribs,
                     save_full_vc=False,
                     save_full_ic=False,
                     save_full_nc=False,
                     compute_ic_ib=False,
                     save_IBs=False,
                     save_VBs=False,
                     segmented_out_dir='',
                     segmented_base_name='',
                     out_tract_type='tck',
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
    compute_ic_ib:
        segment IC results into IB.
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
    out_tract_type: str
        extension for the output tractograms.
    verbose : bool
        indicates if the algorithm needs to be verbose when logging messages.

    Returns
    ---------
    scores : dict
        dictionnary containing a score for each metric
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        # Silencing SFT's logger if our logging is in DEBUG mode, because it
        # typically produces a lot of outputs!
        set_sft_logger_level('WARNING')

    # Prepare needed scoring data
    logging.info('Preparing GT data: Loading and computing centroids')
    masks_dir = os.path.join(base_data_dir, "masks")
    rois_dir = os.path.join(masks_dir, "rois")
    bundles_dir = os.path.join(base_data_dir, "bundles")
    bundles_masks_dir = os.path.join(masks_dir, "bundles")
    ref_anat_fname = os.path.join(masks_dir, "wm.nii.gz")

    ROIs = [nib.load(os.path.join(rois_dir, f))
            for f in sorted(os.listdir(rois_dir))]

    # Get the dict with 'name', 'threshold', 'streamlines',
    # 'cluster_map' and 'mask' for each bundle.
    ref_bundles = _prepare_gt_bundles_info(bundles_dir,
                                           bundles_masks_dir,
                                           basic_bundles_attribs,
                                           ref_anat_fname)

    logging.info('Loading submission data')
    sft = load_tractogram(streamlines_fname, ref_anat_fname,
                          bbox_valid_check=False, trk_header_check=False)
    sft.to_vox()
    sft.to_center()
    total_strl_count = len(sft.streamlines)

    # Extract VCs and VBs, compute OL, OR, f1 for each.
    logging.info("Starting VC, VB scoring")
    VC_indices, found_vbs_info = auto_extract_VCs(sft, ref_bundles)
    VC = len(VC_indices)

    if save_VBs or save_full_vc:
        save_valid_connections(found_vbs_info, sft, segmented_out_dir,
                               segmented_base_name, out_tract_type,
                               save_vbs=save_VBs,
                               save_full_vc=save_full_vc)

    candidate_ic_strl_indices = np.setdiff1d(range(total_strl_count),
                                             VC_indices)
    if compute_ic_ib:
        logging.info("Starting IC, IB scoring")

        candidate_ic_indices = []
        rejected_indices = []

        # Chosen from GT dataset
        length_thres = 35.

        # Filter streamlines that are too short, consider them as NC
        for idx in candidate_ic_strl_indices:
            if slength(sft.streamlines[idx]) >= length_thres:
                candidate_ic_indices.append(idx)
            else:
                rejected_indices.append(idx)

        logging.debug('Found {} candidate IC'
                      .format(len(candidate_ic_indices)))
        logging.debug('Found {} streamlines that were too short'
                      .format(len(rejected_indices)))

        ic_counts = 0
        nb_ib = 0

        if len(candidate_ic_indices):
            additional_rejected_indices, ic_counts, nb_ib = \
                group_and_assign_ibs(sft, candidate_ic_indices,  ROIs,
                                     save_IBs, save_full_ic, segmented_out_dir,
                                     segmented_base_name, ref_anat_fname,
                                     out_tract_type)

            rejected_indices.extend(additional_rejected_indices)

        if ic_counts != len(candidate_ic_strl_indices) - len(rejected_indices):
            raise ValueError("Some streamlines were not correctly assigned to "
                             "NC")

        IC = len(candidate_ic_strl_indices) - len(rejected_indices)
    else:
        IC = 0
        nb_ib = 0
        rejected_indices = candidate_ic_strl_indices

    if len(rejected_indices) > 0 and save_full_nc:
        out_nc_fname = os.path.join(
            segmented_out_dir,
            '{}_NC.{}'.format(segmented_base_name, out_tract_type))

        save_tractogram(sft[rejected_indices], out_nc_fname, bbox_valid_check=False)

    logging.debug("Preparing summary of results")

    nb_VB_found = [v['nb_streamlines'] > 0 for k,
                   v in found_vbs_info.items()].count(True)
    streamlines_per_bundle = {
        k: v['nb_streamlines']
        for k, v in found_vbs_info.items() if v['nb_streamlines'] > 0}

    # Converting np.float to floats for json dumps
    NC = len(rejected_indices)
    scores = {'version': 2,
              'algo_version': 5,
              'VC_count': VC,
              'VC': VC / total_strl_count,
              'IC_count': IC,
              'IC': IC / total_strl_count,
              'NC_count': NC,
              'NC':  NC / total_strl_count,
              'VB': nb_VB_found,
              'IB': nb_ib,
              'streamlines_per_bundle': streamlines_per_bundle,
              'total_streamlines_count': total_strl_count,
              'overlap_per_bundle': {k: float(v["overlap"])
                                     for k, v in found_vbs_info.items()},
              'overreach_per_bundle': {k: float(v["overreach"])
                                       for k, v in found_vbs_info.items()},
              'overreach_norm_gt_per_bundle': {
                  k: v["overreach_norm"] for k, v in found_vbs_info.items()},
              'f1_score_per_bundle': {k: float(v["f1_score"])
                                      for k, v in found_vbs_info.items()}}

    # Compute average bundle overlap, overreach and f1-score.
    scores['mean_OL'] = float(
        np.mean(list(scores['overlap_per_bundle'].values())))
    scores['mean_OR'] = float(
        np.mean(list(scores['overreach_per_bundle'].values())))
    scores['mean_ORn'] = float(
        np.mean(list(scores['overreach_norm_gt_per_bundle'].values())))
    scores['mean_F1'] = float(
        np.mean(list(scores['f1_score_per_bundle'].values())))

    return scores
