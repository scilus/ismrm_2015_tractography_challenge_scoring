#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

from collections import Counter
import logging
import os
import random

import nibabel as nib
import numpy as np

from dipy.tracking.streamline import set_number_of_points
from dipy.segment.clustering import QuickBundles
import dipy.segment.quickbundles as qb
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.metrics import length as slength

from tractconverter.formats.tck import TCK
# TODO remove
from dipy.tracking.vox2track import track_counts as streamlines_count

# TODO check names
from challenge_scoring import NB_POINTS_RESAMPLE
from challenge_scoring.io.streamlines import get_tracts_voxel_space_for_dipy, \
                                       save_tracts_tck_from_dipy_voxel_space
from challenge_scoring.metrics.invalid_connections import get_closest_roi_pairs_for_all_streamlines
from challenge_scoring.metrics.valid_connections import auto_extract_VCs
from challenge_scoring.utils.filenames import get_root_image_name

# TODO remove
#import filter_streamlines


def _save_independant_IB(roi1, roi2, strl_chunk, chunk_it, chunk_size,
                         IC_idx, segmented_out_dir, segmented_base_name,
                         ref_anat_fname):
    roi1_basename = os.path.basename(roi1.get_filename()).replace('.nii.gz', '')
    roi2_basename = os.path.basename(roi2.get_filename()).replace('.nii.gz', '')
    out_fname = os.path.join(segmented_out_dir, segmented_base_name + '_IB_{0}_{1}.tck'.format(roi1_basename, roi2_basename))

    if not os.path.isfile(out_fname):
        ib_f = TCK.create(out_fname)
    else:
        ib_f = TCK(out_fname)

    ic_list = [v - (chunk_it * chunk_size) for v in IC_idx]
    ic_strl = [strl_chunk[idx] for idx in ic_list]

    save_tracts_tck_from_dipy_voxel_space(ib_f, ref_anat_fname, ic_strl)


def _save_independent_VB(bundle_name, strl_chunk,
                         VC_idx, segmented_out_dir, segmented_base_name,
                         ref_anat_fname):
    out_fname = os.path.join(segmented_out_dir, segmented_base_name +
                             '_VB_{0}.tck'.format(bundle_name))

    if not os.path.isfile(out_fname):
        vb_f = TCK.create(out_fname)
    else:
        vb_f = TCK(out_fname)

    # In this case, indices are already realted to chunk
    vc_strl = [strl_chunk[idx] for idx in VC_idx]

    save_tracts_tck_from_dipy_voxel_space(vb_f, ref_anat_fname, vc_strl)


def _save_extracted_VBs(extracted_vb_info, streamlines,
                        segmented_out_dir, basename, ref_anat_fname):

    for bundle_name, bundle_info in extracted_vb_info.iteritems():
        if bundle_info['nb_streamlines'] > 0:
            out_fname = os.path.join(segmented_out_dir, basename +
                                     '_VB_{0}.tck'.format(bundle_name))

            vb_f = TCK.create(out_fname)

            vc_strl = [streamlines[idx] for idx in bundle_info['streamlines_indices']]

            save_tracts_tck_from_dipy_voxel_space(vb_f, ref_anat_fname, vc_strl)


def score_from_files(filename, masks_dir, bundles_dir,
                     tracts_attribs, basic_bundles_attribs,
                     save_segmented=False, save_IBs=False,
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
                      save_segmented=save_segmented, save_IBs=save_IBs,
                      save_VBs=save_VBs,
                      out_segmented_strl_dir=segmented_out_dir,
                      base_out_segmented_strl=segmented_base_name,
                      ref_anat_fname=wm_file)





def score_auto_extract_auto_IBs(streamlines, bundles_masks, ref_bundles, ROIs, wm,
                                save_segmented=False, save_IBs=False,
                                save_VBs=False,
                                out_segmented_strl_dir='',
                                base_out_segmented_strl='',
                                ref_anat_fname=''):
    """
    TODO document


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

    VC_indices, found_vbs_info = auto_extract_VCs(full_strl, ref_bundles)
    VC = len(VC_indices)

    if save_VBs:
        _save_extracted_VBs(found_vbs_info, full_strl, out_segmented_strl_dir,
                            base_out_segmented_strl, ref_anat_fname)

    # TODO might be readded
    # To keep track of streamlines that have been classified
    # classified_streamlines_indices = VC_indices

    # New algorithm
    # Step 1: remove streamlines shorter than threshold (currently 35)
    # Step 2: apply Quickbundle with a distance threshold of 20
    # Step 3: remove singletons
    # Step 4: assign to closest ROIs pair
    logging.debug("Starting IC, IB scoring")

    total_strl_count = len(full_strl)
    candidate_ic_strl_indices = sorted(set(range(total_strl_count)) - VC_indices)

    length_thres = 35.

    candidate_ic_streamlines = []
    rejected_streamlines = []

    for idx in candidate_ic_strl_indices:
        if slength(full_strl[idx]) >= length_thres:
            candidate_ic_streamlines.append(full_strl[idx].astype('f4'))
        else:
            rejected_streamlines.append(full_strl[idx].astype('f4'))

    logging.debug('Found {} candidate IC'.format(len(candidate_ic_streamlines)))
    logging.debug('Found {} streamlines that were too short'.format(len(rejected_streamlines)))

    ic_counts = 0
    ib_pairs = {}

    if len(candidate_ic_streamlines):

        # Fix seed to always generate the same output
        # Shuffle to try to reduce the ordering dependency for QB
        random.seed(0.2)
        random.shuffle(candidate_ic_streamlines)


        # TODO threshold on distance as arg
        out_data = qb.QuickBundles(candidate_ic_streamlines,
                                   dist_thr=20.,
                                   pts=12)
        clusters = out_data.clusters()

        logging.debug("Found {} potential IB clusters".format(len(clusters)))

        # TODO this should be better handled
        rois_info = []
        for roi in ROIs:
            rois_info.append((get_root_image_name(os.path.basename(roi.get_filename())),
                              np.array(np.where(roi.get_data())).T))

        all_ics_closest_pairs = get_closest_roi_pairs_for_all_streamlines(candidate_ic_streamlines, rois_info)

        for c_idx, c in enumerate(clusters):
            closest_for_cluster = [all_ics_closest_pairs[i] for i in clusters[c]['indices']]

            if len(clusters[c]['indices']) > 1:
                ic_counts += len(clusters[c]['indices'])
                occurences = Counter(closest_for_cluster)

                # TODO handle either an equality or maybe a range
                most_frequent = occurences.most_common(1)[0][0]

                val = ib_pairs.get(most_frequent)
                if val is None:
                    # Check if flipped pair exists
                    val1 = ib_pairs.get((most_frequent[1], most_frequent[0]))
                    if val1 is not None:
                        val1.append(c_idx)
                    else:
                        ib_pairs[most_frequent] = [c_idx]
                else:
                    val.append(c_idx)
            else:
                rejected_streamlines.append(candidate_ic_streamlines[clusters[c]['indices'][0]])

        if save_segmented and save_IBs:
            for k, v in ib_pairs.iteritems():
                out_strl = []
                for c_idx in v:
                    out_strl.extend([s for s in np.array(candidate_ic_streamlines)[clusters[c_idx]['indices']]])

                out_fname = os.path.join(out_segmented_strl_dir,
                                         base_out_segmented_strl + \
                                         '_IB_{0}_{1}.tck'.format(k[0], k[1]))

                ib_f = TCK.create(out_fname)
                save_tracts_tck_from_dipy_voxel_space(ib_f, ref_anat_fname,
                                                      out_strl)

    if len(rejected_streamlines) > 0 and save_segmented:
        out_nc_fname = os.path.join(out_segmented_strl_dir,
                                    '{}_NC.tck'.format(base_out_segmented_strl))
        out_file = TCK.create(out_nc_fname)
        save_tracts_tck_from_dipy_voxel_space(out_file, ref_anat_fname,
                                              rejected_streamlines)

    # TODO readd classifed_steamlines_indices to validate
    if ic_counts != len(candidate_ic_strl_indices) - len(rejected_streamlines):
        raise ValueError("Some streamlines were not correctly assigned to NC")

    VC /= total_strl_count
    IC = (len(candidate_ic_strl_indices) - len(rejected_streamlines)) / total_strl_count
    NC = len(rejected_streamlines) / total_strl_count
    VCWP = 0

    # TODO could have sanity check on global extracted streamlines vs all
    # possible indices

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
    scores['IB'] = len(ib_pairs.keys())
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
