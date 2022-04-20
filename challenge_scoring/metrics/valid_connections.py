#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import logging
from itertools import chain

from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.distances import bundles_distances_mdf
from dipy.tracking.streamline import set_number_of_points
from nibabel.streamlines import Tractogram
import numpy as np

from challenge_scoring import NB_POINTS_RESAMPLE
from challenge_scoring.metrics.bundle_coverage import compute_bundle_coverage_scores

# todo. See if this could be bigger nowadays.
CHUNK_SIZE = 5000


def auto_extract(model_cluster_map, submission_cluster_map,
                 number_pts_per_str=NB_POINTS_RESAMPLE,
                 close_centroids_thr=20,
                 clean_thr=7.):
    """
    Classifies streamlines to closest ground truth bundle. Submission
    streamlines are already clustered with quickbundles.

    Parameters
    ----------
    model_cluster_map:
        The gt cluster's map for a given bundle.
    submission_cluster_map:
        Result of the qb clustering.
    number_pts_per_str: int
        Number of points for each streamline.
    close_centroids_thr: int
        Threshold for this bundle. Each cluster from the submission is compared
        to the ground truth's centroid(s) and kept if MDF is smaller than
        close_centroids_thr.
    clean_thr: float
        Threshold for this bundle. ?

    Returns
    -------

    """

    model_centroids = model_cluster_map.centroids

    centroid_matrix = bundles_distances_mdf(model_centroids,
                                            submission_cluster_map.centroids)

    centroid_matrix[centroid_matrix > close_centroids_thr] = np.inf
    mins = np.min(centroid_matrix, axis=0)

    close_clusters_ind = np.where(mins != np.inf)[0]
    close_clusters = [submission_cluster_map[i] for i in close_clusters_ind]
    close_indices_inter = [submission_cluster_map[i].indices
                           for i in close_clusters_ind]
    close_indices = list(chain.from_iterable(close_indices_inter))

    close_streamlines = list(chain(*close_clusters))
    closer_streamlines = close_streamlines

    clean_matrix = bundles_distances_mdf(model_cluster_map.refdata,
                                         closer_streamlines)

    clean_matrix[clean_matrix > clean_thr] = np.inf

    mins = np.min(clean_matrix, axis=0)

    clean_indices = [i for i in np.where(mins != np.inf)[0]]

    # Clean indices refer to the streamlines in closer_streamlines,
    # which are the same as the close_streamlines. Each close_streamline
    # has a related element in close_indices, for which the value
    # is the index of the original streamline in the moved_streamlines.
    final_selected_indices = [close_indices[idx] for idx in clean_indices]

    return final_selected_indices


def auto_extract_VCs(streamlines, ref_bundles):
    """
    Extract valid bundles and associated valid streamline indices.

    Parameters
    ----------
    streamlines: list
        List of all streamlines
    ref_bundles: list[dict]
        List of dict with 'name', 'threshold' and 'streamlines' for each
        bundle.

    Returns
    -------
    VC_idx
    found_vbs_info: dict
        Dict with bundle names as keys and, for each, a sub-dict with keys
        'nb_streamlines' and 'streamlines_indices'.
    """
    # Streamlines =

    VC = 0
    VC_idx = set()

    found_vbs_info = {}
    for bundle in ref_bundles:
        found_vbs_info[bundle['name']] = {'nb_streamlines': 0,
                                          'streamlines_indices': set()}

    # Need to bookkeep because we chunk for big datasets
    processed_strl_count = 0
    chunk_it = 0

    nb_bundles = len(ref_bundles)
    bundles_found = [False] * nb_bundles

    logging.debug("Starting scoring VCs")

    qb = QuickBundles(threshold=20, metric=AveragePointwiseEuclideanMetric())

    # Start loop here for big datasets
    while processed_strl_count < len(streamlines):
        logging.debug("Starting chunk: {0}".format(chunk_it))

        strl_chunk = streamlines[chunk_it * CHUNK_SIZE:
                                 (chunk_it + 1) * CHUNK_SIZE]

        processed_strl_count += len(strl_chunk)
        cur_chunk_VC_idx, cur_chunk_IC_idx, cur_chunk_VCWP_idx = set(), set(), set()

        # Already resample and run quickbundles on the submission chunk,
        # to avoid doing it at every call of auto_extract
        rstreamlines = set_number_of_points(strl_chunk, NB_POINTS_RESAMPLE)

        # qb.cluster had problem with f8
        rstreamlines = [s.astype('f4') for s in rstreamlines]

        chunk_cluster_map = qb.cluster(rstreamlines)
        chunk_cluster_map.refdata = strl_chunk

        logging.debug("Starting VC identification through auto_extract")

        for bundle_idx, ref_bundle in enumerate(ref_bundles):
            # The selected indices are from [0, len(strl_chunk)]
            selected_streamlines_indices = auto_extract(
                ref_bundle['cluster_map'], chunk_cluster_map,
                clean_thr=ref_bundle['threshold'])

            # Remove duplicates, when streamlines are assigned to multiple VBs.
            selected_streamlines_indices = set(selected_streamlines_indices) - \
                                           cur_chunk_VC_idx
            cur_chunk_VC_idx |= selected_streamlines_indices

            nb_selected_streamlines = len(selected_streamlines_indices)

            if nb_selected_streamlines:
                bundles_found[bundle_idx] = True
                VC += nb_selected_streamlines

                # Shift indices to match the real number of streamlines
                global_select_strl_indices = set([v + chunk_it * CHUNK_SIZE
                                                  for v in selected_streamlines_indices])
                vb_info = found_vbs_info.get(ref_bundle['name'])
                vb_info['nb_streamlines'] += nb_selected_streamlines
                vb_info['streamlines_indices'] |= global_select_strl_indices

                VC_idx |= global_select_strl_indices
            else:
                global_select_strl_indices = set()

        chunk_it += 1

    # Compute bundle overlap, overreach and f1_scores and update found_vbs_info
    for bundle_idx, ref_bundle in enumerate(ref_bundles):
        bundle_name = ref_bundle["name"]
        bundle_mask = ref_bundle["mask"]

        vb_info = found_vbs_info[bundle_name]

        # Streamlines are in voxel space since that's how they were
        # loaded in the scoring function.
        tractogram = Tractogram(streamlines=(streamlines[i] for i in vb_info['streamlines_indices']),
                                affine_to_rasmm=bundle_mask.affine)

        scores = {}
        if len(tractogram) > 0:
            scores = compute_bundle_coverage_scores(tractogram, bundle_mask)

        vb_info['overlap'] = scores.get("OL", 0)
        vb_info['overreach'] = scores.get("OR", 0)
        vb_info['overreach_norm'] = scores.get("ORn", 0)
        vb_info['f1_score'] = scores.get("F1", 0)

    return VC_idx, found_vbs_info
