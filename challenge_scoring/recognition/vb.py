#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

from itertools import chain

from dipy.tracking.distances import bundles_distances_mdf
from dipy.tracking.streamline import set_number_of_points
import numpy as np

from challenge_scoring import NB_POINTS_RESAMPLE


def auto_extract(model_cluster_map, submission_cluster_map,
                 number_pts_per_str=NB_POINTS_RESAMPLE,
                 close_centroids_thr=20,
                 clean_thr=7.,
                 disp=False, verbose=False):

    model_centroids = model_cluster_map.centroids

    # TODO Logging
    # print('# Find centroids which are close to the model_centroids')

    centroid_matrix = bundles_distances_mdf(model_centroids,
                                            submission_cluster_map.centroids)

    centroid_matrix[centroid_matrix > close_centroids_thr] = np.inf
    mins = np.min(centroid_matrix, axis=0)
    close_clusters = [submission_cluster_map[i]
                      for i in np.where(mins != np.inf)[0]]
    close_indices_inter = [submission_cluster_map[i].indices
                           for i in np.where(mins != np.inf)[0]]
    close_indices = list(chain.from_iterable(close_indices_inter))

    close_streamlines = list(chain(*close_clusters))
    closer_streamlines = close_streamlines

    # TODO logging
    # print('# Remove streamlines which are a bit far')

    rcloser_streamlines = set_number_of_points(closer_streamlines,
                                               number_pts_per_str)

    clean_matrix = bundles_distances_mdf(model_cluster_map.refdata,
                                         rcloser_streamlines)

    clean_matrix[clean_matrix > clean_thr] = np.inf

    mins = np.min(clean_matrix, axis=0)

    clean_indices = [i for i in np.where(mins != np.inf)[0]]

    # Clean indices refer to the streamlines in closer_streamlines,
    # which are the same as the close_streamlines. Each close_streamline
    # has a related element in close_indices, for which the value
    # is the index of the original streamline in the moved_streamlines.
    final_selected_indices = [close_indices[idx] for idx in clean_indices]

    return final_selected_indices
