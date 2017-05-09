#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

from collections import Counter
import logging
import os
from time import time

import numpy as np
from scipy.spatial.distance import cdist


def find_closest_distance_points_to_region(points, roi_volume):
    # TODO add validations
    # TODO commented because we moved it higher to avoid doing it all the time
    #roi_coords = np.array(np.where(roi_volume)).T
    roi_coords = roi_volume
    dists = cdist(points, roi_coords, 'euclidean')

    return np.min(dists, axis=1)


def find_closest_region(points, rois):
    # points is a 2D array, each row being 1 point. We expect 2 rows (start and end points)
    # rois is a list of (region name, region_data)

    # TODO use type max
    min_global_dists = [100000, 100000]
    closest_region_names = ["", ""]

    for region_name, region_data in rois:
        closest_dists = find_closest_distance_points_to_region(points, region_data)

        if closest_dists[0] < min_global_dists[0]:
            min_global_dists[0] = closest_dists[0]
            closest_region_names[0] = region_name
        if closest_dists[1] < min_global_dists[1]:
            min_global_dists[1] = closest_dists[1]
            closest_region_names[1] = region_name

    return (closest_region_names, min_global_dists)


def get_closest_roi_pairs_for_bundle(streamlines, rois):
    start_point = np.reshape(streamlines[0][0], (-1, 3)) # Needs to be 2D for cdist

    closest_rois_pairs = []

    for s in streamlines:
        endpoints = np.vstack([s[0], s[-1]])
        endpoints_dists = cdist(start_point, endpoints).flatten()

        # Make sure we all start from the same "orientation" for streamlines,
        # to try to get the same region as the first region
        if endpoints_dists[0] > endpoints_dists[1]:
            endpoints = np.vstack([s[-1], s[0]])

        closest_region_names, min_dists = find_closest_region(endpoints, rois)
        closest_rois_pairs.append(tuple(closest_region_names))

        # TODO we could use distance to validate
        # TODO we could also have some condition for regions that overlap, and keep all of those

    occurences = Counter(closest_rois_pairs)

    # TODO handle either an equality or maybe a range
    return occurences.most_common(1)[0][0]


def get_closest_roi_pairs_for_all_streamlines(streamlines, rois):
    start_point = np.reshape(streamlines[0][0], (-1, 3)) # Needs to be 2D for cdist

    closest_rois_pairs = []

    for s in streamlines:
        endpoints = np.vstack([s[0], s[-1]])
        endpoints_dists = cdist(start_point, endpoints).flatten()

        # Make sure we all start from the same "orientation" for streamlines,
        # to try to get the same region as the first region
        if endpoints_dists[0] > endpoints_dists[1]:
            endpoints = np.vstack([s[-1], s[0]])

        closest_region_names, min_dists = find_closest_region(endpoints, rois)
        closest_rois_pairs.append(tuple(closest_region_names))

        # TODO we could use distance to validate
        # TODO we could also have some condition for regions that overlap, and keep all of those

    return closest_rois_pairs
