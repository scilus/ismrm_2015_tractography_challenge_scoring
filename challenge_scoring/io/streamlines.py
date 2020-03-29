#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import (
    Origin, Space, StatefulTractogram)


# TODO: Change name
# TODO: Return tractogram
def _get_tracts_over_grid(
    tract_fname,
    ref_anat_fname,
    origin=Origin.NIFTI
):
    sft = load_tractogram(
        tract_fname, ref_anat_fname, to_space=Space.VOX, to_origin=origin,
        bbox_valid_check=False)
    return sft.streamlines


# TODO: Change name
def get_tracts_voxel_space_for_dipy(
    tract_fname, ref_anat_fname,
):
    return _get_tracts_over_grid(
        tract_fname, ref_anat_fname)


# TODO: Change "tck" to tracts
def save_tracts_tck_from_dipy_voxel_space(
    tract_fname, ref_anat_fname, tracts
):
    # TODO: Save streamline data
    sft = StatefulTractogram(tracts, ref_anat_fname, Space.VOX)
    save_tractogram(sft, tract_fname, bbox_valid_check=False)


# TODO: Use tractograms instead of streamlines to keep data
def save_valid_connections(extracted_vb_info, streamlines,
                           segmented_out_dir, basename, ref_anat_fname,
                           save_vbs=False, save_full_vc=False):

    if not save_vbs and not save_full_vc:
        return

    full_vcs = []
    for bundle_name, bundle_info in extracted_vb_info.items():
        # TODO: Make output agnostic for tck/trk
        if bundle_info['nb_streamlines'] > 0:
            out_fname = os.path.join(segmented_out_dir, basename +
                                     '_VB_{0}.tck'.format(bundle_name))

            # TODO: Remove loops if possible
            vc_strl = [streamlines[idx]
                       for idx in bundle_info['streamlines_indices']]

            if save_full_vc:
                full_vcs.extend(vc_strl)

            if save_vbs:
                save_tracts_tck_from_dipy_voxel_space(
                    out_fname, ref_anat_fname, vc_strl)

    if save_full_vc and len(full_vcs):
        # TODO: Make output agnostic for tck/trk
        out_name = os.path.join(segmented_out_dir, basename + '_VC.tck')
        save_tracts_tck_from_dipy_voxel_space(out_name,
                                              ref_anat_fname,
                                              full_vcs)


# TODO: Use tractograms instead of streamlines to keep data
# TODO: Uniformize with `save_valid_connections`
def save_invalid_connections(ib_info, streamlines, ic_clusters,
                             out_segmented_dir, base_name,
                             ref_anat_fname,
                             save_full_ic=False, save_ibs=False):
    # ib_info is a dictionary containing all the pairs of ROIs that were
    # assigned to some IB. The value of each element is a list containing the
    # clusters indices of clusters that were assigned to that ROI pair.
    if not save_full_ic and not save_ibs:
        return

    full_ic = []

    # TODO: Remove loops if possible and use meaningful variable names
    for k, v in ib_info.items():
        out_strl = []
        for c_idx in v:
            out_strl.extend([s for s in np.array(streamlines)[
                ic_clusters[c_idx].indices]])

        if save_ibs:
            # TODO: Make output agnostic for tck/trk
            out_fname = os.path.join(out_segmented_dir,
                                     base_name +
                                     '_IB_{0}_{1}.tck'.format(k[0], k[1]))

            save_tracts_tck_from_dipy_voxel_space(out_fname, ref_anat_fname,
                                                  out_strl)

        if save_full_ic:
            full_ic.extend(out_strl)

    if save_full_ic and len(full_ic):
        # TODO: Make output agnostic for tck/trk
        out_name = os.path.join(out_segmented_dir, base_name + '_IC.tck')

        save_tracts_tck_from_dipy_voxel_space(out_name,
                                              ref_anat_fname,
                                              full_ic)
