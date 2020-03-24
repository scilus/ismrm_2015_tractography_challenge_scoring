#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import (
    Origin, Space, StatefulTractogram)

from nibabel.streamlines import detect_format
from nibabel.streamlines.tck import TckFile as TCK


def guess_orientation(tract_fname):
    tracts_format = detect_format(tract_fname)
    tracts_file = tracts_format(tract_fname)

    if isinstance(tracts_file, TCK):
        return 'RAS'

    return 'Unknown'


def _get_tracts_over_grid(
    tract_fname,
    ref_anat_fname,
    tract_attributes,
    origin=Origin.NIFTI
):
    sft = load_tractogram(
        tract_fname, ref_anat_fname, to_space=Space.VOX, to_origin=origin,
        bbox_valid_check=False)
    return sft.streamlines


def get_tracts_voxel_space_for_dipy(
    tract_fname, ref_anat_fname, tract_attributes,
):
    return _get_tracts_over_grid(
        tract_fname, ref_anat_fname, tract_attributes)


def save_tracts_tck_from_dipy_voxel_space(
    tract_fname, ref_anat_fname, tracts
):
    sft = StatefulTractogram(tracts, ref_anat_fname, Space.VOX)
    save_tractogram(sft, tract_fname, bbox_valid_check=False)


def save_valid_connections(extracted_vb_info, streamlines,
                           segmented_out_dir, basename, ref_anat_fname,
                           save_vbs=False, save_full_vc=False):

    if not save_vbs and not save_full_vc:
        return

    full_vcs = []
    for bundle_name, bundle_info in extracted_vb_info.items():
        if bundle_info['nb_streamlines'] > 0:
            out_fname = os.path.join(segmented_out_dir, basename +
                                     '_VB_{0}.tck'.format(bundle_name))

            vc_strl = [streamlines[idx]
                       for idx in bundle_info['streamlines_indices']]

            if save_full_vc:
                full_vcs.extend(vc_strl)

            if save_vbs:
                save_tracts_tck_from_dipy_voxel_space(
                    out_fname, ref_anat_fname, vc_strl)

    if save_full_vc and len(full_vcs):
        out_name = os.path.join(segmented_out_dir, basename + '_VC.tck')
        save_tracts_tck_from_dipy_voxel_space(out_name,
                                              ref_anat_fname,
                                              full_vcs)


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

    for k, v in ib_info.items():
        out_strl = []
        for c_idx in v:
            out_strl.extend([s for s in np.array(streamlines)[
                ic_clusters[c_idx].indices]])

        if save_ibs:
            out_fname = os.path.join(out_segmented_dir,
                                     base_name +
                                     '_IB_{0}_{1}.tck'.format(k[0], k[1]))

            save_tracts_tck_from_dipy_voxel_space(out_fname, ref_anat_fname,
                                                  out_strl)

        if save_full_ic:
            full_ic.extend(out_strl)

    if save_full_ic and len(full_ic):
        out_name = os.path.join(out_segmented_dir, base_name + '_IC.tck')

        save_tracts_tck_from_dipy_voxel_space(out_name,
                                              ref_anat_fname,
                                              full_ic)
