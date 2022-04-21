#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from dipy.io.streamline import save_tractogram


def save_valid_connections(extracted_vb_info, sft,
                           segmented_out_dir, basename,
                           out_tract_type, save_vbs=False, save_full_vc=False):
    """
    Loops on extracted VB and saves them.

    Parameters
    ----------
    extracted_vb_info: dict
        Keys are the bundle names, values are a dict with 'nb_streamlines' and
        'streamlines_indices'.
    sft: StatefulTractogram
        Total submission sft.
    segmented_out_dir: str
        Directory.
    basename: str
        Prefix
    out_tract_type: str
        Extension (.trk or .tck)
    save_vbs: bool
    save_full_vc: bool
    """

    if not save_vbs and not save_full_vc:
        return

    full_vcs_idx = []

    for bundle_name, bundle_info in extracted_vb_info.items():
        if bundle_info['nb_streamlines'] > 0:
            out_fname = os.path.join(segmented_out_dir, basename +
                                     '_VB_{}.{}'.format(
                                         bundle_name, out_tract_type))

            idx = list(bundle_info['streamlines_indices'])

            if save_full_vc:
                full_vcs_idx.extend(idx)

            if save_vbs:
                sub_sft = sft[idx]
                save_tractogram(sub_sft, out_fname)

    if save_full_vc and len(full_vcs_idx):
        out_name = os.path.join(
            segmented_out_dir, basename + '_VC.{}'.format(out_tract_type))
        sub_sft = sft[full_vcs_idx]
        save_tractogram(sub_sft, out_name)


def save_invalid_connections(ib_info, id_invalids, sft, ic_clusters,
                             out_segmented_dir, base_name, out_tract_type,
                             save_full_ic=False, save_ibs=False):
    """
    Loops on extracted IB and saves them.

    Parameters
    ----------
    ib_info: dict
        Dictionary containing all the pairs of ROIs that were assigned to some
        IB. The value of each element is a list containing the clusters indices
        of clusters that were assigned to that ROI pair.
    id_invalids: list
        List of rejected streamlines.
    sft: StatefulTractogram
        Total submission sft
    ic_clusters: list
        List of indices for each IB.
    out_segmented_dir: str
        Path.
    base_name: str
        Prefix
    out_tract_type:
        Extension (.tck or .trk)
    save_full_ic: bool
    save_ibs: bool
    """

    # ib_info is a
    if not save_full_ic and not save_ibs:
        return

    full_ic_idx = []
    for k, v in ib_info.items():
        idx = []
        for c_idx in v:
            idx.extend(ic_clusters[c_idx].indices)

        # Get idx from invalid streamline ids
        invalid_idx = [id_invalids[i] for i in idx]

        # Get actual invalid streamlines
        ib_sft = sft[invalid_idx]

        if save_ibs:
            out_fname = os.path.join(out_segmented_dir,
                                     base_name +
                                     '_IB_{0}_{1}.{2}'.format(
                                         k[0], k[1], out_tract_type))

            save_tractogram(ib_sft, out_fname)

        if save_full_ic:
            full_ic_idx.extend(invalid_idx)

    if save_full_ic and len(full_ic_idx):
        out_name = os.path.join(out_segmented_dir,
                                base_name + '_IC.{}'.format(out_tract_type))
        ic_sft = sft[full_ic_idx]
        save_tractogram(ic_sft, out_name)
