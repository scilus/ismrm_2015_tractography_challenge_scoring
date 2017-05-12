#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import nibabel as nb
import numpy as np
from numpy import linalg
from numpy.lib.index_tricks import c_
import tractconverter as tc
from tractconverter.formats.tck import TCK


def _get_tracts_over_grid(tract_fname, ref_anat_fname, tract_attributes,
                           start_at_corner=True):
    # TODO move to only get the attribute
    # Tract_attributes is a dictionary containing various information
    # about a dataset. Currently using:
    # - "orientation" (should be LPS or RAS)
    tracts_format = tc.detect_format(tract_fname)
    tracts_file = tracts_format(tract_fname)

    # Get information on the supporting anatomy
    ref_img = nb.load(ref_anat_fname)

    index_to_world_affine = ref_img.get_header().get_best_affine()

    if isinstance(tracts_file, tc.formats.vtk.VTK):
        # For VTK files, we need to check the orientation.
        # Considered to be in world space. Use the orientation to correct the
        # affine to bring back to voxel.
        # Since the affine from Nifti goes from voxel to RAS, we need to
        # *-1 the 2 first rows if we are in LPS.
        orientation = tract_attributes.get("orientation", None)
        if orientation is None:
            raise AttributeError('Missing the "orientation" attribute for VTK')
        elif orientation == "NOT_FOUND":
            raise ValueError('Invalid value of "NOT_FOUND" for orientation')
        elif orientation == "LPS":
            index_to_world_affine[0,:] *= -1.0
            index_to_world_affine[1,:] *= -1.0

    # Transposed for efficient computations later on.
    index_to_world_affine = index_to_world_affine.T.astype('<f4')
    world_to_index_affine = linalg.inv(index_to_world_affine)

    # Load tracts
    if isinstance(tracts_file, tc.formats.tck.TCK)\
        or isinstance(tracts_file, tc.formats.vtk.VTK):
        if start_at_corner:
            shift = 0.5
        else:
            shift = 0.0

        for s in tracts_file:
            transformed_s = np.dot(c_[s, np.ones([s.shape[0], 1], dtype='<f4')],
                                   world_to_index_affine)[:, :-1] + shift
            yield transformed_s
    elif isinstance(tracts_file, tc.formats.trk.TRK):
         # Use nb.trackvis to read directly in correct space
         # TODO this should be made more robust, using
         # all fields in header.
         # Currently, load in rasmm space, and then bring back to LPS vox
        try:
            streamlines, _ = nb.trackvis.read(tract_fname,
                                              as_generator=True,
                                              points_space='rasmm')
        except nb.trackvis.HeaderError as er:
            print(er)
            raise ValueError("\n------ ERROR ------\n\n" +\
                  "TrackVis header is malformed or incomplete.\n" +\
                  "Please make sure all fields are correctly set.\n\n" +\
                  "The error message reported by Nibabel was:\n" +\
                  str(er))

        if start_at_corner:
            shift = 0.0
        else:
            shift = 0.5

        for s in streamlines:
            transformed_s = np.dot(c_[s[0], np.ones([s[0].shape[0], 1], dtype='<f4')],
                                   world_to_index_affine)[:, :-1] + shift
            yield transformed_s


def get_tracts_voxel_space(tract_fname, ref_anat_fname, tract_attributes):
    return _get_tracts_over_grid(tract_fname, ref_anat_fname, tract_attributes,
                                 True)


def get_tracts_voxel_space_for_dipy(tract_fname, ref_anat_fname, tract_attributes):
    return _get_tracts_over_grid(tract_fname, ref_anat_fname, tract_attributes,
                                 False)


def save_tracts_tck_from_dipy_voxel_space(tract_outobj, ref_anat_fname,
                                          tracts):
    # TODO validate that tract_outobj is a TCK file.
    # Get information on the supporting anatomy
    ref_img = nb.load(ref_anat_fname)

    index_to_world_affine = ref_img.get_header().get_best_affine()

    # Transposed for efficient computations later on.
    index_to_world_affine = index_to_world_affine.T.astype('<f4')

    # Do not shift, because we save as TCK, and dipy expect shifted tracts.
    transformed = [np.dot(c_[s, np.ones([s.shape[0], 1], dtype='<f4')],
                          index_to_world_affine)[:, :-1] for s in tracts]

    tract_outobj += transformed


def save_valid_connections(extracted_vb_info, streamlines,
                           segmented_out_dir, basename, ref_anat_fname,
                           save_vbs=False, save_full_vc=False):

    if not save_vbs and not save_full_vc:
        return

    full_vcs = []
    for bundle_name, bundle_info in extracted_vb_info.iteritems():
        if bundle_info['nb_streamlines'] > 0:
            out_fname = os.path.join(segmented_out_dir, basename +
                                     '_VB_{0}.tck'.format(bundle_name))

            vc_strl = [streamlines[idx]
                       for idx in bundle_info['streamlines_indices']]

            if save_full_vc:
                full_vcs.extend(vc_strl)

            if save_vbs:
                vb_f = TCK.create(out_fname)
                save_tracts_tck_from_dipy_voxel_space(vb_f, ref_anat_fname, vc_strl)

    if save_full_vc and len(full_vcs):
        out_name = os.path.join(segmented_out_dir, basename + '_VC.tck')
        tract_file = TCK.create(out_name)

        save_tracts_tck_from_dipy_voxel_space(tract_file,
                                              ref_anat_fname,
                                              full_vcs)
