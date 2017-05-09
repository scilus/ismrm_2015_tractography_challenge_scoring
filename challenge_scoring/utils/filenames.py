#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import basename, splitext

def get_root_image_name(filename):
    # Used for nifti images, to remove .nii.gz
    return splitext(splitext(basename(filename))[0])[0]
