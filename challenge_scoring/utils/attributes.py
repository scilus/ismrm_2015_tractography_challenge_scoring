#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json


def load_attribs(attribs_filepath):
    with open(attribs_filepath, 'rb') as attribs_file:
        attribs = json.load(attribs_file)
    return attribs
