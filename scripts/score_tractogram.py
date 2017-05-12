#!/usr/bin/env python

from __future__ import division

import argparse
import os
import pickle
import logging

# TODO change this is ugly
import challenge_scoring.metrics.metrics as metrics
from challenge_scoring.io.results import save_results
from challenge_scoring.utils.attributes import get_attribs_for_file,\
                                               load_attribs
from challenge_scoring.utils.filenames import mkdir


###############
# Script part #
###############
DESCRIPTION = 'Scoring script for ISMRM tractography challenge.\n\n' \
              'Dissociated from the basic scoring.py because the behaviors\n' \
              'and prerequisite are not the same.\n' \
              'Once behaviors are uniformized (if they ever are), we could merge.\n\n' \
              'NB: bundle overlap and bundle overreach scores are only computed when\n' \
              'algorithm 5 is used.'

# TODO updat description and say that algo is
# 5: VC: auto_extract -> IC: length threshold -> QB -> ' +
#                        'singleton removal -> nearest regions classification.


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('tractogram', action='store',
                   metavar='TRACTS', type=str, help='Tractogram file')

    # TODO where do we host this?
    p.add_argument('base_dir',   action='store',
                   metavar='BASE_DIR',  type=str,
                   help='base directory for scoring data')

    p.add_argument('metadata_file', action='store',
                   metavar='SUBMISSIONS_ATTRIBUTES', type=str,
                   help='attributes file of the submissions. ' +\
                        'Needs to contain the orientation.\n' +\
                        'Normally, use metadata/ismrm_challenge_2015/' +\
                        'anon_submissions_attributes.json.\n' +\
                        'Can be computed with ' +\
                        'ismrm_compute_submissions_attributes.py.')

    p.add_argument('basic_bundles_attribs', action='store',
                   metavar='GT_ATTRIBUTES', type=str,
                   help='attributes of the basic bundles. ' +\
                        'Same format as SUBMISSIONS_ATTRIBUTES')

    p.add_argument('out_dir',    action='store',
                   metavar='OUT_DIR',  type=str,
                   help='directory where to send score files')

    p.add_argument('--save_full_vc', action='store_true',
                   help='save one file containing all VCs')
    p.add_argument('--save_full_ic', action='store_true',
                   help='save one file containing all ICs')
    p.add_argument('--save_full_nc', action='store_true',
                   help='save one file containing all NCs')

    p.add_argument('--save_ib', action='store_true',
                   help='save IB independently.')
    p.add_argument('--save_vb', action='store_true',
                   help='save VB independently.')

    #Other
    p.add_argument('-f', dest='is_forcing', action='store_true',
                   required=False, help='overwrite output files')
    p.add_argument('-v', dest='is_verbose', action='store_true',
                   required=False, help='produce verbose output')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    tractogram = args.tractogram
    base_dir = args.base_dir
    attribs_file = args.metadata_file
    out_dir = args.out_dir

    isForcing = args.is_forcing
    isVerbose = args.is_verbose

    if isVerbose:
        #helper.VERBOSE = True
        # TODO check level to send to scoring
        logging.basicConfig(level=logging.DEBUG)

    if not os.path.isfile(tractogram):
        parser.error('"{0}" must be a file!'.format(tractogram))

    if not os.path.isdir(base_dir):
        parser.error('"{0}" must be a directory!'.format(base_dir))

    if not os.path.isfile(attribs_file):
        parser.error('"{0}" must be a file!'.format(attribs_file))

    if not os.path.isfile(args.basic_bundles_attribs):
        parser.error('"{0}" is not a file!'.format(args.basic_bundles_attribs))

    if out_dir is not None:
        out_dir = mkdir(out_dir + "/").replace("//", "/")


    scores_dir = mkdir(os.path.join(out_dir, "scores"))
    # TODO remove all pkl mentions
    scores_filename = scores_dir + tractogram.split('/')[-1][:-4] + ".pkl"

    if os.path.isfile(scores_filename):
        if isForcing:
            os.remove(scores_filename)
        else:
            print "Skipping... {0}".format(scores_filename)
            return

    # TODO support just giving the orientation attribute
    tracts_attribs = get_attribs_for_file(attribs_file, os.path.basename(tractogram))
    basic_bundles_attribs = load_attribs(args.basic_bundles_attribs)

    # TODO remove files in out_dir segmented
    if args.save_full_vc or args.save_full_ic or args.save_ib or args.save_vb \
       or args.save_full_nc:
        segments_dir = mkdir(os.path.join(out_dir, "segmented"))
        base_name = os.path.splitext(os.path.basename(tractogram))[0]
    else:
        segments_dir = ''
        base_name = ''

    scores = metrics.score_submission(tractogram, tracts_attribs,
                                      base_dir, basic_bundles_attribs,
                                      args.save_full_vc,
                                      args.save_full_ic,
                                      args.save_full_nc,
                                      args.save_ib, args.save_vb,
                                      segments_dir, base_name, isVerbose)

    if scores is not None:
        save_results(scores_filename[:-4] + '.json', scores)

    if isVerbose:
        print(scores)

if __name__ == "__main__":
    main()
