#!/usr/bin/env python

from __future__ import division

import argparse
import logging
import os

from challenge_scoring.io.results import save_results
from challenge_scoring.metrics.scoring import score_submission
from challenge_scoring.utils.attributes import get_attribs_for_file,\
                                               load_attribs
from challenge_scoring.utils.filenames import mkdir


DESCRIPTION = """
    Score a submission for the ISMRM 2015 tractography challenge.

    This is based on the ISMRM 2015 tractography challenge, see
    http://www.tractometer.org/ismrm_2015_challenge/

    This script scores a submission following the method presented in
    https://doi.org/10.1101/084137

    This method differs from the classical Tractometer approach
    (https://doi.org/10.1016/j.media.2013.03.009). Instead of only using
    masks to define the ground truth and classify streamlines in the
    submission, bundles are extracted using a bundle recognition technique.

    More details are provided in the documentation here:
    https://github.com/scilus/ismrm_2015_tractography_challenge_scoring

    The algorithm has 6 main steps:
        1: extract all streamlines that are valid, which are classified as
           Valid Connections (VC) making up Valid Bundles (VB).
        2: remove streamlines shorter than an threshold based on the GT dataset
        3: cluster the remaining streamlines
        4: remove singletons
        5: assign each cluster to the closest ROIs pair. Those make up the
           Invalid Connections (IC), grouped as Invalid Bundles (IB).
        6: streamlines that are neither in VC nor IC are classified as
           No Connection (NC).
"""


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('tractogram', action='store',
                   metavar='TRACTS', type=str, help='Tractogram file')

    p.add_argument('base_dir', action='store',
                   metavar='BASE_DIR', type=str,
                   help='base directory for scoring data.\n'
                        'See www.tractometer.org/downloads/downloads/'
                        'scoring_data_tractography_challenge.tar.gz')

    p.add_argument('metadata_file', action='store',
                   metavar='TRACT_ATTRIBUTES', type=str,
                   help='attributes file of the submission.'
                        'Needs to contain the orientation.\n'
                        'Normally, use metadata/ismrm_challenge_2015/' +
                        'anon_submissions_attributes.json.\n' +
                        'Can be computed with ' +
                        'ismrm_compute_submissions_attributes.py.')

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
        # TODO check level to send to scoring
        logging.basicConfig(level=logging.DEBUG)

    if not os.path.isfile(tractogram):
        parser.error('"{0}" must be a file!'.format(tractogram))

    if not os.path.isdir(base_dir):
        parser.error('"{0}" must be a directory!'.format(base_dir))

    if not os.path.isfile(attribs_file):
        parser.error('"{0}" must be a file!'.format(attribs_file))

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
    tracts_attribs = get_attribs_for_file(attribs_file,
                                          os.path.basename(tractogram))


    # Basic bundle attributes should be stored in the scoring data directory.
    gt_bundles_attribs_path = os.path.join(args.base_dir,
                                           'gt_bundles_attributes.json')
    if not os.path.isfile(gt_bundles_attribs_path):
        parser.error('Missing the "gt_bundles_attributes.json" file in the '
                     'provided base directory.')
    basic_bundles_attribs = load_attribs(gt_bundles_attribs_path)

    # TODO remove files in out_dir segmented
    if args.save_full_vc or args.save_full_ic or args.save_ib or args.save_vb \
       or args.save_full_nc:
        segments_dir = mkdir(os.path.join(out_dir, "segmented"))
        base_name = os.path.splitext(os.path.basename(tractogram))[0]
    else:
        segments_dir = ''
        base_name = ''

    scores = score_submission(tractogram, tracts_attribs,
                              base_dir, basic_bundles_attribs,
                              args.save_full_vc,
                              args.save_full_ic,
                              args.save_full_nc,
                              args.save_ib, args.save_vb,
                              segments_dir, base_name, isVerbose)

    if scores is not None:
        save_results(scores_filename[:-4] + '.json', scores)


if __name__ == "__main__":
    main()
