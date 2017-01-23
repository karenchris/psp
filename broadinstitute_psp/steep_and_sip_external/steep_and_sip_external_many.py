"""
steep_and_sip_external_many.py

Runs steep_and_sip_external.py for each cell line in the corpus. Most of the
arguments come from the config file. The default config file points to the
latest QCNORM and SIM directories.

"""

import ConfigParser
import argparse
import datetime
import logging
import os
import sys
import time
import traceback

import broadinstitute_cmap.io.pandasGEXpress.parse as pg
import broadinstitute_cmap.io.pandasGEXpress.write_gct as wg

import broadinstitute_psp.steep_and_sip_external.steep_and_sip_external as steep_and_sip_external
import broadinstitute_psp.utils.setup_logger as setup_logger

__author__ = "Lev Litichevskiy"
__email__ = "lev@broadinstitute.org"

logger = logging.getLogger(setup_logger.LOGGER_NAME)

INTERNAL_GCT_FORMAT = "{assay}_{cell}_QCNORM.gct"
BG_GCT_FORMAT = "{assay}_{cell}_SIM.gct"
OUT_STEEP_FORMAT = "{cell}_SIM.gct"
OUT_SIP_FORMAT = "{cell}_CONN.gct"
OUT_DIR_FORMAT = "steep_and_sip_external_many_{time}"

# TODO(LL): rename this, make this easy for AO's use case (e.g. annotation)?
# TODO(LL): need to figure out how we will pass -fae through

def build_parser():
    """Build argument parser."""

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required args
    parser.add_argument("--assay", "-a", required=True, choices=["GCP", "P100"],
                        help="which assay's data to query")
    parser.add_argument("--external_gct_path", "-e", required=True,
                        help="path to gct file of external profiles")
    parser.add_argument("--out_dir", "-o", required=True,
                        help="output directory")
    parser.add_argument("--psp_on_clue_config_path", "-p",
                        default="clue/psp_on_clue.cfg",
                        help="filepath to psp_on_clue.cfg")

    # Optional args
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="whether to increase the # of messages reported")

    return parser


def main(args):

    # Record start_time
    start_time = datetime.datetime.now()

    # Create output directory using start_time down to the millisecond
    time_for_out_dir = start_time.strftime("%Y_%m_%d_%H%M%S%f")
    actual_out_dir = os.path.join(args.out_dir, OUT_DIR_FORMAT.format(time=time_for_out_dir))
    os.makedirs(actual_out_dir)

    try:

        # Read and unpack config file
        (cells, internal_gct_dir, bg_gct_dir, fields_to_aggregate_for_external_profiles,
         fields_to_aggregate_for_internal_profiles, similarity_metric,
         connectivity_metric) = read_config_file(args.psp_on_clue_config_path)

        # Read in the external profiles only once
        external_gct = pg.parse(args.external_gct_path, convert_neg_666=False, make_multiindex=True)

        # Loop over cell lines in corpus
        for cell in cells:

            # Import gct with the internal profiles for this cell line
            internal_gct_path = os.path.join(internal_gct_dir, INTERNAL_GCT_FORMAT.format(
                assay=args.assay, cell=cell))
            internal_gct = pg.parse(internal_gct_path, convert_neg_666=False, make_multiindex=True)

            # Import gct with the similarity matrix for this cell line
            bg_gct_path = os.path.join(bg_gct_dir, BG_GCT_FORMAT.format(
                assay=args.assay, cell=cell))
            bg_gct = pg.parse(bg_gct_path, convert_neg_666=False, make_multiindex=True)

            (sim_gct, conn_gct) = steep_and_sip_external.do_steep_and_sip(
                external_gct, internal_gct, bg_gct, "spearman",
                "ks_test", fields_to_aggregate_for_external_profiles,
                fields_to_aggregate_for_internal_profiles)

            # Write output gcts
            out_steep_name = os.path.join(actual_out_dir, OUT_STEEP_FORMAT.format(cell=cell))
            out_sip_name = os.path.join(actual_out_dir, OUT_SIP_FORMAT.format(cell=cell))
            wg.write(sim_gct, out_steep_name, data_null="NaN", metadata_null="NA", filler_null="NA")
            wg.write(conn_gct, out_sip_name, data_null="NaN", filler_null="NaN", metadata_null="NaN")

        # Write success.txt with timestamp
        success_path = os.path.join(actual_out_dir, "success.txt")
        write_success(success_path)

        # Return how much time it took
        end_time = datetime.datetime.now()
        minutes_elapsed = (end_time - start_time).seconds / 60.
        logger.info("steep_and_sip_external_many.py completed in {:.1f} min.".format(minutes_elapsed))


    except Exception:
        failure_path = os.path.join(actual_out_dir, "failure.txt")
        msg = "steep_and_sip_external_many.py failed. See {} for stacktrace.".format(failure_path)

        # Write failure.txt
        write_failure(failure_path)

        # Raise exception
        logger.error(msg)
        raise Exception(msg)


def read_config_file(config_path):

    assert os.path.exists(config_path), (
        "Config file can't be found. config_path: {}".format(config_path))

    # Read config file
    config_parser = ConfigParser.RawConfigParser()
    config_parser.read(config_path)

    # Return config fields as dictionarires
    config_corpus = dict(config_parser.items("corpus"))
    config_metadata = dict(config_parser.items("metadata"))
    config_algorithms = dict(config_parser.items("algorithms"))

    # Unpack the config file
    cells = eval(config_corpus["cells"])
    internal_gct_dir = config_corpus["qcnorm_dir"]
    bg_gct_dir = config_corpus["sim_dir"]
    fields_to_aggregate_for_external_profiles = eval(config_metadata["fields_to_aggregate_for_external_profiles"])
    fields_to_aggregate_for_internal_profiles = eval(config_metadata["fields_to_aggregate_for_internal_profiles"])
    similarity_metric = config_algorithms["similarity_metric"]
    connectivity_metric = config_algorithms["connectivity_metric"]

    return cells, internal_gct_dir, bg_gct_dir, \
           fields_to_aggregate_for_external_profiles, \
           fields_to_aggregate_for_internal_profiles, \
           similarity_metric, connectivity_metric


def write_success(file_name):
    # Create timestamp
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    # Write timestamp to file_name
    f = open(file_name, 'w')
    f.write("steep_and_sip_external_many.py completed at {}".format(timestamp))
    f.close()


def write_failure(file_name):
    # Record stacktrace
    _, _, exc_traceback = sys.exc_info()

    # Write stacktrace to file_name
    f = open(file_name, "w")
    traceback.print_exc(exc_traceback, file=f)
    f.close()


if __name__ == "__main__":
    args = build_parser().parse_args(sys.argv[1:])
    setup_logger.setup(verbose=args.verbose)

    main(args)