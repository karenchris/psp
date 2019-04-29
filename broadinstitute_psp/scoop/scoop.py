"""
scoop.py

For datasets with multiple perturbagens, converts corresponding metadata fields between array and separated forms.
If the user does not specify the direction of conversion, by default scoop will determine the direction.

Required input is a path to a gct file. Output is a gct file with updated metadata.
"""

import argparse
import logging
import numpy as np
import os
import pandas as pd
import sys
import re

import cmapPy.pandasGEXpress.GCToo as GCToo
import cmapPy.pandasGEXpress.write_gct as wg
import broadinstitute_psp.utils.psp_utils as psp_utils
import broadinstitute_psp.utils.setup_logger as setup_logger

__author__ = "Karen Christianson"
__email__ = "karen@broadinstitute.org"

# Setup logger
logger = logging.getLogger(setup_logger.LOGGER_NAME)

# Default output suffix
DEFAULT_SCOOP_SUFFIX = ".scoop.gct"


def build_parser():
    """Build argument parser."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arg
    parser.add_argument("--in_gct_path", "-i", required=True,
                        help="filepath to input gct")

    # Optional args
    parser.add_argument("--direction", "-d", default="guess",
                        help="direction of metadata conversion; options are join, separate, guess")
    parser.add_argument("--out_dir", "-o", default=".",
                        help="path to save directory")
    parser.add_argument("--out_base_name", "-ob", default=None,
                        help="base name of output gct file (default is <INPUT_GCT>.scoop.gct")
    parser.add_argument("--psp_config_path", "-p", default="~/psp_production.cfg",
                        help="filepath to PSP config file")
    parser.add_argument("--force_assay", "-f",
                        choices=["GCP", "P100"], default=None,
                        help=("directly specify assay type here " +
                              "(overrides first entry of provenance code)"))

    # Parameters
    parser.add_argument("-verbose", "-v", action="store_true", default=False,
                        help="increase the number of messages reported")

    return parser


def main(args):
    """THE MAIN METHOD. Create new metadata fields .

    Args:
        args (argparse.Namespace object): fields as defined in build_parser()

    Returns:
        out_gct (GCToo object): output gct object
    """
    ### READ GCT AND CONFIG FILE
    (in_gct, assay_type, prov_code, config_io, config_metadata, config_parameters) = (
        read_dry_gct_and_config_file(
            args.in_gct_path, args.psp_config_path, args.force_assay))

    ### DETERMINE DIRECTION OF CONVERSION
    direction = check_prov_code_and_determine_direction(in_gct, prov_code, config_metadata['scoop_prov_code_entry'], args.direction)

    ### CONVERT COLUMN METADATA BASED ON DIRECTION AND UPDATE PROVENANCE CODE
    out_gct = metadata_conversion_with_updated_prov_code(in_gct, direction, prov_code,
                                                         config_metadata['scoop_prov_code_entry'],
                                                         config_metadata["prov_code_field"],
                                                         config_metadata["prov_code_delimiter"])

    ### CONFIGURE OUT NAMES
    out_gct_name = configure_out_names(
        args.in_gct_path, args.out_base_name)

    ### WRITE OUTPUT GCT
    write_output_gct(out_gct, args.out_dir, out_gct_name,
                     config_io["data_null"], config_io["filler_null"])

    return out_gct


# this function copied from dry.py
def read_dry_gct_and_config_file(in_gct_path, config_path, forced_assay_type):
    """ Read gct and config file.

    Uses the utility function read_gct_and_config_file from psp_utils.
    Provenance code is extracted from the col metadata. It must be non-empty
    and the same for all samples. If forced_assay_type is not None,
    assay_type is set to forced_assay_type.

    Args:
        in_gct_path (string): filepath to gct file
        config_path (string): filepath to config file
        forced_assay_type (string, or None)

    Returns:
        gct (GCToo object)
        assay_type (string)
        prov_code (list of strings)
        config_io (dictionary)
        config_metadata (dictionary)
        config_parameters (dictionary)
    """
    # Read gct and config file
    (gct, config_io, config_metadata, config_parameters) = psp_utils.read_gct_and_config_file(in_gct_path, config_path)

    # Extract the plate's provenance code
    prov_code = psp_utils.extract_prov_code(gct.col_metadata_df,
                                  config_metadata["prov_code_field"],
                                  config_metadata["prov_code_delimiter"])

    # If forced_assay_type is not None, set assay_type to forced_assay_type.
    # Otherwise, the first entry of the provenance code is the assay_type.
    if forced_assay_type is not None:
        assay_type = forced_assay_type
    else:
        assay_type = prov_code[0]

    # Make sure assay_type is one of the allowed values
    p100_assay_types = eval(config_metadata["p100_assays"])
    gcp_assay_types = eval(config_metadata["gcp_assays"])
    assay_type_out = check_assay_type(assay_type, p100_assay_types, gcp_assay_types)

    return gct, assay_type_out, prov_code, config_io, config_metadata, config_parameters


# this function copied from dry.py
def check_assay_type(assay_type, p100_assays, gcp_assays):
    """Verify that assay is one of the allowed types. Return either P100 or GCP.

    Args:
        assay_type_in (string)
        p100_assays (list of strings)
        gcp_assays (list of strings)

    Returns:
        assay_type_out (string): choices = {"P100", "GCP"}
    """
    if assay_type in p100_assays:
        assay_type_out = "p100"
    elif assay_type in gcp_assays:
        assay_type_out = "gcp"
    else:
        err_msg = ("The assay type is not a recognized P100 or GCP assay. " +
                   "assay_type: {}, p100_assays: {}, gcp_assays: {}")
        logger.error(err_msg.format(assay_type, p100_assays, gcp_assays))
        raise(Exception(err_msg.format(assay_type, p100_assays, gcp_assays)))

    return assay_type_out


def check_prov_code_and_determine_direction(gct, prov_code, prov_code_entry, input_direction):
    """Check if scoop has already been performed on this dataset by looking at provenance code and then determine the
    direction of conversion based on input args. Returns 'join' 'separate' or 'do nothing'.

    Args:
        gct (pandas df)
        prov_code (list of strings)
        prov_code_entry (string)
        input_direction (string)

    Returns:
        direction (string): choices = {"join", "separate", "do nothing"}
    """
    if prov_code_entry in prov_code:
        logger.info("{} has already occurred.".format(prov_code_entry))
        if input_direction == 'join' or input_direction == 'separate':
            direction = determine_direction_from_input(gct, input_direction)
        else:
            raise ValueError("Cannot guess direction if scoop has already occurred. Must input 'join' or 'separate'.")
    else:
        direction = determine_direction(gct, input_direction)

    return direction


def determine_direction(gct, input_direction):
    """Determines the direction of conversion based on input args. Returns 'join' 'separate' or 'do nothing'.

    Args:
        gct (pandas df)
        input_direction (string)

    Returns:
        direction (string): choices = {"join", "separate", "do nothing"}
    """
    if 'pert_multiplicity' not in gct.col_metadata_df.columns.values:
        raise ValueError("Column metadata is missing the required field 'pert_multiplicity'.")
    elif gct.col_metadata_df['pert_multiplicity'].max() == 1:
        logger.info('No metadata fields to convert. Returning original GCT.')
        direction = "do nothing"
    elif gct.col_metadata_df['pert_multiplicity'].max() > 1:
        if ("pert_iname" in gct.col_metadata_df.columns.values) & (len(filter(re.compile('pert_iname').match,
                                                                              gct.col_metadata_df.columns.values)) > 1):
            logger.info('GCT already contains all metadata field conversions. Returning original GCT.')
            direction = "do nothing"
        elif ("pert_iname" not in gct.col_metadata_df.columns.values) & (len(filter(re.compile('pert_iname').match,
                                                                                 gct.col_metadata_df.columns.values)) == 0):
            raise ValueError("Column metadata must have a field containing 'pert_iname'.")
        else:
            direction = determine_direction_from_input(gct, input_direction)
    else:
        raise ValueError("pert_multiplicity values cannot be less than 1")
    # TODO case where pert_multiplicity max = 0

    return direction


def determine_direction_from_input(gct, input_direction):
    """ Determines the direction of conversion based on input args. Returns 'join' 'separate' or 'do nothing'.
    If input is 'guess', this function will determine 'join' or 'separate'.

    Args:
        gct (pandas df)
        input_direction (string)

    Returns:
        direction (string): choices = {"join", "separate", "do nothing"}
    """
    if input_direction == 'join':
        if len(filter(re.compile('pert_iname').match,gct.col_metadata_df.columns.values)) > 1:
            direction = 'join'
        else:
            raise ValueError("Joined metadata already exists. Did you mean to input 'separate'?")
    elif input_direction == 'separate':
        if "pert_iname" in gct.col_metadata_df.columns.values:
            direction = 'separate'
        else:
            raise ValueError("Separated metadata already exists. Did you mean to input 'join'?")
    elif input_direction == 'guess':
        direction = guess_direction(gct)
    else:
        raise ValueError("Not a valid input direction. Valid input directions are 'join', 'separate', or 'guess'.")

    return direction


def guess_direction(gct):
    """Determines the direction of conversion if input is 'guess'.
    Args:
        gct (pandas df)

    Returns:
        direction (string): choices = {"join", "separate", "do nothing"}
    """
    if "pert_iname" in gct.col_metadata_df.columns.values:
        direction = "separate"
    elif "pert_iname" not in gct.col_metadata_df.columns.values:
        direction = "join"
    else:
        raise ValueError("Cannot guess direction: column metadata must have a field containing 'pert_iname'.")

    return direction


def metadata_conversion_with_updated_prov_code(gct, direction, prov_code, prov_code_entry, prov_code_field,
                                               prov_code_delimiter):
    """Does metadata conversion and updates the provenance code.

    Args:
        gct (pandas df)
        direction (string)
        prov_code (string)
        prov_code_entry (string)
        prov_code_field (string)
        prov_code_delimiter (string)

    Returns:
        out_gct (GCToo object)
    """
    (out_col_metadata, updated_prov_code) = do_metadata_conversion(gct, direction, prov_code, prov_code_entry)
    out_gct = update_metadata_and_prov_code(gct.data_df, gct.row_metadata_df, out_col_metadata, updated_prov_code,
                                            prov_code_field, prov_code_delimiter)

    return out_gct


def do_metadata_conversion(gct, direction, prov_code, prov_code_entry):
    """ Performs the metadata conversion. If direction is 'join', will combine multiple fields into one in array format.
    If direction is 'separate', will separate one metadata field with multiple entries into several. Only acts on column
    metadata.

    Args:
        gct (pandas df): gct that is read in first step
        direction (string): the direction of conversion as determined by determine_direction() function
        prov_code (list of strings)
        prov_code_entry (string)

    Returns:
        out_gct (GCToo object)
    """
    if direction == 'join':
        out_col_metadata = join_metadata(gct.col_metadata_df)
        updated_prov_code = prov_code + [prov_code_entry]
    elif direction == 'separate':
        out_col_metadata = separate_metadata(gct.col_metadata_df)
        updated_prov_code = prov_code + [prov_code_entry]
    else:
        out_col_metadata = gct.col_metadata_df.copy()
        updated_prov_code = prov_code

    return out_col_metadata, updated_prov_code


def join_metadata(df):
    """Creates a new column metadata field that combines multiple of the same metadata field into one array

    Args:
        df(pandas dataframe): column metadata
    Returns:
        new_col_metadata: edited column metadata with joined elements (pandas dataframe)
    """
    col_metadata = df.copy()

    # Create a list of metadata fields with multiple entries that need to be joined
    fields_to_join =  determine_fields_to_join(col_metadata)

    # Subset the common fields to be joined within the large list based on pert_multiplicity
    subset_fields_to_join=[]
    m=col_metadata['pert_multiplicity'].max()
    for i in range(m, len(fields_to_join)+1, m):
        sub = fields_to_join[(i-m):i]
        subset_fields_to_join.append(sub)

    # Use subset list to join the fields into array
    for field in subset_fields_to_join:
        df_subset = col_metadata[field]
        name = re.sub('_\d+$', '', field[0])
        col_metadata[name]=df_subset.apply(lambda r: list(np.array(r)), axis=1)

    return col_metadata


def determine_fields_to_join(col_metadata):
    """Determines which fields are duplicates and need to be joined into array format based on numerical tags. Input is
    column metadata df, output is a list of duplicate metadata fields.

    Args:
        col_metadata(pandas df)

    Returns:
         fields_to_join: list of duplicate metadata fields that will be joined into array
    """
    # Make list of the metadata column fields containing numeric elements
    fields_to_join=[]
    for i in col_metadata.columns.values:
        if re.search('_\d+$', i):
            fields_to_join.append(i)
    fields_to_join.sort()

    return fields_to_join


def separate_metadata(df):
    """Creates a new column metadata field separating arrays into multiple numbered fields

    Args:
        df(pandas dataframe): column metadata
    Returns:
        new_col_metadata: edited column metadata with separated elements (pandas dataframe)
    """
    col_metadata = df.copy()

    # Make a list of metadata fields that need to be separated
    fields_to_separate = determine_fields_to_separate(col_metadata)

    # Do the separation
    for field in fields_to_separate:
        m = col_metadata.pert_multiplicity.max()
        new = col_metadata[field].str.replace('\[','')
        new = new.str.replace('\]', '')
        new = new.str.split(',', n=(m-1), expand=True)
        new = new.fillna('NA')
        for i in range(m):
            name = field+"_%02d"%(i+1)
            col_metadata[name]=new[i]

    return col_metadata


def determine_fields_to_separate(col_metadata):
    """Determines which fields are contain arrays and need to be separated out into individual columns. Input is
        column metadata df, output is a list of metadata fields to be separated.

        Args:
            col_metadata(pandas df)

        Returns:
             fields_to_separate: list of metadata fields containing arrays that will be separated
        """
    # Make list of the metadata column fields with entries that contain arrays
    fields_to_separate=[]
    for column in col_metadata.columns.values:
        if sum(col_metadata[column].astype(str).str.match('\[')) > 0:
            fields_to_separate.append(column)

    return fields_to_separate


def update_metadata_and_prov_code(data_df, row_metadata_df, col_metadata_df, updated_prov_code, prov_code_field, prov_code_delimiter):
    """Update metadata with the already sliced data_df, and update the prov_code.

    Args:
        data_df (pandas df)
        row_meta_df (pandas df)
        col_meta_df (pandas df)
        prov_code_entry (string)
        prov_code (list of strings)

    Returns:
        out_gct (GCToo object): updated
        updated_prov_code (list of strings)

    """
    col_metadata = insert_prov_code(col_metadata_df, updated_prov_code, prov_code_field, prov_code_delimiter)
    out_gct = GCToo.GCToo(data_df=data_df, row_metadata_df=row_metadata_df, col_metadata_df=col_metadata)

    return out_gct


def insert_prov_code(col_metadata, updated_prov_code, prov_code_field, prov_code_delimiter):
    """Insert offsets into output gct and update provenance code in metadata.
    Args:
        gct (GCToo object)
        prov_code (list of strings)
        prov_code_field (string): name of col metadata field containing the provenance code
        prov_code_delimiter (string): what string to use as delimiter in prov_code
    Returns:
        gct (GCToo object): updated metadata
    """
    # Convert provenance code to delimiter separated string
    prov_code_str = prov_code_delimiter.join(updated_prov_code)

    # Update the provenance code in col_metadata_df
    col_metadata.loc[:, prov_code_field] = prov_code_str

    return col_metadata


def configure_out_names(in_gct_path, out_base_name_from_args):
    """If out_base_name_from_args is None, append DEFAULT_GCT_SUFFIX to the input gct name to generate the gct file.

    Args:
        in_gct_path:
        out_base_name_from_args:

    Returns:
        out_gct_name (file path)
    """

    if out_base_name_from_args is None:
        input_basename = os.path.splitext(os.path.basename(in_gct_path))[0]
        out_gct_name = input_basename + DEFAULT_SCOOP_SUFFIX
    else:
        input_basename = os.path.splitext(os.path.basename(out_base_name_from_args))[0]
        out_gct_name = input_basename + DEFAULT_SCOOP_SUFFIX

    return out_gct_name

# this function copied from dry.py
def write_output_gct(gct, out_dir, out_gct_name, data_null, filler_null):
    """Write output gct file.

    Args:
        gct (GCToo object)
        out_dir (string): path to save directory
        out_gct_name (string): name of output gct
        data_null (string): string with which to represent NaN in data
        filler_null (string): string with which to fill the empty top-left quadrant in the output gct

    Returns:
        None

    """
    out_fname = os.path.join(out_dir, out_gct_name)
    wg.write(gct, out_fname, data_null=data_null, filler_null=filler_null, data_float_format=None)



if __name__ == "__main__":
    args = build_parser().parse_args(sys.argv[1:])
    setup_logger.setup(verbose=args.verbose)
    logger.debug("args: {}".format(args))

    main(args)
