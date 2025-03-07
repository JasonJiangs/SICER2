# Developed by Zang Lab at University of Virginia - 2018

# Author: Jin Yong Yoo

import os
import shutil
import sys
import tempfile
import multiprocessing as mp
from functools import partial

import pandas as pd
import time

curr_path = os.getcwd()

# From SICER Package
from sicer.lib import GenomeData
from sicer.src import remove_redundant_reads
from sicer.src import run_make_graph_file_by_chrom
from sicer.src import find_islands_in_pr
from sicer.src import associate_tags_with_chip_and_control_w_fc_q
from sicer.src import filter_islands_by_significance
from sicer.src import make_normalized_wig
from sicer.src import filter_raw_tags_by_islands
from sicer.clipper_addon import utils as clipper_utils
import sicer.clipper_addon.clipper as clipper
from sicer.clipper_addon.clipper import Clipper

''' args: ArgumentParser object formed form command line parameters
    df_run: If df_run is true, then this instance of SICER is called by SICER-DF module.
            Default value is False.'''


def main(args, df_run=False):
    # Checks if there is a control library
    control_lib_exists = True
    if args.control_file is None:
        control_lib_exists = False

    # Creates temporary directory to contain all intermediate files.
    try:
        temp_dir = tempfile.mkdtemp()
        # Change current working directory to temp_dir
        os.chdir(temp_dir)
    except:
        sys.exit(
            "Temporary directory required for SICER cannot be created. Check if directories can be created in %s." % curr_path)
    try:
        # Step 0: create Pool object for parallel-Processing
        num_chroms = len(GenomeData.species_chroms[args.species])
        pool = mp.Pool(processes=min(args.cpu, num_chroms))

        # Step 1: Remove redundancy reads in input file according to input threshold
        # Output is the total number of reads retained. Represents size of library.
        treatment_file_name = os.path.basename(args.treatment_file)
        print("Preprocess the", treatment_file_name, "file to remove redundancy with threshold of",
              args.redundancy_threshold, "\n")
        total_treatment_read_count = remove_redundant_reads.main(args, args.treatment_file, pool)
        args.treatment_file = treatment_file_name
        print('\n')

        # Step 2: Remove redundancy reads in control library according to input threshold
        if control_lib_exists:
            control_file_name = os.path.basename(args.control_file)
            print("Preprocess the", control_file_name, "file to remove redundancy with threshold of",
                  args.redundancy_threshold, "\n")
            total_control_read_count = remove_redundant_reads.main(args, args.control_file, pool)
            args.control_file = control_file_name
            print('\n')

        # Step 3: Partition the genome in windows and generate graph files for each chromosome
        print("Partition the genome in windows and generate summary files... \n")
        total_tag_in_windows = run_make_graph_file_by_chrom.main(args, pool)
        print("\n")

        # Step4+5: Normalize and generate WIG file
        print("Normalizing graphs by total island filitered reads per million and generating summary WIG file...\n")
        output_WIG_name = (treatment_file_name.replace('.bed', '') + "-W" + str(args.window_size) + "-normalized.wig")
        make_normalized_wig.main(args, output_WIG_name, pool)

        # Step 6: Find candidate islands exhibiting clustering
        print("Finding candidate islands exhibiting clustering...\n")
        find_islands_in_pr.main(args, total_tag_in_windows, pool)
        print("\n")

        # Running SICER with a control library
        if control_lib_exists:
            # Step 7
            print("Calculating significance of candidate islands using the control library... \n")
            q_cal_time_start = time.time()
            associate_tags_with_chip_and_control_w_fc_q.main(args=args,
                                                             chip_library_size=total_treatment_read_count,
                                                             control_library_size=total_control_read_count,
                                                             pool=pool)
            q_cal_time_end = time.time()

            print('============================================Clipper Test========================================\n')
            print("Identify significant islands using Clipper FDR control\n")
            test_start_time = time.time()
            # output read union
            clipper_utils.island_bdg_union(args, False, 'reads', GenomeData.species_chroms[args.species], pool)

            # calculate Clipper FDR control threshold and filter the SICER peak with it
            clipper_significant_read_count = clipper.main(args=args, pool=pool)
            test_end_time = time.time()
            print("Total number of significant islands identified by Clipper: ", clipper_significant_read_count, "\n")
            print("Time to filter islands by Clipper FDR control: ", test_end_time - test_start_time, " seconds\n")
            print('============================================Clipper Test========================================\n')

            # Step 8: Filter out any significant islands whose pvalue is greater than the false discovery rate
            print("Identify significant islands using FDR criterion\n")
            # value 7 represents the ith column we want to filtered by
            q_filter_test_start_time = time.time()
            significant_read_count = filter_islands_by_significance.main(args, 7, pool)
            q_filter_test_end_time = time.time()
            total_time = (q_cal_time_end - q_cal_time_start) + (q_filter_test_end_time - q_filter_test_start_time)
            print("Time to filter islands by p-value FDR control: ", total_time, " seconds\n")
            print("Out of the ", total_treatment_read_count, " reads in ", treatment_file_name, ", ",
                  significant_read_count, " reads are in significant islands\n")

        # Optional Outputs
        if (args.significant_reads):
            # Step 9: Filter treatment reads by the significant islands found from step 8
            print("Filtering reads with identified significant islands...\n")
            filter_raw_tags_by_islands.main(args, pool)

            # Step 10: Produce graph file based on the filtered reads from step 9
            print("Making summary graph with filtered reads...\n")
            run_make_graph_file_by_chrom.main(args, pool, True)
            # Step 11: Produce Normalized WIG file
            print(
                "\nNormalizing graphs by total island filitered reads per million and generating summary WIG file...\n")
            output_WIG_name = (treatment_file_name.replace('.bed', '') + "-W" + str(args.window_size) + "-G" + str(
                args.gap_size) + "-FDR" + str(args.false_discovery_rate) + "-islandfiltered-normalized.wig")
            make_normalized_wig.main(args, output_WIG_name, pool)


        pool.close()
        pool.join()
        # Final Step
        if (df_run == True):
            return temp_dir, total_treatment_read_count
        else:
            print("End of SICER")
    finally:
        if df_run == False:
            print("Removing temporary directory and all files in it.")
            # ##################### Debugging #####################
            # view the files in the directory
            for f in os.listdir(temp_dir):
                print(f)
            # #####################################################
            shutil.rmtree(temp_dir)
