import numpy as np
import pandas as pd
import warnings
from .utils import clipper1sided, index_bp_conversion_back
from sicer.lib import GenomeData
import os

def Clipper(analysis='enrichment', FDR=0.05, procedure=None, contrast_score=None,
            n_permutation=None, seed=12345, args=None, pool=None, chroms=None):

    if analysis == 'enrichment':
        if procedure is None:
            procedure = 'BC'

        if contrast_score is None:
            if procedure == 'BC':
                contrast_score = 'diff'

        if procedure == 'BC':
            re = clipper1sided(score_exp=None, score_back=None, FDR=FDR,
                               nknockoff=n_permutation,
                               importanceScore_method=contrast_score,
                               FDR_control_method=procedure, ifpowerful=False,
                               seed=seed, args=args, chroms=chroms, pool=pool)

        FDR_nodisc = [len(re_i['discovery'][0]) == 0 for re_i in re['results']]
        if any(FDR_nodisc) and procedure != 'aBH':
            warnings.warn(
                'At FDR = '+ str(FDR) +', no discovery has been found using current procedure. '
                'To make more discoveries, switch to aBH procedure or increase the FDR threshold.')

    return [result['thre'] for result in re['results']][0]


def filter_island_with_clipper(args, chroms, pool=None):
    threshold = Clipper(args=args, chroms=chroms, FDR=args.false_discovery_rate, pool=pool)
    for chrom in chroms:
        chrom_island_load = args.treatment_file.replace('.bed', '') + '_' + args.control_file.replace('.bed', '') + '_' + \
                        f'reads_{chrom}' + '_island_summary_clipper.npy'
        chrom_read_union_load = args.treatment_file.replace('.bed', '') + '_' + \
                                 args.control_file.replace('.bed', '') + '_' + f'reads_{chrom}' + '_union_sup.npy'
        try:
            chrom_island_list = pd.DataFrame(np.load(chrom_island_load, allow_pickle=True))
            read_union = pd.DataFrame(np.load(chrom_read_union_load, allow_pickle=True))
        except:
            continue

        s1 = np.repeat(read_union[3], read_union[2] - read_union[1]).to_numpy().reshape(-1, 1).astype(np.int16)
        s2 = np.repeat(read_union[4], read_union[2] - read_union[1]).to_numpy().reshape(-1, 1).astype(np.int16)

        mean_peak = [np.mean(s1[start:end] - s2[start:end])
                       for start, end in zip(chrom_island_list[1], chrom_island_list[2])]
        clipper_peak = chrom_island_list[np.array(mean_peak) >= threshold]
        np.save(chrom_island_load, clipper_peak)


def main(args, pool=None):
    chroms = GenomeData.species_chroms[args.species]
    df_call = args.df  # Determines if this function was called by SICER or SICER-DF
    columnindex = 7  # temp, need to be revisited
    island_summary_length = 0

    filter_island_with_clipper(args, chroms, pool=pool)

    outfile_name = (args.treatment_file.replace('.bed', '') + '-W' + str(args.window_size) + '-G'
                    + str(args.gap_size) + '-ClipperFDR' + str(args.false_discovery_rate) + '-island.bed')
    outfile_path = os.path.join(args.output_directory, outfile_name)

    with open(outfile_path, 'w') as outfile:
        for chrom in chroms:
            if df_call:  # TODO: need to be revisited
                island_file_name = chrom + '_union_island_summary_filtered' + str(columnindex) + '.npy'
            else:
                island_file_name = args.treatment_file.replace('.bed', '') + '_' + \
                                   args.control_file.replace('.bed', '') + '_' + f'reads_{chrom}' + \
                                   '_island_summary_clipper.npy'

            try:
                filtered_island = pd.DataFrame((np.load(island_file_name, allow_pickle=True)),
                                               columns=['chrom', 'start', 'end', 'treat', 'ctrl']).astype(
                    {'chrom': str, 'start': int, 'end': int, 'treat': int, 'ctrl': int})
            except:
                continue

            island_summary_length += len(filtered_island)
            filtered_island = index_bp_conversion_back(filtered_island).to_numpy()

            for island in filtered_island:
                output_line = ''
                for i in range(0, len(island)):
                    output_line += str(island[i]) + '\t'
                output_line += '\n'
                outfile.write(output_line)

    return island_summary_length
