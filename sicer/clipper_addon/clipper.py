import numpy as np
import pandas as pd
import warnings
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from .utils import clipper2sided, clipper1sided
import time
from sicer.lib import GenomeData
import os
from functools import partial

def Clipper(score_exp=None, score_back=None, analysis='enrichment', FDR=0.05, procedure=None, contrast_score=None,
            n_permutation=None, seed=12345, args=None, pool=None):

    analysis = analysis.lower()
    if analysis not in ['differential', 'enrichment']:
        raise ValueError("'analysis' must be 'differential' or 'enrichment'")

    # if analysis == 'differential':
    #     if contrast_score is None:
    #         contrast_score = 'max'
    #     elif contrast_score not in ['diff', 'max']:
    #         raise ValueError("'contrast_score' must be 'diff' or 'max'")
    #
    #     if procedure is None:
    #         procedure = 'GZ'
    #     elif procedure not in ['BC', 'aBH', 'GZ']:
    #         raise ValueError("'procedure' must be 'BC', 'aBH', or 'GZ'")
    #
    #     re = clipper2sided(score_exp=score_exp, score_back=score_back, FDR=FDR,
    #                        nknockoff=n_permutation,
    #                        contrastScore_method=contrast_score, importanceScore_method='diff',
    #                        FDR_control_method=procedure, ifpowerful=False, seed=seed)
    #
    #     FDR_nodisc = [len(re_i['discovery']) == 0 for re_i in re['results']]
    #     if any(FDR_nodisc) and contrast_score == 'max':
    #         warnings.warn(
    #             'At FDR = {}, no discovery has been found using max contrast score. To make more discoveries, switch to diff contrast score or increase the FDR threshold.'.format(
    #                 ', '.join(map(str, np.array(FDR)[FDR_nodisc]))))

    ##### we can move it inside for loop process for each chromosome
    # if np.ndim(score_exp) == 1:
    #     score_exp = np.reshape(score_exp, (-1, 1))
    # if np.ndim(score_back) == 1:
    #     score_back = np.reshape(score_back, (-1, 1))
    ##### there are lots of decision code here, we can move it if we actually only need one of them
    if analysis == 'enrichment':
        if procedure is None:
            procedure = 'BC'
        elif procedure not in ['BC', 'aBH', 'GZ']:
            raise ValueError("'procedure' must be 'BC', 'aBH', or 'GZ'")

        # if score_exp.shape[1] != score_back.shape[1]:
        #     procedure = 'GZ'

        if contrast_score is None:
            if procedure == 'BC':
                contrast_score = 'diff'
            if procedure == 'GZ':
                contrast_score = 'max'
        elif contrast_score not in ['diff', 'max']:
            raise ValueError("'contrast_score' must be 'diff' or 'max'")

        if procedure == 'BC':
            re = clipper1sided(score_exp=score_exp, score_back=score_back, FDR=FDR,
                               nknockoff=n_permutation,
                               importanceScore_method=contrast_score,
                               FDR_control_method=procedure, ifpowerful=False, seed=seed, args=args)
        if procedure == 'GZ':
            re = clipper1sided(score_exp=score_exp, score_back=score_back, FDR=FDR,
                               nknockoff=n_permutation,
                               contrastScore_method=contrast_score,
                               FDR_control_method=procedure, ifpowerful=False, seed=seed)
        del score_back, score_exp

        FDR_nodisc = [len(re_i['discovery'][0]) == 0 for re_i in re['results']]
        if any(FDR_nodisc) and procedure != 'aBH':
            warnings.warn(
                'At FDR = '+ str(FDR) +', no discovery has been found using current procedure. '
                'To make more discoveries, switch to aBH procedure or increase the FDR threshold.')

    contrast_score_value = re['contrastScore']
    thre = [result['thre'] for result in re['results']]
    discoveries = [result['discovery'] for result in re['results']]
    q = re['results'][0]['q']

    # return {'contrast_score': contrast_score,
    #         'contrast_score_value': contrast_score_value,
    #         'FDR': FDR,
    #         'contrast_score_threshold': thre,
    #         'discoveries': discoveries,
    #         'q': q}
    return thre

def insert_gaps(table):
    new_table = []

    # Insert starting row from 0 to the start of the first row, if it doesn't start at 0
    if table[0][1] != 0:
        new_table.append((table[0][0], 0, table[0][1]-1, 1.0, 0.0))
    # else:
    #     table[0][1] = 1

    for i in range(len(table)):
        new_table.append(table[i])
        if i < len(table) - 1 and table[i][0] == table[i + 1][0]:  # same chromosome
            end_current = table[i][2]
            start_next = table[i + 1][1]
            if end_current + 1 < start_next:  # check for gap
                new_row = (table[i][0], end_current+1, start_next-1, 0.0, 0.0)
                new_table.append(new_row)
    new_table = pd.DataFrame(new_table)
    new_table[1] = new_table[1] - 1
    return new_table


def clipper_filter(args, chrom):
    print('Processing chromosome ' + chrom)
    island_file_name = args.treatment_file.replace('.bed', '') + '_' + chrom + '_island_summary.npy'
    chrom_island_list = pd.DataFrame(np.load(island_file_name, allow_pickle=True))
    try:
        chrom_island_list[1] = chrom_island_list[1] - 1
    except:
        chrom_island_save_name = args.treatment_file.replace('.bed', '') + '_' + \
                                 chrom + '_clipper_island_summary.npy'
        np.save(chrom_island_save_name, chrom_island_list)
        return

    indicator = 'reads'
    union_read_file_name = args.treatment_file.replace('.bed', '') + '_' + args.control_file.replace('.bed', '') + \
                           '_' + f'{indicator}_{chrom}' + '_union.npy'
    chrom_union_read = insert_gaps(np.load(union_read_file_name, allow_pickle=True))

    s1 = np.repeat(chrom_union_read[4], chrom_union_read[2] - chrom_union_read[1]).to_numpy().reshape(-1, 1)
    s2 = np.repeat(chrom_union_read[3], chrom_union_read[2] - chrom_union_read[1]).to_numpy().reshape(-1, 1)

    # return Clipper contrast score for this chromosome
    chrom_contrast_score = Clipper(score_exp=s1, score_back=s2, args=args, FDR=args.false_discovery_rate)

    # filter the island using this contrast score
    median_peak = [np.median(s1[start:end] - s2[start:end]) for start, end in
                   zip(chrom_island_list[1] + 1, chrom_island_list[2] + 1)]

    if not isinstance(chrom_contrast_score[0], (int, float)):
        clipper_filtered_island = chrom_island_list.copy()
    else:
        try:
            clipper_filtered_island = chrom_island_list[median_peak >= chrom_contrast_score[0]].copy()
        except:
            clipper_filtered_island = chrom_island_list.copy()

    # save the filtered island
    clipper_filtered_island.loc[:, 1] = clipper_filtered_island.loc[:, 1] + 1
    chrom_island_save_name = args.treatment_file.replace('.bed', '') + '_' + chrom + '_clipper_island_summary.npy'
    np.save(chrom_island_save_name, clipper_filtered_island)


def main(args, pool=None):
    chroms = GenomeData.species_chroms[args.species]
    df_call = args.df  # Determines if this function was called by SICER or SICER-DF
    columnindex = 7  # temp, need to be revisited
    total_island_count, total_read_count = 0, 0
    island_summary_length = 0

    filter_by_clipper_partial = partial(clipper_filter, args)
    pool.map(filter_by_clipper_partial, chroms)

    # for chrom in chroms:
    #     print('Processing chromosome ' + chrom)
    #     island_file_name = args.treatment_file.replace('.bed', '') + '_' + chrom + '_island_summary.npy'
    #     chrom_island_list = pd.DataFrame(np.load(island_file_name, allow_pickle=True))
    #     try:
    #         chrom_island_list[1] = chrom_island_list[1] - 1
    #     except:
    #         chrom_island_save_name = args.treatment_file.replace('.bed','') + '_' + \
    #                                  chrom + '_clipper_island_summary.npy'
    #         np.save(chrom_island_save_name, chrom_island_list)
    #         continue
    #
    #     indicator = 'reads'
    #     union_read_file_name = args.treatment_file.replace('.bed', '') + '_' + args.control_file.replace('.bed', '') + \
    #                            '_' + f'{indicator}_{chrom}' + '_union.npy'
    #     chrom_union_read = insert_gaps(np.load(union_read_file_name, allow_pickle=True))
    #
    #     s1 = np.repeat(chrom_union_read[4], chrom_union_read[2] - chrom_union_read[1]).to_numpy().reshape(-1, 1)
    #     s2 = np.repeat(chrom_union_read[3], chrom_union_read[2] - chrom_union_read[1]).to_numpy().reshape(-1, 1)
    #
    #     # return Clipper contrast score for this chromosome
    #     chrom_contrast_score = Clipper(score_exp=s1, score_back=s2, args=args, FDR=args.false_discovery_rate)
    #
    #     # filter the island using this contrast score
    #     median_peak = [np.median(s1[start:end] - s2[start:end]) for start, end in zip(chrom_island_list[1]+1, chrom_island_list[2]+1)]
    #
    #     if not isinstance(chrom_contrast_score[0], (int, float)):
    #         clipper_filtered_island = chrom_island_list.copy()
    #     else:
    #         try:
    #             clipper_filtered_island = chrom_island_list[median_peak >= chrom_contrast_score[0]].copy()
    #         except:
    #             clipper_filtered_island = chrom_island_list.copy()
    #
    #     # save the filtered island
    #     clipper_filtered_island.loc[:, 1] = clipper_filtered_island.loc[:, 1] + 1
    #     chrom_island_save_name = args.treatment_file.replace('.bed', '') + '_' + chrom + '_clipper_island_summary.npy'
    #     np.save(chrom_island_save_name, clipper_filtered_island)

    outfile_name = (args.treatment_file.replace('.bed', '') + '-W' + str(args.window_size) + '-G'
                    + str(args.gap_size) + '-ClipperFDR' + str(args.false_discovery_rate) + '-island.bed')
    outfile_path = os.path.join(args.output_directory, outfile_name)

    with open(outfile_path, 'w') as outfile:
        for chrom in chroms:
            if df_call:
                island_file_name = chrom + '_union_island_summary_filtered' + str(columnindex) + '.npy'
            else:
                island_file_name = args.treatment_file.replace('.bed', '') + '_' + chrom + '_clipper_island_summary.npy'

            island_summary_length += len(np.load(island_file_name, allow_pickle=True))

            for island in np.load(island_file_name, allow_pickle=True):
                output_line = ''
                for i in range(0, len(island)):
                    output_line += str(island[i]) + '\t'
                output_line += '\n'
                outfile.write(output_line)
                total_island_count += 1
                total_read_count += island[3]

    print("For Clipper FDR control method, given significance", str(args.false_discovery_rate),
          ", there are", total_island_count, "significant islands out of total", island_summary_length, "islands.")

    return total_island_count


def clipper_test():
    dataurl = 'http://www.stat.ucla.edu/~jingyi.li/data/Clipper/'

    experimental = pd.read_table(dataurl + 'experimental_treat_pileup.bdg', header=None)
    print(len(experimental))
    print(experimental.head())

    control = pd.read_table(dataurl + 'control_treat_pileup.bdg', header=None)
    print(len(control))
    print(control.head())

    macs2_peak = pd.read_table(dataurl + 'twosample_peaks.narrowPeak', header=None)
    print(len(macs2_peak))
    print(macs2_peak.head())

    # save
    experimental.to_csv('experimental.csv', index=False, header=False)
    control.to_csv('control.csv', index=False, header=False)
    macs2_peak.to_csv('macs2_peak.csv', index=False, header=False)

    s1 = np.repeat(experimental[3], experimental[2] - experimental[1])
    s2 = np.repeat(control[3], control[2] - control[1])

    if len(s1) < len(s2):
        s1 = np.pad(s1, (0, len(s2) - len(s1)), 'constant')

    # convert from series to numpy array
    s2 = np.array(s2)

    re = Clipper(score_exp=s1.reshape(-1, 1), score_back=s2.reshape(-1, 1), analysis="enrichment", FDR=0.05)
    # re = Clipper(score_exp=s1.reshape(-1, 1), analysis="enrichment", FDR=0.05)
    print('Threshold is', re[0])

    # 使用与Clipper相关的阈值来筛选候选峰值
    median_peak = [np.median(s1[start:end] - s2[start:end]) for start, end in zip(macs2_peak[1] + 1, macs2_peak[2])]

    start_indices = [i + 1 for i in macs2_peak[1]]
    end_indices = macs2_peak[2]
    median_peak = []
    for start, end in zip(start_indices, end_indices):
        diff_slice = s1[start:end] - s2[start:end]
        median_value = np.median(diff_slice)
        median_peak.append(median_value)

    clipper_peak = macs2_peak[median_peak >= re[0]]
    # save clipper_peak
    clipper_peak.to_csv('clipper_peak.csv', index=False)
    print('Clipper peak is', clipper_peak)


if __name__ == '__main__':
    clipper_test()