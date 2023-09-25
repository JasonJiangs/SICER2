import pandas as pd
import numpy as np
import warnings

from functools import partial
from sicer.lib import GenomeData
import os

def expand_intervals(data_array):
    # expanded data use ndarray to store
    expanded_data = []

    for row in data_array:
        chrom, start, end, val1, val2, val3 = row
        start, end = int(start), int(end)

        length = end - start

        for i in range(length):
            new_start = start + i
            new_end = new_start + 1
            expanded_data.append([chrom, int(new_start), int(new_end), float(val1), float(val2), float(val3)])

    return np.array(expanded_data, dtype=object)

def island_bdg_union_partial(args, chrom):
    """
    Preprocess the island summary file for Clipper
    """
    try:
        island_file = args.treatment_file.replace('.bed', '') + f'_{chrom}_island_summary.npy'
        island_bdg = pd.DataFrame(np.load(island_file, allow_pickle=True)[:, :5],
                              columns=['chrom', 'start', 'end', 'treat', 'ctrl']).astype(
                                {'chrom': str, 'start': int, 'end': int, 'treat': int, 'ctrl': int})
    except:
        return
    island_bdg = index_bp_conversion(island_bdg, window_size=args.window_size)
    # save to npy file
    np.save(island_file.replace('_island_summary.npy', '_island_summary_clipper.npy'), island_bdg.to_numpy())


def island_bdg_union(args, chroms, pool):
    """
    island_bdg_union: union two bdg files,
    make sure the regions are one-to-one mapping and at least non-zero in one of the counterparts
    ctrl_bdg: a bedgraph file generated from control group
    treat_bdg: a bedgraph file generated from treatment group

    Args:
        args: arguments from command line
        chroms: list of chromosomes
        pool: multiprocessing pool
    """
    island_bdg_union_pl = partial(island_bdg_union_partial, args)
    pool.map(island_bdg_union_pl, chroms)


def index_bp_conversion(bdg_file, window_size=200):
    """
    Scale down the index by window size to reduce the computation burden for Clipper
    """
    return bdg_file.assign(
        start=(bdg_file['start'] / window_size).astype(int),
        end=((bdg_file['end'] + 1) / window_size).astype(int)
    )

def process_chrom(args, t, compare, chrom):
    # union_read_file_name = args.treatment_file.replace('.bed', '') + f'_{chrom}_island_summary_clipper.npy'
    union_read_file_name = args.treatment_file.replace('.bed', '') + '_' + \
                            args.control_file.replace('.bed', '') + '_' + f'reads_{chrom}' + '_union.npy'
    try:
        union_read_file = np.load(union_read_file_name, allow_pickle=True)
        union_read_file[:, 1:6] = union_read_file[:, 1:6].astype(np.int32)

        if compare == 'lr':
            return np.sum(union_read_file[:, 5] >= t)
        elif compare == 'sl':
            return np.sum(union_read_file[:, 5] <= -t)
        else:
            return 0
    except Exception as e:
        # print(f"An error occurred: {e}")
        return 0


def sum_contrast_score(args, chroms, t, compare, pool):
    partial_process_chrom = partial(process_chrom, args, t, compare)
    sum_values = pool.map(partial_process_chrom, chroms)

    return np.sum(sum_values)


def get_discovery(args, chroms, thre):
    discovery = []
    for chrom in chroms:
        # union_read_file_name = args.treatment_file.replace('.bed', '') + f'_{chrom}_island_summary_clipper.npy'
        union_read_file_name = args.treatment_file.replace('.bed', '') + '_' + \
                                args.control_file.replace('.bed', '') + '_' + f'reads_{chrom}' + '_union.npy'
        try:
            union_read_file = np.load(union_read_file_name, allow_pickle=True)
            union_read_file[:, 1:6] = union_read_file[:, 1:6].astype(np.int32)

        except:
            continue

        discovery.append(np.where(union_read_file[:, 5] >= thre))

        if len(discovery) != 0:
            return discovery

    return None


def clipper_BC(contrastScore=None, FDR=0.05, args=None, chroms=None, pool=None):
    # TODO: can be parallelized
    c_abs = np.array([])
    for chrom in chroms:
        try:
            # union_read_file_name = args.treatment_file.replace('.bed', '') + f'_{chrom}_island_summary_clipper.npy'
            union_read_file_name = args.treatment_file.replace('.bed', '') + '_' + \
                                     args.control_file.replace('.bed', '') + '_' + f'reads_{chrom}' + '_union.npy'
            union_read_file = np.load(union_read_file_name, allow_pickle=True)
        except:
            continue
        contrast_score = np.nan_to_num(union_read_file[:, 5], nan=0)
        c_abs = np.append(c_abs, np.sort(np.unique(np.abs(contrast_score[contrast_score != 0]))))

    c_abs = np.sort(np.unique(c_abs))

    emp_fdp = np.full(len(c_abs), np.nan)
    emp_fdp[0] = 1
    for i in range(1, len(c_abs)):
        t = c_abs[i]
        emp_fdp[i] = min((1 + sum_contrast_score(args, chroms, t, 'sl', pool)) /
                         sum_contrast_score(args, chroms, t, 'lr', pool), 1)
        emp_fdp[i] = min(emp_fdp[i], emp_fdp[i - 1])

    c_abs = c_abs[~np.isnan(emp_fdp)]
    emp_fdp = emp_fdp[~np.isnan(emp_fdp)]
    q = None

    re = []
    indices = np.where(emp_fdp <= FDR)
    if indices[0].size == 0:
        raise ValueError("Do not find valid index for the given FDR cutoff.")
    else:
        thre = c_abs[np.min(indices)]
    re_i = {'FDR': FDR, 'FDR_control': 'BC', 'thre': thre, 'q': q, 'discovery': get_discovery(args, chroms, thre)}
    re.append(re_i)

    return re

def compute_importance_score(args, chrom):
    # chrom_island_load_name = args.treatment_file.replace('.bed', '') + f'_{chrom}_island_summary_clipper.npy'
    chrom_island_load_name = args.treatment_file.replace('.bed', '') + '_' + \
                                args.control_file.replace('.bed', '') + '_' + f'reads_{chrom}' + '_union.npy'
    try:
        chrom_island_load = np.load(chrom_island_load_name, allow_pickle=True)
    except:
        return
    chrom_island_load = np.c_[chrom_island_load, np.zeros(len(chrom_island_load))]
    chrom_island_load[:, 5] = chrom_island_load[:, 3] - chrom_island_load[:, 4]
    # unfold the island to one by one window sized region
    unfold_chrom_island_load = expand_intervals(chrom_island_load)
    np.save(chrom_island_load_name, unfold_chrom_island_load)


def compute_importanceScore_wsinglerep(score_exp, score_back, importanceScore_method,
                                       args=None, chroms=None, pool=None):
    if importanceScore_method == 'diff':
        compute_importance_score_partial = partial(compute_importance_score, args)
        pool.map(compute_importance_score_partial, chroms)
    else:
        raise ValueError(f"Unsupported importance score method: {importanceScore_method}")

def clipper_1sided_woknockoff(score_exp=None, score_back=None, r1=None, r2=None, FDR=0.05, aggregation_method='mean',
                              importanceScore_method=None, FDR_control_method=None, args=None,
                              chroms=None, pool=None):
    contrast_score = compute_importanceScore_wsinglerep(score_exp=None, score_back=None,
                                                       importanceScore_method=importanceScore_method, args=args,
                                                       chroms=chroms, pool=pool)

    if FDR_control_method == 'BC':
        re = clipper_BC(contrastScore=contrast_score, FDR=FDR, args=args, chroms=chroms, pool=pool)

    re = {'importanceScore': contrast_score, 'importanceScore_method': importanceScore_method,
          'contrastScore': contrast_score, 'contrastScore_method': importanceScore_method, 'results': re}

    return re


def clipper1sided(score_exp=None, score_back=None, FDR=0.05, ifuseknockoff=None, nknockoff=None,
                  contrastScore_method=None, importanceScore_method='diff', FDR_control_method=None,
                  ifpowerful=True, seed=12345, args=None, chroms=None,pool=None):
    if not ifuseknockoff:
        re = clipper_1sided_woknockoff(score_exp=None, score_back=None,
                                       r1=None, r2=None, FDR=FDR,
                                       importanceScore_method=importanceScore_method,
                                       FDR_control_method=FDR_control_method, args=args, chroms=chroms, pool=pool)
    return re


def clipper_threshold(analysis='enrichment', FDR=0.05, procedure=None, contrast_score=None,
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

def insert_gaps(table):
    # table to a list
    table = table.to_numpy()
    new_table = []

    # Insert starting row from 0 to the start of the first row, if it doesn't start at 0
    if table[0][1] != 0:
        new_table.append((table[0][0], 0, table[0][1]-1, 0, 0))

    for i in range(len(table)):
        new_table.append(table[i])
        if i < len(table) - 1 and table[i][0] == table[i + 1][0]:  # same chromosome
            end_current = table[i][2]
            start_next = table[i + 1][1]
            if end_current + 1 < start_next:  # check for gap
                new_row = (table[i][0], end_current+1, start_next-1, 0, 0)
                new_table.append(new_row)
    new_table = pd.DataFrame(new_table, columns=['chrom', 'start', 'end', 'treat', 'ctrl']).astype(
        {'chrom': str, 'start': int, 'end': int, 'treat': int, 'ctrl': int})
    return new_table

def filter_and_expand_rows(base_df, filter_df):
    expanded_rows = []

    for _, filter_row in filter_df.iterrows():
        starts = np.arange(filter_row['start'], filter_row['end'])
        ends = starts + 1

        mask = base_df['start'].isin(starts) & base_df['end'].isin(ends)
        matched_rows = base_df[mask]

        expanded_rows.extend(matched_rows.values)

    expanded_df = pd.DataFrame(expanded_rows, columns=base_df.columns)
    return expanded_df

def islands_bdg_union_partial(args, chrom):
    treat_file = args.treatment_file.replace('.bed', '') + f'_{chrom}' + '_graph.npy'
    ctrl_file = args.control_file.replace('.bed', '') + f'_{chrom}' + '_graph.npy'
    island_file = args.treatment_file.replace('.bed', '') + f'_{chrom}_island_summary.npy'

    treat_file = 'reads_' + treat_file
    ctrl_file = 'reads_' + ctrl_file

    treat_bdg = np.load(treat_file, allow_pickle=True)
    ctrl_bdg = np.load(ctrl_file, allow_pickle=True)

    try:  # TODO: no 'CUTLL-H3K27me3-SRR243558.sort_chrM_island_summary.npy' found when the window size is large, e.g. W50000, 100000
        island_bdg = pd.DataFrame(np.load(island_file, allow_pickle=True)[:, :5],
                                  columns=['chrom', 'start', 'end', 'treat', 'ctrl']).astype(
            {'chrom': str, 'start': int, 'end': int, 'treat': int, 'ctrl': int})
    except:
        return

    if treat_bdg.shape[0] == 0 and ctrl_bdg.shape[0] == 0:  # this chromosome is empty
        return
    elif treat_bdg.shape[0] == 0:
        treat_bdg = pd.DataFrame(columns=['chrom', 'start', 'end', 'treat'])
        ctrl_bdg = pd.DataFrame(ctrl_bdg[ctrl_bdg[:, 3] != 0], columns=['chrom', 'start', 'end', 'ctrl'])
    elif ctrl_bdg.shape[0] == 0:
        ctrl_bdg = pd.DataFrame(columns=['chrom', 'start', 'end', 'ctrl'])
        treat_bdg = pd.DataFrame(treat_bdg[treat_bdg[:, 3] != 0], columns=['chrom', 'start', 'end', 'treat'])
    else:
        treat_bdg = pd.DataFrame(treat_bdg[treat_bdg[:, 3] != 0], columns=['chrom', 'start', 'end', 'treat'])
        ctrl_bdg = pd.DataFrame(ctrl_bdg[ctrl_bdg[:, 3] != 0], columns=['chrom', 'start', 'end', 'ctrl'])

    ctrl_bdg = ctrl_bdg.astype({"chrom": str, "start": int, "end": int, "ctrl": int})
    treat_bdg = treat_bdg.astype({"chrom": str, "start": int, "end": int, "treat": int})
    merged_df_outer_join = pd.merge(ctrl_bdg, treat_bdg, how='outer', on=['chrom', 'start', 'end']).fillna(0)
    merged_df_outer_join = merged_df_outer_join.sort_values(by=['start'])
    # reindex
    merged_df_outer_join.index = range(merged_df_outer_join.shape[0])
    merged_df_outer_join = merged_df_outer_join[['chrom', 'start', 'end', 'treat', 'ctrl']]
    # insert gaps and convert index to bp
    merged_df_outer_join = insert_gaps(merged_df_outer_join)
    merged_df_outer_join = index_bp_conversion(merged_df_outer_join, window_size=args.window_size)
    island_bdg = index_bp_conversion(island_bdg, window_size=args.window_size)

    # name_for_save_union_read = args.treatment_file.replace('.bed', '') + '_' + args.control_file.replace('.bed','') + \
    #                            '_' + f'reads_{chrom}' + '_union.npy'
    name_for_save_island = args.treatment_file.replace('.bed', '') + '_' + args.control_file.replace('.bed', '') + '_' + \
                           f'read_{chrom}' + '_island_summary_clipper_converted.npy'
    # TODO: might can be deleted
    np.save(name_for_save_island, island_bdg.to_numpy())
    # np.save(name_for_save_union_read, merged_df_outer_join.to_numpy())

    save_union_read_candidate_island_wide = args.treatment_file.replace('.bed', '') + '_' + \
                                            args.control_file.replace('.bed','') + '_' + f'reads_{chrom}' + '_union.npy'

    union_read_candidate_island_wide = filter_and_expand_rows(merged_df_outer_join, island_bdg)
    # save to npy file
    np.save(save_union_read_candidate_island_wide, union_read_candidate_island_wide.to_numpy())


def islands_bdg_union(args, chroms, pool):
    """
    island_bdg_union: union two bdg files,
    make sure the regions are one-to-one mapping and at least non-zero in one of the counterparts
    ctrl_bdg: a bedgraph file generated from control group
    treat_bdg: a bedgraph file generated from treatment group

    Return: directly save the unioned bedgraph file to save_path
    """
    island_bdg_union_pl = partial(islands_bdg_union_partial, args)
    pool.map(island_bdg_union_pl, chroms)


def index_bp_conversion_back(bdg_file, window_size=200):
    return bdg_file.assign(
        start=(bdg_file['start'] * window_size).astype(int),
        end=((bdg_file['end'] * window_size) - 1).astype(int)
    )


def filter_rows_within_range(data_df, start, end):
    filtered_df = data_df[(data_df['start'] < end) & (data_df['end'] > start)]
    return filtered_df

def main(args, pool):
    chroms = GenomeData.species_chroms[args.species]
    import time
    start_time = time.time()
    # union reads
    islands_bdg_union(args, chroms, pool)
    print("--- %s seconds ---" % (time.time() - start_time))

    # treatment_file.replace('.bed', '') + f'_{chrom}_island_summary.npy' -> ’_island_summary_clipper.npy‘
    island_bdg_union(args, chroms, pool)  # TODO: need to be revisited

    # calculate threshold
    threshold = clipper_threshold(args=args, chroms=chroms, FDR=args.false_discovery_rate, pool=pool)
    print('Clipper threshold is: ', threshold)  # TODO: clearify the contrast score calculation method

    # filter the islands by threshold
    for chrom in chroms:
        chrom_island_load = args.treatment_file.replace('.bed', '') + '_' + args.control_file.replace('.bed', '') + '_' + \
                           f'read_{chrom}' + '_island_summary_clipper_converted.npy'
        chrom_read_union_load = args.treatment_file.replace('.bed', '') + '_' + \
                                args.control_file.replace('.bed', '') + '_' + f'reads_{chrom}' + '_union.npy'
        try:
            chrom_island_list = pd.DataFrame(np.load(chrom_island_load, allow_pickle=True),
                                             columns=['chrom', 'start', 'end', 'treat', 'ctrl']).astype(
                {'chrom': str, 'start': int, 'end': int, 'treat': int, 'ctrl': int})
            read_union = pd.DataFrame(np.load(chrom_read_union_load, allow_pickle=True),
                                      columns=['chrom', 'start', 'end', 'treat', 'ctrl', 'diff']).astype(
                {'chrom': str, 'start': int, 'end': int, 'treat': int, 'ctrl': int, 'diff': int})

        except:
            continue

        # filter and find clipper filtered peak
        clipper_peak_save = args.treatment_file.replace('.bed', '') + '_' + args.control_file.replace('.bed', '') + \
                            '_' + f'read_{chrom}' + '_island_clipper_filtered.npy'

        filtered_island_list = []
        for i in range(len(chrom_island_list)):
            filtered_reads = filter_rows_within_range(read_union, chrom_island_list.iloc[i]['start'],
                                                        chrom_island_list.iloc[i]['end'])
            # calculate the mean of difference between treatment and control
            mean_value = np.sum(filtered_reads['treat'] - filtered_reads['ctrl'])  # TODO: mean or sum
            if mean_value >= threshold:
                filtered_island_list.append(chrom_island_list.iloc[i])

        filtered_island_list = pd.DataFrame(filtered_island_list)
        np.save(clipper_peak_save, filtered_island_list.to_numpy())


    # save output filer for clipper filtered island
    island_summary_length = 0
    outfile_name = (args.treatment_file.replace('.bed', '') + '-W' + str(args.window_size) + '-G'
                    + str(args.gap_size) + '-ClipperFDR' + str(args.false_discovery_rate) + '-island.bed')
    outfile_path = os.path.join(args.output_directory, outfile_name)

    with open(outfile_path, 'w') as outfile:
        for chrom in chroms:
            island_file_name = args.treatment_file.replace('.bed', '') + '_' + args.control_file.replace('.bed', '') + \
                               '_' + f'read_{chrom}' + '_island_clipper_filtered.npy'
            try:
                filtered_island = pd.DataFrame((np.load(island_file_name, allow_pickle=True)),
                                               columns=['chrom', 'start', 'end', 'treat', 'ctrl']).astype(
                    {'chrom': str, 'start': int, 'end': int, 'treat': int, 'ctrl': int})
            except:
                continue

            island_summary_length += len(filtered_island)
            filtered_island = index_bp_conversion_back(filtered_island, window_size=args.window_size).to_numpy()

            for island in filtered_island:
                output_line = ''
                for i in range(0, len(island)):
                    output_line += str(island[i]) + '\t'
                output_line += '\n'
                outfile.write(output_line)

    return island_summary_length

