import numpy as np
import pandas as pd

from functools import partial


def clipper1sided(score_exp=None, score_back=None, FDR=0.05, ifuseknockoff=None, nknockoff=None,
                  contrastScore_method=None, importanceScore_method='diff', FDR_control_method=None,
                  ifpowerful=True, seed=12345, args=None, chroms=None,pool=None):
    if not ifuseknockoff:
        re = clipper_1sided_woknockoff(score_exp=None, score_back=None,
                                       r1=None, r2=None, FDR=FDR,
                                       importanceScore_method=importanceScore_method,
                                       FDR_control_method=FDR_control_method, args=args, chroms=chroms, pool=pool)
    return re


def clipper_1sided_woknockoff(score_exp=None, score_back=None, r1=None, r2=None, FDR=0.05, aggregation_method='mean',
                              importanceScore_method=None, FDR_control_method=None, args=None,
                              chroms=None, pool=None):
    contrastscore = compute_importanceScore_wsinglerep(score_exp=None, score_back=None,
                                                       importanceScore_method=importanceScore_method, args=args,
                                                       chroms=chroms, pool=pool)

    if FDR_control_method == 'BC':
        re = clipper_BC(contrastScore=contrastscore, FDR=FDR, args=args, chroms=chroms, pool=pool)

    re = {'importanceScore': contrastscore, 'importanceScore_method': importanceScore_method,
          'contrastScore': contrastscore, 'contrastScore_method': importanceScore_method, 'results': re}

    return re


def compute_importanceScore_wsinglerep(score_exp, score_back, importanceScore_method,
                                       args=None, chroms=None, pool=None):
    if importanceScore_method == 'diff':
        compute_importance_score_partial = partial(compute_importance_score, args)
        pool.map(compute_importance_score_partial, chroms)
    else:
        raise ValueError(f"Unsupported importance score method: {importanceScore_method}")


def compute_importance_score(args, chrom):

    chrom_island_load_name = args.treatment_file.replace('.bed', '') + '_' + args.control_file.replace('.bed', '') + '_' + \
                    f'reads_{chrom}' + '_union.npy'
    try:
        chrom_island_load = np.load(chrom_island_load_name, allow_pickle=True)
    except:
        return

    chrom_island_load = np.c_[chrom_island_load, np.zeros(len(chrom_island_load))]
    chrom_island_load[:, 5] = chrom_island_load[:, 3] - chrom_island_load[:, 4]
    np.save(chrom_island_load_name, chrom_island_load)


def clipper_BC(contrastScore=None, FDR=0.05, args=None, chroms=None, pool=None):
    # contrastScore = np.nan_to_num(contrastScore, nan=0)
    # c_abs = np.sort(np.unique(np.abs(contrastScore[contrastScore != 0])))
    # TODO: can be parallelized
    # empty np array to store c_abs
    c_abs = np.array([])
    for chrom in chroms:
        try:
            union_read_file_name = args.treatment_file.replace('.bed', '') + '_' + args.control_file.replace('.bed', '') + \
                               '_' + f'reads_{chrom}' + '_union.npy'
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


def island_bdg_union_partial(args, indicator, filtered, chrom):
    treat_file = args.treatment_file.replace('.bed', '') + f'_{chrom}'
    ctrl_file = args.control_file.replace('.bed', '') + f'_{chrom}'
    island_file = args.treatment_file.replace('.bed', '') + f'_{chrom}_island_summary.npy'

    if filtered:
        treat_file += '_filtered'
        ctrl_file += '_filtered'
    else:
        treat_file += '_graph.npy'
        ctrl_file += '_graph.npy'

    if indicator == 'reads':
        treat_file = 'reads_' + treat_file
        ctrl_file = 'reads_' + ctrl_file

    treat_bdg = np.load(treat_file, allow_pickle=True)
    ctrl_bdg = np.load(ctrl_file, allow_pickle=True)

    try:  # TODO: no 'CUTLL-H3K27me3-SRR243558.sort_chrM_island_summary.npy' found when the window size is large, e.g. 50000, 100000
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
    name_for_save_union_read = args.treatment_file.replace('.bed', '') + '_' + args.control_file.replace('.bed',
                                                                                                         '') + '_' + \
                               f'{indicator}_{chrom}' + '_union.npy'
    np.save(name_for_save_union_read, index_bp_conversion(merged_df_outer_join, window_size=args.window_size).to_numpy())

    merged_df_outer_join = insert_gaps(merged_df_outer_join)
    merged_df_outer_join = index_bp_conversion(merged_df_outer_join, window_size=args.window_size)
    island_bdg = index_bp_conversion(island_bdg, window_size=args.window_size)

    name_for_save_union_read = args.treatment_file.replace('.bed', '') + '_' + args.control_file.replace('.bed',
                                                                                                         '') + '_' + \
                               f'{indicator}_{chrom}' + '_union_sup.npy'
    name_for_save_island = args.treatment_file.replace('.bed', '') + '_' + args.control_file.replace('.bed', '') + '_' + \
                           f'{indicator}_{chrom}' + '_island_summary_clipper.npy'

    np.save(name_for_save_island, island_bdg.to_numpy())
    np.save(name_for_save_union_read, merged_df_outer_join.to_numpy())


def island_bdg_union(args, filtered, indicator, chroms, pool):
    """
    island_bdg_union: union two bdg files,
    make sure the regions are one-to-one mapping and at least non-zero in one of the counterparts
    ctrl_bdg: a bedgraph file generated from control group
    treat_bdg: a bedgraph file generated from treatment group

    Return: directly save the unioned bedgraph file to save_path
    """
    island_bdg_union_pl = partial(island_bdg_union_partial, args, indicator, filtered)
    pool.map(island_bdg_union_pl, chroms)


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


def index_bp_conversion(bdg_file, window_size=200):
    return bdg_file.assign(
        start=(bdg_file['start'] / window_size).astype(int),
        end=((bdg_file['end'] + 1) / window_size).astype(int)
    )


def index_bp_conversion_back(bdg_file, window_size=200):
    return bdg_file.assign(
        start=(bdg_file['start'] * window_size).astype(int),
        end=((bdg_file['end'] * window_size) - 1).astype(int)
    )


def index_append_conversion(bdg_file, value_append):
    return bdg_file.assign(
        start=bdg_file['start'] + value_append,
        end=bdg_file['end'] + value_append
    )


# def sum_contrast_score(args, chroms, t, compare):
#     sum_value = 0
#
#     comparator = {
#         'lr': lambda x: x >= t,
#         'sl': lambda x: x <= -t,
#     }.get(compare, lambda x: False)
#
#     for chrom in chroms:
#         union_read_file_name = args.treatment_file.replace('.bed', '') + '_' + args.control_file.replace('.bed', '') + \
#                                '_' + f'reads_{chrom}' + '_union.npy'
#         try:
#             union_read_file = np.load(union_read_file_name, allow_pickle=True)
#         except:
#             continue
#
#         sum_value += np.sum(comparator(union_read_file[:, 5]))
#
#     return sum_value


def process_chrom(args, t, compare, chrom):
    union_read_file_name = args.treatment_file.replace('.bed', '') + '_' + args.control_file.replace('.bed', '') + \
                           '_' + f'reads_{chrom}' + '_union.npy'
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
        print(f"An error occurred: {e}")
        return 0


def sum_contrast_score(args, chroms, t, compare, pool):
    partial_process_chrom = partial(process_chrom, args, t, compare)
    sum_values = pool.map(partial_process_chrom, chroms)

    return np.sum(sum_values)


def get_discovery(args, chroms, thre):
    discovery = []
    for chrom in chroms:
        union_read_file_name = args.treatment_file.replace('.bed', '') + '_' + args.control_file.replace('.bed', '') + \
                               '_' + f'reads_{chrom}' + '_union.npy'
        try:
            union_read_file = np.load(union_read_file_name, allow_pickle=True)
            union_read_file[:, 1:6] = union_read_file[:, 1:6].astype(np.int32)

        except:
            continue

        discovery.append(np.where(union_read_file[:, 5] >= thre))

        if len(discovery) != 0:
            return discovery

    return None
