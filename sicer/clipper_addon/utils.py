import numpy as np
import pandas as pd
import warnings
from scipy.special import comb
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import os

from sicer.lib import GenomeData

def clipper1sided(score_exp=None, score_back=None, FDR=0.05, ifuseknockoff=None, nknockoff=None, contrastScore_method=None,
                  importanceScore_method='diff', FDR_control_method=None, ifpowerful=True, seed=12345, args=None):
    # Shift all measurements to be non-negative
    # if (score_exp.min() < 0) or (score_back.min() < 0):
    #     shift = min(score_exp.min(), score_back.min())
    #     score_exp = score_exp - shift
    #     score_back = score_back - shift

    # Convert score_exp and score_back to matrices if they are numerical vectors
    score_exp = np.atleast_2d(score_exp)
    score_back = np.atleast_2d(score_back)
    r1, r2 = score_exp.shape[1], score_back.shape[1]

    # Check if score_exp and score_back have the same number of instances
    if score_exp.shape[0] != score_back.shape[0]:
        raise ValueError('score_exp and score_back must have the same number of rows (features)')

    # Default: use knockoffs when r1 neq r2.
    if ifuseknockoff is None:
        ifuseknockoff = r1 != r2

    # Use clipper_1sided_woknockoff
    if not ifuseknockoff:
        if r1 != r2:
            warnings.warn(
                'Caution: no knockoffs are constructed when the numbers of replicates are different; FDR control is not guaranteed')
        if FDR_control_method == 'GZ':
            warnings.warn('FDR_control_method cannot be GZ when no knockoffs are constructed. Switching to BC.')
            FDR_control_method = 'BC'

        # Assuming clipper_1sided_woknockoff is a function defined elsewhere in the script
        re = clipper_1sided_woknockoff(score_exp=score_exp, score_back=score_back,
                                       r1=r1, r2=r2, FDR=FDR,
                                       importanceScore_method=importanceScore_method,
                                       FDR_control_method=FDR_control_method, args=args)

        # if FDR_control_method = BC or GZ but fail to identify any discovery at some FDR levels, switch to BH at those FDR levels.
        if ifpowerful and FDR_control_method != 'BH':
            FDR_nodisc = [len(re_i['discovery']) == 0 for re_i in re['results']]
            if any(FDR_nodisc):
                warnings.warn(
                    'At FDR = {}, no discovery has been found using FDR control method {}; switching to BH...'.format(
                        FDR[FDR_nodisc], FDR_control_method))
                re_clipperbh = clipper_BH(re['contrastScore'], FDR[
                    FDR_nodisc])  # Assuming clipper_BH is a function defined elsewhere in the script
                re['results'][FDR_nodisc] = re_clipperbh

    # Use clipper_1sided_wknockoff
    if ifuseknockoff:
        if r1 == 1 and r2 == 1:
            raise ValueError(
                'Cannot generate knockoffs when both score_exp and score_back have one column. Please rerun clipper1sided by setting ifuseknockoff = False')

        # Check if nknockoff is reasonable
        nknockoff_max = comb(r1 + r2, r1) // 2 - 1 if r1 == r2 else comb(r1 + r2, r1) - 1
        if nknockoff is not None:
            if not (1 <= nknockoff <= nknockoff_max and isinstance(nknockoff, int)):
                warnings.warn(
                    'nknockoff must be a positive integer and must not exceed the maximum number of knockoff; using the exhaustive knockoffs instead.')
                nknockoff = min(nknockoff, nknockoff_max)
        else:
            warnings.warn(
                'nknockoff is not supplied; generate the maximum number of knockoffs: {}'.format(nknockoff_max))
            nknockoff = nknockoff_max if contrastScore_method == "diff" else 1

        # Assuming clipper_1sided_wknockoff is a function defined elsewhere in the script
        re = clipper_1sided_wknockoff(score_exp, score_back, r1, r2, FDR, importanceScore_method, contrastScore_method,
                                      nknockoff, nknockoff_max, seed)

        # if FDR_control_method = BC or GZ but fail to identify any discovery at some FDR levels, switch to BH at those FDR levels.
        if ifpowerful and FDR_control_method != 'BH':
            FDR_nodisc = [len(re_i['discovery']) == 0 for re_i in re['results']]
            if any(FDR_nodisc):
                warnings.warn(
                    'At FDR = {}, no discovery has been found using FDR control method {}; switching to BH...'.format(
                        FDR[FDR_nodisc], FDR_control_method))
                re_clipperbh = clipper_BH(re['contrastScore'], nknockoff, FDR[
                    FDR_nodisc])  # Assuming clipper_BH is a function defined elsewhere in the script
                re['results'][FDR_nodisc] = re_clipperbh

    return re


def clipper2sided(score_exp, score_back, FDR=0.05, nknockoff=None, contrastScore_method='max',
                  importanceScore_method='diff', FDR_control_method='GZ', ifpowerful=True, seed=12345):
    # Convert score_exp and score_back to matrices if they are numerical vectors
    score_exp = np.atleast_2d(score_exp)
    score_back = np.atleast_2d(score_back)
    r1, r2 = score_exp.shape[1], score_back.shape[1]

    if r1 == 1 and r2 == 1:
        raise ValueError(
            'clipper_addon is not yet able to perform two sided identification when either condition has one replicate')

    # Check if nknockoff is reasonable
    nknockoff_max = min(comb(r1 + r2, r1) // 2 - 1 if r1 == r2 else comb(r1 + r2, r1) - 1, 200)
    if nknockoff is not None:
        if not (1 <= nknockoff <= nknockoff_max and isinstance(nknockoff, int)):
            warnings.warn(
                'nknockoff must be a positive integer and must not exceed the maximum number of knockoff; using the maximal number of knockoffs instead.')
        nknockoff = min(nknockoff, nknockoff_max)
    else:
        nknockoff = min(nknockoff_max, 50) if contrastScore_method == "diff" else 1

    # Assuming generate_knockoffidx and compute_taukappa are functions defined elsewhere in the script
    knockoffidx = generate_knockoffidx(r1, r2, nknockoff, nknockoff_max, seed)
    kappatau_ls = compute_taukappa(score_exp, score_back, r1, r2, True, knockoffidx, importanceScore_method,
                                   contrastScore_method)

    # Assuming clipper_GZ is a function defined elsewhere in the script
    re = clipper_GZ(kappatau_ls['tau'], kappatau_ls['kappa'], nknockoff, FDR)
    re = {'knockoffidx': knockoffidx, 'importanceScore_method': importanceScore_method,
          'importanceScore': kappatau_ls['importanceScore'], 'contrastScore_method': contrastScore_method,
          'contrastScore': (2 * kappatau_ls['kappa'] - 1) * np.abs(kappatau_ls['tau']), 'results': re}

    # if FDR_control_method = BC or GZ but fail to identify any discovery at some FDR levels, switch to BH at those FDR levels.
    if ifpowerful and FDR_control_method != 'BH':
        FDR_nodisc = [len(re_i['discovery']) == 0 for re_i in re['results']]
        if any(np.array(FDR_nodisc) & (contrastScore_method == 'max')):
            warnings.warn(
                'At FDR = {}, no discovery has been found using FDR control method {}; switching to BH...'.format(
                    FDR[np.array(FDR_nodisc)], FDR_control_method))
            # Assuming clipper_BH is a function defined elsewhere in the script
            re_clipperbh = clipper_BH(re['contrastScore'], nknockoff, FDR[np.array(FDR_nodisc)])
            re['results'][np.array(FDR_nodisc)] = re_clipperbh

    return re


def clipper_1sided_woknockoff(score_exp, score_back, r1, r2, FDR=0.05, aggregation_method='mean',
                              importanceScore_method=None, FDR_control_method=None, args=None):
    # Aggregate multiple replicates into single replicate
    if r1 > 1:
        # Assuming aggregate_clipper is a function defined elsewhere in the script
        score_exp = aggregate_clipper(score=score_exp, aggregation_method=aggregation_method)
    if r2 > 1:
        score_back = aggregate_clipper(score=score_back, aggregation_method=aggregation_method)

    # Assuming compute_importanceScore_wsinglerep is a function defined elsewhere in the script
    contrastscore = compute_importanceScore_wsinglerep(score_exp, score_back, importanceScore_method, args=args)

    if FDR_control_method == 'BC':
        # Assuming clipper_BC is a function defined elsewhere in the script
        re = clipper_BC(contrastScore=contrastscore, FDR=FDR, args=args)

    if FDR_control_method == 'BH':
        # Assuming clipper_BH is a function defined elsewhere in the script
        re = clipper_BH(contrastScore=contrastscore, FDR=FDR)

    re = {'importanceScore': contrastscore, 'importanceScore_method': importanceScore_method,
          'contrastScore': contrastscore, 'contrastScore_method': importanceScore_method, 'results': re}

    return re


def clipper_1sided_wknockoff(score_exp, score_back, r1, r2, FDR=0.05, importanceScore_method=None,
                             contrastScore_method=None, nknockoff=None, nknockoff_max=None, seed=None):
    # Assuming generate_knockoffidx is a function defined elsewhere in the script
    knockoffidx = generate_knockoffidx(r1, r2, nknockoff, nknockoff_max, seed)

    # Assuming compute_taukappa is a function defined elsewhere in the script
    kappatau_ls = compute_taukappa(score_exp, score_back, r1, r2, False, knockoffidx, importanceScore_method,
                                   contrastScore_method)

    # Assuming clipper_GZ is a function defined elsewhere in the script
    re = clipper_GZ(kappatau_ls['tau'], kappatau_ls['kappa'], nknockoff, FDR)
    re = {'knockoffidx': knockoffidx, 'importanceScore_method': importanceScore_method,
          'importanceScore': kappatau_ls['importanceScore'], 'contrastScore_method': contrastScore_method,
          'contrastScore': (2 * kappatau_ls['kappa'] - 1) * np.abs(kappatau_ls['tau']), 'results': re}

    return re


def aggregate_clipper(score, aggregation_method):
    if aggregation_method == 'mean':
        score_single = np.nanmean(score, axis=1)
    elif aggregation_method == 'median':
        score_single = np.nanmedian(score, axis=1)
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

    return score_single


def compute_importanceScore_wsinglerep(score_exp, score_back, importanceScore_method, args=None):
    if importanceScore_method == 'diff':
        contrastScore = score_exp - score_back
    elif importanceScore_method == 'max':
        contrastScore = np.maximum(score_exp, score_back) * np.sign(score_exp - score_back)
    else:
        raise ValueError(f"Unsupported importance score method: {importanceScore_method}")

    return contrastScore.flatten()


def clipper_BC(contrastScore, FDR, args=None):
    # supress warning
    warnings.filterwarnings("ignore")
    # Impute missing contrast scores with 0
    contrastScore = np.nan_to_num(contrastScore, nan=0)
    c_abs = np.sort(np.unique(np.abs(contrastScore[contrastScore != 0])))

    emp_fdp = np.full(len(c_abs), np.nan)
    emp_fdp[0] = 1
    for i in range(1, len(c_abs)):
        t = c_abs[i]
        ### RuntimeWarning: divide by zero encountered in scalar divide
        emp_fdp[i] = min((1 + np.sum(contrastScore <= -t)) / np.sum(contrastScore >= t), 1)
        emp_fdp[i] = min(emp_fdp[i], emp_fdp[i - 1])

    c_abs = c_abs[~np.isnan(emp_fdp)]
    emp_fdp = emp_fdp[~np.isnan(emp_fdp)]
    q = None
    # series = pd.Series(emp_fdp, index=c_abs)
    # q = series.reindex(contrastScore)
    # q = q.fillna(1).values  # unimportant, not processed yet
    # q = pd.Series(emp_fdp, index=c_abs).reindex(contrastScore).fillna(1).values  ## kill

    if not isinstance(FDR, (list, np.ndarray)):
        FDR = [FDR]

    re = []  # -------
    for FDR_i in FDR:  # -------
        # thre = c_abs[np.min(np.where(emp_fdp <= FDR_i))]
        indices = np.where(emp_fdp <= FDR_i)  # -------
        if indices[0].size == 0:
            thre = 9999999999
            # raise ValueError("Do not find valid index for the given FDR cutoff.")
        else:
            thre = c_abs[np.min(indices)]  # -------
        re_i = {'FDR': FDR_i, 'FDR_control': 'BC', 'thre': thre, 'q': q, 'discovery': np.where(contrastScore >= thre)}  # -------
        if thre == 9999999999:
            re_i['thre'] = None
        re.append(re_i)  # -------

    # supress warning stop
    warnings.filterwarnings("default")
    return re


def clipper_BH(contrastScore, nknockoff=None, FDR=None):
    if isinstance(contrastScore, dict):
        n = len(contrastScore[0])
        kappa = contrastScore['kappa']
        tau = contrastScore['tau']
        idx_na = np.isnan(tau) | np.isnan(kappa)
        tau, kappa = tau[~idx_na], kappa[~idx_na]
        pval = [np.sum(np.logical_not(kappa) & tau >= tau[i],
                       where=~np.isnan(tau)) / np.sum(np.logical_not(kappa),
                                                      where=~np.isnan(tau)) * nknockoff / (nknockoff + 1) for i in
                range(n)]
    else:
        n = len(contrastScore)
        idx_na = np.isnan(contrastScore)
        contrastScore_nomiss = contrastScore[~idx_na]
        cs_negative = contrastScore_nomiss[contrastScore_nomiss < 0]
        cs_null = np.concatenate((cs_negative, -cs_negative))
        pval = [np.mean(x <= cs_null) for x in contrastScore_nomiss]

    qvalue = multipletests(pval, method='fdr_bh')[1]

    re = [{'FDR': FDR_i, 'FDR_control': 'BH', 'discovery': np.where(~idx_na)[0][np.where(qvalue <= FDR_i)], 'q': qvalue}
          for FDR_i in FDR]

    return re


def generate_knockoffidx(r1, r2, nknockoff, nknockoff_max, seed):
    np.random.seed(seed)

    if nknockoff_max == 200:
        knockoffidx = []
        i_knockoff = 0
        while i_knockoff < nknockoff:
            temp = np.random.choice(r1 + r2, r1, replace=False) + 1
            if (any(~np.isin(temp, np.arange(1, r1 + 1))) and any(np.isin(temp, np.arange(1, r1 + 1)))):
                knockoffidx.append(temp)
                i_knockoff += 1
    else:
        combination_all = list(combinations(np.arange(r1 + r2) + 1, r1))

        if r1 == r2:
            combination_all = combination_all[:len(combination_all) // 2][1:]
        else:
            combination_all = combination_all[1:]

        knockoffidx = np.random.choice(combination_all, nknockoff, replace=False)

    return knockoffidx


def compute_taukappa(score_exp, score_back, r1, r2, if2sided, knockoffidx, importanceScore_method,
                     contrastScore_method):
    perm_idx = [np.arange(r1)] + knockoffidx
    score_tot = np.concatenate((score_exp, score_back), axis=1)

    imp_ls = []
    for x in perm_idx:
        se = score_tot[:, x]
        sb = score_tot[:, np.setdiff1d(np.arange(r1 + r2), x)]
        se = aggregate_clipper(se, aggregation_method='mean')
        sb = aggregate_clipper(sb, aggregation_method='mean')
        imp = compute_importanceScore_wsinglerep(se, sb, importanceScore_method)
        if if2sided:
            imp = np.abs(imp)
        imp_ls.append(imp)

    kappatau_ls = []
    for x in np.array(imp_ls).T:
        kappa = not any(x[1:] == max(x))  # kappa should be true iff the maximum occurs only at the first index
        if len(kappa) == 0:
            kappa = np.nan
        x_sorted = np.sort(x)[::-1]
        if contrastScore_method == 'diff':
            tau = x_sorted[0] - x_sorted[1]
        elif contrastScore_method == 'max':
            tau = x_sorted[0]
        else:
            raise ValueError(f"Unsupported contrast score method: {contrastScore_method}")
        kappatau_ls.append({'kappa': kappa, 'tau': tau})

    re = {'importanceScore': imp_ls,
          'kappa': [x['kappa'] for x in kappatau_ls],
          'tau': [x['tau'] for x in kappatau_ls]}

    return re


def clipper_GZ(tau, kappa, nknockoff, FDR):
    n = len(tau)
    contrastScore = (2 * kappa - 1) * np.abs(tau)
    contrastScore = np.nan_to_num(contrastScore, nan=0)
    c_abs = np.sort(np.unique(np.abs(contrastScore[contrastScore != 0])))

    emp_fdp = np.empty(len(c_abs))
    emp_fdp[0] = 1
    for i in range(1, len(c_abs)):
        t = c_abs[i]
        emp_fdp[i] = min((1 / nknockoff + 1 / nknockoff * np.sum(contrastScore <= -t)) / np.sum(contrastScore >= t), 1)
        emp_fdp[i] = min(emp_fdp[i], emp_fdp[i - 1])

    c_abs = c_abs[~np.isnan(emp_fdp)]
    emp_fdp = emp_fdp[~np.isnan(emp_fdp)]
    q = pd.Series(emp_fdp, index=c_abs).reindex(contrastScore, method='nearest').fillna(1).values

    re = []
    for FDR_i in FDR:
        thre = c_abs[np.min(np.where(emp_fdp <= FDR_i))]
        re_i = {'FDR': FDR_i, 'FDR_control': 'BC', 'thre': thre, 'discovery': np.where(contrastScore >= thre), 'q': q}
        re.append(re_i)

    return re


def island_bdg_union(args, filtered, indicator, chroms):
    """
    island_bdg_union: union two bdg files,
    make sure the regions are one-to-one mapping and at least non-zero in one of the counterparts
    ctrl_bdg: a bedgraph file generated from control group
    treat_bdg: a bedgraph file generated from treatment group

    Return: directly save the unioned bedgraph file to save_path
    """

    whole_genome_union = pd.DataFrame(columns=['col1', 'col2', 'col3', 'ctrl', 'treat'])

    for chrom in chroms:
        treat_file = args.treatment_file.replace('.bed', '') + f'_{chrom}'
        ctrl_file = args.control_file.replace('.bed', '') + f'_{chrom}'

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

        if treat_bdg.shape[0] == 0 and ctrl_bdg.shape[0] == 0:
            continue
        elif treat_bdg.shape[0] == 0:
            treat_bdg = pd.DataFrame(columns=['col1', 'col2', 'col3', 'treat'])
            ctrl_bdg = pd.DataFrame(ctrl_bdg[ctrl_bdg[:, 3] != 0], columns=['col1', 'col2', 'col3', 'ctrl'])
        elif ctrl_bdg.shape[0] == 0:
            ctrl_bdg = pd.DataFrame(columns=['col1', 'col2', 'col3', 'ctrl'])
            treat_bdg = pd.DataFrame(treat_bdg[treat_bdg[:, 3] != 0], columns=['col1', 'col2', 'col3', 'treat'])
        else:
            treat_bdg = pd.DataFrame(treat_bdg[treat_bdg[:, 3] != 0], columns=['col1', 'col2', 'col3', 'treat'])
            ctrl_bdg = pd.DataFrame(ctrl_bdg[ctrl_bdg[:, 3] != 0], columns=['col1', 'col2', 'col3', 'ctrl'])

        ctrl_bdg = ctrl_bdg.astype({"col1": str, "col2": int, "col3": int, "ctrl": int})
        treat_bdg = treat_bdg.astype({"col1": str, "col2": int, "col3": int, "treat": int})
        merged_df_outer_join = pd.merge(ctrl_bdg, treat_bdg, how='outer', on=['col1', 'col2', 'col3']).fillna(0)
        merged_df_outer_join = merged_df_outer_join.sort_values(by=['col2'])
        merged_df_outer_join['col2'] = merged_df_outer_join['col2']


        whole_genome_union = pd.concat([whole_genome_union, merged_df_outer_join], axis=0)

        # del first second and third columns
        # merged_df_outer_join = merged_df_outer_join.iloc[:, 3:]
        merged_df_outer_join = merged_df_outer_join.to_numpy()
        # save as temp npu file
        name_for_save = args.treatment_file.replace('.bed', '') + '_' + args.control_file.replace('.bed', '') + '_' + \
                        f'{indicator}_{chrom}' + '_union.npy'

        np.save(name_for_save, merged_df_outer_join)

    outfile_path = os.path.join(args.output_directory, args.treatment_file.replace('.bed', '') + '-' +
                                args.treatment_file.replace('.bed', '') + '-' +
                                f'{indicator}_union.bed')
    # save as a bed file
    whole_genome_union.to_csv(outfile_path, sep='\t', header=False, index=False)
    print(f'{indicator} union bedgraph file created')
