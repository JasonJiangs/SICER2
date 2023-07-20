import numpy as np
import warnings
from utils import clipper2sided, clipper1sided


def clipper(score_exp, score_back, analysis, FDR=0.05,
            procedure=None, contrast_score=None,
            n_permutation=None, seed=12345):
    analysis_choices = ['differential', 'enrichment']
    assert analysis in analysis_choices, f"Invalid value for analysis: {analysis}. Must be one of: {analysis_choices}"

    if analysis == 'differential':
        contrast_score_choices = ['diff', 'max']
        contrast_score = 'max' if contrast_score is None else contrast_score
        assert contrast_score in contrast_score_choices, f"Invalid value for contrast_score: {contrast_score}. Must be one of: {contrast_score_choices}"

        procedure_choices = ['BC', 'aBH', 'GZ']
        procedure = 'GZ' if procedure is None else procedure
        assert procedure in procedure_choices, f"Invalid value for procedure: {procedure}. Must be one of: {procedure_choices}"

        re = clipper2sided(score_exp, score_back, FDR, n_permutation,
                           contrast_score, 'diff', procedure, False, seed)

        FDR_nodisc = [len(re_i['discovery']) == 0 for re_i in re['results']]
        if any(FDR_nodisc) and contrast_score == 'max':
            warnings.warn(
                'At FDR = {}, no discovery has been found using max contrast score. To make more discoveries, switch to diff contrast score or increase the FDR threshold.'.format(
                    ', '.join(map(str, FDR))))

    if np.ndim(score_exp) == 1:
        score_exp = np.reshape(score_exp, (-1, 1))

    if np.ndim(score_back) == 1:
        score_back = np.reshape(score_back, (-1, 1))

    if analysis == 'enrichment':
        procedure_choices = ['BC', 'aBH', 'GZ']
        procedure = 'BC' if procedure is None else procedure
        assert procedure in procedure_choices, f"Invalid value for procedure: {procedure}. Must be one of: {procedure_choices}"

        if score_exp.shape[1] != score_back.shape[1]:
            procedure = 'GZ'

        contrast_score_choices = ['diff', 'max']
        if procedure == 'BC':
            contrast_score = 'diff'
        elif procedure == 'GZ':
            contrast_score = 'max'

        contrast_score = 'max' if contrast_score is None else contrast_score
        assert contrast_score in contrast_score_choices, f"Invalid value for contrast_score: {contrast_score}. Must be one of: {contrast_score_choices}"

        re = clipper1sided(score_exp, score_back, FDR, n_permutation,
                           contrast_score, procedure, False, seed)

        FDR_nodisc = [len(re_i['discovery']) == 0 for re_i in re['results']]
        if any(FDR_nodisc) and procedure != 'aBH':
            warnings.warn(
                'At FDR = {}, no discovery has been found using current procedure. To make more discoveries, switch to aBH procedure or increase the FDR threshold.'.format(
                    ', '.join(map(str, FDR))))

    contrast_score_value = re['contrastScore']
    thre = [re_i['thre'] for re_i in re['results']]
    discoveries = [re_i['discovery'] for re_i in re['results']]
    q = re['results'][0]['q']

    re = {'contrast.score': contrast_score,
          'contrast.score.value': contrast_score_value,
          'FDR': FDR,
          'contrast.score.thre': thre,
          'discoveries': discoveries,
          'q': q}
    return re