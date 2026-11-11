def duration_in_minutes(frame, feature):
    """Use the unchanged linear backend with explicit minute input/output labels.

The linear backend expects columns ending _s. Here those two temporary
backend slots contain minutes, and all returned column/reference/formula labels
are renamed. No sample or outcome value is changed.
"""
    import numpy as np
    from .sensitivity import fit_hrv_duration_adjusted
    for field in ('pre_duration_s', 'post_duration_s'):
        if not np.isfinite(frame[field]).all() or not (frame[field] > 0).all():
            raise ValueError('Duration needs finite positive original support')
    converted = frame.copy()
    for field in ('pre_duration_s', 'post_duration_s'):
        converted[field] = converted[field] / 60
    result = fit_hrv_duration_adjusted(converted, feature, hrv_duration_rule_confirmed=True)
    replacements = {'pre_duration_s_centered': 'duration_pre_min_centered',
                    'post_duration_s_centered': 'duration_post_min_centered',
                    'pre_duration_s': 'duration_pre_min', 'post_duration_s': 'duration_post_min'}
    def rename(value):
        if isinstance(value, dict):
            return {replacements.get(k, k): rename(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rename(v) for v in value]
        if isinstance(value, str):
            for before, after in replacements.items():
                value = value.replace(before, after)
        return value
    result = rename(result)
    result.update(duration_covariate_unit='minutes', duration_basis='original_source_supported_complete_answer_seconds_divided_by_60',
                  backend_reused_without_algorithm_change=True)
    return result
