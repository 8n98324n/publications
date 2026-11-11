from collections import defaultdict
import math
from .jitter_diagnostics import read_log,assess

def audit_support(log):
    """Only assess physical-support consistency; standalone pass is insufficient."""
    errors = []
    try:
        data = read_log(log)
    except (ValueError, KeyError, OverflowError) as exc:
        return {'support_contract_pass': False, 'errors': [str(exc)], 'error_count': 1,
                'frames_checked': 0, 'accepted_periods_checked': 0,
                'research_feature_released': False}

    def error(row, message):
        errors.append(f"line {row.get('_line', '?')}: {message}")

    def integers(row, names):
        ok = all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) and float(row[k]).is_integer() for k in names)
        if not ok:
            error(row, 'Nonfinite/noninteger physical sample or identity field: ' + ','.join(names))
        return ok

    begins, frames, searches, periods = {}, {}, {}, defaultdict(list)
    previous = {}
    for kind, destination, fields in [
        ('JITTER_V2_BEGIN', begins, ('instance', 'frame_id')),
        ('JITTER_V2_FRAME', frames, ('instance', 'frame_id')),
        ('JITTER_V2_SEARCH', searches, ('instance', 'attempt_id')),
    ]:
        for row in data[kind]:
            key = tuple(row[k] for k in fields)
            if key in destination:
                error(row, 'Duplicate support identity')
            destination[key] = row
    for row in data['JITTER_V2_PERIOD']:
        key = (row['instance'], row['attempt_id'])
        if key in previous:
            error(row, 'Duplicate accepted-period identity')
        previous[key] = row
        periods[row['instance'], row['frame_id']].append(row)
    if not frames:
        errors.append('No diagnostic FRAME records')
    if set(begins) != set(frames):
        errors.append('BEGIN/FRAME key mismatch')
    if set(periods) - set(frames):
        errors.append('Accepted period outside a closed frame')

    for key, frame in frames.items():
        begin = begins.get(key)
        if begin is None:
            continue
        if not integers(begin, ('frame_id', 'wave_read_start', 'wave_read_actual')) or not integers(frame, ('frame_id', 'wave_read_start', 'wave_read_actual_end_exclusive', 'all_support_start', 'all_support_end_exclusive')):
            continue
        lo, hi = begin['wave_read_start'], begin['wave_read_start'] + begin['wave_read_actual']
        if lo < 0 or hi <= lo:
            error(begin, 'Invalid actual waveform interval')
            continue
        if frame['wave_read_start'] != lo or frame['wave_read_actual_end_exclusive'] != hi:
            error(frame, 'BEGIN actual length and FRAME read endpoint disagree')
        envelope_lo, envelope_hi = frame['all_support_start'], frame['all_support_end_exclusive']
        if envelope_lo < 0 or envelope_lo > lo or envelope_hi < hi:
            error(frame, 'Conservative envelope does not contain actual current read')
        for period in periods[key]:
            if not integers(period, ('attempt_id', 'frame_id', 'period_start', 'period_end_exclusive', 'search_start', 'search_end_exclusive', 'previous_attempt_id')):
                continue
            search = searches.get((period['instance'], period['attempt_id']))
            if search is None or search['kind'] != 'accepted' or (search['instance'], search['frame_id']) != key:
                error(period, 'Accepted PERIOD has no matching accepted SEARCH in its frame')
                continue
            if not integers(search, ('search_start', 'search_end_exclusive', 'period_start', 'period_end_exclusive')):
                continue
            if any(period[k] != search[k] for k in ('period_start', 'period_end_exclusive', 'search_start', 'search_end_exclusive')):
                error(period, 'PERIOD and SEARCH physical intervals disagree')
            if not (lo <= search['search_start'] < search['search_end_exclusive'] <= hi):
                error(search, 'Accepted SEARCH outside actual waveform read')
            if not (search['search_start'] <= period['period_start'] < period['period_end_exclusive'] <= search['search_end_exclusive']):
                error(period, 'PERIOD nominal interval outside accepted SEARCH physical support')
            # A no-pair first period retains -1 prior sentinels. Never require a
            # nonexistent prior interval, including after an unvoiced reset.
            if period['algorithm_uses_pair'] == 1:
                prior = previous.get((period['instance'], period['previous_attempt_id']))
                if prior is None or prior['_line'] >= period['_line']:
                    error(period, 'Contributing pair has no chronologically prior accepted event')
                    continue
                if not integers(prior, ('search_start', 'search_end_exclusive')):
                    continue
                if envelope_lo > prior['search_start'] or envelope_hi < prior['search_end_exclusive']:
                    error(frame, 'Conservative envelope omits referenced previous SEARCH support')
    return {'support_contract_pass': not errors, 'errors': errors[:100], 'error_count': len(errors),
            'frames_checked': len(frames), 'accepted_periods_checked': len(data['JITTER_V2_PERIOD']),
            'source_envelope_is_not_effective_duration': True,
            'interpolated_period_not_equated_to_integer_interval_length': True,
            'research_feature_released': False, 'human_review_completed': False}

def combined_assessment(log, jitter_csv):
    original, _ = assess(log, jitter_csv)
    extension = audit_support(log)
    return {'original_parser_checks_pass': original['parser_checks_pass'],
            'original_parser_errors': original['errors'], 'support_extension': extension,
            'combined_parser_and_support_contract_pass': original['parser_checks_pass'] and extension['support_contract_pass'],
            'required_branch_coverage_complete': all(original['branch_coverage'].values()) if original['branch_coverage'] else False,
            'branch_coverage_note': 'Per-log coverage only; the frozen runner must separately establish full predeclared-suite coverage and ON/OFF equivalence.',
            'research_feature_released': False, 'human_review_completed': False}
