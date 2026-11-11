FEATURES=("smile_prevalence_score","smile_intensity_score")
RATING_KEYS=set(FEATURES)|{"scorable","quality_flags","visible_evidence"}

def aggregate(accepted,identities):
    """identities: blind_code, participant_id, timepoint; two original ratings."""
    codes=[r['blind_code'] for r in identities]
    if len(codes)!=len(set(codes)):raise ValueError('Duplicate blind code')
    index=index_accepted(accepted,codes);stages=[];repeats=[]
    for r in identities:
        calls=[index.get(r['blind_code']+f'-r{k}') for k in (1,2)]
        good=all(x is not None and x['rating']['scorable'] for x in calls)
        for feature in FEATURES:
            values=[x['rating'][feature] for x in calls if x is not None and x['rating']['scorable']]
            stages.append(dict(participant_id=r['participant_id'],timepoint=r['timepoint'],feature_id=feature,analysis_usable=good,analysis_value=sum(values)/2 if good else None))
            for k,x in enumerate(calls,1):
                if x is not None and x['rating']['scorable']:
                    repeats.append(dict(participant_id=r['participant_id'],timepoint=r['timepoint'],feature_id=feature,repeat_id=k,value=x['rating'][feature]))
    return stages,repeats

def validate_rating(value):
    if not isinstance(value, dict) or set(value) != RATING_KEYS:
        raise ValueError('accepted rating schema mismatch')
    if type(value['scorable']) is not bool:
        raise ValueError('scorable must be boolean')
    if not isinstance(value['quality_flags'], list) or not all(
            isinstance(flag, str) for flag in value['quality_flags']):
        raise ValueError('quality_flags must be a string list')
    if not isinstance(value['visible_evidence'], str):
        raise ValueError('visible_evidence must be a string')
    a, b = (value[feature] for feature in FEATURES)
    if not value['scorable']:
        if a is not None or b is not None:
            raise ValueError('unscorable scores must remain null')
    elif any(type(v) is not int or not 0 <= v <= 4 for v in (a, b)):
        raise ValueError('scorable scores must be integer 0..4')
    elif (a == 0) != (b == 0):
        raise ValueError('inconsistent smile presence')

def index_accepted(accepted, codes):
    result = {}
    for response in accepted:
        code, repeat = response['blind_code'], response['repeat']
        if code not in codes or type(repeat) is not int or repeat not in (1, 2):
            raise ValueError('unknown clip or repeat in accepted response')
        slot = f'{code}-r{repeat}'
        attempt = response['attempt']
        if type(attempt) is not int or not 1 <= attempt <= 3:
            raise ValueError('attempt must be 1..3')
        if response['slot_id'] != slot or response['custom_id'] != f'{slot}-a{attempt}':
            raise ValueError('accepted slot/custom_id mismatch')
        if slot in result:
            raise ValueError('duplicate accepted slot: refusing to choose a score')
        validate_rating(response['rating'])
        expected = 'valid_scorable' if response['rating']['scorable'] else 'valid_unscorable'
        if response['status'] != expected:
            raise ValueError('accepted status and rating disagree')
        if response['returned_model'] != 'qwen3.5-omni-plus':
            raise ValueError('unexpected returned model')
        result[slot] = response
    return result
