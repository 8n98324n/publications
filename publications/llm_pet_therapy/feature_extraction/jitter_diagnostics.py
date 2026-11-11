from collections import Counter,defaultdict
from pathlib import Path
import csv,json,math,struct
HERE=Path(__file__).resolve().parent

def read_log(path):
    schema=json.loads((HERE/'diagnostic_schema.json').read_text(encoding='utf-8'))['columns']
    rows=defaultdict(list)
    with Path(path).open(encoding='utf-8',errors='replace') as f:
        for lineno,line in enumerate(f,1):
            if not line.startswith('JITTER_V2_'):continue
            values=next(csv.reader([line.strip()]));kind=values.pop(0)
            if kind not in schema:raise ValueError(f'Unknown diagnostic type at line {lineno}: {kind}')
            if len(values)!=len(schema[kind]):raise ValueError(f'Wrong field count at line {lineno}: {kind} {len(values)} expected {len(schema[kind])}')
            row={key:value if key in {'instance','kind'} else float(value) for key,value in zip(schema[kind],values)}
            row['_line']=lineno;rows[kind].append(row)
    return rows

def close(a,b,absolute=2e-7,relative=3e-6):return math.isclose(a,b,rel_tol=relative,abs_tol=absolute)

def assess(log,jitter_csv=None):
    data=read_log(log); errors=[]; derived=[]
    for kind,rows in data.items():
        for row in rows:
            for field,value in row.items():
                if isinstance(value,float) and not math.isfinite(value):
                    errors.append(f'{kind} line{row["_line"]}: nonfinite {field}')
    if errors:
        return {'parser_checks_pass':False,'errors':errors[:100],'error_count':len(errors),
            'diagnostic_frames':0,'frame_branch_counts':{},'technical_supported_frames':0,
            'branch_coverage':{},'wave_read_failures':len(data['JITTER_V2_READ_FAILURE']),
            'research_feature_released':False,'human_review_completed':False},[]
    config={r['instance']:r for r in data['JITTER_V2_CONFIG']}
    begins={(r['instance'],r['frame_id']):r for r in data['JITTER_V2_BEGIN']}
    periods=defaultdict(list);searches=defaultdict(list);lookup={}
    for r in data['JITTER_V2_PERIOD']:
        periods[(r['instance'],r['frame_id'])].append(r);lookup[(r['instance'],r['attempt_id'])]=r
    for r in data['JITTER_V2_SEARCH']:searches[(r['instance'],r['frame_id'])].append(r)
    framekeys=[(r['instance'],r['frame_id']) for r in data['JITTER_V2_FRAME']]
    searchkeys=[(r['instance'],r['attempt_id']) for r in data['JITTER_V2_SEARCH']]
    if len(config)!=len(data['JITTER_V2_CONFIG']):errors.append('Duplicate instance configuration')
    if len(begins)!=len(data['JITTER_V2_BEGIN']):errors.append('Duplicate frame begin')
    if len(framekeys)!=len(set(framekeys)):errors.append('Duplicate frame end')
    if len(searchkeys)!=len(set(searchkeys)):errors.append('Duplicate search attempt identifier')
    if len(lookup)!=len(data['JITTER_V2_PERIOD']):errors.append('Duplicate accepted-period identifier')
    if not data['JITTER_V2_FRAME']:errors.append('No diagnostic frames; cannot validate an empty trace')
    if data['JITTER_V2_READ_FAILURE']:errors.append('Waveform read failures occurred; investigate rather than skip frames')
    # Current config requires onlyVoiced=0 and jitterLocal=1, so normal BEGIN
    # records must all produce FRAME records. A read failure precedes BEGIN.
    if set(begins)!=set(framekeys):errors.append('Normal BEGIN/FRAME records do not close')
    if set(periods)-set(framekeys) or set(searches)-set(framekeys):errors.append('Orphan SEARCH/PERIOD record outside a completed frame')
    csv_values={}
    if jitter_csv is not None:
        with Path(jitter_csv).open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f,delimiter=';'):
                stamp=round(float(r['frameTime']),7);value=float(r['jitterLocal'])
                if not math.isfinite(stamp) or not math.isfinite(value):errors.append('Nonfinite CSV time/output')
                if stamp in csv_values:errors.append('Duplicate CSV frame time')
                csv_values[stamp]=value
        diagnostic_times=[round(r['time_sec'],7) for r in data['JITTER_V2_FRAME']]
        if len(csv_values)!=len(diagnostic_times) or set(csv_values)!=set(diagnostic_times):errors.append('CSV/diagnostic frame times do not match exactly')
    for c in config.values():
        if c['useBrokenJitterThresh']!=0 or c['jitterLocal']!=1 or c['jitterLocalEnv']!=0 or c['onlyVoiced']!=0:
            errors.append('Configuration differs from required diagnostic contract')
        if c['sample_period_sec']<=0 or c['minNumPeriods']!=2 or not close(c['minCC'],.5) or not close(c['searchRangeRel'],.1) or c['usePeakToPeakPeriodLength']!=0:
            errors.append('Period configuration differs from the fixed production-family probe')
    # Reconstruct the diagnostic gap state from the ordered event stream,
    # independently of each PERIOD record's declared gap flag.
    gap_state={};gap_before_period={}
    stream=sorted([(r['_line'],kind,r) for kind,rows in data.items() for r in rows],key=lambda item:item[0])
    for _,kind,row in stream:
        instance=row['instance']
        if kind=='JITTER_V2_BEGIN' and row['input_index_discontinuity']==1:gap_state[instance]=True
        elif kind=='JITTER_V2_READ_FAILURE':gap_state[instance]=True
        elif kind=='JITTER_V2_SEARCH' and row['kind'] in {'rejected_cc','no_peak'}:gap_state[instance]=True
        elif kind=='JITTER_V2_PERIOD':
            gap_before_period[(instance,row['attempt_id'])]=gap_state.get(instance,False)
            gap_state[instance]=False
        elif kind=='JITTER_V2_FRAME' and row['F0_Hz']<=0:gap_state[instance]=False
    latest_accepted={}
    for f in data['JITTER_V2_FRAME']:
        key=(f['instance'],f['frame_id']); pp=periods[key]; ss=searches[key]; begin=begins.get(key)
        prefix=f'{key}: '
        if begin is None:errors.append(prefix+'Missing frame begin');continue
        if f['instance'] not in config:errors.append(prefix+'Missing component configuration');continue
        if f['_line']<=begin['_line']:errors.append(prefix+'FRAME does not follow BEGIN')
        if any(f[k]!=begin[k] for k in ('time_sec','F0_Hz','sample_period_sec','wave_read_start','input_index_discontinuity')):
            errors.append(prefix+'BEGIN/FRAME identity disagreement')
        if f['sample_period_sec']<=0 or f['F0_Hz']<0:errors.append(prefix+'Invalid pitch or sample period')
        if f['wave_read_actual_end_exclusive']<=f['wave_read_start'] or f['wave_read_start']<0:errors.append(prefix+'Invalid waveform read support')
        if f['all_support_start']<0 or f['all_support_start']>f['wave_read_start'] or f['all_support_end_exclusive']<f['wave_read_actual_end_exclusive']:
            errors.append(prefix+'Invalid conservative support envelope')
        if not close(begin['pitch_step_sec'],.01,1e-9,0):errors.append(prefix+'Unexpected time step')
        used=[p for p in pp if p['algorithm_uses_pair']==1]
        unsupported=[p for p in used if p['pair_support_complete']!=1]
        accepted={p['attempt_id'] for p in pp}
        if accepted!={s['attempt_id'] for s in ss if s['kind']=='accepted'}:errors.append(prefix+'Accepted-search/period mismatch')
        if len(pp)!=f['accepted_periods']:errors.append(prefix+'nPeriods mismatch')
        if len(used)!=f['algorithm_pair_count'] or len(used)!=f['diagnostic_pair_count']:errors.append(prefix+'nPeriodsLocal mismatch')
        if len(unsupported)!=f['unsupported_pair_count']:errors.append(prefix+'Unsupported-pair count mismatch')
        if sum(s['kind']=='rejected_cc' for s in ss)!=f['rejected_attempts']:errors.append(prefix+'Rejected-search count mismatch')
        if sum(s['kind']=='no_peak' for s in ss)!=f['no_peak_attempts']:errors.append(prefix+'No-peak count mismatch')
        search_by_attempt={s['attempt_id']:s for s in ss}
        for s in ss:
            if s['kind'] not in {'accepted','rejected_cc','no_peak'}:errors.append(prefix+'Unknown search outcome')
            if not begin['_line']<s['_line']<f['_line'] or s['time_sec']!=f['time_sec']:errors.append(prefix+'SEARCH event lies outside its frame')
            if s['search_start']<begin['wave_read_start'] or s['search_end_exclusive']>begin['wave_read_start']+begin['wave_read_actual'] or s['search_end_exclusive']<=s['search_start']:
                errors.append(prefix+'Search outside actual waveform support')
            if not close(s['threshold'],config[f['instance']]['minCC']):errors.append(prefix+'Search threshold differs from fixed component configuration')
            if s['kind']=='accepted' and not s['cc_peak']>s['threshold']:errors.append(prefix+'Accepted correlation does not pass actual threshold')
            if s['kind']=='rejected_cc' and s['cc_peak']>s['threshold']:errors.append(prefix+'Rejected correlation passes actual threshold')
        nbytes=int(config[f['instance']]['float_bytes'])
        if nbytes not in {4,8}:errors.append(prefix+'Unsupported FLOAT_DMEM size');continue
        rnd=(lambda x:struct.unpack('f',struct.pack('f',x))[0]) if nbytes==4 else float
        period_sum=diff_sum=0.
        for p in pp:
            if not begin['_line']<p['_line']<f['_line'] or p['time_sec']!=f['time_sec']:errors.append(prefix+'PERIOD event lies outside its frame')
            if any(p[k] not in {0,1} for k in ('algorithm_uses_pair','gap_since_previous_accepted','nominal_intervals_adjacent','previous_period_tracking_matches','pair_support_complete','input_index_discontinuity')):
                errors.append(prefix+'Invalid boolean diagnostic field')
            if p['period_sec']<=0 or p['period_end_exclusive']<=p['period_start'] or p['abs_period_diff_sec']<0:
                errors.append(prefix+'Nonpositive period or invalid difference/support')
            if bool(p['algorithm_uses_pair'])!=(p['previous_algorithm_period_sec']>0):errors.append(prefix+'Original pair predicate mismatch')
            if bool(p['gap_since_previous_accepted'])!=gap_before_period[(p['instance'],p['attempt_id'])]:errors.append(prefix+'Declared gap differs from independent event-state reconstruction')
            search=search_by_attempt.get(p['attempt_id'])
            if search is None or search['kind']!='accepted':errors.append(prefix+'Accepted period lacks unique accepted SEARCH')
            elif any(p[k]!=search[k] for k in ('period_start','period_end_exclusive','search_start','search_end_exclusive')) or search['_line']>=p['_line']:
                errors.append(prefix+'SEARCH/PERIOD interval or order disagreement')
            expected_prior=latest_accepted.get(p['instance'])
            if expected_prior is None:
                if p['previous_attempt_id']!=-1 or p['algorithm_uses_pair']!=0:
                    errors.append(prefix+'First accepted period after process/unvoiced reset carries an old pair')
            elif p['previous_attempt_id']!=expected_prior['attempt_id']:
                errors.append(prefix+'Previous-period reference skips the latest accepted period')
            period_sum=rnd(period_sum+p['period_sec'])
            if p['algorithm_uses_pair']:
                diff_sum=rnd(diff_sum+p['abs_period_diff_sec'])
                prev=lookup.get((p['instance'],p['previous_attempt_id']))
                if prev is not None and prev['_line']>=p['_line']:errors.append(prefix+'Previous period is not chronologically prior in the diagnostic stream')
                if prev is not None and any(p[a]!=prev[b] for a,b in (
                    ('previous_period_start','period_start'),('previous_period_end_exclusive','period_end_exclusive'),
                    ('previous_search_start','search_start'),('previous_search_end_exclusive','search_end_exclusive'))):
                    errors.append(prefix+'Referenced previous-period support differs from original event')
                tracked=prev is not None and prev['period_sec']==p['previous_algorithm_period_sec']
                adjacent=prev is not None and prev['period_end_exclusive']==p['period_start']
                supported=tracked and adjacent and p['gap_since_previous_accepted']==0 and p['input_index_discontinuity']==0
                if bool(p['previous_period_tracking_matches'])!=tracked:errors.append(prefix+'Previous-period identity mismatch')
                if bool(p['nominal_intervals_adjacent'])!=adjacent:errors.append(prefix+'Adjacency flag mismatch')
                if bool(p['pair_support_complete'])!=supported:errors.append(prefix+'Pair-support flag mismatch')
                # The C++ subtraction occurs in FLOAT_DMEM before fabs/cast.
                reference_diff=rnd(abs(rnd(p['previous_algorithm_period_sec']-p['period_sec'])))
                if not close(reference_diff,p['abs_period_diff_sec']):errors.append(prefix+'Original per-pair absolute difference mismatch')
            elif p['abs_period_diff_sec']!=0:errors.append(prefix+'No-pair period has a nonzero difference')
            latest_accepted[p['instance']]=p
        if not close(period_sum,f['period_sum_sec']):errors.append(prefix+'Accepted-period sum mismatch')
        if not close(diff_sum,f['pair_difference_sum_before_division_sec']):errors.append(prefix+'Difference sum mismatch')
        branch=int(f['output_branch'])
        expected_branch=0 if f['F0_Hz']<=0 else 1 if f['accepted_periods']<=0 else 2 if f['algorithm_pair_count']<=0 else 3
        if branch!=expected_branch:errors.append(prefix+'Output branch mismatch')
        if branch==0:latest_accepted.pop(f['instance'],None)
        if branch==3:
            if f['algorithm_pair_count']<=0 or f['accepted_periods']<=0 or period_sum<=0:
                errors.append(prefix+'Current estimate has no positive denominator')
            else:
                reference=rnd(rnd(diff_sum/f['algorithm_pair_count'])/rnd(period_sum/f['accepted_periods']))
                if not close(reference,f['unclamped_last_jitter_ratio']):errors.append(prefix+'Frame formula mismatch')
            expected_output=min(f['unclamped_last_jitter_ratio'],1.)
        elif branch==1:expected_output=min(f['previous_jitter_ratio'],1.)
        else:expected_output=0.
        if not close(expected_output,f['output_jitter_ratio']):errors.append(prefix+'Output/carry/placeholder mismatch')
        if jitter_csv is not None:
            actual=csv_values.get(round(f['time_sec'],7))
            if actual is None or not close(actual,f['output_jitter_ratio'],1e-6,1e-5):errors.append(prefix+'CSV jitter output mismatch')
        usable=branch==3 and f['algorithm_pair_count']>0 and f['unsupported_pair_count']==0 and f['input_index_discontinuity']==0 and f['period_sum_sec']>0
        derived.append({k:v for k,v in f.items() if k!='_line'} | {
            'technical_current_adjacent_pair_support':usable,
            'true_zero_with_supported_period_pairs':usable and f['output_jitter_ratio']==0,
            'requires_source_event_mask_before_research_use':True,
            'research_feature_released':False})
    counter=Counter(int(r['output_branch']) for r in derived)
    coverage={
        'current_estimate':counter[3]>0,'unvoiced_placeholder':counter[0]>0,
        'carry_forward_no_current_period':counter[1]>0,'accepted_period_without_pair_placeholder':counter[2]>0,
        'true_zero_with_period_support':any(r['true_zero_with_supported_period_pairs'] for r in derived),
        'correlation_rejection':any(r['kind']=='rejected_cc' for r in data['JITTER_V2_SEARCH']),
        'no_peak':any(r['kind']=='no_peak' for r in data['JITTER_V2_SEARCH']),
        'nonadjacent_or_unreliable_counted_pair':any(r['unsupported_pair_count']>0 for r in derived),
    }
    if errors:
        for row in derived:
            row['technical_current_adjacent_pair_support']=False
            row['true_zero_with_supported_period_pairs']=False
        coverage['true_zero_with_period_support']=False
    summary={'parser_checks_pass':not errors,'errors':errors[:100], 'error_count':len(errors),'diagnostic_frames':len(derived),
        'frame_branch_counts':dict(counter),'technical_supported_frames':sum(r['technical_current_adjacent_pair_support'] for r in derived),
        'branch_coverage':coverage,'wave_read_failures':len(data['JITTER_V2_READ_FAILURE']),
        'research_feature_released':False,'human_review_completed':False}
    return summary,derived
