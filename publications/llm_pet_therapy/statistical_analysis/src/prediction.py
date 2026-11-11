"""Repeated five-fold predictions with training-fold median imputation."""
from .data_io import prediction_inputs,write_json,write_csv
from .models import ml_core as core
def run(data,config,out,*,demo=False):
    out.mkdir(parents=True,exist_ok=False)
    inputs=prediction_inputs(data);inputs['config']=config['ml']
    memberships,blocks=core.expected_splits(inputs)
    write_csv(out/'cv/split_membership.csv',memberships)
    write_json(out/'config.json',config['ml'])
    for matrix in ('delta','post'):
        import pandas as pd
        frame=pd.DataFrame(inputs[matrix],columns=core.FEATURES);frame.insert(0,'participant_id',inputs['ids'])
        write_csv(out/'inputs'/('X_'+matrix+'_unimputed.csv'),frame)
    checkpoints=[]
    # The demo keeps fold/repeat structure but uses fewer trees. Formal settings
    # are never overwritten in the saved study configuration.
    old=core.COMMON_RF['n_estimators']
    if demo:core.COMMON_RF['n_estimators']=8
    try:
        for mid,(version,_,_,_) in core.MODEL_SPECS.items():
            for repeat in range(10):
                for fold in range(5):
                    train,test=blocks[version,repeat,fold]
                    result=core.fit_fold(mid,repeat,fold,train,test,inputs)
                    # Computed predictions are retained; fitted trees need not
                    # remain resident once their parameters/MDI are recorded.
                    result.pop('model',None)
                    checkpoints.append(result)
                print(f'ML {mid}: repeat {repeat+1}/10',flush=True)
        summary=core.write_results(out,checkpoints,inputs)
    finally:core.COMMON_RF['n_estimators']=old
    summary.update(synthetic=demo,n_estimators=8 if demo else 500,environment=core.environment())
    write_json(out/'run_summary.json',summary)
    return summary
