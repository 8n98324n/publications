"""Single entry point for data validation, inference and prediction."""
import argparse,os,sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','TBB_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','stats','ml','all'])
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--demo',action='store_true',help='Only for synthetic DEMO IDs; separate demo_ output directory required')
    args=parser.parse_args()
    if args.demo and not args.output.name.startswith('demo_'):parser.error('Demo output directory must start with demo_')
    if args.output.exists():parser.error('Choose a new output directory; existing results are not overwritten')
    from src.data_io import load_inputs,check,read_json,write_json
    data=load_inputs(args.data,demo=args.demo)
    if args.demo and not all(p.startswith('DEMO') for p in data['participants']):parser.error('Demo mode requires clearly synthetic DEMO identifiers')
    config=read_json(Path(__file__).parent/'config/analysis.json')
    args.output.mkdir(parents=True)
    write_json(args.output/'input_check.json',check(data))
    if args.command in ('stats','all'):
        from src.statistics import run
        run(data,config,args.output/'statistics',demo=args.demo)
    if args.command in ('ml','all'):
        from src.prediction import run
        run(data,config,args.output/'machine_learning',demo=args.demo)
    print('Completed:',args.output)
if __name__=='__main__':main()
