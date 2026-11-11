"""Files/Batch requests, four per file. No realtime inference route."""
import argparse,base64,json,os,random,time
from pathlib import Path
from .smile import validate_rating
CONFIG=Path(__file__).resolve().parents[2]/'config'

def prepare(manifest,output):
    import csv
    manifest=Path(manifest);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((CONFIG/'smile_batch.json').read_text(encoding='utf-8'));prompt=(CONFIG/'prompts/smile_coding.txt').read_text(encoding='utf-8')
    with manifest.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len({r['blind_code'] for r in rows})!=len(rows):raise ValueError('Duplicate blind code')
    slots=[(r,k) for r in rows for k in (1,2)];random.Random(cfg['random_seed']).shuffle(slots)
    if not slots or len(slots)%4:raise ValueError('Each batch must contain exactly four requests')
    for first in range(0,len(slots),4):
        items=[]
        for r,k in slots[first:first+4]:
            source=manifest.parent/r['video_path'];url='data:video/mp4;base64,'+base64.b64encode(source.read_bytes()).decode('ascii')
            body={'model':cfg['model'],'messages':[{'role':'system','content':cfg['system_prompt']},{'role':'user','content':[{'type':'text','text':prompt},{'type':'video_url','video_url':{'url':url},'fps':2.0}]}],'modalities':['text'],'temperature':0,'max_tokens':600,'stream':False}
            items.append({'custom_id':f'{r["blind_code"]}-r{k}-a1','method':'POST','url':cfg['batch_endpoint'],'body':body})
        write_batch(output/f'batch_{first//4:04d}.jsonl',items)
    (output/'state.json').write_text('{}\n',encoding='utf-8')

def write_batch(path,items):
    if len(items)!=4:raise ValueError('Exactly four requests are required; no successful filler ratings')
    lines=[json.dumps(x,ensure_ascii=False)+'\n' for x in items]
    if any(len(s.encode('utf-8'))>6000000 for s in lines):raise ValueError('Request exceeds the fixed 6 MB line limit')
    payload=''.join(lines).encode('utf-8')
    if len(payload)>500000000:raise ValueError('Batch file exceeds the fixed file limit')
    with Path(path).open('xb') as f:f.write(payload)

def run(directory):
    directory=Path(directory);lock=directory/'.batch_runner.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    try:return _run(directory)
    finally:lock.unlink()

def _run(directory):
    import requests
    directory=Path(directory);cfg=json.loads((CONFIG/'smile_batch.json').read_text(encoding='utf-8'))
    key=os.environ.get('DASHSCOPE_API_KEY')
    if not key:raise ValueError('Set DASHSCOPE_API_KEY in the environment')
    session=requests.Session();session.headers['Authorization']='Bearer '+key
    base=cfg['base_url'];statepath=directory/'state.json';state=json.loads(statepath.read_text(encoding='utf-8'))
    def save():statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    def request(method,route,**kwargs):
        response=session.request(method,base+route,timeout=(20,60),allow_redirects=False,**kwargs)
        if not response.ok:raise RuntimeError(f'Batch HTTP {response.status_code}; inspect state before any retry')
        return response
    # Network submission is serial, with at most four pending cloud batches.
    files=[p for p in sorted(directory.glob('batch_*.jsonl')) if not p.stem.endswith(('_responses','_errors'))]
    while True:
        for path in files:
            item=state.get(path.name,{})
            if item.get('state') in ('submitting','uploading'):raise RuntimeError('An interrupted POST has unknown outcome; reconcile the cloud job before resuming')
            if item.get('batch_id') and item.get('state')!='terminal':
                info=request('GET','/batches/'+item['batch_id']).json();item['cloud_status']=info['status']
                if info['status'] in ('completed','failed','expired','cancelled'):
                    for tag,keyname in [('responses','output_file_id'),('errors','error_file_id')]:
                        if info.get(keyname):(directory/(path.stem+'_'+tag+'.jsonl')).write_bytes(request('GET','/files/'+info[keyname]+'/content').content)
                    item['state']='terminal'
                state[path.name]=item;save()
        if all(state.get(p.name,{}).get('state')=='terminal' for p in files):return
        active=sum(x.get('batch_id') is not None and x.get('state')!='terminal' for x in state.values())
        for path in files:
            if active>=4:break
            if state.get(path.name,{}).get('batch_id'):continue
            item=state.setdefault(path.name,{})
            if not item.get('file_id'):
                item['state']='uploading';save()
                with path.open('rb') as f:upload=request('POST','/files',files={'file':(path.name,f,'application/jsonl')},data={'purpose':'batch'}).json()
                item.update(file_id=upload['id'],state='uploaded');save()
            item['state']='submitting';save()
            batch=request('POST','/batches',json={'input_file_id':item['file_id'],'endpoint':cfg['batch_endpoint'],'completion_window':'24h','metadata':{'ds_name':path.stem,'ds_description':'Blinded visible-smile ratings; independent video requests'}}).json()
            item.update(batch_id=batch['id'],state='submitted');save();active+=1
        time.sleep(30)

def parse(directory,output):
    directory=Path(directory);accepted=[];errors=[];seen=set();known={};slots=set()
    inputs=[p for p in sorted(directory.glob('batch_*.jsonl')) if not p.stem.endswith(('_responses','_errors'))]
    for p in inputs:
        for line in p.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);cid=r['custom_id']
            if cid in known:raise ValueError('Duplicate request ID')
            known[cid]=r
    for p in sorted([*directory.glob('batch_*_responses.jsonl'),*directory.glob('batch_*_errors.jsonl')]):
        for line in p.read_text(encoding='utf-8').splitlines():
            response=json.loads(line);cid=response.get('custom_id','')
            if cid not in known or cid in seen:raise ValueError('Unknown or duplicate response ID')
            seen.add(cid)
            try:
                env=response['response'];body=env['body']
                if response.get('error') or body.get('error') or env['status_code']!=200:raise ValueError('service_failure')
                if body.get('model') not in (None,'qwen3.5-omni-plus'):raise RuntimeError('Returned model changed; do not automatically retry')
                choices=body['choices']
                if len(choices)!=1 or choices[0]['finish_reason']!='stop':raise ValueError('incomplete_or_ambiguous_response')
                content=choices[0]['message']['content']
                if isinstance(content,list):content=''.join(x.get('text','') for x in content if isinstance(x,dict) and isinstance(x.get('text'),str))
                rating=json.loads(content);validate_rating(rating)
                if body['model']!='qwen3.5-omni-plus':raise ValueError('Unexpected returned model')
                slot,attempt=cid.rsplit('-a',1);code,repeat=slot.rsplit('-r',1)
                if slot in slots:raise RuntimeError('A successful independent slot was repeated')
                slots.add(slot)
                accepted.append(dict(blind_code=code,repeat=int(repeat),attempt=int(attempt),slot_id=slot,custom_id=cid,rating=rating,status='valid_scorable' if rating['scorable'] else 'valid_unscorable',returned_model=body['model']))
            except (KeyError,ValueError,TypeError,IndexError) as e:errors.append({'custom_id':cid,'reason':type(e).__name__})
    state=json.loads((directory/'state.json').read_text(encoding='utf-8'))
    terminal=all(state.get(p.name,{}).get('state')=='terminal' for p in inputs)
    if terminal:errors += [{'custom_id':cid,'reason':'missing_terminal_response'} for cid in sorted(set(known)-seen)]
    result={'accepted':accepted,'technical_errors':errors,'all_batches_terminal':terminal}
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result,known

def prepare_retries(directory):
    directory=Path(directory);report,known=parse(directory,directory/'ratings.json')
    if not report['all_batches_terminal']:raise ValueError('All batches must be terminal and downloaded before retry preparation')
    valid={x['slot_id'] for x in report['accepted']};latest={}
    for r in report['technical_errors']:
        slot,attempt=r['custom_id'].rsplit('-a',1)
        if slot not in latest or int(attempt)>latest[slot]:latest[slot]=int(attempt)
    requests=[]
    for slot,attempt in sorted(latest.items()):
        if slot in valid or attempt>=3:continue
        request=dict(known[slot+f'-a{attempt}']);request['custom_id']=slot+f'-a{attempt+1}';requests.append(request)
    if len(requests)%4:raise ValueError('Technical retry tail is not a multiple of four; retain missing slots, do not repeat successful ratings as filler')
    n=sum(1 for p in directory.glob('batch_*.jsonl') if not p.stem.endswith(('_responses','_errors')))
    for offset in range(0,len(requests),4):write_batch(directory/f'batch_{n+offset//4:04d}.jsonl',requests[offset:offset+4])
    return len(requests)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare');a.add_argument('--manifest',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('run');a.add_argument('--directory',required=True)
    a=sub.add_parser('parse');a.add_argument('--directory',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('retry');a.add_argument('--directory',required=True)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.manifest,a.output)
    elif a.action=='run':run(a.directory)
    elif a.action=='parse':parse(a.directory,a.output)
    else:prepare_retries(a.directory)
if __name__=='__main__':main()
