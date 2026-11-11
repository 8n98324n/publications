"""Fixed feature metadata and equal-weight factorial contrasts."""
import json
from pathlib import Path
REGISTRY={r['feature_id']:r for r in json.loads((Path(__file__).resolve().parents[2]/'config/registry.json').read_text(encoding='utf-8'))}
FEATURES=list(REGISTRY)
CONDITIONS={'observation':(0,0),'petting':(1,0),'feeding':(0,1),'combined':(1,1)}
PARAMETERS=['Intercept','pre_centered','Petting','Feeding','Petting:Feeding']
CONTRASTS={'Petting_average':[0,0,1,0,.5],'Feeding_average':[0,0,0,1,.5],'Interaction':[0,0,0,0,1]}
