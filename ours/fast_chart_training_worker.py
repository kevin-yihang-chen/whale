"""Read actual native rollout receiver weights at compact E4 stage boundaries."""
import json
import os
from pathlib import Path

from .native_visual_model_worker import NativeVisualModelWorker
from .probe_updated_vllm import worker_embedding_samples
from .native_resume_observation import publish
from .visual_task import file_sha256
from .local_completion import checkpoint_tensor


class CompactChartModelWorker(NativeVisualModelWorker):
    def record_compact_weights(self,global_step):
        import torch
        path=Path(os.environ['VETO_COMPACT_TRAINING_PLAN'])
        plan=json.loads(path.read_text())
        coordinates=plan['worker_coordinates'];distinguishes=False
        digest=None
        if global_step==0:
            expected=[p['expected'] for p in coordinates]
        else:
            directory=(Path(plan['resume_checkpoint']['directory']) if global_step==4 and plan['stage']==2
                       else Path(plan['output'])/f'checkpoints/global_step_{global_step}')
            model_path=directory/'actor/model_world_size_1_rank_0.pt'
            state=torch.load(model_path,map_location='cpu',weights_only=True,mmap=True)
            embedding=state['model.language_model.embed_tokens.weight']
            initial=checkpoint_tensor(Path(plan['model']['path']),'model.language_model.embed_tokens.weight')
            changed=[]
            for start in range(0,len(initial),128):
                hits=torch.nonzero(embedding[start:start+128].to(torch.bfloat16)!=initial[start:start+128],as_tuple=False)[:8-len(changed)]
                changed.extend({'token_id':start+int(row),'column':int(col)} for row,col in hits)
                if len(changed)==8:break
            if changed:
                coordinates=changed;distinguishes=True
            expected=[float(embedding[p['token_id'],p['column']].to(torch.bfloat16)) for p in coordinates]
            digest=file_sha256(model_path)
        actual=worker_embedding_samples(self.model_runner.get_model(),coordinates)
        if actual['values']!=expected or actual['embedding_dtype']!='torch.bfloat16':
            raise ValueError('Native receiver differs from the stage checkpoint')
        publish(Path(plan['output'])/f'state/receiver-step{global_step}-{os.getpid()}.json',{
            'status':'PASS_COMPACT_RECEIVER','plan_sha256':file_sha256(path),'global_step':global_step,
            'actual':actual,'expected':expected,'native_checkpoint_sha256':digest,
            'coordinates_distinguish_initial_model':distinguishes,
            'limitation':'Up to eight changed receiver coordinates; full exported graph comparison is a separate required check.'})
        return 'PASS_COMPACT_RECEIVER'
