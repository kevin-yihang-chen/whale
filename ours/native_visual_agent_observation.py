"""Record actual E1/E4 image-bearing calls without changing native generation."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import uuid

import ray
from .training_bootstrap import prepare_worker
prepare_worker()
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker, AsyncLLMServerManager


class ObservedVisualServerManager(AsyncLLMServerManager):
    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
        directory = Path(os.environ['VETO_NATIVE_EVALUATION_OUTPUT'])/'requests'
        ident = uuid.uuid4().hex
        start = {'id': ident, 'request_id': request_id, 'prompt_ids': list(prompt_ids),
            'sampling_params': deepcopy(sampling_params),
            'images': [{'mode': image.mode, 'size': list(image.size),
                'pixels_sha256': hashlib.sha256(image.tobytes()).hexdigest()} for image in (image_data or [])],
            'videos_present': bool(video_data)}
        def record(suffix, value):
            with (directory/f'{ident}.{suffix}.json').open('x') as stream:
                json.dump(value, stream, allow_nan=False)
                stream.write('\n')
        record('request', start)
        try:
            result = await super().generate(request_id, prompt_ids=prompt_ids, sampling_params=sampling_params,
                image_data=image_data, video_data=video_data)
            record('response', {'id': ident, 'token_ids': list(result.token_ids),
                'log_probs': result.log_probs, 'num_preempted': result.num_preempted})
            return result
        except BaseException as error:
            record('failure', {'id': ident, 'error_type': type(error).__name__, 'error': str(error)})
            raise


class ObservedVisualAgentWorker(AgentLoopWorker):
    def __init__(self, config, servers, load_balancer_handle, reward_loop_worker_handles=None):
        super().__init__(config, servers, load_balancer_handle, reward_loop_worker_handles)
        self.server_manager = ObservedVisualServerManager(config, servers, load_balancer_handle)


class ObservedVisualAgentManager(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(ObservedVisualAgentWorker)
        super().__init__(*args, **kwargs)
