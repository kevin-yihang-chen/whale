"""Native Chess chat requests and exact E4 batch records around original methods.

The current pilot is not retroactively instrumented. Install these subclasses
explicitly in a later frozen run. They do not select samples or change rewards,
generation arguments, masks, optimizer calls or returned model responses. The
separate token-ID generate interface is outside this chat-request recorder.
"""
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from uuid import uuid4


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def encoded(value):
    return (json.dumps(json_value(value), ensure_ascii=False, allow_nan=False) + '\n').encode()


def summarize_requests(directory):
    """Return verified lower-bound usage; interrupted or errored calls stay unknown."""
    started, finished, partial_tails = {}, {}, 0
    paths = sorted(Path(directory).glob('requests-*.jsonl'))
    for path in paths:
        lines = path.read_bytes().splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.endswith(b'\n'):
                if index != len(lines) - 1:
                    raise ValueError('Invalid journal framing')
                partial_tails += 1
                continue
            event = json.loads(line)
            if event.get('kind') != 'native_training_request':
                raise ValueError('Unexpected journal event')
            key, stage = event['trace_id'], event['stage']
            if stage == 'started':
                if key in started:
                    raise ValueError('Repeated request start')
                started[key] = event
            elif stage in ('completed', 'error'):
                if key not in started or key in finished:
                    raise ValueError('Orphan or repeated terminal request event')
                if event['request_id'] != started[key]['request_id'] or event['context'] != started[key]['context']:
                    raise ValueError('Request identity changed')
                finished[key] = event
            else:
                raise ValueError('Unknown request stage')
    completed = [event for event in finished.values() if event['stage'] == 'completed']
    measured = []
    for event in completed:
        count = (event['response'].get('usage') or {}).get('completion_tokens')
        if count is not None:
            if type(count) is not int or count < 0:
                raise ValueError('Invalid returned token usage')
            measured.append(count)
    pending = len(started) - len(finished)
    errors = len(finished) - len(completed)
    complete = bool(paths) and bool(started) and not (pending or errors or partial_tails) and len(measured) == len(completed)
    return {'kind': 'native_training_request_accounting', 'status': 'SNAPSHOT',
            'recorded_starts': len(started), 'completed': len(completed), 'errors': errors,
            'pending_or_interrupted': pending, 'partial_tail_records': partial_tails,
            'known_completion_tokens': sum(measured), 'completed_without_usage': len(completed) - len(measured),
            'observed_requests_fully_accounted': complete,
            'limitations': ['This is not a job-completion check; a live producer can append more requests.',
                            'Counts exclude unparseable partial records and are lower bounds when incomplete.',
                            'Errored or interrupted calls may have generated unreturned tokens.']}


class NativeTrainingJournal:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def event(self, value):
        record = {'time_unix_ns': time.time_ns(), 'pid': os.getpid(), **value}
        with (self.directory / f'requests-{os.getpid()}.jsonl').open('ab') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(encoded(record))
            stream.flush()
            os.fsync(stream.fileno())

    async def call(self, invoke, *, request_id, messages, sampling_params, context):
        trace_id = uuid4().hex
        shared = {'kind': 'native_training_request', 'trace_id': trace_id,
                  'request_id': request_id, 'context': context}
        self.event({**shared, 'stage': 'started', 'messages': messages,
                    'sampling_params': sampling_params})
        started = time.monotonic()
        try:
            result = await invoke(request_id=request_id, messages=messages,
                                  sampling_params=sampling_params)
        except BaseException as error:
            self.event({**shared, 'stage': 'error', 'error_type': type(error).__name__,
                        'error': str(error), 'seconds': time.monotonic() - started})
            raise
        self.event({**shared, 'stage': 'completed', 'response': result,
                    'seconds': time.monotonic() - started})
        return result

    def batch(self, batch, *, step, context):
        """Keep actual padded tensors and events; publish a receipt only after close."""
        if int(step) != step or step < 1:
            raise ValueError('A positive native optimizer-phase index is required')
        fields = ('prompts', 'responses', 'response_mask', 'attention_mask', 'position_ids',
                  'input_ids', 'token_level_scores')
        missing = set(fields) - set(batch.batch.keys())
        if missing:
            raise ValueError(f'Native rollout batch lacks {sorted(missing)}')
        count = len(batch.batch['responses'])
        if not count or any(len(batch.batch[key]) != count for key in fields):
            raise ValueError('Incomplete native rollout tensor batch')
        path = self.directory / f'batch-{int(step)}.jsonl.gz'
        schema = {key: {'shape': list(batch.batch[key].shape), 'dtype': str(batch.batch[key].dtype)}
                  for key in fields}
        with path.open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
                stream.write(encoded({'kind': 'native_training_batch_header', 'step': int(step),
                                      'count': count, 'tensors': schema, 'context': context,
                                      'meta_info': batch.meta_info}))
                for index in range(count):
                    row = {'kind': 'native_training_trajectory', 'index': index,
                           'tensors': {key: batch.batch[key][index] for key in fields},
                           'metadata': {key: values[index] for key, values in batch.non_tensor_batch.items()
                                        if key in ('uid', 'extra_info', 'extras', 'reward_extra_info', 'request_id')}}
                    stream.write(encoded(row))
            raw.flush()
            os.fsync(raw.fileno())
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        receipt = {'kind': 'native_training_batch_receipt', 'status': 'COMPLETE', 'step': int(step),
                   'count': count, 'file': path.name, 'sha256': digest,
                   'recorder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'role': 'training', 'contains_reference_metadata': True,
                   'limitations': ['A completed trace is not proof of a successful optimizer update.',
                                   'Model paths do not prove actual loaded parameter values.',
                                   'These private training records are not proposer-safe exports.']}
        with (self.directory / f'batch-{int(step)}.receipt.json').open('xb') as stream:
            stream.write(encoded(receipt))
            stream.flush()
            os.fsync(stream.fileno())
        return receipt


def install_native_recorders():
    """Called only by an explicitly selected future training/worker entrypoint."""
    from omegaconf import OmegaConf
    import verl.experimental.agent_loop.agent_loop as agent_module
    import verl.trainer.main_textarena_disagg_rsft as trainer_module

    if getattr(agent_module.AsyncLLMServerManager, '_native_training_recorder', False):
        if not getattr(trainer_module.DisaggregatedRayTrainer, '_native_training_recorder', False):
            raise RuntimeError('Only one of the native recorder classes is installed')
        return
    if getattr(trainer_module.DisaggregatedRayTrainer, '_native_training_recorder', False):
        raise RuntimeError('Only one of the native recorder classes is installed')

    def recording(config):
        config_dict = OmegaConf.to_container(config, resolve=True)
        context = {'configured_model_path': config_dict['actor_rollout_ref']['model']['path'],
                   'configuration_sha256': hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest(),
                   'job_id': os.environ.get('SLURM_JOB_ID'), 'recording_only': True}
        root = Path(config_dict['trainer']['default_local_dir']) / 'audit'
        return NativeTrainingJournal(root), context

    class TracePreservingServerManager(agent_module.AsyncLLMServerManager):
        _native_training_recorder = True

        async def chat_completion(self, request_id, *, messages, sampling_params):
            journal, context = recording(self.config)
            return await journal.call(super().chat_completion, request_id=request_id,
                                      messages=messages, sampling_params=sampling_params, context=context)

    class TracePreservingDisaggregatedTrainer(trainer_module.DisaggregatedRayTrainer):
        _native_training_recorder = True

        def _log_rollout_data(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir):
            journal, context = recording(self.config)
            journal.batch(batch, step=self.global_steps, context=context)
            return super()._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

    agent_module.AsyncLLMServerManager = TracePreservingServerManager
    trainer_module.DisaggregatedRayTrainer = TracePreservingDisaggregatedTrainer
