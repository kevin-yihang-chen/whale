"""Replay E4 visual inputs and success selection after native checkpoint resume.

No model or optimizer is invoked. This audit checks pixels, task identity,
returned token sequences, rewards and the accepted subset. Parameter changes,
checkpoint recovery and scientific accuracy require separate evidence.
"""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .visual_task import binary_answer_verifier, file_sha256


def signature(prompt, images, response):
    return json.dumps({'prompt': prompt, 'images': images, 'response': response}, sort_keys=True)


async def audit(plan_path, output):
    from .training_bootstrap import prepare_worker
    prepare_worker()
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoProcessor, AutoTokenizer
    from verl import DataProto
    from verl.trainer.main_ppo import create_rl_dataset
    from verl.experimental.agent_loop.agent_loop import apply_chat_template
    plan = json.loads(plan_path.read_text())
    assert plan['kind'] == 'native_visual_rsft_resume' and plan['role'] == 'W'
    from .visual_training_resume import certify_handoff
    handoff = plan['handoff']
    _, verified = certify_handoff(Path(handoff['search_root']), Path(handoff['first_plan']), Path(handoff['followup_plan']))
    assert handoff == verified
    for name, digest in {**plan['source_sha256'], **plan['inputs_sha256']}.items():
        assert file_sha256(Path(name)) == digest
    cfg = OmegaConf.create(plan['config'])
    assert cfg.data.tool_config_path is None and cfg.actor_rollout_ref.rollout.multi_turn.tool_config_path is None
    root = Path(plan['output'])
    start = json.loads((root / 'start.json').read_text())
    assert start['plan_sha256'] == file_sha256(plan_path)
    sources = {str(plan_path): file_sha256(plan_path)}
    batches = {}
    for kind in ('rollout', 'accepted'):
        path = root / 'checkpoints/audit' / f'{kind}-step2.pkl'
        receipt_path = path.with_suffix('.json')
        receipt = json.loads(receipt_path.read_text())
        assert file_sha256(path) == receipt['sha256']
        assert receipt['job_id'] == start['job_id']
        batches[kind] = DataProto.load_from_disk(path)
        sources.update({str(p): file_sha256(p) for p in (path, receipt_path)})
    batch, accepted = batches['rollout'], batches['accepted']
    assert len(batch) == 64
    assert len(plan['cpu_preflight']['next_eight_ids']) == 8
    meta = batch.non_tensor_batch
    expected_ids = [sample for sample in plan['cpu_preflight']['next_eight_ids'] for _ in range(8)]
    assert meta['visual_sample_id'].tolist() == expected_ids
    assert set(meta['visual_harness_sha256']) == {plan['harness_sha256']}
    assert all(c == 1 for c in meta['visual_policy_calls'])
    assert all(not trace for trace in meta['visual_harness_tool_trace'])
    model = cfg.actor_rollout_ref.model.path
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    dataset = create_rl_dataset(cfg.data.train_files, cfg.data, tokenizer, processor,
                                is_train=True, max_samples=32)
    assert list(dataset.dataframe['visual_sample_id']) == plan['cpu_preflight']['all_sample_ids']
    rows = {dataset.dataframe[i]['visual_sample_id']: i for i in range(len(dataset))}
    expected_inputs = {}
    for sample in dict.fromkeys(expected_ids):
        row = dataset[rows[sample]]
        messages = deepcopy(row['raw_prompt'])
        images, videos = await dataset.process_vision_info(messages, image_patch_size=16, config=cfg.data)
        assert len(images) == 1 and not videos
        raw = apply_chat_template(processor, messages, tools=None, add_generation_prompt=True,
                                  tokenize=False, **cfg.data.apply_chat_template_kwargs)
        encoded = processor(text=[raw], images=images, videos=None, video_metadata=None,
                            return_tensors='pt', do_sample_frames=False)
        image_records = [{'mode': im.mode, 'size': list(im.size),
                          'pixels_sha256': hashlib.sha256(im.tobytes()).hexdigest()} for im in images]
        expected_inputs[sample] = (row, encoded, image_records)
    expected_requests, correctness = Counter(), []
    for i, sample in enumerate(expected_ids):
        row, encoded, images = expected_inputs[sample]
        prompt_width = batch.batch['prompts'].shape[1]
        prompt = batch.batch['prompts'][i][batch.batch['attention_mask'][i, :prompt_width].bool()].tolist()
        assert prompt == encoded['input_ids'][0].tolist()
        mmi = meta['multi_modal_inputs'][i]
        assert torch.equal(mmi['pixel_values'], encoded['pixel_values'])
        assert torch.equal(mmi['image_grid_thw'], encoded['image_grid_thw'])
        mask = batch.batch['response_mask'][i]
        assert bool(((mask == 0) | (mask == 1)).all())
        response = batch.batch['responses'][i][mask.bool()].tolist()
        raw_answer = tokenizer.decode(response, skip_special_tokens=True)
        assert raw_answer == meta['visual_final_answer'][i]
        committed = dataset.visual_harness.invoke('parse_answer', text=raw_answer)
        assert committed == meta['visual_committed_answer'][i]
        truth = row['reward_model']['ground_truth']
        assert meta['reward_model'][i]['ground_truth'] == truth
        score = int(binary_answer_verifier(committed, truth))
        assert batch.batch['token_level_scores'][i].sum().item() == score
        assert len(response) == meta['visual_generated_tokens'][i]
        correctness.append(score)
        expected_requests[signature(prompt, images, response)] += 1
    actual_requests, total_tokens = Counter(), 0
    starts = list((root / 'requests').glob('*.request.json'))
    assert len(starts) == len(list((root / 'requests').glob('*.response.json'))) == 64
    assert not list((root / 'requests').glob('*.failure.json'))
    for path in starts:
        request = json.loads(path.read_text())
        response_path = path.with_name(request['id'] + '.response.json')
        response = json.loads(response_path.read_text())
        assert request['id'] == response['id'] and not request['videos_present']
        assert all(request['sampling_params'].get(k) == v for k, v in
                   {'temperature': 1., 'top_p': 1., 'top_k': 20}.items())
        actual_requests[signature(request['prompt_ids'], request['images'], response['token_ids'])] += 1
        total_tokens += len(response['token_ids'])
    assert actual_requests == expected_requests
    selected = [i for i, score in enumerate(correctness) if score > cfg.trainer.online_rsft.score_threshold]
    assert len(accepted) == len(selected) > 0
    expected = batch.select_idxs(selected)
    assert set(expected.batch.keys()) == set(accepted.batch.keys())
    for key in expected.batch.keys():
        assert torch.equal(expected.batch[key], accepted.batch[key]), key
    for i, source_index in enumerate(selected):
        assert accepted.non_tensor_batch['visual_sample_id'][i] == expected_ids[source_index]
        for key, value in meta['multi_modal_inputs'][source_index].items():
            assert torch.equal(value, accepted.non_tensor_batch['multi_modal_inputs'][i][key])
    workers = list((root / 'workers').glob('model-*.json'))
    assert len(workers) == 1
    worker = json.loads(workers[0].read_text())
    assert worker['status'] == 'PASS_LOADED_COORDINATES' and worker['seed'] == plan['seed']
    assert worker['plan_sha256'] == file_sha256(plan_path)
    # Restoration precedes every rollout; the second native synchronization
    # follows the saved step-2 checkpoint. Neither observation draws samples.
    resume_receipts = {}
    for name in ('actor', 'loader', 'transport-step1', 'transport-step2'):
        receipt_path = root / 'resume' / f'{name}.json'
        receipt = json.loads(receipt_path.read_text())
        assert receipt['plan_sha256'] == file_sha256(plan_path)
        resume_receipts[name] = receipt
        sources[str(receipt_path)] = file_sha256(receipt_path)
    assert resume_receipts['actor']['status'] == 'PASS'
    assert resume_receipts['actor']['device'] == 'cuda'
    assert resume_receipts['loader']['status'] == 'PASS_NATIVE_VISUAL_PENDING_LOADER'
    assert resume_receipts['loader']['actor_receipt_sha256'] == sources[str(root / 'resume/actor.json')]
    assert resume_receipts['loader']['pending_state_digest'] == plan['cpu_preflight']['pending_loader']['pending_state_digest']
    for step in (1, 2):
        transport = resume_receipts[f'transport-step{step}']
        assert transport['status'] == 'PASS_NATIVE_WEIGHT_TRANSPORT_OBSERVED' and transport['global_step'] == step
        receivers = list((root / 'resume').glob(f'receiver-step{step}-*.json'))
        assert len(receivers) == 1 and file_sha256(receivers[0]) == transport['receiver_sha256']
        receiver = json.loads(receivers[0].read_text())
        assert receiver['status'] == 'PASS_NATIVE_RECEIVER_COORDINATES'
        assert receiver['job_id'] == start['job_id'] and receiver['plan_sha256'] == file_sha256(plan_path)
        checkpoint = (Path(plan['resume_checkpoint']['directory']) if step == 1 else root / 'checkpoints/global_step_2')
        assert file_sha256(checkpoint / 'actor/model_world_size_1_rank_0.pt') == receiver['native_checkpoint_sha256']
        sources[str(receivers[0])] = file_sha256(receivers[0])
    update_path = root / 'checkpoints/audit/update-step2.json'
    update = json.loads(update_path.read_text())
    assert update['status'] == 'NATIVE_SFT_CALL_RETURNED'
    accepted_receipt = json.loads((root / 'checkpoints/audit/accepted-step2.json').read_text())
    assert update['accepted_receipt'] == accepted_receipt
    assert update['metrics']['actor/sft_token_count'] == [float(accepted.batch['response_mask'].sum())]
    assert all(x > 0 for x in update['metrics']['actor/sft_updates'])
    sources[str(update_path)] = file_sha256(update_path)
    value = {'status': 'AUDITED_NATIVE_VISUAL_RSFT_RESUMED_BATCH', 'job_id': start['job_id'], 'role': 'W',
        'plan_sha256': file_sha256(plan_path), 'native_step': 2, 'trajectories': 64, 'unique_training_questions': 8,
        'accepted': len(selected), 'accepted_indices': selected, 'correctness': correctness,
        'actual_model_calls': len(starts), 'generated_tokens': total_tokens,
        'accepted_assistant_loss_tokens': int(accepted.batch['response_mask'].sum()),
        'source_sha256': sources, 'audit_source_sha256': file_sha256(Path(__file__)),
        'input_pixels_and_tokens_replayed': True, 'native_success_subset_exact': True,
        'native_restore_and_both_transports_observed': True, 'optimizer_updates': update['metrics']['actor/sft_updates'],
        'parameter_change_verified': False, 'scientific_method_verified': False}
    with output.open('x') as stream:
        json.dump(value, stream, indent=2)
    print(json.dumps({k: v for k, v in value.items() if k not in ('source_sha256', 'correctness', 'accepted_indices')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(audit(args.plan.resolve(), args.output.resolve()))
