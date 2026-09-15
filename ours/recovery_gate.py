"""Freeze source, base weights, cached batch and config for interrupted E4 recovery."""
import argparse
import json
import os
from pathlib import Path

from .audit_native_training_batch import require
from .visual_task import file_sha256


def repository_path(path):
    return Path(__file__).resolve().parents[1] / path


def native_runtime_config(config):
    """Apply the released pre-init migrations, for this one-step native layout."""
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    config = migrate_legacy_reward_impl(config)
    require(not config.reward.reward_model.enable_resource_pool, 'Unexpected reward resource pool')
    config.reward.reward_model.nnodes = config.trainer.nnodes
    config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
    require(config.trainer.total_training_steps in (None, 1), 'Unexpected total step override')
    require(config.trainer.total_epochs == 1 and config.data.train_batch_size == 8, 'Unexpected data schedule')
    config.actor_rollout_ref.actor.optim.total_training_steps = 1
    config.critic.optim.total_training_steps = 1
    return config


def check_config(plan, actual, *, runtime=False):
    from omegaconf import OmegaConf
    expected = repository_path(plan['resolved_config']).read_text()
    expected = expected.replace('replay-cpu-preflight', f"replay-{os.environ.get('SLURM_JOB_ID', 'cpu-preflight')}")
    expected = OmegaConf.create(expected)
    if runtime:
        expected = native_runtime_config(expected)
    require(actual == OmegaConf.to_container(expected, resolve=True),
            'Effective recovery configuration differs from frozen configuration')


def check(path, effective_config=None):
    from .rsft_pilot_gate import check as check_original
    from .recovered_training_bootstrap import restore_batch
    plan = json.loads(path.read_text())
    require(plan['kind'] == 'native_rsft_recovery_plan' and plan['new_model_calls'] == 0, 'Wrong recovery plan')
    for name, digest in plan['source_sha256'].items():
        require(file_sha256(Path(name)) == digest, f'Frozen recovery source changed: {name}')
    source = check_original(Path(plan['source_plan']))
    require(source['model'] == plan['model'] and source['weights_sha256'] == plan['weights_sha256'], 'Different base model')
    require(file_sha256(Path(plan['source_plan'])) == plan['source_plan_sha256'], 'Source plan changed')
    require(file_sha256(Path(plan['batch_audit'])) == plan['batch_audit_sha256'], 'Batch audit changed')
    audit = json.loads(Path(plan['batch_audit']).read_text())
    require(audit['plan_sha256'] == plan['source_plan_sha256'], 'Audited source plan differs')
    batch, receipt = restore_batch(plan['batch_directory'], plan['batch_audit'])
    require(receipt['sha256'] == plan['batch_sha256'], 'Recovery batch differs')
    if effective_config:
        from omegaconf import OmegaConf
        raw = effective_config.read_text()
        check_config(plan, OmegaConf.to_container(OmegaConf.create(raw[raw.index('model_engine: dp\n'):]), resolve=True))
    report = {'kind': 'native_rsft_recovery_start', 'status': 'PASS', 'job_id': os.environ.get('SLURM_JOB_ID'),
              'plan_sha256': file_sha256(path), 'cached_trajectories': len(batch), 'new_model_calls': 0,
              'completed_optimizer_steps': 0}
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--effective-config', type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    if not args.check_only:
        import torch
        require(bool(os.environ.get('SLURM_JOB_ID')) and torch.cuda.device_count() == 2, 'Expected two-GPU Slurm job')
    report = check(args.plan, args.effective_config)
    if not args.check_only:
        with Path(f"results/native-rsft-recovery-start-{report['job_id']}.json").open('x') as f:
            json.dump(report, f, indent=2)
            f.write('\n')
    print(json.dumps(report), flush=True)
