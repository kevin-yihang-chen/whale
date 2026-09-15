"""Execute the explicitly approved A inventory and retain a per-path receipt.

Reproduction asset maintenance: E1--E4 computations are unaffected. This is a
single-use operation bound to the reviewed paths, not a generic cache cleaner.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat


ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = Path('/userhome/cs3/yihangc')
OUT = ROOT / 'results/veto-authorized-a-cleanup-20260910-v1'
PROPOSAL = ROOT / 'results/veto-cache-cleanup-proposal-20260910-v1/proposal.json'
DEDUP = ROOT / 'results/veto-checkpoint-deduplication-review-20260910-v1.json'
PIP = ACCOUNT / '.cache/pip/http-v2'
B = ACCOUNT / 'Data/MIMIC-CXR-JPG/files'
CACHES = {
    'qwen3_vl_32b': ACCOUNT / 'Data/hf_cache/hub/models--Qwen--Qwen3-VL-32B-Instruct',
    'qwen25_vl_7b': ACCOUNT / 'Data/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct',
}
CHECKPOINTS = [ROOT / ('data/native-rsft/controlled-weight-only-seed42-222801/'
                      f'global_step_{i}/actor/model_world_size_1_rank_0.pt') for i in (1, 2)]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def identity(path):
    s = path.lstat()
    return dict(path=str(path), device=s.st_dev, inode=s.st_ino, mode=s.st_mode,
                logical_bytes=s.st_size, allocated_bytes=s.st_blocks * 512,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, nlink=s.st_nlink)


def unchanged(record):
    assert identity(Path(record['path'])) == record, record['path']


def real_directory(path):
    assert path.is_dir() and path.resolve() == path, str(path)


def write_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def event(value):
    with (OUT / 'operations.jsonl').open('a') as f:
        f.write(json.dumps(dict(at_utc=now(), **value), ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def open_local_targets(paths):
    """Check this account's login-node descriptors; Slurm is checked separately."""
    wanted = {str(p) for p in paths}
    hits = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            for fd in (proc / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                except (FileNotFoundError, PermissionError):
                    continue
                if target in wanted:
                    hits.append(dict(pid=int(proc.name), fd=fd.name, path=target))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return hits


def prepare():
    OUT.mkdir(exist_ok=False)
    write_new(OUT / 'authorization.json', {
        'recorded_at_utc': now(), 'source': 'User message in this conversation',
        'user_statement': '我授权A组，B组不授权。不清理就运行不下去了吗？',
        'authorized': ['A1 listed 32B weight links and blobs',
                       'A2 listed 7B weight links and blobs', 'A3 pip http-v2 cache',
                       'A4 replace step2 model by a hard link to identical step1'],
        'not_authorized': [str(B), 'any other cleanup'],
        'proposal_sha256': digest(PROPOSAL), 'dedup_review_sha256': digest(DEDUP),
        'executor_sha256': digest(Path(__file__)),
        'live_slurm_check': {'job_id': 223964, 'name': 'whale-joint-mh',
                            'state': 'RUNNING', 'node': 'gpucluster-g4',
                            'source': 'Read-only squeue immediately before preparation'},
    })
    proposal, dedup = json.loads(PROPOSAL.read_text()), json.loads(DEDUP.read_text())
    assert {c['id'] for c in proposal['candidates']} == set(CACHES)
    state = dict(prepared_at_utc=now(), disk_before=shutil.disk_usage(ACCOUNT)._asdict(),
                 executor_sha256=digest(Path(__file__)), weights=[], metadata=[],
                 pip_files=[], pip_directories=[], checkpoints=[], frozen_sources=[])
    state['excluded_b_directory'] = identity(B)
    real_directory(B)
    for candidate in proposal['candidates']:
        cache = CACHES[candidate['id']]
        assert str(cache) == candidate['cache_root']
        real_directory(cache)
        expected_count = 14 if candidate['id'] == 'qwen3_vl_32b' else 5
        assert len(candidate['weights']) == expected_count
        listed_links = set()
        listed_blobs = set()
        for w in candidate['weights']:
            blob, link = Path(w['blob_path']), Path(w['snapshot_path'])
            assert blob.parent == cache / 'blobs'
            assert link.is_relative_to(cache / 'snapshots') and link.is_symlink()
            assert os.readlink(link) == w['link_target'] and link.resolve() == blob
            real_directory(blob.parent)
            real_directory(link.parent)
            bs = identity(blob)
            assert stat.S_ISREG(bs['mode']) and bs['nlink'] == 1
            assert bs['inode'] == w['inode'] and bs['logical_bytes'] == w['logical_bytes']
            state['weights'].append(dict(blob=bs, link=identity(link), target=w['link_target']))
            listed_links.add(link)
            listed_blobs.add(blob)
        for entry in (cache / 'snapshots').rglob('*'):
            if entry.is_symlink() and entry.resolve() in listed_blobs:
                assert entry in listed_links, f'Unlisted snapshot reference: {entry}'
        for m in candidate['retained_metadata']:
            assert digest(Path(m['source'])) == m['sha256']
            assert digest(Path(m['retained_copy'])) == m['sha256']
            state['metadata'].append(m)
    assert Path(proposal['pip_cache']['path']) == PIP
    real_directory(PIP)
    for base, dirs, files in os.walk(PIP, followlinks=False):
        directory = Path(base)
        real_directory(directory)
        state['pip_directories'].append(identity(directory))
        for name in files:
            p = directory / name
            item = identity(p)
            assert stat.S_ISREG(item['mode']) and item['nlink'] == 1, str(p)
            state['pip_files'].append(item)
        for name in dirs:
            real_directory(directory / name)
    assert [Path(x['path']) for x in dedup['files']] == CHECKPOINTS
    for prior, checkpoint in zip(dedup['files'], CHECKPOINTS):
        real_directory(checkpoint.parent)
        before = identity(checkpoint)
        assert stat.S_ISREG(before['mode']) and before['nlink'] == 1
        assert before['inode'] == prior['inode'] and before['logical_bytes'] == prior['logical_bytes']
        print(f'Hashing {checkpoint.parent.parent.name}', flush=True)
        sha = digest(checkpoint)
        unchanged(before)
        assert sha == prior['sha256']
        state['checkpoints'].append(dict(identity=before, sha256=sha))
        for p in checkpoint.parent.parent.rglob('*'):
            if p.is_file() and p != checkpoint and p.stat().st_size < 64 * 1024**2:
                state['metadata'].append(dict(source=str(p), sha256=digest(p)))
    assert state['checkpoints'][0]['sha256'] == state['checkpoints'][1]['sha256']
    assert state['checkpoints'][0]['identity']['device'] == state['checkpoints'][1]['identity']['device']
    for name in ('veto-milestone-plan-20260910-v1.json',
                 'native-visual-service-plan-20260910-v1.json',
                 'controlled-whale-seed42-transport-recovery-20260910-v1.json'):
        plan_path = ROOT / 'results' / name
        plan = json.loads(plan_path.read_text())
        state['metadata'].append(dict(source=str(plan_path), sha256=digest(plan_path)))
        for path, sha in plan.get('source_sha256', {}).items():
            source = ROOT / path
            assert digest(source) == sha, str(source)
            state['frozen_sources'].append(dict(source=str(source), sha256=sha))
    targets = [Path(w['blob']['path']) for w in state['weights']] + CHECKPOINTS
    targets += [Path(x['path']) for x in state['pip_files']]
    assert not open_local_targets(targets), 'Open local target: defer cleanup'
    state['expected_file_allocated_bytes_released'] = (
        sum(w['blob']['allocated_bytes'] for w in state['weights']) +
        sum(x['allocated_bytes'] for x in state['pip_files']) +
        state['checkpoints'][1]['identity']['allocated_bytes'])
    write_new(OUT / 'preflight.json', state)
    print(json.dumps({k: state[k] for k in ('prepared_at_utc', 'disk_before',
          'expected_file_allocated_bytes_released')}, indent=2), flush=True)


def verify_retained(state):
    unchanged(state['excluded_b_directory'])
    for m in state['metadata'] + state['frozen_sources']:
        assert digest(Path(m['source'])) == m['sha256'], m['source']
        if 'retained_copy' in m:
            assert digest(Path(m['retained_copy'])) == m['sha256']


def execute():
    state = json.loads((OUT / 'preflight.json').read_text())
    auth = json.loads((OUT / 'authorization.json').read_text())
    assert digest(PROPOSAL) == auth['proposal_sha256']
    assert digest(DEDUP) == auth['dedup_review_sha256']
    assert digest(Path(__file__)) == state['executor_sha256'] == auth['executor_sha256']
    assert not (OUT / 'operations.jsonl').exists(), 'No automatic retry after any mutation'
    assert not (OUT / 'result.json').exists()
    for w in state['weights']:
        unchanged(w['blob'])
        unchanged(w['link'])
    for x in state['pip_files'] + state['pip_directories']:
        unchanged(x)
    for c in state['checkpoints']:
        unchanged(c['identity'])
    verify_retained(state)
    targets = [Path(w['blob']['path']) for w in state['weights']] + CHECKPOINTS
    targets += [Path(x['path']) for x in state['pip_files']]
    assert not open_local_targets(targets), 'Open local target: defer cleanup'
    event(dict(action='begin_authorized_A', preflight_sha256=digest(OUT / 'preflight.json')))
    # Creating the link before atomic replacement keeps both public names present.
    first, second = CHECKPOINTS
    temporary = second.with_name(second.name + '.authorized-a-link')
    assert not temporary.exists() and not temporary.is_symlink()
    os.link(first, temporary, follow_symlinks=False)
    unchanged(state['checkpoints'][1]['identity'])
    os.replace(temporary, second)
    assert first.stat().st_ino == second.stat().st_ino and first.stat().st_nlink == 2
    event(dict(action='A4_atomic_hardlink_dedup', source=str(first), retained_path=str(second),
               retained_sha256=state['checkpoints'][0]['sha256'],
               allocated_bytes_released=state['checkpoints'][1]['identity']['allocated_bytes']))
    for w in state['weights']:
        unchanged(w['link'])
        unchanged(w['blob'])
        Path(w['link']['path']).unlink()
        event(dict(action='A1_A2_unlink_snapshot', path=w['link']['path']))
        Path(w['blob']['path']).unlink()
        event(dict(action='A1_A2_unlink_blob', path=w['blob']['path'],
                   allocated_bytes_released=w['blob']['allocated_bytes']))
    for x in state['pip_files']:
        unchanged(x)
        Path(x['path']).unlink()
        event(dict(action='A3_unlink_pip_cache_file', path=x['path'],
                   allocated_bytes_released=x['allocated_bytes']))
    for x in sorted(state['pip_directories'], key=lambda x: len(Path(x['path']).parts), reverse=True):
        # rmdir refuses any new or unlisted contents; never recurse at execution.
        Path(x['path']).rmdir()
    verify_retained(state)
    for w in state['weights']:
        assert not os.path.lexists(w['link']['path']) and not os.path.lexists(w['blob']['path'])
    assert not PIP.exists()
    assert first.stat().st_ino == second.stat().st_ino and first.stat().st_nlink == 2
    # Rehash the one retained inode after replacement, covering both names.
    final_sha = digest(first)
    assert final_sha == state['checkpoints'][0]['sha256']
    after = shutil.disk_usage(ACCOUNT)._asdict()
    result = dict(status='COMPLETED_AUTHORIZED_A_ONLY', completed_at_utc=now(),
                  authorization_sha256=digest(OUT / 'authorization.json'),
                  preflight_sha256=digest(OUT / 'preflight.json'),
                  weights_removed=len(state['weights']), pip_files_removed=len(state['pip_files']),
                  checkpoints_after=[identity(p) for p in CHECKPOINTS], checkpoint_sha256=final_sha,
                  metadata_hashes_verified=len(state['metadata']),
                  frozen_source_hashes_verified=len(state['frozen_sources']),
                  excluded_b_directory_unchanged=True,
                  allocated_file_bytes_released=state['expected_file_allocated_bytes_released'],
                  disk_before=state['disk_before'], disk_after=after,
                  observed_free_bytes_change=after['free'] - state['disk_before']['free'],
                  disk_delta_note='Live filesystem delta may include concurrent job writes and directory blocks.',
                  hardlink_contract='Both checkpoint model paths share one immutable inode; copy before any future in-place edit.')
    event(dict(action='completed', result=result))
    write_new(OUT / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'execute'))
    args = parser.parse_args()
    prepare() if args.phase == 'prepare' else execute()
