"""Native multi-candidate E3 proposals in a disposable audited-MH archive.

The caller must verify the MH snapshot before invoking this paid boundary.
All allocated slots and the real iteration reach the original proposer. Only
their code, normalized native metadata and current report can be imported back.
"""
import json
import os
from pathlib import Path
import re
import shutil
import sys
from unittest.mock import patch

from .audit_native_training_batch import require
from .glm_gateway import MODEL, relay
from .native_search import tree_hashes, write_json


def allocated_slots(names):
    require(isinstance(names, list) and 1 <= len(names) <= 3 and
            all(isinstance(n, str) and re.fullmatch(r'h[1-9][0-9]*', n) for n in names) and
            len(set(names)) == len(names), 'Invalid native candidate allocation')
    return tuple(names)


def normalize_candidates(payload, names, parents):
    names = allocated_slots(list(names))
    entries = payload.get('candidates') if isinstance(payload, dict) else None
    require(isinstance(entries, list) and 1 <= len(entries) <= len(names), 'Expected allocated candidate metadata')
    normalized, seen = [], set()
    for entry in entries:
        require(isinstance(entry, dict), 'Invalid candidate metadata')
        aliases = [entry[k] for k in ('name', 'slot') if k in entry]
        require(bool(aliases) and all(isinstance(n, str) and n in names for n in aliases) and
                len(set(aliases)) == 1, 'Candidate name is outside allocation or aliases conflict')
        name = aliases[0]
        require(name not in seen, 'Duplicate candidate metadata')
        require(entry.get('path', f'harnesses/{name}/harness.py') == f'harnesses/{name}/harness.py', 'Candidate path differs')
        if 'parent' in entry:
            require(isinstance(entry['parent'], str) and entry['parent'] in parents, 'Unknown candidate parent')
        seen.add(name)
        normalized.append(dict(entry, name=name))
    return dict(payload, candidates=normalized)


def check_proposal(view, before, names, iteration):
    names = allocated_slots(list(names))
    after = tree_hashes(view, ignore_cli_state=True)
    allowed = {f'harnesses/{name}/harness.py' for name in names}
    allowed.update({'pending_eval.json', f'logs/iteration_{iteration:03d}/report.md'})
    require(all(after.get(name) == digest for name, digest in before.items()), 'Proposer changed protected evidence')
    for name in after.keys() - before.keys():
        require(name in allowed or name.startswith('logs/claude_sessions/'), f'Unexpected proposer artifact: {name}')
    created = []
    for name in names:
        path = view / f'harnesses/{name}/harness.py'
        if path.exists():
            require(path.is_file() and 0 < path.stat().st_size <= 65536, 'Candidate is empty or oversized')
            created.append(name)
    require(bool(created), 'No allocated candidate was created')
    pending = view / 'pending_eval.json'
    if pending.exists():
        require(pending.stat().st_size <= 49152, 'Oversized candidate metadata')
        parents = {Path(name).parts[1] for name in before if re.fullmatch(r'harnesses/h[0-9]+/harness.py', name)}
        normalize_candidates(json.loads(pending.read_text()), names, parents)
    return sorted(name for name in allowed if (view / name).is_file())


def isolated_native_proposal(native, plan, output, journal, **kwargs):
    names = allocated_slots(kwargs.pop('next_names'))
    iteration = kwargs['iteration']
    require(type(iteration) is int and 1 <= iteration <= 5, 'Outside frozen pilot search iterations')
    require(kwargs['proposer_model'] == MODEL and plan['data_role'] == 'mh_val', 'Wrong proposer or data role')
    require(all(type(plan[k]) is int and 1 <= plan[k] <= 12 for k in ('max_api_requests', 'max_cli_turns')),
            'Unbounded proposer calls or turns')
    require(type(kwargs['timeout_seconds']) is int and 1 <= kwargs['timeout_seconds'] <= 300, 'Unbounded proposer timeout')
    original_run = Path(kwargs['run_dir']).resolve()
    output = Path(output).resolve()
    original_before = tree_hashes(original_run)
    require(all(f'harnesses/{name}/harness.py' not in original_before for name in names), 'Candidate slot already occupied')
    output.mkdir(parents=True, exist_ok=False)
    view = output / 'proposer-workspace'
    shutil.copytree(original_run, view)
    # Native keeps previous pending metadata at the run root. It is already in
    # the evidence snapshot; the proposer must produce this iteration's slots.
    (view / 'pending_eval.json').unlink(missing_ok=True)
    before = tree_hashes(view)
    state = view / '.cli-state'
    for name in ('config', 'tmp', 'empty_plugins'):
        (state / name).mkdir(parents=True)
    root, wrapper = Path.cwd(), native.claude_wrapper
    original_build = wrapper.build_command

    def bounded_command(*args, **options):
        command = original_build(*args, **options)
        command.remove('--dangerously-skip-permissions')
        command[0] = str(root / 'data/claude-runtime-2.1.236/claude')
        command += ['--bare', '--no-session-persistence', '--max-turns', str(plan['max_cli_turns']),
                    '--permission-mode', 'dontAsk', '--debug-file', str(state / 'debug.log')]
        return [sys.executable, str(root / 'ours/confined_exec.py'), '--workspace', str(view), '--', *command]

    capture = output / 'raw-sse'
    capture.mkdir()
    with relay(journal, '/userhome/cs3/yihangc/.config/whale-delta/credentials.json',
               max_requests=plan['max_api_requests'], capture_directory=capture) as gateway:
        env = {k: v for k, v in os.environ.items() if not k.endswith(('API_KEY', 'AUTH_TOKEN')) and k not in
               {'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_HUB_TOKEN'}}
        env.update(ANTHROPIC_BASE_URL=gateway['base_url'], ANTHROPIC_API_KEY=gateway['token'],
            ANTHROPIC_DEFAULT_OPUS_MODEL=MODEL, ANTHROPIC_DEFAULT_SONNET_MODEL=MODEL,
            ANTHROPIC_DEFAULT_HAIKU_MODEL=MODEL, CLAUDE_CONFIG_DIR=str(state / 'config'),
            TMPDIR=str(state / 'tmp'), CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1',
            DISABLE_AUTOUPDATER='1', DISABLE_UPDATES='1', CLAUDE_CODE_MAX_OUTPUT_TOKENS='4096',
            CLAUDE_SETTING_SOURCES='project')
        bypass = ','.join(filter(None, [env.get('NO_PROXY', env.get('no_proxy', '')), '127.0.0.1', 'localhost']))
        env.update(NO_PROXY=bypass, no_proxy=bypass)
        with patch.dict(os.environ, env, clear=True), patch.object(wrapper, 'build_command', bounded_command), \
             patch.object(wrapper, '_EMPTY_PLUGIN_DIR', state / 'empty_plugins'):
            result = native.propose_claude(**dict(kwargs, run_dir=view))
    write_json(output / 'proposer-result.json', {'exit_code': result.exit_code,
        'duration_seconds': result.duration_seconds, 'budget': journal.summary(),
        'iteration': iteration, 'allocated_slots': list(names), 'cli_usd_estimate_is_provider_bill': False})
    allowed = check_proposal(view, before, names, iteration)
    require(tree_hashes(original_run) == original_before, 'Original search changed during proposer execution')
    parents = {Path(name).parts[1] for name in before if re.fullmatch(r'harnesses/h[0-9]+/harness.py', name)}
    for name in allowed:
        destination = original_run / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name == 'pending_eval.json':
            write_json(destination, normalize_candidates(json.loads((view / name).read_text()), names, parents))
        else:
            destination.write_bytes((view / name).read_bytes())
    metadata_source = 'proposer'
    if 'pending_eval.json' not in allowed:
        recovered = native.recover_pending_candidates(original_run, list(names))
        require(bool(recovered), 'Native candidate recovery found no new slots')
        write_json(original_run / 'pending_eval.json', normalize_candidates({'candidates': recovered}, names, parents))
        metadata_source = 'native_recovery'
    shutil.copytree(view / 'logs/claude_sessions', original_run / 'logs/claude_sessions', dirs_exist_ok=True)
    write_json(output / 'proposal-integrity.json', {'status': 'PASS', 'iteration': iteration,
        'allocated_slots': list(names), 'imported': allowed, 'metadata_source': metadata_source,
        'protected_files': len(before), 'original_before_sha256': original_before,
        'landlock_scope': 'disposable audited MH archive only'})
    return result
