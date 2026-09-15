"""One stateless callback request after kernel confinement, via bounded JSON."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT/'upstream/WHALE/domains/chess_puzzles')]

from ours.callback_confinement import confine
from ours.visual_harness import load_visual_harness
# Load the pinned trusted validator before directory enumeration is restricted.
# Candidate source is not opened or executed until after kernel confinement.
import autoharness_chess_puzzle.harness


def main():
    raw = sys.stdin.buffer.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError('Visual callback request exceeds its bound')
    request = json.loads(raw)
    candidate = Path(request['path']).resolve()
    reference = Path(request['prompt_reference']).resolve() if request.get('prompt_reference') else None
    policy = confine([ROOT/'ours', ROOT/'upstream/WHALE/domains/chess_puzzles/autoharness_chess_puzzle/harness.py',
        ROOT/'upstream/WHALE/domains/chess_puzzles/autoharness_chess_puzzle/__init__.py',
        candidate, *([reference] if reference else [])])
    harness = load_visual_harness(candidate, prompt_reference=reference)
    if harness.sha256 != request['sha256']:
        raise ValueError('Candidate changed before confined execution')
    if request['operation'] == 'describe':
        value = {'system_prompt': harness.system_prompt}
    elif request['operation'] == 'invoke':
        value = harness.invoke(request['name'], **request['visible'])
    else:
        raise ValueError('Unknown visual callback operation')
    reply = json.dumps({'sha256': harness.sha256, 'value': value, 'confinement': policy}, allow_nan=False)
    if len(reply.encode()) > 1024 * 1024:
        raise ValueError('Visual callback response exceeds its bound')
    print(reply)


if __name__ == '__main__':
    main()
