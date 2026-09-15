"""Kernel restrictions are exercised independently of AST rejection."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ours.isolated_visual_harness import load_isolated_visual_harness
from ours.visual_harness import BASE, load_visual_harness


class IsolatedVisualHarnessTests(unittest.TestCase):
    def test_baseline_callbacks_equal_and_source_mutation_rejected(self):
        native = str(Path(__file__).resolve().parents[2]/'upstream/WHALE/domains/chess_puzzles')
        with patch.object(sys, 'path', [native, *sys.path]):
            direct = load_visual_harness(BASE)
        isolated = load_isolated_visual_harness(BASE)
        self.assertEqual(isolated.system_prompt, direct.system_prompt)
        inputs = {'format_observation': {'question': 'Is the blue bar taller?'},
            'prepare_tool': {'arguments': {'bbox_2d': [0, 0, 32, 32]}, 'image_size': [64, 64]},
            'format_feedback': {'text': 'Visible crop.'}, 'parse_answer': {'text': 'A'},
            'nudge': {'text': 'A', 'assistant_turns': 1}}
        for name, visible in inputs.items():
            with self.subTest(name=name):
                self.assertEqual(isolated.invoke(name, **visible), direct.invoke(name, **visible))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'candidate.py'
            path.write_bytes(BASE.read_bytes())
            candidate = load_isolated_visual_harness(path)
            path.write_text(path.read_text()+'\n')
            with self.assertRaises(ValueError):
                candidate.invoke('parse_answer', text='A')

    def test_kernel_denies_reads_writes_truncate_metadata_network_and_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protected = root/'protected.txt'
            protected.write_text('private fixture')
            allowed = root/'allowed.txt'
            allowed.write_text('public fixture')
            package_root = str(Path(__file__).resolve().parents[2])
            script = '''
import sys,json,os,socket
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from ours.callback_confinement import confine
allowed,protected=sys.argv[2:4]
policy=confine([allowed])
assert Path(allowed).read_text()=='public fixture'
attempts={
 'read':lambda:Path(protected).read_text(),
 'write':lambda:Path(allowed).write_text('changed'),
 'truncate':lambda:os.truncate(protected,0),
 'chmod':lambda:os.chmod(protected,0o777),
 'network':lambda:socket.socket(socket.AF_INET,socket.SOCK_STREAM),
 'process':lambda:os.fork()}
denied=[]
for name,attempt in attempts.items():
 try: attempt()
 except PermissionError: denied.append(name)
 else: raise AssertionError('Kernel did not deny '+name)
print(json.dumps({'denied':denied,'policy':policy}))
'''
            result = subprocess.run([sys.executable, '-I', '-B', '-c', script, package_root, str(allowed), str(protected)],
                capture_output=True, text=True, timeout=10, env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(set(json.loads(result.stdout)['denied']), {'read','write','truncate','chmod','network','process'})
            self.assertEqual(protected.read_text(), 'private fixture')
            self.assertEqual(allowed.read_text(), 'public fixture')

    def test_nonterminating_candidate_is_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'candidate.py'
            source = BASE.read_text().replace('def parse_answer(text):', 'def parse_answer(text):\n    while True:\n        pass')
            path.write_text(source)
            candidate = load_isolated_visual_harness(path)
            with self.assertRaises((ValueError, subprocess.TimeoutExpired)):
                candidate.invoke('parse_answer', text='A')


if __name__ == '__main__':
    unittest.main()
