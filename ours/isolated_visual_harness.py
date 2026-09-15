"""Stateless JSON callback execution outside native data and model processes."""
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

from .visual_harness import BASE, SIGNATURES
from .visual_task import file_sha256


@dataclass(frozen=True)
class IsolatedVisualHarness:
    path: Path
    sha256: str
    source: str
    prompt_reference: str | None
    system_prompt: str

    def unchanged(self):
        if file_sha256(self.path) != self.sha256:
            raise ValueError('Isolated visual harness changed after construction')

    def invoke(self, name, **visible):
        self.unchanged()
        if name not in SIGNATURES or set(visible) != set(SIGNATURES[name]):
            raise ValueError('Unexpected isolated callback inputs')
        return execute(self.path, self.sha256, self.prompt_reference, 'invoke', name=name, visible=visible)['value']


def execute(path, digest, reference, operation, **kwargs):
    request = json.dumps({'path': str(path), 'sha256': digest, 'prompt_reference': reference,
        'operation': operation, **kwargs}, allow_nan=False)
    if len(request.encode()) > 1024 * 1024:
        raise ValueError('Visual callback request exceeds its bound')
    worker = Path(__file__).with_name('visual_callback_worker.py')
    result = subprocess.run([sys.executable, '-I', '-B', str(worker)], input=request, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, close_fds=True,
        cwd=worker.parent, env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})
    if result.returncode:
        raise ValueError(f'Isolated visual callback failed (exit {result.returncode}): {result.stderr[-2000:]}')
    if len(result.stdout.encode()) > 1024 * 1024:
        raise ValueError('Visual callback response exceeds its bound')
    reply = json.loads(result.stdout)
    if reply['sha256'] != digest or reply['confinement']['network'] != 'seccomp_denied':
        raise ValueError('Unverified callback execution identity or confinement')
    return reply


def load_isolated_visual_harness(path=None, *, prompt_reference=None):
    path = Path(path or BASE).resolve()
    source = path.read_text()
    if path.stat().st_size > 65536:
        raise ValueError('Visual candidate exceeds 64KiB')
    digest = file_sha256(path)
    reference = str(Path(prompt_reference).resolve()) if prompt_reference else None
    description = execute(path, digest, reference, 'describe')
    harness = IsolatedVisualHarness(path, digest, source, reference, description['value']['system_prompt'])
    harness.unchanged()
    return harness
