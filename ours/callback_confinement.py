"""Read-only Landlock plus seccomp for isolated visual callbacks on Linux.

Cluster kernel ABI1 needs explicit seccomp denial of truncate and metadata
mutations. No network/process creation is allowed after confinement. This is a
shared execution boundary, not a visual acceptance mechanism.
"""
import ctypes
import errno
import os
from pathlib import Path
import platform
import resource
import sys


def confine(read_paths):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Visual callback confinement requires Linux x86_64')
    libc = ctypes.CDLL(None, use_errno=True)
    seccomp = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise RuntimeError('Landlock unavailable; refusing callback execution')
    class Ruleset(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]
    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int)]
    handled = (1 << (15 if abi >= 3 else 14 if abi >= 2 else 13)) - 1
    attr = Ruleset(handled)
    ruleset = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), 'Cannot create read-only callback ruleset')
    allowed = ['/usr', '/lib', '/lib64', '/bin', '/etc/ld.so.cache', '/dev/urandom', '/dev/null',
        sys.prefix, sys.base_prefix, *map(str, read_paths)]
    try:
        for name in dict.fromkeys(allowed):
            path = Path(name).resolve()
            if not path.exists():
                continue
            access = 1 | 4 | (8 if path.is_dir() else 0)
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(access, fd)
                if libc.syscall(445, ruleset, 1, ctypes.byref(rule), 0) != 0:
                    raise OSError(ctypes.get_errno(), 'Cannot apply callback read allowance')
            finally:
                os.close(fd)
        if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.syscall(446, ruleset, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Cannot enforce callback Landlock')
    finally:
        os.close(ruleset)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    context = seccomp.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError('Cannot create callback syscall filter')
    blocked = ('socket', 'socketpair', 'connect', 'bind', 'listen', 'accept', 'accept4',
        'clone', 'clone3', 'fork', 'vfork', 'execve', 'execveat', 'ptrace',
        'process_vm_readv', 'process_vm_writev', 'kill', 'tkill', 'tgkill',
        'truncate', 'ftruncate', 'chmod', 'fchmod', 'fchmodat', 'fchmodat2',
        'chown', 'fchown', 'lchown', 'fchownat', 'utime', 'utimes', 'futimesat', 'utimensat',
        'setxattr', 'lsetxattr', 'fsetxattr', 'removexattr', 'lremovexattr', 'fremovexattr',
        'mount', 'umount2', 'pivot_root', 'unshare', 'setns', 'bpf', 'userfaultfd',
        'io_uring_setup', 'io_uring_enter', 'io_uring_register')
    try:
        for name in blocked:
            number = seccomp.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and seccomp.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0) != 0:
                raise RuntimeError('Cannot add callback syscall restriction')
        if seccomp.seccomp_load(context) != 0:
            raise RuntimeError('Cannot enforce callback syscall restriction')
    finally:
        seccomp.seccomp_release(context)
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return {'landlock_abi': abi, 'filesystem': 'declared_read_only', 'network': 'seccomp_denied',
        'process_creation': 'seccomp_denied', 'cpu_seconds': 2, 'address_space_mib': 512}
