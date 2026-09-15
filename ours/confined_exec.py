"""Restrict a disposable proposer subprocess with Linux Landlock before exec.

ABI 1 protects file reads/writes below the allowed paths. It does not provide
network confinement or control metadata-only operations. This launcher never
continues without successful kernel enforcement. No API credential is loaded.
"""
import argparse
import ctypes
import os
from pathlib import Path
import platform


def restrict(workspace, executable, extra_read=()):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This launcher requires Linux x86_64")
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise RuntimeError("Landlock is unavailable; refusing unconfined execution")

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]

    handled = (1 << 13) - 1  # All ABI 1 filesystem actions.
    attr = Ruleset(handled)
    ruleset = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "Create Landlock ruleset failed")
    readonly = ["/usr", "/lib", "/lib64", "/bin", "/etc/ld.so.cache", "/etc/ssl",
                "/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf", "/etc/passwd",
                "/etc/group", "/etc/localtime", "/proc/self", "/proc/thread-self",
                "/proc/cpuinfo", "/proc/meminfo", "/proc/stat", "/dev/urandom", "/dev/random",
                str(Path(executable).resolve()), *map(str, extra_read)]
    paths = [(p, 1 | 4 | 8) for p in readonly]
    paths += [(str(Path(workspace).resolve()), handled), ("/dev/null", 2 | 4)]
    try:
        for text, access in paths:
            path = Path(text)
            if not path.exists():
                continue
            if not path.is_dir():
                access &= 1 | 2 | 4
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(access, fd)
                if libc.syscall(445, ruleset, 1, ctypes.byref(rule), 0) != 0:
                    raise OSError(ctypes.get_errno(), "Add Landlock path rule failed")
            finally:
                os.close(fd)
        if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.syscall(446, ruleset, 0) != 0:
            raise OSError(ctypes.get_errno(), "Enforce Landlock failed")
    finally:
        os.close(ruleset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--read", type=Path, action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not Path(command[0]).is_absolute():
        parser.error("An absolute executable path is required")
    restrict(args.workspace, command[0], args.read)
    os.execv(command[0], command)
