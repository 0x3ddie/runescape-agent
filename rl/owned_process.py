"""Contain only processes launched by this transport; never search other jobs.

Windows launches a Python trampoline that cannot spawn until its Job Object is
assigned. Closing the parent's non-inherited job handle kills all descendants,
including after an abrupt parent exit. POSIX uses a dedicated process session;
cleanup signals its group (descendants that deliberately detach are outside it).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path


def console_python(executable=None):
    """Use this interpreter's console sibling for subprocess protocol streams.

    A Windows GUI can be hosted by pythonw, but its workers and launch handshake
    need normal stdin/stdout. CREATE_NO_WINDOW still hides their console windows.
    """
    path = Path(executable or sys.executable)
    if path.name.lower() == "pythonw.exe":
        path = path.with_name("python.exe")
        if not path.is_file():
            raise RuntimeError(f"console Python interpreter missing: {path}")
    return str(path)


class _WindowsJob:
    def __init__(self):
        import ctypes
        from ctypes import wintypes as w

        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", w.DWORD), ("min_ws", ctypes.c_size_t),
                        ("max_ws", ctypes.c_size_t), ("active_limit", w.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", w.DWORD),
                        ("scheduling", w.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", IO), ("process_mem", ctypes.c_size_t),
                        ("job_mem", ctypes.c_size_t), ("peak_process", ctypes.c_size_t),
                        ("peak_job", ctypes.c_size_t)]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, args, result in (
            ("CreateJobObjectW", [w.LPVOID, w.LPCWSTR], w.HANDLE),
            ("SetInformationJobObject", [w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD], w.BOOL),
            ("AssignProcessToJobObject", [w.HANDLE, w.HANDLE], w.BOOL),
            ("CloseHandle", [w.HANDLE], w.BOOL),
        ):
            fn = getattr(self.api, name)
            fn.argtypes, fn.restype = args, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, proc):
        import ctypes
        if not self.api.AssignProcessToJobObject(self.handle, int(proc._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


class OwnedProcess:
    """Popen owner with terminate()/kill() covering its contained descendants.

    The caller supplies stdin=PIPE and text=True, owns the returned proc streams,
    and must call kill() even after a normal wait to release the containment.
    No other process handles or process-name searches are used.
    """
    def __init__(self, command: list[str], **kwargs):
        if kwargs.get("stdin") != subprocess.PIPE or not kwargs.get("text"):
            raise ValueError("OwnedProcess requires stdin=PIPE and text=True")
        self._released = False
        self.job = _WindowsJob() if os.name == "nt" else None
        self.proc = None
        try:
            if self.job:
                # -I -S excludes user startup hooks before the launch handshake.
                command = [console_python(), "-I", "-S", str(Path(__file__).resolve()), *command]
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            else:
                kwargs["start_new_session"] = True
            self.proc = subprocess.Popen(command, **kwargs)
            if self.job:
                self.job.assign(self.proc)
                self.proc.stdin.write("G")
                self.proc.stdin.flush()
        except BaseException:
            self.kill()
            if self.proc:
                self.proc.wait(timeout=10)
                for stream in (self.proc.stdin, self.proc.stdout):
                    if stream:
                        try:
                            stream.close()
                        except OSError:
                            pass
            raise

    def _signal_group(self, sig):
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Darwin reports EPERM when only the exited, unreaped leader remains.
            if self.proc.poll() is None:
                raise

    def kill(self):
        if self._released:
            return
        self._released = True
        if self.job:
            self.job.close()
            # Assignment may have failed; the blocked trampoline is still ours.
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
        elif self.proc:
            self._signal_group(signal.SIGKILL)


if __name__ == "__main__":
    # Read exactly one byte, avoiding buffered read-ahead of protocol requests.
    if os.read(sys.stdin.fileno(), 1) != b"G":
        raise SystemExit(1)
    raise SystemExit(subprocess.call(sys.argv[1:]))
