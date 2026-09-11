"""Windows and AMD GPU telemetry collector using native DXGI, PDH, and PSAPI."""
from __future__ import annotations

import csv
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional


class LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", wintypes.WCHAR * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", LUID),
        ("Flags", wintypes.UINT),
    ]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


class PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [
        ("CStatus", wintypes.DWORD),
        ("Padding", wintypes.DWORD),
        ("largeValue", ctypes.c_int64),
    ]

_AMD_VENDOR_ID = 0x1002
_DXGI_ADAPTER_FLAG_SOFTWARE = 0x2
_MIN_MONITORED_DEDICATED_BYTES = 2 ** 30


def _select_monitored_adapter(adapters, target_substring=None):
    """Pick one enumerated adapter for Windows monitoring, else None.

    Each adapter is a mapping with description, vendor_id, flags and
    dedicated_bytes (extra keys carried through). An explicit substring
    must match exactly one description (case-insensitive); without a
    filter, auto-select requires exactly one physical AMD adapter
    (VendorId 0x1002, software flag bit clear) with dedicated VRAM
    above 1 GiB. Ambiguous or missing candidates yield None so the
    caller reports unknown capacity instead of choosing silently.
    """
    if target_substring:
        matches = [a for a in adapters
                   if str(target_substring).lower() in str(a.get("description", "")).lower()]
        return matches[0] if len(matches) == 1 else None
    candidates = [a for a in adapters
                  if a.get("vendor_id") == _AMD_VENDOR_ID
                  and not (int(a.get("flags", 0)) & _DXGI_ADAPTER_FLAG_SOFTWARE)
                  and int(a.get("dedicated_bytes", 0)) > _MIN_MONITORED_DEDICATED_BYTES]
    return candidates[0] if len(candidates) == 1 else None


def _release_com(ptr) -> None:
    """Release one IUnknown-derived COM pointer; never raises."""
    try:
        if ptr is None or not getattr(ptr, "value", None):
            return
        vtbl = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        release_fn = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(vtbl[2])
        release_fn(ptr)
    except Exception:
        pass


def _discover_gpu_adapter(target_substring: Optional[str] = None) -> tuple[str, str, int, int]:
    """Discover the Windows monitoring adapter, returning (luid_str, description, vram_bytes, shared_bytes).

    With no substring, auto-selects only when exactly one physical AMD
    adapter with dedicated VRAM above 1 GiB exists; an explicit
    substring must also match exactly one adapter. Ambiguous or missing
    candidates return unknown capacity ("", "", 0, 0) instead of
    choosing silently.
    """
    if os.name != "nt":
        return "", "Non-Windows fallback", 0, 0
    try:
        dxgi = ctypes.oledll.dxgi
        ole32 = ctypes.oledll.ole32

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", wintypes.BYTE * 8),
            ]

        iid_factory1 = GUID()
        ole32.IIDFromString(wintypes.LPCWSTR("{770aae78-f26f-4dba-a829-253c83d1b387}"), ctypes.byref(iid_factory1))
        factory = ctypes.c_void_p()
        hr = dxgi.CreateDXGIFactory1(ctypes.byref(iid_factory1), ctypes.byref(factory))
        if hr != 0:
            return "", "", 0, 0
        adapters: list[dict[str, Any]] = []
        try:
            vtbl = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            enum_fn = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))(vtbl[12])
            idx = 0
            while True:
                adapter = ctypes.c_void_p()
                try:
                    hr = enum_fn(factory, idx, ctypes.byref(adapter))
                    if hr != 0:
                        break
                except OSError:
                    break
                idx += 1
                try:
                    avtbl = ctypes.cast(adapter, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
                    get_desc_fn = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, ctypes.POINTER(DXGI_ADAPTER_DESC1))(avtbl[10])
                    desc = DXGI_ADAPTER_DESC1()
                    get_desc_fn(adapter, ctypes.byref(desc))
                    adapters.append({
                        "luid": f"luid_0x{desc.AdapterLuid.HighPart:08x}_0x{desc.AdapterLuid.LowPart:08x}",
                        "description": desc.Description,
                        "vendor_id": int(desc.VendorId),
                        "flags": int(desc.Flags),
                        "dedicated_bytes": int(desc.DedicatedVideoMemory),
                        "shared_bytes": int(desc.SharedSystemMemory),
                    })
                finally:
                    _release_com(adapter)
        finally:
            _release_com(factory)
        selected = _select_monitored_adapter(adapters, target_substring)
        if selected is None:
            return "", "", 0, 0
        return (selected["luid"], selected["description"],
                int(selected["dedicated_bytes"]), int(selected["shared_bytes"]))
    except Exception:
        pass
    return "", "", 0, 0


def _expand_pdh_wildcard(counter_mask: str) -> list[str]:
    if os.name != "nt":
        return []
    try:
        pdh = ctypes.windll.pdh
        buf_size = wintypes.DWORD(0)
        pdh.PdhExpandWildCardPathW(None, counter_mask, None, ctypes.byref(buf_size), 0)
        if buf_size.value == 0:
            return []
        buf = (wintypes.WCHAR * buf_size.value)()
        res = pdh.PdhExpandWildCardPathW(None, counter_mask, buf, ctypes.byref(buf_size), 0)
        if res == 0:
            raw = buf[: buf_size.value]
            return [s for s in raw.split("\x00") if s]
    except Exception:
        pass
    return []


class TelemetryCollector:
    """Collects continuous GPU and host memory metrics during local inference experiments."""

    def __init__(
        self,
        run_id: str,
        target_gpu_substring: Optional[str] = None,
        target_pid: Optional[int] = None,
        interval_sec: float = 0.1,
    ):
        self.run_id = run_id
        self.target_gpu_substring = target_gpu_substring
        self.target_pid = target_pid
        self.interval_sec = interval_sec
        self.current_stage = "idle"
        self.stage_notes: str = ""
        self.prompt_tokens: Optional[int] = None
        self.generated_tokens: Optional[int] = None

        self.gpu_luid = ""
        self.gpu_name = ""
        self.gpu_total_dedicated_bytes = 0
        self.gpu_total_shared_bytes = 0

        self._samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._pdh_query = None
        self._h_local = None
        self._h_non_local = None
        self._init_backend()

    def _init_backend(self) -> None:
        if os.name != "nt":
            return
        luid, name, dedicated, shared = _discover_gpu_adapter(self.target_gpu_substring)
        self.gpu_luid = luid
        self.gpu_name = name
        self.gpu_total_dedicated_bytes = dedicated
        self.gpu_total_shared_bytes = shared

        if not luid:
            return

        try:
            pdh = ctypes.windll.pdh
            h_query = wintypes.HANDLE()
            if pdh.PdhOpenQueryW(None, 0, ctypes.byref(h_query)) != 0:
                return
            self._pdh_query = h_query

            local_paths = _expand_pdh_wildcard(rf"\GPU Local Adapter Memory(*{luid}*)\Local Usage")
            if local_paths:
                h_c = wintypes.HANDLE()
                if pdh.PdhAddEnglishCounterW(h_query, local_paths[0], 0, ctypes.byref(h_c)) == 0:
                    self._h_local = h_c

            non_local_paths = _expand_pdh_wildcard(rf"\GPU Non Local Adapter Memory(*{luid}*)\Non Local Usage")
            if non_local_paths:
                h_nc = wintypes.HANDLE()
                if pdh.PdhAddEnglishCounterW(h_query, non_local_paths[0], 0, ctypes.byref(h_nc)) == 0:
                    self._h_non_local = h_nc
        except Exception:
            pass

    def set_pid(self, pid: int) -> None:
        self.target_pid = pid

    def set_stage(
        self,
        stage: str,
        notes: str = "",
        prompt_tokens: Optional[int] = None,
        generated_tokens: Optional[int] = None,
    ) -> None:
        with self._lock:
            self.current_stage = stage
            self.stage_notes = notes
            if prompt_tokens is not None:
                self.prompt_tokens = prompt_tokens
            if generated_tokens is not None:
                self.generated_tokens = generated_tokens

    def sample_now(self) -> dict[str, Any]:
        ts = datetime.now(timezone.utc).isoformat()
        gpu_local_bytes = None
        gpu_non_local_bytes = None

        if self._pdh_query and os.name == "nt":
            try:
                pdh = ctypes.windll.pdh
                if pdh.PdhCollectQueryData(self._pdh_query) == 0:
                    fmt_large = 0x00000400
                    dw_type = wintypes.DWORD()
                    if self._h_local:
                        val = PDH_FMT_COUNTERVALUE()
                        if pdh.PdhGetFormattedCounterValue(self._h_local, fmt_large, ctypes.byref(dw_type), ctypes.byref(val)) == 0:
                            gpu_local_bytes = int(val.largeValue)
                    if self._h_non_local:
                        val_nl = PDH_FMT_COUNTERVALUE()
                        if pdh.PdhGetFormattedCounterValue(self._h_non_local, fmt_large, ctypes.byref(dw_type), ctypes.byref(val_nl)) == 0:
                            gpu_non_local_bytes = int(val_nl.largeValue)
            except Exception:
                pass

        host_avail_bytes = None
        host_total_bytes = None
        if os.name == "nt":
            try:
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)) != 0:
                    host_avail_bytes = int(stat.ullAvailPhys)
                    host_total_bytes = int(stat.ullTotalPhys)
            except Exception:
                pass

        proc_ws_bytes = None
        proc_commit_bytes = None
        if self.target_pid and os.name == "nt":
            try:
                psapi = ctypes.windll.psapi
                kernel32 = ctypes.windll.kernel32
                process_query_limited_information = 0x1000
                process_vm_read = 0x0010
                h_proc = kernel32.OpenProcess(process_query_limited_information | process_vm_read, False, self.target_pid)
                if h_proc:
                    try:
                        pmc = PROCESS_MEMORY_COUNTERS_EX()
                        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
                        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
                        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
                        if psapi.GetProcessMemoryInfo(h_proc, ctypes.byref(pmc), pmc.cb):
                            proc_ws_bytes = int(pmc.WorkingSetSize)
                            proc_commit_bytes = int(pmc.PrivateUsage)
                    finally:
                        kernel32.CloseHandle(h_proc)
            except Exception:
                pass

        device_free_bytes = None
        if gpu_local_bytes is not None and self.gpu_total_dedicated_bytes > 0:
            device_free_bytes = max(0, self.gpu_total_dedicated_bytes - gpu_local_bytes)

        with self._lock:
            record = {
                "timestamp_utc": ts,
                "run_id": self.run_id,
                "stage": self.current_stage,
                "collector": "pdh+dxgi+psapi",
                "scope": "device+process" if self.target_pid else "device",
                "pid": self.target_pid if self.target_pid else "",
                "dedicated_gpu_bytes": gpu_local_bytes if gpu_local_bytes is not None else "",
                "shared_gpu_bytes": gpu_non_local_bytes if gpu_non_local_bytes is not None else "",
                "host_rss_bytes": proc_ws_bytes if proc_ws_bytes is not None else "",
                "host_available_bytes": host_avail_bytes if host_avail_bytes is not None else "",
                "device_free_bytes": device_free_bytes if device_free_bytes is not None else "",
                "prompt_tokens": self.prompt_tokens if self.prompt_tokens is not None else "",
                "generated_tokens": self.generated_tokens if self.generated_tokens is not None else "",
                "notes": self.stage_notes,
            }
            self._samples.append(record)
            return record

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self.sample_now()
            self._stop_event.wait(self.interval_sec)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self.sample_now()
        self._thread = threading.Thread(target=self._loop, name="telemetry_collector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.sample_now()
        if self._pdh_query and os.name == "nt":
            try:
                ctypes.windll.pdh.PdhCloseQuery(self._pdh_query)
            except Exception:
                pass
            self._pdh_query = None

    def summary(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._samples)

        if not samples:
            return {}

        dedicated_vals = [int(s["dedicated_gpu_bytes"]) for s in samples if s.get("dedicated_gpu_bytes")]
        shared_vals = [int(s["shared_gpu_bytes"]) for s in samples if s.get("shared_gpu_bytes")]
        rss_vals = [int(s["host_rss_bytes"]) for s in samples if s.get("host_rss_bytes")]
        avail_vals = [int(s["host_available_bytes"]) for s in samples if s.get("host_available_bytes")]

        stages = {}
        for s in samples:
            st = s.get("stage", "unknown")
            ded = int(s["dedicated_gpu_bytes"]) if s.get("dedicated_gpu_bytes") else 0
            if st not in stages:
                stages[st] = {"min_dedicated_bytes": ded, "peak_dedicated_bytes": ded}
            else:
                stages[st]["peak_dedicated_bytes"] = max(stages[st]["peak_dedicated_bytes"], ded)
                if ded > 0:
                    stages[st]["min_dedicated_bytes"] = (
                        min(stages[st]["min_dedicated_bytes"], ded)
                        if stages[st]["min_dedicated_bytes"] > 0
                        else ded
                    )

        return {
            "gpu_name": self.gpu_name,
            "gpu_luid": self.gpu_luid,
            "gpu_total_dedicated_bytes": self.gpu_total_dedicated_bytes,
            "gpu_total_shared_bytes": self.gpu_total_shared_bytes,
            "sample_count": len(samples),
            "idle_dedicated_bytes": dedicated_vals[0] if dedicated_vals else 0,
            "peak_dedicated_bytes": max(dedicated_vals) if dedicated_vals else 0,
            "final_dedicated_bytes": dedicated_vals[-1] if dedicated_vals else 0,
            "peak_shared_bytes": max(shared_vals) if shared_vals else 0,
            "peak_proc_rss_bytes": max(rss_vals) if rss_vals else 0,
            "min_host_available_bytes": min(avail_vals) if avail_vals else 0,
            "stage_breakdown": stages,
        }

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "timestamp_utc",
            "run_id",
            "stage",
            "collector",
            "scope",
            "pid",
            "dedicated_gpu_bytes",
            "shared_gpu_bytes",
            "host_rss_bytes",
            "host_available_bytes",
            "device_free_bytes",
            "prompt_tokens",
            "generated_tokens",
            "notes",
        ]
        with self._lock:
            rows = list(self._samples)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
