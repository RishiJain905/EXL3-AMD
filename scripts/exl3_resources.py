"""Portable resource policy; Torch-free, CPU-only.

Detected totals drive all byte thresholds. Allocator and monitor
fractions stay explicit CLI in (0,1), with positive finite GiB guards
for free RAM and disk, so any GPU and host work without code changes.
"""
import math

GIB = 2 ** 30

GPU_MEMORY_FRACTION_DEFAULT = 0.90
MAX_GPU_MEMORY_FRACTION_DEFAULT = 0.95
MAX_HOST_MEMORY_FRACTION_DEFAULT = 0.50
MIN_FREE_RAM_GIB_DEFAULT = 1.0
MIN_FREE_DISK_GIB_DEFAULT = 1.0

MONITOR_KEYS = ('max_host_memory_fraction', 'max_gpu_memory_fraction',
                'min_free_ram_gib', 'min_free_disk_gib')

_DEFAULTS = {
    'max_host_memory_fraction': MAX_HOST_MEMORY_FRACTION_DEFAULT,
    'max_gpu_memory_fraction': MAX_GPU_MEMORY_FRACTION_DEFAULT,
    'min_free_ram_gib': MIN_FREE_RAM_GIB_DEFAULT,
    'min_free_disk_gib': MIN_FREE_DISK_GIB_DEFAULT,
}


def validate_fraction(value, name):
    """Finite 0<f<1 allocator/monitor fraction, else ValueError."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError('%s must be a float in (0,1)' % name)
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise ValueError('%s must be a finite float in (0,1)' % name)
    return number


def validate_positive_gib(value, name):
    """Finite positive GiB guard, else ValueError."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError('%s must be a positive finite GiB value' % name)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError('%s must be a positive finite GiB value' % name)
    return number


def allocator_bytes(fraction, total_bytes):
    return int(float(fraction) * int(total_bytes))


def rss_limit_bytes(fraction, memtotal_bytes):
    return int(float(fraction) * int(memtotal_bytes))


def free_bytes(gib):
    return int(float(gib) * GIB)


def add_allocator_arg(parser):
    parser.add_argument('--gpu-memory-fraction', type=float,
                        default=GPU_MEMORY_FRACTION_DEFAULT,
                        help='Torch allocator fraction of detected device total in (0,1)')
    return parser


def add_monitor_args(parser):
    parser.add_argument('--max-gpu-memory-fraction', type=float,
                        default=MAX_GPU_MEMORY_FRACTION_DEFAULT,
                        help='Windows adapter dedicated usage fraction of detected total in (0,1)')
    parser.add_argument('--max-host-memory-fraction', type=float,
                        default=MAX_HOST_MEMORY_FRACTION_DEFAULT,
                        help='WSL process-group RSS fraction of detected MemTotal in (0,1)')
    parser.add_argument('--min-free-ram-gib', type=float,
                        default=MIN_FREE_RAM_GIB_DEFAULT,
                        help='Minimum free RAM guard in GiB; positive finite')
    parser.add_argument('--min-free-disk-gib', type=float,
                        default=MIN_FREE_DISK_GIB_DEFAULT,
                        help='Minimum free disk guard in GiB; positive finite')
    return parser


def add_resource_args(parser):
    add_allocator_arg(parser)
    add_monitor_args(parser)
    return parser


def limits_from_args(args):
    """Monitor limits dict serialized into request.json."""
    return {key: float(getattr(args, key)) for key in MONITOR_KEYS}


def wsl_limits(request):
    """WSL worker limits with portable defaults for older saved requests.

    Requests written before limits serialization lack request['limits'];
    fall back to module defaults and sanitize non-finite or
    out-of-range values so monitoring stays bounded on any host.
    """
    saved = request.get('limits') if isinstance(request, dict) else None
    if not isinstance(saved, dict):
        saved = {}
    limits = {}
    for key, default in _DEFAULTS.items():
        try:
            value = float(saved.get(key, default))
        except (TypeError, ValueError):
            value = default
        if key in ('max_host_memory_fraction', 'max_gpu_memory_fraction'):
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                value = default
        else:
            if not math.isfinite(value) or value <= 0.0:
                value = default
        limits[key] = value
    return limits


def parse_meminfo(text):
    """(MemTotal, MemAvailable) in bytes; None each when missing."""
    total = available = None
    for line in str(text).splitlines():
        if line.startswith('MemTotal:'):
            try:
                total = int(line.split()[1]) * 1024
            except (ValueError, IndexError):
                total = None
        elif line.startswith('MemAvailable:'):
            try:
                available = int(line.split()[1]) * 1024
            except (ValueError, IndexError):
                available = None
    return total, available
