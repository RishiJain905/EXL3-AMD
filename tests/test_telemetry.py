"""CPU-only telemetry adapter selection tests; no DXGI, GPU, or network."""
import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quantlab import telemetry

GIB = 2 ** 30
AMD = 0x1002
NVIDIA = 0x10DE
SOFTWARE_FLAG = 0x2


def _adapter(description, *, vendor_id=AMD, flags=0, dedicated_gib=16,
             shared_gib=8, luid="luid-test"):
    return {"luid": luid, "description": description, "vendor_id": vendor_id,
            "flags": flags, "dedicated_bytes": int(dedicated_gib * GIB),
            "shared_bytes": int(shared_gib * GIB)}


class SelectionTests(unittest.TestCase):
    def test_unique_amd_auto_selected(self):
        dgpu = _adapter("AMD Radeon RX 7800 XT", luid="luid-dgpu")
        selected = telemetry._select_monitored_adapter([dgpu])
        self.assertIs(selected, dgpu)
        self.assertEqual(selected["luid"], "luid-dgpu")

    def test_igpu_plus_dgpu_selects_dgpu(self):
        igpu = _adapter("AMD Radeon Graphics", dedicated_gib=0.5, luid="luid-igpu")
        dgpu = _adapter("AMD Radeon RX 9070 XT", dedicated_gib=16, luid="luid-dgpu")
        self.assertIs(telemetry._select_monitored_adapter([igpu, dgpu]), dgpu)
        self.assertIs(telemetry._select_monitored_adapter([dgpu, igpu]), dgpu)

    def test_ambiguous_amd_dgpus_yield_none(self):
        first = _adapter("AMD Radeon RX 7800 XT", luid="luid-first")
        second = _adapter("AMD Radeon RX 9070 XT", luid="luid-second")
        self.assertIsNone(telemetry._select_monitored_adapter([first, second]))

    def test_explicit_match_selects_one(self):
        first = _adapter("AMD Radeon RX 7800 XT", luid="luid-first")
        second = _adapter("AMD Radeon RX 9070 XT", luid="luid-second")
        self.assertIs(telemetry._select_monitored_adapter([first, second], "9070"), second)
        self.assertIs(telemetry._select_monitored_adapter([first, second], "radeon rx 7800"), first)

    def test_explicit_match_requires_unique(self):
        first = _adapter("AMD Radeon RX 7800 XT", luid="luid-first")
        second = _adapter("AMD Radeon RX 9070 XT", luid="luid-second")
        self.assertIsNone(telemetry._select_monitored_adapter([first, second], "radeon"))
        self.assertIsNone(telemetry._select_monitored_adapter([first, second], ""))

    def test_no_match_yields_none(self):
        dgpu = _adapter("AMD Radeon RX 7800 XT")
        self.assertIsNone(telemetry._select_monitored_adapter([dgpu], "GeForce"))
        self.assertIsNone(telemetry._select_monitored_adapter([], "Radeon"))
        self.assertIsNone(telemetry._select_monitored_adapter([]))

    def test_software_adapter_ignored(self):
        warp = _adapter("Microsoft Basic Render Driver", vendor_id=AMD,
                        flags=SOFTWARE_FLAG, dedicated_gib=16, luid="luid-warp")
        self.assertIsNone(telemetry._select_monitored_adapter([warp]))
        dgpu = _adapter("AMD Radeon RX 7800 XT", luid="luid-dgpu")
        self.assertIs(telemetry._select_monitored_adapter([warp, dgpu]), dgpu)

    def test_non_amd_ignored(self):
        nvidia = _adapter("NVIDIA GeForce RTX 5090", vendor_id=NVIDIA,
                          dedicated_gib=32, luid="luid-nvidia")
        self.assertIsNone(telemetry._select_monitored_adapter([nvidia]))
        dgpu = _adapter("AMD Radeon RX 7800 XT", luid="luid-dgpu")
        self.assertIs(telemetry._select_monitored_adapter([nvidia, dgpu]), dgpu)

    def test_boundary_1gib_not_selected(self):
        small = _adapter("AMD Radeon Graphics", dedicated_gib=1.0)
        self.assertIsNone(telemetry._select_monitored_adapter([small]))


class DefaultTests(unittest.TestCase):
    def test_discovery_defaults_to_no_name_constraint(self):
        params = inspect.signature(telemetry._discover_gpu_adapter).parameters
        self.assertIn("target_substring", params)
        self.assertIsNone(params["target_substring"].default)

    def test_collector_defaults_to_no_name_constraint(self):
        params = inspect.signature(telemetry.TelemetryCollector).parameters
        self.assertIn("target_gpu_substring", params)
        self.assertIsNone(params["target_gpu_substring"].default)


class CollectorTests(unittest.TestCase):
    def test_unknown_capacity_without_dxgi(self):
        with patch.object(telemetry.os, "name", "nt"), patch.object(telemetry, "_discover_gpu_adapter",
                          return_value=("", "", 0, 0)) as discover:
            collector = telemetry.TelemetryCollector("run-under-test")
        discover.assert_called_once_with(None)
        self.assertEqual(collector.gpu_luid, "")
        self.assertEqual(collector.gpu_name, "")
        self.assertEqual(collector.gpu_total_dedicated_bytes, 0)
        self.assertEqual(collector.gpu_total_shared_bytes, 0)

    def test_explicit_filter_forwarded_without_dxgi(self):
        with patch.object(telemetry.os, "name", "nt"), patch.object(telemetry, "_discover_gpu_adapter",
                          return_value=("", "", 0, 0)) as discover:
            collector = telemetry.TelemetryCollector("run-under-test",
                                                     target_gpu_substring="Radeon")
        discover.assert_called_once_with("Radeon")
        self.assertEqual(collector.target_gpu_substring, "Radeon")
        self.assertEqual(collector.gpu_total_dedicated_bytes, 0)


if __name__ == "__main__":
    unittest.main()
