from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live_mvp_app import format_inches_compact

LIVE_SCRIPT = (REPO_ROOT / "static" / "js" / "live.js").read_text(encoding="utf-8")
HISTORY_SCRIPT = (REPO_ROOT / "static" / "js" / "history.js").read_text(
    encoding="utf-8"
)


class MeasurementFormatTests(unittest.TestCase):
    def test_compact_format_reduces_fraction_to_sixteenths(self) -> None:
        cases = {
            489.0625: "40' 9 1/16\"",
            121.5: "10' 1 1/2\"",
            120.0: "10' 0\"",
            0.03125: "0' 0 1/16\"",
            -0.0625: "-0' 0 1/16\"",
        }

        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(format_inches_compact(value), expected)

        self.assertEqual(format_inches_compact(None), "-")

    def test_live_and_history_use_compact_measurement_format(self) -> None:
        self.assertIn("function compactMeasurement(value)", LIVE_SCRIPT)
        self.assertIn(
            "compactMeasurement(piece.measurement?.measurement_in)",
            LIVE_SCRIPT,
        )
        self.assertNotIn("return `${feet} ft", HISTORY_SCRIPT)
        self.assertIn("return `${sign}${feet}' ${inchText}\"`;", HISTORY_SCRIPT)


if __name__ == "__main__":
    unittest.main()
