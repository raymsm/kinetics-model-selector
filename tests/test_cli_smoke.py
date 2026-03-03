from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class CLISmokeTests(unittest.TestCase):
    def test_end_to_end_creates_artifacts(self) -> None:
        fixture = Path("tests/fixtures/pfo_clean.csv")
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            cmd = [
                "python",
                "-m",
                "kinetics_model_selector.cli",
                "--input",
                str(fixture),
                "--outdir",
                str(outdir),
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            summary = outdir / "summary.json"
            mc_samples = outdir / "mc_samples.csv"
            self.assertTrue(summary.exists())
            self.assertTrue(mc_samples.exists())

            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertIn("best_model", payload)
            self.assertIn("intervals", payload)


if __name__ == "__main__":
    unittest.main()
