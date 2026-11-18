#!/usr/bin/env python3
"""Run two designs of the independent test through the MCP workflow tool evaluate_design (Section 5.2).

  A046  north + west, floor-to-ceiling, 1.05 m wide, 0.62 m overhang  -> expected: MLP prediction accepted
  B003  north + south, floor-to-ceiling, 0.70 m wide, 1.28 m overhang -> expected: range contains 55 %, Radiance

The same function as the MCP tool is called (5_mcp/mcp_server.py), with the same decision rule, models and
Radiance settings; nothing in the workflow is changed. Drawing in Rhino is done separately.

    python 6_extensions/workflow_examples_independent.py

Output: thesis_data/results/extensions/workflow_examples/<id>_tool_result.json (full tool result with the fixed report),
        <id>_report.txt, and for the Radiance run the folder written by the tool in thesis_data/results/mcp_runs/.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "5_mcp"))
import mcp_server  # noqa: E402

OUT = ROOT / "thesis_data" / "results" / "extensions" / "workflow_examples"
DESIGNS = {
    "A046": dict(active_facades=["north", "west"], window_type="floor_to_ceiling", window_width_m=1.05,
                 overhang_depth_m=0.62),
    "B003": dict(active_facades=["north", "south"], window_type="floor_to_ceiling", window_width_m=0.70,
                 overhang_depth_m=1.28),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, design in DESIGNS.items():
        start = time.perf_counter()
        result = json.loads(mcp_server.evaluate_design(**design, draw_in_rhino=False))
        result["elapsed_s"] = time.perf_counter() - start
        (OUT / f"{name}_tool_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (OUT / f"{name}_report.txt").write_text(result["report"], encoding="utf-8")
        f = result.get("final") or {}
        print(f"{name}: decision {result.get('decision')}, source {f.get('source')}, "
              f"sDA {f.get('sDA_300_50_percent')}, ASE {f.get('ASE_1000_250_percent')}, "
              f"LEED {f.get('leed_level')}, {result['elapsed_s']:.1f} s")


if __name__ == "__main__":
    main()
