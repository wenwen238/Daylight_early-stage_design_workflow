#!/usr/bin/env python3
"""Fixed reports of the two independent-test workflow examples (A046, B003) for Appendix A.

The reports are rebuilt with the workflow's own report function (5_mcp/report.py) from the saved tool results
(thesis_data/results/extensions/workflow_examples/<id>_tool_result.json), with the design name in the title. Both designs were drawn
in Rhino after the evaluation (same drawing function as the MCP tool), so the report states the Rhino output
as the tool does when it draws. No value is recalculated and no new Radiance run is made.

    python 6_extensions/workflow_examples_reports.py

Output: thesis_data/results/extensions/workflow_examples/<id>_report_appendix.md
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "5_mcp"))
import report  # noqa: E402

DIR = ROOT / "thesis_data" / "results" / "extensions" / "workflow_examples"
RHINO = "drawn in Rhino: 24 surfaces, DA300 map, ASE map on a copy of the room 8 m to the east"


def main():
    for name, maps in (("A046", "predicted"), ("B003", "simulated")):
        result = json.loads((DIR / f"{name}_tool_result.json").read_text(encoding="utf-8"))
        result.pop("report", None)
        result.pop("elapsed_s", None)
        result["rhino"] = RHINO
        text = report.evaluation_report(result, maps, name=name)
        (DIR / f"{name}_report_appendix.md").write_text(text, encoding="utf-8")
        print(text, "\n")


if __name__ == "__main__":
    main()
