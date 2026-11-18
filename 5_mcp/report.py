"""Fixed report of the workflow tools: the text the assistant shows to the designer.

The report is built here from the tool results, so its structure does not depend on the assistant
and every value comes from decision.py, suggest.py or Radiance. Nothing is calculated here except
which metric limits the design.

evaluation_report(result, maps, name)
    Six parts in a fixed order: 1 Result, 2 Design inputs, 3 Basis of the result, 4 Daylight maps
    (only with maps), 5 Notes, 6 Next step. Parts 3 and 4 depend on the route of the design.
rejection_report(problems)
    Short message for a design that cannot be built; there is no result to report.
suggestion_report(result)
    Current design, target and the list of suggested changes.
"""

SDA, ASE = "sDA_300_50_percent", "ASE_1000_250_percent"
LABEL = {SDA: "sDA300/50%", ASE: "ASE1000,250h"}
BENCHMARK_NOTE = "The LEED level is a comparative benchmark, not a certification result."
MAPS = {
    "predicted": "Predicted 10 × 10 maps of DA300 and of direct-sun hours (PCA + MLP). "
                 "They are advisory and do not affect the decision.",
    "simulated": "Simulated 32 × 32 maps of DA300 and of direct-sun hours from the Radiance run.",
    "stored": "Stored 32 × 32 maps of DA300 and of direct-sun hours from the Radiance result of this case.",
}


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _drawn(result):
    return str(result.get("rhino", "")).startswith("drawn")


def _result_part(result):
    final = result.get("final")
    if not final:
        return "**1. Result**\nNot available: a Radiance simulation is required but was not run."
    return "**1. Result**\n" + _table(
        [LABEL[SDA], LABEL[ASE], "LEED level", "Source"],
        [[f"{final[SDA]:.1f}%", f"{final[ASE]:.1f}%", final["leed_level"], final["source"]]])


def _inputs_part(result, heading="2. Design inputs"):
    d, coverage = result["design"], result["coverage"]
    window = (f"{d['window_type'].replace('_', '-')} window, {d['window_width_m']:.2f} m wide × "
              f"{d['window_height_m']:.2f} m high, sill {d['sill_height_m']:.2f} m")
    overhang = f"overhang {d['overhang_depth_m']:.2f} m" if d["overhang_depth_m"] > 0 else "no overhang"
    check = ("passed (inside the sampled design space)" if coverage["in_design_space"]
             else "not passed (" + "; ".join(coverage["outside"]) + ")")
    return (f"**{heading}**\nWindows on: {', '.join(d['active_facades'])}; {window}; {overhang}.\n"
            f"Input coverage check: {check}.")


def _prediction_table(result):
    final = result.get("final") or {}
    simulated = final.get("source") != "MLP ensemble" and bool(final)
    header = ["Metric", "Prediction", "90% range", "Thresholds in range", "Spread level"]
    rows = []
    for t in (SDA, ASE):
        m = result["mlp"][t]
        thresholds = ", ".join(f"{v:g}%" for v in m["thresholds_in_range"]) or "none"
        rows.append([LABEL[t], f"{m['prediction']:.1f}%", f"{m['range'][0]:.1f}–{m['range'][1]:.1f}%", thresholds,
                     f"{m['spread_level']} ({m['spread_pp']:.2f} pp)"] + ([f"{final[t]:.1f}%"] if simulated else []))
    return _table(header + (["Radiance"] if simulated else []), rows)


def _basis_part(result):
    decision = result.get("decision")
    if decision == "mlp":
        lines = ["MLP prediction accepted: no LEED threshold within the 90% error margins.", _prediction_table(result)]
    elif decision == "radiance" and "mlp" in result:
        lines = ["Radiance simulation: a LEED threshold lies within the 90% range of a prediction, "
                 "so the LEED outcome is uncertain.", _prediction_table(result)]
    elif decision == "radiance":
        lines = ["Radiance simulation: the design is outside the sampled design space, so no prediction is available."]
    else:
        lines = ["Radiance simulation requested by the designer."]
    if "stored_radiance" in result and decision == "mlp":
        s = result["stored_radiance"]
        lines.append(f"Stored Radiance result of this case ({result['split']} set): sDA {s[SDA]:.1f}%, "
                     f"ASE {s[ASE]:.1f}%, LEED level {s['leed_level']}.")
    if result.get("radiance_output"):
        lines.append(f"Simulation files: {result['radiance_output']}")
    return "**3. Basis of the result**\n" + "\n\n".join(lines)


def _maps_part(result, maps):
    if not maps:
        return None
    text = MAPS[maps]
    if _drawn(result):
        text += (" In Rhino: DA300 map on the work plane (white line: 50%); direct-sun-hour map "
                 "on a copy of the room to the east (cyan line: 250 h).")
    return "**4. Daylight maps**\n" + text


def _notes_part(result):
    # the warnings concern the MLP prediction, which a Radiance result replaces
    final = result.get("final") or {}
    warnings = result.get("warnings", []) if final.get("source", "MLP ensemble") == "MLP ensemble" else []
    lines = [f"Warnings: {'; '.join(warnings) if warnings else 'none'}.", BENCHMARK_NOTE]
    if result.get("rhino") and not _drawn(result):
        lines.append(str(result["rhino"]))
    return "**5. Notes**\n" + " ".join(lines)


def _limiting_metric(final):
    sda, ase = final[SDA], final[ASE]
    if ase > 10.0:
        return f"Limiting metric: ASE ({ase:.1f}% > 10%)."
    if sda < 55.0:
        return f"Limiting metric: sDA ({sda:.1f}% < 55%)."
    if sda < 75.0:
        return f"2 points reached; sDA would have to reach 75% for 3 points (now {sda:.1f}%)."
    return "3 points reached, the highest level of the benchmark."


def _next_part(result):
    final = result.get("final")
    if not final:
        text = "Run the Radiance simulation to obtain a result."
    elif final[SDA] >= 75.0 and final[ASE] <= 10.0:
        text = _limiting_metric(final) + " You can change the design and evaluate it again."
    else:
        text = _limiting_metric(final) + " You can ask for suggested changes, or change the design and evaluate it again."
    return "**6. Next step**\n" + text


def evaluation_report(result, maps=None, name=None):
    """Report of an evaluated design. maps: None, "predicted", "simulated" or "stored"."""
    title = "**Design evaluation" + (f": {name}" if name else "") + "**"
    parts = [title, _result_part(result), _inputs_part(result), _basis_part(result),
             _maps_part(result, maps), _notes_part(result), _next_part(result)]
    return "\n\n".join(p for p in parts if p)


def rejection_report(problems):
    return "**Design rejected**\n\nThe design cannot be built: " + "; ".join(problems) + ". Please change the input."


def suggestion_report(result):
    """Report of suggest_changes."""
    current = result["current"]
    if SDA in current:
        now = f"sDA {current[SDA]:.1f}%, ASE {current[ASE]:.1f}%"
        if "leed_level" in current:
            now += f", LEED level {current['leed_level']}"
    else:
        now = current["note"].rstrip(".")
        now = now[0].upper() + now[1:]
    parts = ["**Design suggestions**", _inputs_part(result, "Design inputs"), f"**Current design**\n{now}."]
    if result.get("target") is None:
        parts.append(f"**Suggestions**\nNone: {result['note']}.")
        return "\n\n".join(parts)
    parts.append(f"**Target**\n{result['target']}.\nDesigns searched: {result['designs_searched']}; "
                 f"reaching the target: {result['designs_reaching_target']}.")
    lines = [result["note"][0].upper() + result["note"][1:] + "."] if result.get("note") else []
    lines.append(_table(
        ["#", "Change", "sDA", "ASE", "Spread level (sDA / ASE)", "LEED level", "Change cost"],
        [[i, s["change"], f"{s['predicted_' + SDA]:.1f}%", f"{s['predicted_' + ASE]:.1f}%",
          f"{s['spread_level']['sDA']} / {s['spread_level']['ASE']}", s["leed_level"], f"{s['change_cost']:.2f}"]
         for i, s in enumerate(result["suggestions"], 1)]))
    parts.append("**Suggestions (MLP predictions)**\n" + "\n\n".join(lines))
    check = result.get("radiance_check_of_first_suggestion")
    if check:
        parts.append(f"**Radiance check of suggestion 1**\nsDA {check[SDA]:.1f}%, ASE {check[ASE]:.1f}%, "
                     f"LEED level {check['leed_level']}. Simulation files: {check['radiance_output']}")
    advisory = result["advisory"]
    parts.append("**Notes**\n" + advisory[0].upper() + advisory[1:] + " " + BENCHMARK_NOTE)
    return "\n\n".join(parts)
