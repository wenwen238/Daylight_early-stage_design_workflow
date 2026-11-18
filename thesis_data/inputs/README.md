# Input data

| File | Content |
|---|---|
| `fixed_design_cases_560.csv` | The 560 sampled façade designs (thesis Section 4.3.2): façade pattern, window type, width, height, sill, overhang depth and design stratum of each `model_NNN` |
| `fixed_design_cases_560.json` | The same designs in the format read by the Radiance and decision scripts |
| `weather/milan.wea` | Weather data of Milano Linate converted for Radiance (`gendaymtx`); this is the file read by the scripts |

**Weather data source.** `milan.wea` was converted from the TMYx weather file `ITA_LM_Milano-Linate.AP.160800_TMYx.epw` (Lawrie, L.K., Crawley, D.B., Development of Global Typical Meteorological Years (TMYx), https://climate.onebuilding.org). The EPW file itself is not included in this repository; it can be downloaded from climate.onebuilding.org (WMO Region 6 – Europe → Italy → Lombardia). The files on that website are updated from time to time, so a newly downloaded file may differ slightly from the one used in the thesis; `milan.wea` is the version that was used.
