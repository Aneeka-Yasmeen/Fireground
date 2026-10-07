# Fireground AI reference dataset

Rows: 142

This dataset is a **research/reference dataset**, not a validated operational water-demand model.

## Primary sources

1. FSRI, *Impact of Fire Attack Utilizing Interior and Exterior Streams on Firefighter Safety and Occupant Survival: Water Mapping* (Part I). The study reports 83 no-fire water-distribution experiments and includes nozzle settings, pressure/flow conditions, expected water, and experimentally collected water in Table A.1. (61 rows, `record_type = water_mapping`)
2. FSRI *Coordinated Fire Attack* — Single-Family Homes (`Coord_Tactics_Single_Family_Homes.pdf`, 437 pages). All 20 live-fire experiments (Methods 1-6) extracted case-by-case: nozzle (1¾ in. hoseline, combination nozzle, straight stream, 150 gpm — stated identically for every experiment), total suppression water used, and tactic timing. Pressure (psi) is confirmed **not reported anywhere** in this document. Six per-method average rows (Table 5.2) are kept separate as `live_fire_aggregate`. (26 rows)
3. FSRI *Coordinated Fire Attack* — Multi-Family Dwellings (`Coord_Tactics_Multi_Family.pdf`, 347 pages). All 13 live-fire experiments (1A-1E, 2A, 2B, 3A, 3B, 4A, 4B, 5, 6) plus two post-suppression hydraulic-ventilation water-use records and four report-level averages (Table in §"Water Usage"). Nozzle/hose/flow/50 psi detail is stated as applying to 12 of the 13 experiments in the report's own general equipment description. (20 rows, plus the pre-existing Experiment 1B row refined with a burst-by-burst nuance the report gives on that estimate)
4. FSRI *Coordinated Fire Attack* — Strip Malls (`Coord_Tactics_Strip_Malls.pdf`, 196 pages). The pre-existing 7 water-use rows (Table 6.3 totals) were expanded with the nozzle/hose/flow detail given in the report's per-experiment narrative (Section 3) — confirmed **no nozzle/pump/discharge pressure (psi) is reported anywhere** in this document (all "psi" figures in the text are structural pressure-transducer readings, unrelated to hose/nozzle pressure). Added temperature-response rows (Table 6.2), one aerial-master-stream tactic row, and two study-level water-use aggregates. (16 rows total)
5. FSRI *Understanding and Fighting Basement Fires* (`Understanding_and_Fighting_Basement_Fires.pdf`). The pre-existing 8 temperature-response rows were completed with their nozzle flow/pressure specifications (stated in the report's methodology section, not previously captured). Added 8 further temperature-response/method rows covering Experiments 6-11 (additional suppression actions) and Experiment 10 (a residential-sprinkler comparison case). This report contains no gallons-used or no-fire water-mapping data — confirmed absent. (17 rows total)
6. FSRI Residential aggregate (`FSRI-Residential`, 2 rows): 31 of 33 residential experiments used a 1¾-in. hoseline with either a combination nozzle at 150 gpm/50 psi or a 7/8-in. smooth-bore nozzle at 160 gpm/50 psi; primary suppression water use averaged 145 gal ± 51 gal (range 73-256 gal). **Citation caveat:** this statistic was re-checked against `Coord_Tactics_Single_Family_Homes.pdf` during the September 2026 extraction pass and does **not** appear anywhere in that document (that report's own aggregate is different — see source 2 above, Table 5.2). It most likely originates from the Water Mapping report family (source 1) or a related FSRI synthesis document, not from the four Coordinated Fire Attack PDFs reviewed in this pass. Treat this row's source attribution as unverified until traced to its exact origin — do not delete it, but do not treat "FSRI-Residential" as confirmed to mean any specific one of the four PDFs above.

## Important interpretation rule

Do **not** train a model with a row like "HIGH fire state -> 650 L/min" from this file.

The dataset distinguishes:
- no-fire water-distribution measurements (`water_mapping`),
- live-fire suppression cases (`live_fire_experiment`),
- live-fire suppression tactics with a water-use figure that is not the *initial-knockdown* flow, e.g. hydraulic ventilation after the fire is already out (`live_fire_method`),
- fire-side/stairwell temperature responses to a suppression action (`live_fire_temperature_response`),
- total-water-used-only records without nozzle/flow/pressure detail (`live_fire_usage`),
- and aggregate/average live-fire water-use statistics spanning multiple experiments (`live_fire_aggregate`).

Missing values are intentionally blank. No unreported fire conditions, flow rates, pressures, or nozzle settings were invented. Where a report explicitly states one nozzle/flow/pressure specification applies uniformly across a named group of experiments (e.g. "a 1¾ in. hoseline flowing 150 gpm at 50 psi was used for initial knockdown and suppression" across 12 of 13 multi-family experiments), that specification was applied to each of those experiments' rows and cited accordingly — this is a source-explicit statement, not an invented or estimated value. Aggregate/average statistics (means, ranges, per-method averages) were never applied to any individual experiment's row; they get their own `live_fire_aggregate` row.

The converted columns (`flow_lpm`, `pressure_bar`, `expected_water_l`, `experimental_water_l`) are unit conversions of source values (1 US gal = 3.785411784 L, 1 psi = 0.0689475729 bar), not independently measured values.

## `duration_seconds` / `duration_basis` columns

Added after reviewing the dataset: `flow_gpm` turned out to be nearly constant across live-fire cases (150 or 160 gpm dominates — a fixed nozzle/equipment choice, not something that scales with fire severity), while `experimental_water_gal` and elapsed time both vary by an order of magnitude across scenarios. `duration_seconds` captures that elapsed time — e.g. "water-on to knockdown," "front-door-open to fire-room entry," or an application window in seconds — so that water volume and duration can be considered together as a scenario-dependent signal, instead of only having a flow-rate column that barely varies.

This field is populated only for 47 of 142 rows: single, continuous, unambiguous actions with an explicit start/end time in the source. It is deliberately left blank (not estimated or summed across a gap) for:
- multi-phase actions bundled into one row where the sub-phases aren't contiguous (e.g. an indirect then direct attack with a gap in between),
- rows with only a start time and no stated duration,
- aggregate/average rows (duration isn't a single-case fact),
- and `water_mapping` rows (no-fire nozzle-pattern tests don't have a comparable "time to knockdown").

`duration_basis` is a short label of what the timestamp measures (e.g. "front-door-open to suppression" vs. "hydraulic ventilation run time") — these are not directly comparable to each other without reading the basis, since some measure time-to-knockdown and others measure a post-knockdown tactic's own duration.

## Extraction methodology (September 2026 expansion)

The four Coordinated Fire Attack / basement-fire PDFs in `research/fsri/` were reviewed page-by-page (via text extraction, since the PDF page-image tool was unavailable in that pass's environment) specifically for experiment-level tables and per-experiment narrative describing nozzle type, hose diameter, flow, pressure, water used, and temperature response. Every added or corrected value in this pass is traceable to a specific page and source sentence/table cell (citations were recorded during extraction but are not stored in this CSV to keep it machine-readable; see project notes for the full citation list if needed). Rows were added only where the source explicitly stated a value; fields the source did not report were left blank rather than estimated. One previously captured row (`FSRI-Single-Family`, "Experiments 10-12", a coarse combined summary) was replaced by three precise per-experiment rows (Experiments 10, 11, 12) that the source actually supports individually.

## Suggested next modelling step

Use the live-fire records as **reference/validation cases**, then build a separate target dataset in which an independently established case-level target is available for each scenario (for example measured suppression flow, measured water-use duration, or a physically defined application-rate target). Validate against held-out fire scenarios before presenting any water-demand estimate as operational.
