# Minimal timing constraint for fireground_cnn_top.
#
# Without a defined clock, Synplify has no real timing paths to preserve
# and its optimizer can prune large parts of the design as "hanging
# logic" (which is what happened in the first synthesis run -- 0 LUTs,
# entire FSM removed, see the "no specified timing constraint" /
# "No IO constraint found" warnings in the synthesis log).
#
# 100 MHz is just a starting point -- this design has no throughput
# pressure (see docs/DESIGN_NOTES.md: ~14,500 cycles/inference is far
# faster than the ~1 Hz sensor rate even at a few MHz), so pick whatever
# clock your board/design actually drives this core with and adjust the
# period accordingly. What matters here is that a clock is defined at
# all before synthesizing.
create_clock -name clk -period 10.000 [get_ports clk]
