// =============================================================================
// tb_fireground_cnn_top.v
//
// Self-checking testbench. Loads the one saved test vector
// (sim/test_input.hex, the first window of the real X_test.npy holdout
// set) into the DUT, runs one inference, and checks the three argmax
// outputs against the values computed independently in Python
// (fpga_accelerator/tools/export_weights_fixedpoint.py, which itself
// agreed with the float32 Keras model on all 200 held-out test windows --
// see sim/expected_output.json).
//
// RUN THIS FIRST, before synthesizing, in Libero's simulator (or any
// Verilog simulator). No Verilog simulator was available in the
// environment this RTL was written in, so this testbench has not actually
// been executed yet -- the RTL's correctness was instead verified by hand
// -tracing the FSM's cycle-by-cycle behavior against the Python
// fixed-point reference model (see docs/DESIGN_NOTES.md for the two bugs
// that hand-tracing caught). Please run this before trusting synthesis
// results.
//
// Expected result for sim/test_input.hex (from expected_output.json):
//   activity_argmax      = 2
//   environment_argmax   = 1
//   physiological_argmax = 1
// =============================================================================
`timescale 1ns/1ps

module tb_fireground_cnn_top;

    reg clk = 0;
    reg rst_n = 0;
    reg start = 0;

    wire done;
    wire [1:0] activity_argmax;
    wire [1:0] environment_argmax;
    wire [1:0] physiological_argmax;

    // Not used in this testbench (input is preloaded directly into the
    // DUT's x_mem via a hierarchical reference below, for simplicity);
    // tie off.
    reg        x_wr_en   = 1'b0;
    reg [7:0]  x_wr_addr = 8'd0;
    reg signed [15:0] x_wr_data = 16'sd0;

    integer expected_activity      = 2;
    integer expected_environment   = 1;
    integer expected_physiological = 1;
    integer errors = 0;
    integer cycle_count = 0;

    fireground_cnn_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .x_wr_en(x_wr_en),
        .x_wr_addr(x_wr_addr),
        .x_wr_data(x_wr_data),
        .done(done),
        .activity_argmax(activity_argmax),
        .environment_argmax(environment_argmax),
        .physiological_argmax(physiological_argmax)
    );

    always #5 clk = ~clk; // 100 MHz sim clock (only for simulation timing; synthesis fmax is separate)

    always @(posedge clk) begin
        if (dut.state != 0) cycle_count = cycle_count + 1;
    end

    initial begin
        // Preload the DUT's input scratchpad directly (simulation-only;
        // the synthesizable path for loading a real input window is the
        // x_wr_en/x_wr_addr/x_wr_data port, driven by whatever host
        // interface is added around this core).
        $readmemh("test_input.hex", dut.x_mem);

        rst_n = 0;
        start = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        start = 1;
        @(posedge clk);
        start = 0;

        // Safety timeout so a stuck FSM doesn't hang simulation forever.
        fork
            begin
                wait (done == 1'b1);
            end
            begin
                #200000;
                $display("TIMEOUT: 'done' never asserted -- FSM likely stuck.");
                $finish;
            end
        join_any

        @(posedge clk); // let outputs settle one more cycle

        $display("==================================================");
        $display("Fireground AI CNN accelerator -- testbench result");
        $display("==================================================");
        $display("Cycles to complete inference: %0d", cycle_count);
        $display("activity_argmax      = %0d (expected %0d)", activity_argmax, expected_activity);
        $display("environment_argmax   = %0d (expected %0d)", environment_argmax, expected_environment);
        $display("physiological_argmax = %0d (expected %0d)", physiological_argmax, expected_physiological);

        if (activity_argmax !== expected_activity) begin
            $display("FAIL: activity_argmax mismatch");
            errors = errors + 1;
        end
        if (environment_argmax !== expected_environment) begin
            $display("FAIL: environment_argmax mismatch");
            errors = errors + 1;
        end
        if (physiological_argmax !== expected_physiological) begin
            $display("FAIL: physiological_argmax mismatch");
            errors = errors + 1;
        end

        if (errors == 0)
            $display("PASS: all three outputs match the Python fixed-point reference.");
        else
            $display("FAIL: %0d mismatch(es) -- do not proceed to synthesis until this passes.", errors);

        $finish;
    end

endmodule
