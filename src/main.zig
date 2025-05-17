//! By convention, main.zig is where your main function lives in the case that
//! you are building an executable. If you are making a library, the convention
//! is to delete this file and start with root.zig instead.
const linreg = @import("linReg.zig");
const std = @import("std");
const rnd = std.crypto.random;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const height = [_]f16{ 63, 67, 70, 72 };
    const weight = [_]f16{ 120, 140, 150, 170 };
    const slope: f16 = 0.184615;
    const learning_rate: f16 = 0.00001;

    const test_pred: f16 = linreg.pred_height(39, slope, weight[0]);
    std.debug.print("Test_pred: {}\n", .{test_pred});

    const input_height: f16 = 67;
    const input_pred_height: f16 = 69;
    const test_sr: f16 = linreg.squared_residual(input_height, input_pred_height);

    std.debug.print("SSR: {}\n", .{test_sr});

    const ssr_new: f16 = try linreg.ssr(allocator, weight[0..], height[0..], 39, 0.1846);
    std.debug.print("SSR_NEW: {}\n", .{ssr_new});

    // const derivative: f32 = try linreg.get_deriv(allocator, 39, weight[0..], height[0..], learning_rate, slope);
    // std.debug.print("Derivative: {}", .{derivative});

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    // random intercept
    var intercept_param: f16 = @as(f16, @floatFromInt(rand.intRangeAtMost(u8, 20, 50)));

    // var slope_param: f16 = @as(f16, rand.float(f16));
    var slope_param: f16 = 0.05;
    // var slope_param: f32 = std.crypto.random.float(f32);
    for (0..60000) |_| {
        const step: [2]f16 = try linreg.sgd(intercept_param, weight[0..], height[0..], learning_rate, slope_param);
        // const step: [2]f16 = try linreg.get_deriv(allocator, intercept_param, weight[0..], height[0..], learning_rate, slope_param);
        intercept_param -= step[0];
        slope_param -= step[1];
    }

    std.debug.print("Final inter: {}\n", .{intercept_param});
    std.debug.print("Final slope: {}\n", .{slope_param});
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // Try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "use other module" {
    try std.testing.expectEqual(@as(i32, 150), lib.add(100, 50));
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}

/// This imports the separate module containing `root.zig`. Take a look in `build.zig` for details.
const lib = @import("learn_zig_lib");
