const std = @import("std");
const rnd = std.crypto.random;
var prng = std.Random.DefaultPrng.init(42);
const rand = prng.random();
const testing = std.testing;
const expectApproxEqAbs = testing.expectApproxEqAbs;

pub const LRData = struct {
    vec: []u8,

    pub fn init(allocator: std.mem.Allocator, dataset_size: u16) !LRData {
        var data_arr = try allocator.alloc(u8, dataset_size);
        for (0..dataset_size) |i| {
            data_arr[i] = std.crypto.random.int(u8);
            std.debug.print("rand num:  {}\n", .{data_arr[i]});
        }
        return LRData{ .vec = data_arr };
    }
};

pub fn linear_fn(comptime T: type, y_int: T, slope: T, x: T) T {
    return y_int + (slope * x);
}

pub fn squared_residual(comptime T: type, y: T, predicted_y: T) T {
    const abs_diff = if (y > predicted_y)
        y - predicted_y
    else
        predicted_y - y;
    return abs_diff * abs_diff;
}

pub fn ssr(comptime T: type, allocator: std.mem.Allocator, x: []const T, y: []const T, y_int: T, slope: T) !T {
    var residual_list: []T = try allocator.alloc(T, x.len);
    defer allocator.free(residual_list);

    for (0..x.len) |i| {
        const ssr_pred_height: f16 = linear_fn(y_int, slope, x[i]);
        residual_list[i] = squared_residual(y[i], ssr_pred_height);
    }

    var sum: f16 = 0.0;
    for (residual_list) |residual| {
        sum += residual;
    }

    return sum;
}

pub fn get_deriv(allocator: std.mem.Allocator, y_int: f16, weight: []const f16, height: []const f16, learning_rate: f16, slope: f16) ![2]f16 {
    var gradient_int_list: []f16 = try allocator.alloc(f16, weight.len);
    var gradient_slope_list: []f16 = try allocator.alloc(f16, weight.len);
    defer allocator.free(gradient_int_list);
    defer allocator.free(gradient_slope_list);

    for (0..weight.len) |i| {
        const pred: f16 = linear_fn(y_int, slope, weight[i]);
        const gradient_int: f16 = -2 * (height[i] - pred);
        const gradient_slope: f16 = -2 * weight[i] * (height[i] - pred);
        gradient_int_list[i] = gradient_int;
        gradient_slope_list[i] = gradient_slope;
    }
    var step_size_int: f16 = 0.0;
    var step_size_slope: f16 = 0.0;
    for (0..gradient_int_list.len) |i| {
        step_size_int += gradient_int_list[i];
        step_size_slope += gradient_slope_list[i];
    }
    step_size_int *= learning_rate;
    step_size_slope *= learning_rate;
    return .{ step_size_int, step_size_slope };
}
pub fn sgd(y_int: f16, weight: []const f16, height: []const f16, learning_rate: f16, slope: f16) ![2]f16 {
    // Random index
    const random_idx: u64 = rand.intRangeAtMost(u64, 0, weight.len - 1);

    const pred: f16 = linear_fn(y_int, slope, weight[random_idx]);
    var gradient_int: f16 = -2 * (height[random_idx] - pred);
    var gradient_slope: f16 = -2 * weight[random_idx] * (height[random_idx] - pred);

    gradient_int *= learning_rate;
    gradient_slope *= learning_rate;
    return .{ gradient_int, gradient_slope };
}

pub fn main() void {
    const linear_output: f16 = linear_fn(f16, 1, 2, 67);
    std.debug.print("Linear out: {}", .{linear_output});
}
test "linear_fn basic test" {
    // Test case 1: y = 2 + 3x, x = 4 should give 14
    const result1 = linear_fn(f16, 2.0, 3.0, 4.0);
    try expectApproxEqAbs(@as(f16, 14.0), result1, 0.001);

    // Test case 2: y = -1 + 2x, x = 5 should give 9
    const result2 = linear_fn(f16, -1.0, 2.0, 5.0);
    try expectApproxEqAbs(@as(f16, 9.0), result2, 0.001);

    // Test case 3: y = 0 + 0x, x = 10 should give 0
    const result3 = linear_fn(f16, 0.0, 0.0, 10.0);
    try expectApproxEqAbs(@as(f16, 0.0), result3, 0.001);
}
