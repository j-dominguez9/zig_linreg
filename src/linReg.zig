const std = @import("std");
const rnd = std.crypto.random;
var prng = std.Random.DefaultPrng.init(42);
const rand = prng.random();

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

pub fn linear_fn(y_int: f16, slope: f16, x: f16) f16 {
    const result: f16 = y_int + (slope * x);
    return result;
}

pub fn squared_residual(observed_height: f16, predicted_height: f16) f16 {
    const abs_diff = if (observed_height > predicted_height)
        observed_height - predicted_height
    else
        predicted_height - observed_height;
    return abs_diff * abs_diff;
}

pub fn ssr(allocator: std.mem.Allocator, x: []const f16, y: []const f16, y_int: f16, slope: f16) !f16 {
    var residual_list: []f16 = try allocator.alloc(f16, x.len);
    defer allocator.free(residual_list);

    for (0..x.len) |i| {
        const ssr_pred_height: f16 = pred_height(y_int, slope, x[i]);
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
        const pred: f16 = pred_height(y_int, slope, weight[i]);
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

    const pred: f16 = pred_height(y_int, slope, weight[random_idx]);
    var gradient_int: f16 = -2 * (height[random_idx] - pred);
    var gradient_slope: f16 = -2 * weight[random_idx] * (height[random_idx] - pred);

    gradient_int *= learning_rate;
    gradient_slope *= learning_rate;
    return .{ gradient_int, gradient_slope };
}

test pred_height {
    const input_weight: f16 = 67;
    const input_slope: f16 = 1;
    std.testing.expect(pred_height(y_int: f16, slope: f16, weight: f16))
}
