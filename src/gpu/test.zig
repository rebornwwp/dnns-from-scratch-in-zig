const std = @import("std");
const gpu = @import("nn.zig");
const Cuda = @import("cudaz");

const Device = Cuda.Device;

fn randomWeights(allocator: std.mem.Allocator, inputs: usize, outputs: usize) ![]f32 {
    const weights = try allocator.alloc(f32, inputs * outputs);
    var prng = std.Random.DefaultPrng.init(123);
    const scale = @sqrt(2.0 / @as(f32, @floatFromInt(inputs)));
    for (weights) |*w| w.* = prng.random().floatNorm(f32) * scale;
    return weights;
}

test "GPU forward/backward matches CPU reference" {
    const allocator = std.testing.allocator;
    const device = try Device.default();
    defer device.deinit();

    var net = try gpu.GpuNet.init(allocator, device);
    defer net.deinit();

    const I = gpu.INPUT_SIZE;
    const H = gpu.HIDDEN_SIZE;
    const O = gpu.OUTPUT_SIZE;
    const N = gpu.BATCH_SIZE;

    // Use a known set of weights so the CPU reference is comparable.
    const w1 = try randomWeights(allocator, I, H);
    defer allocator.free(w1);
    const w2 = try randomWeights(allocator, H, O);
    defer allocator.free(w2);
    net.d_w1.free();
    net.d_w1 = try device.htodCopy(f32, w1);
    net.d_w2.free();
    net.d_w2 = try device.htodCopy(f32, w2);

    const x_host = try allocator.alloc(f32, N * I);
    defer allocator.free(x_host);
    var prng = std.Random.DefaultPrng.init(42);
    for (x_host) |*v| v.* = prng.random().float(f32);

    const y_host = try allocator.alloc(u8, N);
    defer allocator.free(y_host);
    for (y_host, 0..) |*v, i| v.* = @intCast(i % O);

    const d_x = try device.htodCopy(f32, x_host);
    defer d_x.free();
    const d_y = try device.htodCopy(u8, y_host);
    defer d_y.free();

    // ---- CPU reference ----
    const z1 = try allocator.alloc(f32, N * H);
    defer allocator.free(z1);
    const a1 = try allocator.alloc(f32, N * H);
    defer allocator.free(a1);
    const z2 = try allocator.alloc(f32, N * O);
    defer allocator.free(z2);
    const dz2 = try allocator.alloc(f32, N * O);
    defer allocator.free(dz2);
    const da1 = try allocator.alloc(f32, N * H);
    defer allocator.free(da1);
    const dw1 = try allocator.alloc(f32, I * H);
    defer allocator.free(dw1);
    const dw2 = try allocator.alloc(f32, H * O);
    defer allocator.free(dw2);
    @memset(dw1, 0);
    @memset(dw2, 0);

    for (0..N) |b| {
        for (0..H) |h| {
            var sum: f32 = 0;
            for (0..I) |i| sum += x_host[b * I + i] * w1[i * H + h];
            z1[b * H + h] = sum;
            a1[b * H + h] = if (sum < 0) 0.01 * sum else sum;
        }
        for (0..O) |o| {
            var sum: f32 = 0;
            for (0..H) |h| sum += a1[b * H + h] * w2[h * O + o];
            z2[b * O + o] = sum;
        }

        var max_v = z2[b * O];
        for (1..O) |o| max_v = @max(max_v, z2[b * O + o]);
        var sum_e: f32 = 0;
        for (0..O) |o| sum_e += @exp(z2[b * O + o] - max_v);
        for (0..O) |o| {
            const p = @exp(z2[b * O + o] - max_v) / sum_e;
            dz2[b * O + o] = p - (if (o == y_host[b]) @as(f32, 1) else 0);
        }

        for (0..H) |h| {
            var sum: f32 = 0;
            for (0..O) |o| sum += dz2[b * O + o] * w2[h * O + o];
            da1[b * H + h] = if (z1[b * H + h] < 0) 0.01 * sum else sum;
        }

        for (0..H) |h| {
            for (0..O) |o| dw2[h * O + o] += dz2[b * O + o] * a1[b * H + h] / @as(f32, @floatFromInt(N));
        }
        for (0..I) |i| {
            for (0..H) |h| dw1[i * H + h] += da1[b * H + h] * x_host[b * I + i] / @as(f32, @floatFromInt(N));
        }
    }

    // ---- GPU ----
    try net.trainBatch(d_x, d_y, N);

    const got_z2 = try allocator.alloc(f32, N * O);
    defer allocator.free(got_z2);
    try Device.dtohCopyInto(f32, net.d_z2, got_z2);
    const got_dw1 = try allocator.alloc(f32, I * H);
    defer allocator.free(got_dw1);
    try Device.dtohCopyInto(f32, net.d_dw1, got_dw1);
    const got_dw2 = try allocator.alloc(f32, H * O);
    defer allocator.free(got_dw2);
    try Device.dtohCopyInto(f32, net.d_dw2, got_dw2);

    for (got_z2, z2) |got, want| try std.testing.expectApproxEqAbs(want, got, 1e-4);
    for (got_dw1, dw1) |got, want| try std.testing.expectApproxEqAbs(want, got, 1e-4);
    for (got_dw2, dw2) |got, want| try std.testing.expectApproxEqAbs(want, got, 1e-4);

    // SGD actually updated the weights.
    const got_w1 = try allocator.alloc(f32, I * H);
    defer allocator.free(got_w1);
    try Device.dtohCopyInto(f32, net.d_w1, got_w1);
    for (got_w1, w1, dw1) |got, initial, grad| {
        try std.testing.expectApproxEqAbs(initial - gpu.LEARNING_RATE * grad, got, 1e-4);
    }
}
