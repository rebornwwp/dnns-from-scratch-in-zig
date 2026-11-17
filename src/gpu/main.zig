const std = @import("std");
const mnist = @import("mnist");
const gpu = @import("nn.zig");
const Cuda = @import("cudaz");

const Device = Cuda.Device;

const EPOCHS: usize = 25;
const TRAIN_SAMPLES: usize = 60000;
const TEST_SAMPLES: usize = 10000;

fn toF32(allocator: std.mem.Allocator, src: []const f64) ![]f32 {
    const dst = try allocator.alloc(f32, src.len);
    for (src, dst) |v, *d| d.* = @floatCast(v);
    return dst;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    const device = try Device.default();
    defer device.deinit();

    // Load MNIST (host-side f64) and convert to f32 for the GPU.
    const data = try mnist.readMnist(allocator, init.io);
    defer data.deinit(allocator);

    const train_x_f32 = try toF32(allocator, data.train_images);
    defer allocator.free(train_x_f32);
    const test_x_f32 = try toF32(allocator, data.test_images);
    defer allocator.free(test_x_f32);

    // Upload the whole dataset once; batches are just views into these buffers.
    const d_train_x = try device.htodCopy(f32, train_x_f32);
    defer d_train_x.free();
    const d_train_y = try device.htodCopy(u8, data.train_labels);
    defer d_train_y.free();
    const d_test_x = try device.htodCopy(f32, test_x_f32);
    defer d_test_x.free();
    const d_test_y = try device.htodCopy(u8, data.test_labels);
    defer d_test_y.free();

    var net = try gpu.GpuNet.init(allocator, device);
    defer net.deinit();

    std.debug.print("Training on GPU: {s}\n", .{"RTX (cudaz)"});
    var epoch: usize = 0;
    while (epoch < EPOCHS) : (epoch += 1) {
        var i: usize = 0;
        while (i < TRAIN_SAMPLES / gpu.BATCH_SIZE) : (i += 1) {
            const x = gpu.GpuNet.batchView(f32, d_train_x, i, gpu.BATCH_SIZE, gpu.INPUT_SIZE);
            const y = gpu.GpuNet.batchView(u8, d_train_y, i, gpu.BATCH_SIZE, 1);
            try net.trainBatch(x, y, gpu.BATCH_SIZE);
        }

        const accuracy = try net.validate(d_test_x, d_test_y, TEST_SAMPLES);
        std.debug.print("Epoch {d}: validation accuracy = {d:.4}\n", .{ epoch + 1, accuracy });
    }
}
