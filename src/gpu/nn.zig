const std = @import("std");
const Cuda = @import("cudaz");

const Device = Cuda.Device;
const Compile = Cuda.Compile;
const LaunchConfig = Cuda.LaunchConfig;
const Cudaslice = Cuda.Cudaslice;

pub const INPUT_SIZE: usize = 784;
pub const HIDDEN_SIZE: usize = 100;
pub const OUTPUT_SIZE: usize = 10;
pub const BATCH_SIZE: usize = 32;
pub const LEARNING_RATE: f32 = 0.01;

const kernel_src = @embedFile("kernels.cu");

const Kernels = struct {
    linear_forward: Cuda.Function,
    leaky_relu_forward: Cuda.Function,
    nll_grad: Cuda.Function,
    linear_input_grad: Cuda.Function,
    linear_weight_grad: Cuda.Function,
    leaky_relu_backward: Cuda.Function,
    sgd_update: Cuda.Function,
    accuracy: Cuda.Function,

    fn init(allocator: std.mem.Allocator) !Kernels {
        const ptx = try Compile.cudaText(kernel_src, .{}, allocator);
        defer allocator.free(ptx);
        const module = try Device.loadPtxText(ptx);
        return .{
            .linear_forward = try module.getFunc("linear_forward"),
            .leaky_relu_forward = try module.getFunc("leaky_relu_forward"),
            .nll_grad = try module.getFunc("nll_grad"),
            .linear_input_grad = try module.getFunc("linear_input_grad"),
            .linear_weight_grad = try module.getFunc("linear_weight_grad"),
            .leaky_relu_backward = try module.getFunc("leaky_relu_backward"),
            .sgd_update = try module.getFunc("sgd_update"),
            .accuracy = try module.getFunc("accuracy_kernel"),
        };
    }
};

fn launchConfig(elems: usize) LaunchConfig {
    return LaunchConfig.for_num_elems(@intCast(elems));
}

/// Non-owning view into a device allocation. Never call free() on a view.
fn view(comptime T: type, base: Cudaslice(T), offset: usize, len: usize) Cudaslice(T) {
    const Ptr = @TypeOf(base.device_ptr);
    return .{
        .device_ptr = base.device_ptr + @as(Ptr, @intCast(offset * @sizeOf(T))),
        .len = len,
        .device = base.device,
    };
}

fn randomWeights(allocator: std.mem.Allocator, inputs: usize, outputs: usize) ![]f32 {
    const weights = try allocator.alloc(f32, inputs * outputs);
    var prng = std.Random.DefaultPrng.init(123);
    const scale = @sqrt(2.0 / @as(f32, @floatFromInt(inputs)));
    for (weights) |*w| {
        w.* = prng.random().floatNorm(f32) * scale;
    }
    return weights;
}

pub const GpuNet = struct {
    device: Device,
    kernels: Kernels,

    d_w1: Cudaslice(f32),
    d_w2: Cudaslice(f32),
    d_dw1: Cudaslice(f32),
    d_dw2: Cudaslice(f32),

    d_z1: Cudaslice(f32),
    d_a1: Cudaslice(f32),
    d_z2: Cudaslice(f32),
    d_dz2: Cudaslice(f32),
    d_da1: Cudaslice(f32),
    d_dz1: Cudaslice(f32),
    d_loss: Cudaslice(f32),

    d_test_z1: Cudaslice(f32),
    d_test_logits: Cudaslice(f32),
    d_correct: Cudaslice(u32),

    pub fn init(allocator: std.mem.Allocator, device: Device) !GpuNet {
        const kernels = try Kernels.init(allocator);

        const w1 = try randomWeights(allocator, INPUT_SIZE, HIDDEN_SIZE);
        defer allocator.free(w1);
        const w2 = try randomWeights(allocator, HIDDEN_SIZE, OUTPUT_SIZE);
        defer allocator.free(w2);

        const d_w1 = try device.htodCopy(f32, w1);
        const d_w2 = try device.htodCopy(f32, w2);

        return .{
            .device = device,
            .kernels = kernels,
            .d_w1 = d_w1,
            .d_w2 = d_w2,
            .d_dw1 = try device.alloc(f32, INPUT_SIZE * HIDDEN_SIZE),
            .d_dw2 = try device.alloc(f32, HIDDEN_SIZE * OUTPUT_SIZE),
            .d_z1 = try device.alloc(f32, BATCH_SIZE * HIDDEN_SIZE),
            .d_a1 = try device.alloc(f32, BATCH_SIZE * HIDDEN_SIZE),
            .d_z2 = try device.alloc(f32, BATCH_SIZE * OUTPUT_SIZE),
            .d_dz2 = try device.alloc(f32, BATCH_SIZE * OUTPUT_SIZE),
            .d_da1 = try device.alloc(f32, BATCH_SIZE * HIDDEN_SIZE),
            .d_dz1 = try device.alloc(f32, BATCH_SIZE * INPUT_SIZE),
            .d_loss = try device.alloc(f32, BATCH_SIZE),
            .d_test_z1 = try device.alloc(f32, 10000 * HIDDEN_SIZE),
            .d_test_logits = try device.alloc(f32, 10000 * OUTPUT_SIZE),
            .d_correct = try device.alloc(u32, 1),
        };
    }

    pub fn deinit(self: *GpuNet) void {
        self.d_w1.free();
        self.d_w2.free();
        self.d_dw1.free();
        self.d_dw2.free();
        self.d_z1.free();
        self.d_a1.free();
        self.d_z2.free();
        self.d_dz2.free();
        self.d_da1.free();
        self.d_dz1.free();
        self.d_loss.free();
        self.d_test_z1.free();
        self.d_test_logits.free();
        self.d_correct.free();
    }

    fn linearForward(self: *GpuNet, x: Cudaslice(f32), w: Cudaslice(f32), y: Cudaslice(f32), n: usize, i: usize, o: usize) !void {
        var n_arg: c_uint = @intCast(n);
        var i_arg: c_uint = @intCast(i);
        var o_arg: c_uint = @intCast(o);
        try self.kernels.linear_forward.run(.{
            &x.device_ptr,
            &w.device_ptr,
            &y.device_ptr,
            &n_arg,
            &i_arg,
            &o_arg,
        }, launchConfig(n * o));
    }

    pub fn trainBatch(self: *GpuNet, d_x: Cudaslice(f32), d_y: Cudaslice(u8), n: usize) !void {
        // Forward: x -> z1 -> a1 -> z2
        try self.linearForward(d_x, self.d_w1, self.d_z1, n, INPUT_SIZE, HIDDEN_SIZE);
        var elems: c_uint = @intCast(n * HIDDEN_SIZE);
        try self.kernels.leaky_relu_forward.run(.{ &self.d_z1.device_ptr, &elems }, launchConfig(n * HIDDEN_SIZE));
        try self.linearForward(self.d_z1, self.d_w2, self.d_z2, n, HIDDEN_SIZE, OUTPUT_SIZE);

        // Loss gradient w.r.t. logits.
        var n_arg: c_uint = @intCast(n);
        var o_arg: c_uint = @intCast(OUTPUT_SIZE);
        try self.kernels.nll_grad.run(.{
            &self.d_z2.device_ptr,
            &d_y.device_ptr,
            &self.d_dz2.device_ptr,
            &self.d_loss.device_ptr,
            &n_arg,
            &o_arg,
        }, launchConfig(n));

        // Backprop through layer 2.
        var i_hidden: c_uint = @intCast(HIDDEN_SIZE);
        try self.kernels.linear_input_grad.run(.{
            &self.d_dz2.device_ptr,
            &self.d_w2.device_ptr,
            &self.d_da1.device_ptr,
            &n_arg,
            &i_hidden,
            &o_arg,
        }, launchConfig(n * HIDDEN_SIZE));
        try self.kernels.leaky_relu_backward.run(.{
            &self.d_z1.device_ptr,
            &self.d_da1.device_ptr,
            &elems,
        }, launchConfig(n * HIDDEN_SIZE));
        try self.kernels.linear_weight_grad.run(.{
            &self.d_dz2.device_ptr,
            &self.d_z1.device_ptr,
            &self.d_dw2.device_ptr,
            &n_arg,
            &i_hidden,
            &o_arg,
        }, launchConfig(HIDDEN_SIZE * OUTPUT_SIZE));

        // Backprop through layer 1.
        var i_input: c_uint = @intCast(INPUT_SIZE);
        try self.kernels.linear_input_grad.run(.{
            &self.d_da1.device_ptr,
            &self.d_w1.device_ptr,
            &self.d_dz1.device_ptr,
            &n_arg,
            &i_input,
            &i_hidden,
        }, launchConfig(n * INPUT_SIZE));
        try self.kernels.linear_weight_grad.run(.{
            &self.d_da1.device_ptr,
            &d_x.device_ptr,
            &self.d_dw1.device_ptr,
            &n_arg,
            &i_input,
            &i_hidden,
        }, launchConfig(INPUT_SIZE * HIDDEN_SIZE));

        // SGD update.
        var lr: f32 = LEARNING_RATE;
        var w1_elems: c_uint = @intCast(INPUT_SIZE * HIDDEN_SIZE);
        try self.kernels.sgd_update.run(.{
            &self.d_w1.device_ptr,
            &self.d_dw1.device_ptr,
            &lr,
            &w1_elems,
        }, launchConfig(INPUT_SIZE * HIDDEN_SIZE));
        var w2_elems: c_uint = @intCast(HIDDEN_SIZE * OUTPUT_SIZE);
        try self.kernels.sgd_update.run(.{
            &self.d_w2.device_ptr,
            &self.d_dw2.device_ptr,
            &lr,
            &w2_elems,
        }, launchConfig(HIDDEN_SIZE * OUTPUT_SIZE));
    }

    pub fn validate(self: *GpuNet, d_test_x: Cudaslice(f32), d_test_y: Cudaslice(u8), n: usize) !f64 {
        try self.linearForward(d_test_x, self.d_w1, self.d_test_z1, n, INPUT_SIZE, HIDDEN_SIZE);
        var elems: c_uint = @intCast(n * HIDDEN_SIZE);
        try self.kernels.leaky_relu_forward.run(.{ &self.d_test_z1.device_ptr, &elems }, launchConfig(n * HIDDEN_SIZE));
        var n_arg: c_uint = @intCast(n);
        var hidden: c_uint = @intCast(HIDDEN_SIZE);
        var output: c_uint = @intCast(OUTPUT_SIZE);
        try self.kernels.linear_forward.run(.{
            &self.d_test_z1.device_ptr,
            &self.d_w2.device_ptr,
            &self.d_test_logits.device_ptr,
            &n_arg,
            &hidden,
            &output,
        }, launchConfig(n * OUTPUT_SIZE));

        try Device.memsetZeros(u32, self.d_correct);
        try self.kernels.accuracy.run(.{
            &self.d_test_logits.device_ptr,
            &d_test_y.device_ptr,
            &self.d_correct.device_ptr,
            &n_arg,
            &output,
        }, launchConfig(n));

        var correct: [1]u32 = undefined;
        try Device.dtohCopyInto(u32, self.d_correct, &correct);
        return @as(f64, @floatFromInt(correct[0])) / @as(f64, @floatFromInt(n));
    }

    /// Returns a non-owning view of a batch inside a larger device array.
    /// `elems_per_sample` is the number of T elements in one sample (784 for
    /// images, 1 for labels); the view offset is measured in elements, not bytes.
    pub fn batchView(comptime T: type, base: Cudaslice(T), index: usize, batch: usize, elems_per_sample: usize) Cudaslice(T) {
        return view(T, base, index * batch * elems_per_sample, batch * elems_per_sample);
    }
};
