const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ---------- CPU version (original) ----------
    const exe = b.addExecutable(.{
        .name = "simple-feed-forward-neural-networks-in-zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    const run_step = b.step("run", "Run the app (CPU)");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests (CPU)");
    test_step.dependOn(&run_unit_tests.step);

    // ---------- GPU version (cudaz) ----------
    const cudaz_dep = b.dependency("cudaz", .{});

    // Reuse the CPU-side MNIST loader as an importable module.
    const mnist_module = b.addModule("mnist", .{
        .root_source_file = b.path("src/mnist.zig"),
        .target = target,
        .optimize = optimize,
    });

    const gpu_exe = b.addExecutable(.{
        .name = "mnist-gpu",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/gpu/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    gpu_exe.root_module.addImport("cudaz", cudaz_dep.module("cudaz"));
    gpu_exe.root_module.addImport("mnist", mnist_module);
    gpu_exe.root_module.link_libc = true;
    gpu_exe.root_module.linkSystemLibrary("cuda", .{});
    gpu_exe.root_module.linkSystemLibrary("nvrtc", .{});
    b.installArtifact(gpu_exe);

    const run_gpu_cmd = b.addRunArtifact(gpu_exe);
    run_gpu_cmd.step.dependOn(b.getInstallStep());
    run_gpu_cmd.addPassthruArgs();
    const run_gpu_step = b.step("run-gpu", "Train MNIST on GPU via cudaz");
    run_gpu_step.dependOn(&run_gpu_cmd.step);

    const gpu_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/gpu/test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    gpu_tests.root_module.addImport("cudaz", cudaz_dep.module("cudaz"));
    gpu_tests.root_module.link_libc = true;
    gpu_tests.root_module.linkSystemLibrary("cuda", .{});
    gpu_tests.root_module.linkSystemLibrary("nvrtc", .{});
    const run_gpu_tests = b.addRunArtifact(gpu_tests);
    const test_gpu_step = b.step("test-gpu", "Run GPU unit tests");
    test_gpu_step.dependOn(&run_gpu_tests.step);
}
