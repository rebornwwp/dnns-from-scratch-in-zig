This repo is a very simple implementation of deep neural networks written in the Zig programming language.

It does not utilize tensors or any automatic differentiation library as it is only composed of fully connected feed forward layers.

To solve MNIST, create a new directory "data" in the root of the project, download the MNIST files from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/), and unzip and store the files in the "data" directory. Then run:

```
zig build run
```

If the above url is not working for you, checkout: [https://github.com/cvdfoundation/mnist](https://github.com/cvdfoundation/mnist).

## GPU version (cudaz)

An independent GPU implementation lives in `src/gpu/` and uses
[cudaz](https://github.com/rebornwwp/cudaz) (Zig 0.17 fork) with CUDA.
It keeps the same network as the CPU version, but uses `f32` throughout and
runtime-compiles the CUDA kernels in `src/gpu/kernels.cu` with NVRTC.

Requirements: an NVIDIA GPU plus a CUDA installation (`nvrtc`/`cuda` libraries
in `/usr/local/cuda`). The MNIST files in `data/` are shared with the CPU
version.

```
zig build run-gpu     # train for 25 epochs and print validation accuracy
zig build test-gpu    # check kernels against a CPU reference (forward, gradients, SGD)
```

The original CPU version (`zig build run`, `zig build test`) is unchanged.
