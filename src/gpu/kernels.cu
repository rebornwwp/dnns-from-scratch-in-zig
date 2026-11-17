// CUDA kernels for the from-scratch MNIST network.
// Compiled at runtime by cudaz (NVRTC) and loaded as PTX.
//
// Layout conventions match the CPU implementation:
//   X: [N, I]  row-major
//   W: [I, O]  row-major, weight for input i -> output o is W[i*O + o]
//   Y: [N, O]  row-major

extern "C" __global__ void linear_forward(
    const float* X,
    const float* W,
    float* Y,
    unsigned int N,
    unsigned int I,
    unsigned int O)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N * O) return;

    unsigned int b = gid / O;
    unsigned int o = gid % O;

    float sum = 0.0f;
    for (unsigned int i = 0; i < I; ++i)
    {
        sum += X[b * I + i] * W[i * O + o];
    }
    Y[gid] = sum;
}

extern "C" __global__ void leaky_relu_forward(float* X, unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    if (X[gid] < 0.0f) X[gid] = 0.01f * X[gid];
}

// Softmax cross-entropy loss and its gradient w.r.t. the logits.
// Uses the max-subtraction trick for numerical stability.
extern "C" __global__ void nll_grad(
    const float* logits,
    const unsigned char* targets,
    float* grads,
    float* losses,
    unsigned int N,
    unsigned int O)
{
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= N) return;

    const float* row = logits + b * O;
    float max_v = row[0];
    for (unsigned int i = 1; i < O; ++i)
    {
        if (row[i] > max_v) max_v = row[i];
    }

    float sum_e = 0.0f;
    for (unsigned int i = 0; i < O; ++i)
    {
        sum_e += expf(row[i] - max_v);
    }

    unsigned int target = (unsigned int)targets[b];
    losses[b] = max_v + logf(sum_e) - row[target];

    for (unsigned int i = 0; i < O; ++i)
    {
        float p = expf(row[i] - max_v) / sum_e;
        float g = p - ((i == target) ? 1.0f : 0.0f);
        grads[b * O + i] = g;
    }
}

extern "C" __global__ void linear_input_grad(
    const float* grad_out,
    const float* W,
    float* grad_in,
    unsigned int N,
    unsigned int I,
    unsigned int O)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N * I) return;

    unsigned int b = gid / I;
    unsigned int i = gid % I;

    float sum = 0.0f;
    for (unsigned int o = 0; o < O; ++o)
    {
        sum += grad_out[b * O + o] * W[i * O + o];
    }
    grad_in[gid] = sum;
}

extern "C" __global__ void linear_weight_grad(
    const float* grad_out,
    const float* X,
    float* grad_w,
    unsigned int N,
    unsigned int I,
    unsigned int O)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= I * O) return;

    unsigned int i = gid / O;
    unsigned int o = gid % O;

    float sum = 0.0f;
    for (unsigned int b = 0; b < N; ++b)
    {
        sum += grad_out[b * O + o] * X[b * I + i];
    }
    grad_w[gid] = sum / (float)N;
}

extern "C" __global__ void leaky_relu_backward(
    const float* X,
    float* grad,
    unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    if (X[gid] < 0.0f) grad[gid] *= 0.01f;
}

extern "C" __global__ void sgd_update(
    float* W,
    const float* grad_w,
    float lr,
    unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    W[gid] -= lr * grad_w[gid];
}

extern "C" __global__ void accuracy_kernel(
    const float* logits,
    const unsigned char* labels,
    unsigned int* correct,
    unsigned int N,
    unsigned int O)
{
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= N) return;

    unsigned int best_i = 0;
    float best_v = logits[b * O];
    for (unsigned int i = 1; i < O; ++i)
    {
        float v = logits[b * O + i];
        if (v > best_v)
        {
            best_v = v;
            best_i = i;
        }
    }
    if (best_i == (unsigned int)labels[b]) atomicAdd(correct, 1u);
}
