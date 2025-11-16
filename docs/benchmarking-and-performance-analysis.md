# Benchmarking and Performance Analysis for Transformers

This guide covers benchmarking methodologies, performance metrics, analysis techniques, and optimization strategies for transformer models. Whether you're profiling training speed, inference latency, or memory usage, this guide provides the tools and knowledge to systematically measure and improve performance.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Benchmarking Fundamentals](#benchmarking-fundamentals)
3. [Key Performance Metrics](#key-performance-metrics)
4. [Running Go Benchmarks](#running-go-benchmarks)
5. [Profiling Tools](#profiling-tools)
6. [Memory Analysis](#memory-analysis)
7. [GPU Benchmarking](#gpu-benchmarking)
8. [Attention Mechanism Benchmarks](#attention-mechanism-benchmarks)
9. [Matrix Multiplication Optimization](#matrix-multiplication-optimization)
10. [End-to-End Model Benchmarks](#end-to-end-model-benchmarks)
11. [Performance Analysis Workflow](#performance-analysis-workflow)
12. [Optimization Strategies](#optimization-strategies)
13. [Production Monitoring](#production-monitoring)
14. [Appendix: Complete Benchmark Suite](#appendix-complete-benchmark-suite)

---

## Introduction

Performance optimization is critical for transformer models due to their computational intensity. A GPT-3-scale model with 175B parameters requires:

- **Training**: ~3640 petaFLOP-days (314 ZFLOPS-seconds)
- **Inference**: Hundreds of GFLOPs per token generated
- **Memory**: Hundreds of GBs for model weights + activations

Systematic benchmarking helps identify bottlenecks, validate optimizations, and ensure production readiness.

**Key Principles:**
1. **Measure First**: Profile before optimizing to avoid premature optimization
2. **Isolate Components**: Benchmark individual operations (attention, matmul) before full models
3. **Realistic Workloads**: Use production-like batch sizes, sequence lengths, and hardware
4. **Statistical Rigor**: Run multiple iterations, report mean + standard deviation
5. **Track Regressions**: Automated benchmarking in CI/CD prevents performance degradation

---

## Benchmarking Fundamentals

### What to Benchmark

**Computation Time:**
- Wall-clock time (real-world latency)
- CPU time (actual CPU cycles used)
- GPU kernel time (device execution time)

**Throughput:**
- Tokens/second (inference)
- Samples/second (training)
- FLOPS (floating-point operations per second)

**Memory:**
- Peak memory usage (maximum allocation)
- Memory bandwidth (GB/s)
- Cache hit rates (L1/L2/L3)

**Efficiency:**
- Model FLOPS Utilization (MFU): actual FLOPS / theoretical peak FLOPS
- Arithmetic intensity: FLOPs per byte of memory access
- GPU utilization percentage

### Benchmarking Best Practices

**1. Warm-up Runs:**
```go
// Run benchmark several times before measuring to warm up caches
func BenchmarkMatMul(b *testing.B) {
	// Setup
	A := NewRandomMatrix(1024, 1024)
	B := NewRandomMatrix(1024, 1024)
	C := NewMatrix(1024, 1024)

	// Warm-up (not measured)
	for i := 0; i < 10; i++ {
		MatMul(A, B, C)
	}

	// Reset timer to exclude setup and warm-up
	b.ResetTimer()

	// Measured iterations
	for i := 0; i < b.N; i++ {
		MatMul(A, B, C)
	}
}
```

**2. Exclude Setup Time:**
```go
func BenchmarkWithSetup(b *testing.B) {
	// Expensive setup (excluded from timing)
	largeData := generateLargeDataset(1000000)

	// Only measure the operation we care about
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		processData(largeData)
	}
}
```

**3. Measure Allocations:**
```go
func BenchmarkWithAllocations(b *testing.B) {
	// Report allocations per operation
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// This will track heap allocations
		result := expensiveOperation()
		_ = result
	}
}
```

**4. Use Sub-benchmarks:**
```go
func BenchmarkOperations(b *testing.B) {
	sizes := []int{128, 512, 1024, 4096}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			data := make([]float64, size)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				process(data)
			}
		})
	}
}
```

---

## Key Performance Metrics

### 1. Latency

**Definition**: Time to complete a single operation or request.

**Measurement**:
```go
start := time.Now()
result := model.Forward(input)
latency := time.Since(start)
```

**Critical for**:
- Real-time inference (chatbots, autocomplete)
- User-facing applications with SLAs
- Latency-sensitive pipelines

**Targets**:
- Interactive inference: <100ms
- Batch inference: <1s per batch
- Training step: Depends on model size and batch size

### 2. Throughput

**Definition**: Number of operations completed per unit time.

**Measurement**:
```go
func MeasureThroughput(model *Transformer, inputs []Tensor, duration time.Duration) float64 {
	start := time.Now()
	processed := 0

	for time.Since(start) < duration {
		for _, input := range inputs {
			model.Forward(input)
			processed++
		}
	}

	elapsed := time.Since(start).Seconds()
	return float64(processed) / elapsed // ops/sec
}
```

**Critical for**:
- Batch inference pipelines
- Training efficiency
- Cost optimization (maximize GPU utilization)

**Targets**:
- Training: Maximize tokens/sec per GPU
- Inference: Depends on batch size and latency constraints

### 3. Model FLOPS Utilization (MFU)

**Definition**: Actual FLOPS achieved / Theoretical peak FLOPS

**Calculation**:
```go
// Example for matrix multiplication (M×K) × (K×N) = M×N
// FLOPs = 2*M*N*K (multiply-add operations)
func ComputeMFU(M, K, N int, duration time.Duration, peakFLOPS float64) float64 {
	flops := float64(2 * M * N * K)
	actualFLOPS := flops / duration.Seconds()
	mfu := actualFLOPS / peakFLOPS
	return mfu
}

// Example: NVIDIA A100 has ~312 TFLOPS FP16 peak
// If we achieve 156 TFLOPS, MFU = 50%
```

**Good MFU Values**:
- Training: 30-50% (state-of-the-art implementations)
- Inference: 10-30% (lower due to memory-bound operations)
- Matrix multiplication: 80-95% (compute-bound, well-optimized)

### 4. Memory Bandwidth

**Definition**: Rate of data transfer to/from memory.

**Measurement**:
```go
func MeasureBandwidth(size int, iterations int) float64 {
	data := make([]float64, size)

	start := time.Now()
	for i := 0; i < iterations; i++ {
		// Memory-bound operation (all reads/writes to memory)
		for j := range data {
			data[j] = data[j] + 1.0
		}
	}
	elapsed := time.Since(start).Seconds()

	// Bytes transferred: 1 read + 1 write per element per iteration
	bytes := float64(size * 8 * 2 * iterations) // 8 bytes per float64
	bandwidth := bytes / elapsed / 1e9 // GB/s
	return bandwidth
}
```

**Typical Values**:
- DDR4 RAM: 25-50 GB/s
- DDR5 RAM: 50-100 GB/s
- GPU HBM2: 900 GB/s (V100)
- GPU HBM2e: 1.6 TB/s (A100)
- GPU HBM3: 3.35 TB/s (H100)

### 5. Arithmetic Intensity

**Definition**: FLOPs per byte of memory accessed (operations:bytes ratio).

**Importance**:
- **High intensity** (>50): Compute-bound (GPU cores are bottleneck)
- **Low intensity** (<10): Memory-bound (memory bandwidth is bottleneck)

**Examples**:

```go
// Matrix multiplication: C[M×N] = A[M×K] × B[K×N]
// FLOPs: 2*M*N*K
// Memory: (M*K + K*N + M*N) * 8 bytes (assuming all data loaded once)
// Arithmetic Intensity = 2*M*N*K / (8*(M*K + K*N + M*N))

func MatMulArithmeticIntensity(M, K, N int) float64 {
	flops := 2.0 * float64(M*N*K)
	bytes := 8.0 * float64(M*K+K*N+M*N) // 8 bytes per float64
	return flops / bytes
}

// Example: 1024×1024 × 1024×1024
// FLOPs = 2 * 1024^3 = 2,147,483,648
// Bytes = 8 * (1024^2 + 1024^2 + 1024^2) = 25,165,824
// Intensity = 85.3 (compute-bound!)
```

**For transformers**:
- Attention (QK^T): Low intensity (memory-bound)
- Matrix multiplication (linear layers): High intensity (compute-bound)
- Softmax: Low intensity (memory-bound)
- LayerNorm/RMSNorm: Low intensity (memory-bound)

---

## Running Go Benchmarks

### Basic Benchmark

```go
// File: operations_test.go
package main

import "testing"

// Benchmark function must start with "Benchmark"
func BenchmarkDotProduct(b *testing.B) {
	// Setup
	x := make([]float64, 1024)
	y := make([]float64, 1024)
	for i := range x {
		x[i] = float64(i)
		y[i] = float64(i * 2)
	}

	// Reset timer (excludes setup time)
	b.ResetTimer()

	// Run operation b.N times
	for i := 0; i < b.N; i++ {
		DotProduct(x, y)
	}
}
```

**Run benchmark:**
```bash
# Run all benchmarks in current directory
go test -bench=.

# Run specific benchmark
go test -bench=BenchmarkDotProduct

# Run with memory allocation stats
go test -bench=. -benchmem

# Run for longer (more accurate results)
go test -bench=. -benchtime=10s

# Run with CPU count
go test -bench=. -cpu=1,2,4,8
```

**Example output:**
```
BenchmarkDotProduct-8    1000000    1043 ns/op    0 B/op    0 allocs/op
```
- `BenchmarkDotProduct-8`: Function name, 8 = GOMAXPROCS
- `1000000`: Number of iterations (b.N)
- `1043 ns/op`: Time per operation
- `0 B/op`: Bytes allocated per operation
- `0 allocs/op`: Number of allocations per operation

### Comparing Benchmarks

```bash
# Save baseline
go test -bench=. -benchmem > old.txt

# Make changes to code

# Run new benchmark
go test -bench=. -benchmem > new.txt

# Compare (requires benchstat: go install golang.org/x/perf/cmd/benchstat@latest)
benchstat old.txt new.txt
```

**Example benchstat output:**
```
name            old time/op    new time/op    delta
DotProduct-8      1.04µs ± 2%    0.52µs ± 1%  -50.00%  (p=0.000 n=10+10)

name            old alloc/op   new alloc/op   delta
DotProduct-8       0.00B          0.00B          ~     (all equal)

name            old allocs/op  new allocs/op  delta
DotProduct-8        0.00           0.00          ~     (all equal)
```

### Benchmark Suite Structure

```go
// operations_test.go
package main

import (
	"fmt"
	"testing"
)

// Benchmark different implementations
func BenchmarkDotProductGo(b *testing.B) {
	benchmarkDotProduct(b, dotProductGo)
}

func BenchmarkDotProductSIMD(b *testing.B) {
	benchmarkDotProduct(b, dotProductSIMD)
}

// Helper function for benchmarking
func benchmarkDotProduct(b *testing.B, fn func([]float64, []float64) float64) {
	// Test multiple sizes
	sizes := []int{128, 512, 1024, 4096, 16384}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			x := make([]float64, size)
			y := make([]float64, size)
			for i := range x {
				x[i] = float64(i)
				y[i] = float64(i)
			}

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				fn(x, y)
			}
		})
	}
}

// Benchmark with varying parameters
func BenchmarkAttention(b *testing.B) {
	configs := []struct {
		seqLen  int
		headDim int
		numHeads int
	}{
		{128, 64, 8},
		{512, 64, 8},
		{1024, 64, 8},
		{2048, 64, 12},
	}

	for _, cfg := range configs {
		name := fmt.Sprintf("seq%d_dim%d_heads%d", cfg.seqLen, cfg.headDim, cfg.numHeads)
		b.Run(name, func(b *testing.B) {
			// Setup
			Q := NewTensor(1, cfg.numHeads, cfg.seqLen, cfg.headDim)
			K := NewTensor(1, cfg.numHeads, cfg.seqLen, cfg.headDim)
			V := NewTensor(1, cfg.numHeads, cfg.seqLen, cfg.headDim)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MultiHeadAttention(Q, K, V)
			}
		})
	}
}
```

---

## Profiling Tools

### 1. CPU Profiling

**Identifies which functions consume the most CPU time.**

```bash
# Generate CPU profile
go test -bench=BenchmarkMatMul -cpuprofile=cpu.prof

# Analyze profile interactively
go tool pprof cpu.prof

# Common pprof commands:
# - top: Show top CPU consumers
# - list <function>: Show line-by-line breakdown
# - web: Generate call graph (requires graphviz)
# - pdf: Generate PDF call graph
```

**Example pprof session:**
```
(pprof) top10
Showing nodes accounting for 8.23s, 95.26% of 8.64s total
Dropped 45 nodes (cum <= 0.04s)
Showing top 10 nodes out of 23
      flat  flat%   sum%        cum   cum%
     3.21s 37.15% 37.15%      3.21s 37.15%  main.MatMul
     2.14s 24.77% 61.92%      2.14s 24.77%  runtime.memmove
     1.23s 14.24% 76.16%      1.23s 14.24%  main.DotProduct
     0.87s 10.07% 86.23%      0.87s 10.07%  math.Exp
     0.45s  5.21% 91.44%      0.45s  5.21%  main.Softmax
```

### 2. Memory Profiling

**Identifies memory allocation hotspots.**

```bash
# Generate memory profile
go test -bench=BenchmarkTransformer -memprofile=mem.prof

# Analyze profile
go tool pprof mem.prof

# Or with allocation sampling
go test -bench=. -memprofilerate=1
```

**Example memory analysis:**
```
(pprof) top
Showing nodes accounting for 2048.17MB, 100% of 2048.17MB total
      flat  flat%   sum%        cum   cum%
 1024.52MB 50.02% 50.02%  1024.52MB 50.02%  main.NewTensor
  512.31MB 25.01% 75.03%   512.31MB 25.01%  main.(*Tensor).Clone
  256.15MB 12.51% 87.54%   256.15MB 12.51%  main.MatMul
  255.19MB 12.46%   100%   255.19MB 12.46%  main.Softmax
```

**Find allocation sites:**
```
(pprof) list NewTensor
Total: 2.00GB
ROUTINE ======================== main.NewTensor in /path/to/tensor.go
   1.00GB     1.00GB (flat, cum) 50.02% of Total
        .          .     10:func NewTensor(dims ...int) *Tensor {
        .          .     11:    size := 1
        .          .     12:    for _, d := range dims {
        .          .     13:        size *= d
        .          .     14:    }
   1.00GB     1.00GB     15:    data := make([]float64, size)  // <-- ALLOCATION HERE
        .          .     16:    return &Tensor{dims: dims, data: data}
        .          .     17:}
```

### 3. Trace Analysis

**Visualizes goroutine execution, blocking, and synchronization.**

```bash
# Generate trace
go test -bench=BenchmarkParallelTraining -trace=trace.out

# View trace (opens browser)
go tool trace trace.out
```

**Trace views:**
- **Goroutine analysis**: See goroutine creation, execution, blocking
- **Network blocking**: Identify network I/O bottlenecks
- **Synchronization blocking**: Find mutex contention
- **Syscall blocking**: See OS-level blocking
- **Scheduler latency**: Measure goroutine scheduling delays

### 4. Flame Graphs

**Visual representation of CPU profile showing call stacks.**

```bash
# Install flamegraph tool
go get -u github.com/google/pprof

# Generate flame graph
go test -bench=. -cpuprofile=cpu.prof
go tool pprof -http=:8080 cpu.prof
# Opens browser with interactive flame graph
```

**Interpreting flame graphs:**
- Width = time spent in function
- Height = call stack depth
- Color = usually just for differentiation (no semantic meaning)
- Click on a box to zoom into that subtree

---

## Memory Analysis

### Memory Usage Components

**For transformer models:**

```go
// Calculate memory footprint
type MemoryFootprint struct {
	ModelParams     int64 // Model weights
	Optimizer       int64 // Optimizer states (Adam: 2x params)
	Gradients       int64 // Gradient tensors
	Activations     int64 // Forward pass activations
	TempBuffers     int64 // Temporary computation buffers
	KVCache         int64 // Key-value cache (inference)
}

func EstimateMemory(config TransformerConfig, batchSize, seqLen int, training bool) MemoryFootprint {
	numParams := config.NumLayers * (
		config.HiddenSize*config.HiddenSize*4 + // Q,K,V,O projections
		config.HiddenSize*config.FFNDim*2 +     // FFN up/down
		config.HiddenSize*4,                    // Biases + norms
	)

	// Model weights: 4 bytes per float32 parameter
	modelMemory := int64(numParams * 4)

	// Optimizer states (Adam): 2x params for momentum and variance
	optimizerMemory := int64(0)
	if training {
		optimizerMemory = modelMemory * 2
	}

	// Gradients: same as model size
	gradientMemory := int64(0)
	if training {
		gradientMemory = modelMemory
	}

	// Activations: depends on batch size and sequence length
	// Rough estimate: 12 * numLayers * batchSize * seqLen * hiddenSize * 4 bytes
	activationMemory := int64(12 * config.NumLayers * batchSize * seqLen * config.HiddenSize * 4)

	// KV cache (inference): 2 (K+V) * numLayers * batchSize * seqLen * headDim * numHeads * 2 bytes (FP16)
	kvCacheMemory := int64(0)
	if !training {
		kvCacheMemory = int64(2 * config.NumLayers * batchSize * seqLen * config.HiddenSize * 2)
	}

	// Temporary buffers (attention scores, etc.): rough estimate
	tempMemory := int64(batchSize * seqLen * seqLen * config.NumHeads * 4)

	return MemoryFootprint{
		ModelParams:  modelMemory,
		Optimizer:    optimizerMemory,
		Gradients:    gradientMemory,
		Activations:  activationMemory,
		TempBuffers:  tempMemory,
		KVCache:      kvCacheMemory,
	}
}

// Example: GPT-2-small (117M params) with batch=8, seq=1024
// Model: 117M * 4 = 468 MB
// Optimizer: 468 MB * 2 = 936 MB
// Gradients: 468 MB
// Activations: ~2 GB (depends on batch/seq)
// Total training: ~4 GB
```

### Memory Optimization Techniques

**1. Gradient Checkpointing (Activation Recomputation):**

```go
// Trade compute for memory: recompute activations in backward pass
// instead of storing them during forward pass

type GradientCheckpointConfig struct {
	CheckpointEveryN int // Checkpoint every N layers
}

func (t *Transformer) ForwardWithCheckpointing(x *Tensor, config GradientCheckpointConfig) *Tensor {
	checkpoints := make([]*Tensor, 0)

	for i, layer := range t.Layers {
		x = layer.Forward(x)

		// Save checkpoint every N layers
		if i%config.CheckpointEveryN == 0 {
			checkpoints = append(checkpoints, x.Clone())
		}
		// Don't save intermediate activations (will recompute in backward pass)
	}

	return x
}

// Memory savings:
// Without checkpointing: O(numLayers * batchSize * seqLen * hiddenSize)
// With checkpointing every N layers: O(numLayers/N * batchSize * seqLen * hiddenSize)
// Example: Checkpoint every 4 layers → 75% memory reduction for activations
```

**2. Mixed Precision Training:**

```go
// Use FP16 for most operations, FP32 for critical operations
// Memory savings: ~50% for activations and weights

type MixedPrecisionConfig struct {
	EnableFP16      bool
	LossScale       float64 // Scale loss to prevent underflow
	DynamicScaling  bool    // Adjust loss scale dynamically
}

func (t *Transformer) ForwardMixedPrecision(x *Tensor, config MixedPrecisionConfig) *Tensor {
	// Convert input to FP16
	x16 := x.ToFloat16()

	// Forward pass in FP16
	for _, layer := range t.Layers {
		x16 = layer.ForwardFP16(x16)
	}

	// Convert back to FP32 for loss computation (critical for numerical stability)
	output := x16.ToFloat32()

	return output
}

// Memory savings: ~50%
// Speed improvement: 2-3x on modern GPUs (V100, A100, H100 with Tensor Cores)
```

**3. CPU Offloading (ZeRO-Offload):**

```go
// Offload optimizer states and gradients to CPU RAM
// Overlap computation and data transfer

type OffloadConfig struct {
	OffloadOptimizer bool // Offload optimizer states to CPU
	OffloadGradients bool // Offload gradients to CPU
	PrefetchLayers   int  // Prefetch N layers ahead
}

func (t *Transformer) TrainWithOffload(batch *Tensor, config OffloadConfig) {
	// Forward pass (keep on GPU)
	output := t.Forward(batch)
	loss := t.ComputeLoss(output)

	// Backward pass
	gradients := t.Backward(loss)

	// Offload gradients to CPU asynchronously
	if config.OffloadGradients {
		go func() {
			t.TransferToCPU(gradients)
		}()
	}

	// Optimizer step (can run on CPU with offloaded states)
	if config.OffloadOptimizer {
		t.OptimizerStepCPU(gradients)
	} else {
		t.OptimizerStepGPU(gradients)
	}
}

// Memory savings: Up to 10x for very large models
// Cost: ~10-20% slower training due to CPU-GPU transfers
```

### Monitoring Memory Usage

```go
import (
	"runtime"
	"runtime/debug"
)

type MemoryStats struct {
	Alloc      uint64 // Bytes allocated and still in use
	TotalAlloc uint64 // Bytes allocated (cumulative)
	Sys        uint64 // Bytes obtained from system
	NumGC      uint32 // Number of GC runs
	HeapAlloc  uint64 // Bytes allocated on heap
	HeapInuse  uint64 // Bytes in use on heap
}

func GetMemoryStats() MemoryStats {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return MemoryStats{
		Alloc:      m.Alloc,
		TotalAlloc: m.TotalAlloc,
		Sys:        m.Sys,
		NumGC:      m.NumGC,
		HeapAlloc:  m.HeapAlloc,
		HeapInuse:  m.HeapInuse,
	}
}

func MonitorMemoryDuringTraining(model *Transformer) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		stats := GetMemoryStats()
		log.Printf("Memory: Alloc=%dMB, Sys=%dMB, NumGC=%d",
			stats.Alloc/1024/1024,
			stats.Sys/1024/1024,
			stats.NumGC)

		// Trigger GC if memory usage is high
		if stats.HeapInuse > 8*1024*1024*1024 { // 8 GB
			runtime.GC()
			debug.FreeOSMemory()
		}
	}
}
```

---

## GPU Benchmarking

### GPU Metrics

**Key metrics for GPU performance:**

1. **SM (Streaming Multiprocessor) Utilization**: Percentage of time SMs are active
2. **Memory Bandwidth Utilization**: Percentage of peak memory bandwidth used
3. **Tensor Core Utilization**: Percentage of time Tensor Cores are active (A100/H100)
4. **Achieved Occupancy**: Ratio of active warps to maximum possible warps

### Profiling with NVIDIA Nsight

```bash
# Profile GPU kernels
nsys profile --stats=true \
  --trace=cuda,nvtx \
  --output=profile.qdrep \
  ./your_program

# View profile (opens GUI)
nsys-ui profile.qdrep

# Command-line stats
nsys stats profile.qdrep
```

**Key metrics to look for:**
- Kernel execution time
- Memory transfer time (host-to-device, device-to-host)
- Kernel launch overhead
- Stream synchronization delays

### Example GPU Benchmark (Go + CUDA)

```go
// gpu_benchmark.go
package main

/*
#cgo LDFLAGS: -lcublas
#include <cuda_runtime.h>
#include <cublas_v2.h>

extern void cuMatMul(float* A, float* B, float* C, int M, int N, int K);
*/
import "C"
import (
	"testing"
	"unsafe"
)

func BenchmarkGPUMatMul(b *testing.B) {
	sizes := []int{128, 512, 1024, 2048, 4096}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			M, N, K := size, size, size

			// Allocate host memory
			A := make([]float32, M*K)
			B := make([]float32, K*N)
			C := make([]float32, M*N)

			// Initialize with random data
			for i := range A {
				A[i] = float32(i)
			}
			for i := range B {
				B[i] = float32(i)
			}

			// Allocate device memory
			var dA, dB, dC unsafe.Pointer
			C.cudaMalloc(&dA, C.size_t(len(A)*4))
			C.cudaMalloc(&dB, C.size_t(len(B)*4))
			C.cudaMalloc(&dC, C.size_t(len(C)*4))

			// Copy data to device
			C.cudaMemcpy(dA, unsafe.Pointer(&A[0]), C.size_t(len(A)*4), C.cudaMemcpyHostToDevice)
			C.cudaMemcpy(dB, unsafe.Pointer(&B[0]), C.size_t(len(B)*4), C.cudaMemcpyHostToDevice)

			// Warm-up
			for i := 0; i < 10; i++ {
				C.cuMatMul((*C.float)(dA), (*C.float)(dB), (*C.float)(dC), C.int(M), C.int(N), C.int(K))
			}
			C.cudaDeviceSynchronize()

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				C.cuMatMul((*C.float)(dA), (*C.float)(dB), (*C.float)(dC), C.int(M), C.int(N), C.int(K))
				C.cudaDeviceSynchronize()
			}

			// Calculate FLOPS
			flops := float64(2 * M * N * K) // multiply-add operations
			elapsed := b.Elapsed().Seconds()
			tflops := flops * float64(b.N) / elapsed / 1e12
			b.ReportMetric(tflops, "TFLOPS")

			// Cleanup
			C.cudaFree(dA)
			C.cudaFree(dB)
			C.cudaFree(dC)
		})
	}
}
```

### Theoretical vs. Actual Performance

```go
// Calculate theoretical peak performance
type GPUSpec struct {
	Name           string
	ComputeUnits   int     // Number of SMs
	CoresPerSM     int     // CUDA cores per SM
	ClockSpeed     float64 // GHz
	TensorCores    int     // Tensor Cores per SM
	TensorTFLOPS   float64 // Peak TFLOPS (FP16 with Tensor Cores)
	MemoryBandwidth float64 // GB/s
}

var A100 = GPUSpec{
	Name:            "NVIDIA A100",
	ComputeUnits:    108,
	CoresPerSM:      64,
	ClockSpeed:      1.41,
	TensorCores:     4,
	TensorTFLOPS:    312,
	MemoryBandwidth: 1555,
}

func CalculateMFU(actualTFLOPS float64, gpu GPUSpec) float64 {
	return actualTFLOPS / gpu.TensorTFLOPS * 100
}

// Example: You measure 156 TFLOPS on A100
// MFU = 156 / 312 = 50% (excellent!)
```

---

## Attention Mechanism Benchmarks

Attention is often the bottleneck in transformers, especially for long sequences. Benchmarking different attention implementations is critical.

### Standard Attention

```go
// Standard O(N²) attention: QK^T → Softmax → × V
func BenchmarkStandardAttention(b *testing.B) {
	configs := []struct {
		batch, heads, seqLen, headDim int
	}{
		{1, 8, 128, 64},
		{1, 8, 512, 64},
		{1, 8, 1024, 64},
		{1, 8, 2048, 64},
		{2, 12, 1024, 64},
	}

	for _, cfg := range configs {
		name := fmt.Sprintf("B%d_H%d_S%d_D%d", cfg.batch, cfg.heads, cfg.seqLen, cfg.headDim)
		b.Run(name, func(b *testing.B) {
			Q := NewTensor(cfg.batch, cfg.heads, cfg.seqLen, cfg.headDim)
			K := NewTensor(cfg.batch, cfg.heads, cfg.seqLen, cfg.headDim)
			V := NewTensor(cfg.batch, cfg.heads, cfg.seqLen, cfg.headDim)

			// Initialize
			for i := range Q.data {
				Q.data[i] = rand.Float64()
				K.data[i] = rand.Float64()
				V.data[i] = rand.Float64()
			}

			scale := 1.0 / math.Sqrt(float64(cfg.headDim))

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				// Q × K^T
				scores := MatMul(Q, K.Transpose())

				// Scale
				scores.Scale(scale)

				// Softmax
				scores.Softmax(-1)

				// Scores × V
				output := MatMul(scores, V)

				_ = output
			}

			// Report complexity
			complexity := float64(cfg.batch * cfg.heads * cfg.seqLen * cfg.seqLen * cfg.headDim)
			b.ReportMetric(complexity/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

// Expected results (CPU, no SIMD):
// B1_H8_S128_D64:   ~100 MFLOPS
// B1_H8_S512_D64:   ~50 MFLOPS (quadratic slowdown!)
// B1_H8_S1024_D64:  ~25 MFLOPS
// B1_H8_S2048_D64:  ~12 MFLOPS
```

### Flash Attention

```go
// Flash Attention: O(N²) but memory-efficient (blocked computation)
func BenchmarkFlashAttention(b *testing.B) {
	configs := []struct {
		batch, heads, seqLen, headDim int
		blockSize                      int
	}{
		{1, 8, 128, 64, 32},
		{1, 8, 512, 64, 64},
		{1, 8, 1024, 64, 64},
		{1, 8, 2048, 64, 128},
		{2, 12, 1024, 64, 64},
	}

	for _, cfg := range configs {
		name := fmt.Sprintf("B%d_H%d_S%d_D%d_Block%d",
			cfg.batch, cfg.heads, cfg.seqLen, cfg.headDim, cfg.blockSize)
		b.Run(name, func(b *testing.B) {
			Q := NewTensor(cfg.batch, cfg.heads, cfg.seqLen, cfg.headDim)
			K := NewTensor(cfg.batch, cfg.heads, cfg.seqLen, cfg.headDim)
			V := NewTensor(cfg.batch, cfg.heads, cfg.seqLen, cfg.headDim)

			// Initialize
			for i := range Q.data {
				Q.data[i] = rand.Float64()
				K.data[i] = rand.Float64()
				V.data[i] = rand.Float64()
			}

			config := NewFlashAttentionConfig(cfg.headDim)
			config.BlockSize = cfg.blockSize

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				output := FlashAttention(Q, K, V, config)
				_ = output
			}
		})
	}
}

// Expected improvements over standard attention:
// - Memory: 5-20x reduction (no full attention matrix)
// - Speed: 2-4x faster (better cache utilization)
// - Especially significant for long sequences (>1024)
```

### Sparse Attention

```go
// Sparse attention: only compute attention for a subset of positions
func BenchmarkSparseAttention(b *testing.B) {
	patterns := []string{"local", "strided", "random", "bigbird"}

	for _, pattern := range patterns {
		b.Run(pattern, func(b *testing.B) {
			seqLen := 1024
			headDim := 64
			numHeads := 8

			Q := NewTensor(1, numHeads, seqLen, headDim)
			K := NewTensor(1, numHeads, seqLen, headDim)
			V := NewTensor(1, numHeads, seqLen, headDim)

			// Generate sparsity pattern
			var mask *SparseMask
			switch pattern {
			case "local":
				mask = LocalAttentionMask(seqLen, 128) // Window size 128
			case "strided":
				mask = StridedAttentionMask(seqLen, 8) // Stride 8
			case "random":
				mask = RandomAttentionMask(seqLen, 0.1) // 10% sparsity
			case "bigbird":
				mask = BigBirdMask(seqLen, 64, 8, 2) // Window, stride, global
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				output := SparseAttention(Q, K, V, mask)
				_ = output
			}
		})
	}
}

// Complexity:
// Standard: O(N²)
// Local (window W): O(N*W)
// Strided: O(N²/stride)
// BigBird: O(N * (window + global))
```

### KV Cache Benchmarks (Inference)

```go
// KV cache: reuse K and V from previous tokens during autoregressive generation
func BenchmarkKVCacheInference(b *testing.B) {
	seqLen := 1024
	headDim := 64
	numHeads := 8
	numLayers := 12

	// Allocate KV cache
	cache := NewKVCache(numLayers, numHeads, seqLen, headDim)

	b.Run("with_cache", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Generate 100 new tokens
			for pos := 0; pos < 100; pos++ {
				Q := NewTensor(1, numHeads, 1, headDim) // Only 1 new token

				// Retrieve cached K, V (don't recompute)
				K, V := cache.Get(0, pos) // Layer 0, position pos

				// Compute attention only for new token
				output := AttentionWithCache(Q, K, V)
				_ = output
			}
		}
	})

	b.Run("without_cache", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Generate 100 new tokens (recompute everything)
			for pos := 0; pos < 100; pos++ {
				// Recompute Q, K, V for ALL tokens (0 to pos)
				Q := NewTensor(1, numHeads, pos+1, headDim)
				K := NewTensor(1, numHeads, pos+1, headDim)
				V := NewTensor(1, numHeads, pos+1, headDim)

				output := StandardAttention(Q, K, V)
				_ = output
			}
		}
	})
}

// Expected results:
// with_cache: ~100x faster (only computes for 1 token, not all previous)
// without_cache: Recomputes O(N²) attention for each new token
```

---

## Matrix Multiplication Optimization

Matrix multiplication (GEMM - General Matrix Multiply) is the core operation in transformers. Optimizing matmul is critical for overall performance.

### Naive vs. Optimized

```go
// Naive implementation: O(M*N*K) with poor cache utilization
func MatMulNaive(A, B, C []float64, M, N, K int) {
	// C[M×N] = A[M×K] × B[K×N]
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := 0.0
			for k := 0; k < K; k++ {
				sum += A[i*K+k] * B[k*N+j]
			}
			C[i*N+j] = sum
		}
	}
}

// Blocked (tiled) implementation: improves cache utilization
func MatMulBlocked(A, B, C []float64, M, N, K int, blockSize int) {
	// Process in blocks to fit in L1/L2 cache
	for ii := 0; ii < M; ii += blockSize {
		for jj := 0; jj < N; jj += blockSize {
			for kk := 0; kk < K; kk += blockSize {
				// Compute block
				iMax := min(ii+blockSize, M)
				jMax := min(jj+blockSize, N)
				kMax := min(kk+blockSize, K)

				for i := ii; i < iMax; i++ {
					for j := jj; j < jMax; j++ {
						sum := C[i*N+j]
						for k := kk; k < kMax; k++ {
							sum += A[i*K+k] * B[k*N+j]
						}
						C[i*N+j] = sum
					}
				}
			}
		}
	}
}

// SIMD implementation: vectorized operations
func MatMulSIMD(A, B, C []float64, M, N, K int) {
	// Uses AVX-512 intrinsics (8 doubles per vector)
	const vecSize = 8

	for i := 0; i < M; i++ {
		for j := 0; j < N; j += vecSize {
			// Load 8 elements at a time
			var vsum [vecSize]float64

			for k := 0; k < K; k++ {
				va := A[i*K+k]
				// Vectorized multiply-add
				for v := 0; v < vecSize && j+v < N; v++ {
					vsum[v] += va * B[k*N+j+v]
				}
			}

			// Store result
			for v := 0; v < vecSize && j+v < N; v++ {
				C[i*N+j+v] = vsum[v]
			}
		}
	}
}
```

### Benchmark Comparison

```go
func BenchmarkMatMulVariants(b *testing.B) {
	sizes := []int{128, 512, 1024}

	for _, size := range sizes {
		M, N, K := size, size, size
		A := make([]float64, M*K)
		B := make([]float64, K*N)
		C := make([]float64, M*N)

		// Initialize
		for i := range A {
			A[i] = float64(i)
			B[i] = float64(i)
		}

		b.Run(fmt.Sprintf("Naive_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulNaive(A, B, C, M, N, K)
			}
			reportGFLOPS(b, M, N, K)
		})

		b.Run(fmt.Sprintf("Blocked_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulBlocked(A, B, C, M, N, K, 64)
			}
			reportGFLOPS(b, M, N, K)
		})

		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatMulSIMD(A, B, C, M, N, K)
			}
			reportGFLOPS(b, M, N, K)
		})
	}
}

func reportGFLOPS(b *testing.B, M, N, K int) {
	flops := float64(2 * M * N * K * b.N)
	elapsed := b.Elapsed().Seconds()
	gflops := flops / elapsed / 1e9
	b.ReportMetric(gflops, "GFLOPS")
}

// Expected results (M=N=K=1024, Intel Core i9):
// Naive:   ~5 GFLOPS (poor cache utilization)
// Blocked: ~20 GFLOPS (4x faster, better cache usage)
// SIMD:    ~60 GFLOPS (12x faster, vectorization)
// GPU:     ~5000 GFLOPS (1000x faster!)
```

### Using Optimized BLAS Libraries

```go
// Use OpenBLAS, Intel MKL, or Apple Accelerate for optimal performance
// These libraries are hand-tuned for specific hardware

/*
#cgo LDFLAGS: -lopenblas
#include <cblas.h>
*/
import "C"

func MatMulBLAS(A, B, C []float64, M, N, K int) {
	// cblas_dgemm: double-precision general matrix multiply
	// C = alpha * A * B + beta * C
	alpha := 1.0
	beta := 0.0

	C.cblas_dgemm(
		C.CblasRowMajor,
		C.CblasNoTrans, C.CblasNoTrans,
		C.int(M), C.int(N), C.int(K),
		C.double(alpha),
		(*C.double)(unsafe.Pointer(&A[0])), C.int(K),
		(*C.double)(unsafe.Pointer(&B[0])), C.int(N),
		C.double(beta),
		(*C.double)(unsafe.Pointer(&C[0])), C.int(N),
	)
}

// Expected performance (M=N=K=1024):
// OpenBLAS: ~100 GFLOPS (near-optimal CPU performance)
// Intel MKL: ~150 GFLOPS (optimized for Intel CPUs)
// Apple Accelerate: ~200 GFLOPS (M1/M2 with AMX)
```

---

## End-to-End Model Benchmarks

### Training Benchmark

```go
func BenchmarkTransformerTraining(b *testing.B) {
	config := TransformerConfig{
		VocabSize:   50000,
		HiddenSize:  768,
		NumLayers:   12,
		NumHeads:    12,
		FFNDim:      3072,
		MaxSeqLen:   512,
		DropoutRate: 0.1,
	}

	model := NewTransformer(config)
	optimizer := NewAdam(0.0001, 0.9, 0.999)

	batchSize := 8
	seqLen := 512

	b.Run("forward", func(b *testing.B) {
		input := NewRandomTensor(batchSize, seqLen)
		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			output := model.Forward(input)
			_ = output
		}
	})

	b.Run("forward_backward", func(b *testing.B) {
		input := NewRandomTensor(batchSize, seqLen)
		target := NewRandomTensor(batchSize, seqLen)

		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			// Forward pass
			output := model.Forward(input)
			loss := CrossEntropyLoss(output, target)

			// Backward pass
			model.Backward(loss)

			// (Don't update weights in benchmark)
		}
	})

	b.Run("full_step", func(b *testing.B) {
		input := NewRandomTensor(batchSize, seqLen)
		target := NewRandomTensor(batchSize, seqLen)

		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			// Forward
			output := model.Forward(input)
			loss := CrossEntropyLoss(output, target)

			// Backward
			model.Backward(loss)

			// Optimizer step
			optimizer.Step(model.Parameters())
			optimizer.ZeroGrad()
		}
	})
}

// Analyze results:
// - Forward: ~50-100ms per batch (depends on hardware)
// - Forward+Backward: ~150-300ms (backward is ~2-3x slower than forward)
// - Full step: ~200-400ms (optimizer adds ~10-20%)
```

### Inference Benchmark

```go
func BenchmarkTransformerInference(b *testing.B) {
	config := TransformerConfig{
		VocabSize:  50000,
		HiddenSize: 768,
		NumLayers:  12,
		NumHeads:   12,
		FFNDim:     3072,
		MaxSeqLen:  512,
	}

	model := NewTransformer(config)
	model.Eval() // Disable dropout

	b.Run("single_token", func(b *testing.B) {
		input := NewTensor(1, 1) // Batch=1, SeqLen=1
		input.data[0] = 42

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			output := model.Forward(input)
			_ = output
		}
	})

	b.Run("batch_inference", func(b *testing.B) {
		batchSizes := []int{1, 8, 32, 128}

		for _, batchSize := range batchSizes {
			b.Run(fmt.Sprintf("batch_%d", batchSize), func(b *testing.B) {
				input := NewRandomTensor(batchSize, 512)

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					output := model.Forward(input)
					_ = output
				}

				// Calculate throughput
				tokensPerBatch := float64(batchSize * 512)
				tokensPerSec := tokensPerBatch * float64(b.N) / b.Elapsed().Seconds()
				b.ReportMetric(tokensPerSec, "tokens/sec")
			})
		}
	})

	b.Run("autoregressive_generation", func(b *testing.B) {
		// Simulate text generation (1 token at a time with KV cache)
		cache := NewKVCache(config.NumLayers, config.NumHeads, 512, config.HiddenSize/config.NumHeads)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Generate 100 tokens
			input := NewTensor(1, 1)
			input.data[0] = 42 // Start token

			for pos := 0; pos < 100; pos++ {
				output := model.ForwardWithCache(input, cache, pos)

				// Sample next token (argmax for simplicity)
				nextToken := output.ArgMax()
				input.data[0] = float64(nextToken)
			}
		}

		// Calculate tokens per second
		tokensPerRun := 100.0
		tokensPerSec := tokensPerRun * float64(b.N) / b.Elapsed().Seconds()
		b.ReportMetric(tokensPerSec, "tokens/sec")
	})
}

// Expected results (GPT-2-small, NVIDIA A100):
// single_token: ~1-2ms (500-1000 tokens/sec)
// batch_8: ~10ms (~4000 tokens/sec)
// batch_32: ~30ms (~5400 tokens/sec)
// autoregressive_generation: ~10-20 tokens/sec (with cache)
```

### Scaling Benchmarks

```go
// Measure how performance scales with model size
func BenchmarkModelSizes(b *testing.B) {
	configs := []struct {
		name       string
		hiddenSize int
		numLayers  int
		numHeads   int
		ffnDim     int
	}{
		{"GPT2-small", 768, 12, 12, 3072},      // 117M params
		{"GPT2-medium", 1024, 24, 16, 4096},    // 345M params
		{"GPT2-large", 1280, 36, 20, 5120},     // 774M params
		{"GPT2-XL", 1600, 48, 25, 6400},        // 1.5B params
	}

	for _, cfg := range configs {
		b.Run(cfg.name, func(b *testing.B) {
			config := TransformerConfig{
				VocabSize:  50000,
				HiddenSize: cfg.hiddenSize,
				NumLayers:  cfg.numLayers,
				NumHeads:   cfg.numHeads,
				FFNDim:     cfg.ffnDim,
				MaxSeqLen:  512,
			}

			model := NewTransformer(config)
			input := NewRandomTensor(8, 512)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				output := model.Forward(input)
				_ = output
			}

			// Report memory usage
			params := model.CountParameters()
			memoryMB := float64(params*4) / 1e6 // 4 bytes per float32
			b.ReportMetric(memoryMB, "memory_MB")
		})
	}
}

// Expected scaling (A100):
// GPT2-small:  ~50ms/batch  (117M params, ~500MB)
// GPT2-medium: ~120ms/batch (345M params, ~1.4GB)
// GPT2-large:  ~220ms/batch (774M params, ~3GB)
// GPT2-XL:     ~400ms/batch (1.5B params, ~6GB)
```

---

## Performance Analysis Workflow

### Step-by-Step Optimization Process

**1. Establish Baseline:**
```bash
# Run baseline benchmarks
go test -bench=. -benchmem -benchtime=10s > baseline.txt
```

**2. Profile to Identify Bottlenecks:**
```bash
# CPU profile
go test -bench=BenchmarkTransformer -cpuprofile=cpu.prof
go tool pprof -top cpu.prof

# Memory profile
go test -bench=BenchmarkTransformer -memprofile=mem.prof
go tool pprof -top mem.prof
```

**3. Analyze Hotspots:**
```
(pprof) top10
Showing nodes accounting for 7.2s, 80% of 9.0s total
      flat  flat%   sum%        cum   cum%
     2.1s 23.33% 23.33%      2.1s 23.33%  main.MatMul            <-- HOTSPOT
     1.8s 20.00% 43.33%      1.8s 20.00%  main.Softmax          <-- HOTSPOT
     1.2s 13.33% 56.67%      1.2s 13.33%  main.LayerNorm
     0.9s 10.00% 66.67%      0.9s 10.00%  runtime.memmove
     0.7s  7.78% 74.44%      0.7s  7.78%  main.Dropout
     0.5s  5.56% 80.00%      0.5s  5.56%  math.Exp
```

**4. Optimize Hotspots:**
- Focus on top 2-3 functions consuming most time
- Try different implementations (SIMD, blocked, library-based)
- Benchmark each optimization individually

**5. Validate Improvements:**
```bash
# Run new benchmarks
go test -bench=. -benchmem -benchtime=10s > optimized.txt

# Compare
benchstat baseline.txt optimized.txt
```

**6. Iterate:**
- Continue profiling → optimizing → validating
- Stop when reaching diminishing returns or target performance

### Example Optimization Session

```go
// BEFORE: Naive softmax implementation
func SoftmaxNaive(x []float64) {
	// Find max (for numerical stability)
	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}

	// Exp and sum
	sum := 0.0
	for i, v := range x {
		x[i] = math.Exp(v - max)
		sum += x[i]
	}

	// Normalize
	for i := range x {
		x[i] /= sum
	}
}

// Profile shows math.Exp is bottleneck (60% of time)

// AFTER: Optimized with SIMD and fast exp approximation
func SoftmaxOptimized(x []float64) {
	// Find max (SIMD)
	max := simdMax(x)

	// Exp and sum (SIMD + fast exp)
	sum := 0.0
	const vecSize = 8
	for i := 0; i < len(x); i += vecSize {
		end := min(i+vecSize, len(x))
		for j := i; j < end; j++ {
			x[j] = fastExp(x[j] - max)
			sum += x[j]
		}
	}

	// Normalize (SIMD)
	simdScale(x, 1.0/sum)
}

// Fast exp approximation (trades accuracy for speed)
func fastExp(x float64) float64 {
	// Based on Schraudolph's method
	if x < -20 {
		return 0
	}
	if x > 20 {
		return math.Exp(20)
	}

	// Linear approximation in log-space
	i := int64(12102203.161561485*x + 1072632447)
	return math.Float64frombits(uint64(i << 32))
}

// Results:
// Before: 1200 ns/op
// After:  320 ns/op (3.75x speedup)
```

---

## Optimization Strategies

### 1. Algorithmic Optimizations

**Flash Attention:**
- Replace O(N²) standard attention with blocked Flash Attention
- **Speedup**: 2-4x
- **Memory savings**: 5-20x

**Kernel Fusion:**
- Combine multiple operations into single GPU kernel
- Example: Fuse LayerNorm + GELU + Dropout
- **Speedup**: 1.5-3x (reduces memory bandwidth)

**Mixed Precision:**
- Use FP16/BF16 for most operations, FP32 for critical ones
- **Speedup**: 2-3x on modern GPUs with Tensor Cores
- **Memory savings**: ~50%

**Gradient Checkpointing:**
- Recompute activations in backward pass instead of storing
- **Memory savings**: 4-10x (enables larger batch sizes)
- **Cost**: ~20% slower training

### 2. Systems Optimizations

**Data Loading:**
```go
// Use buffered channels for asynchronous data loading
func AsyncDataLoader(filenames []string, batchSize int) <-chan *Batch {
	batches := make(chan *Batch, 10) // Buffer 10 batches

	go func() {
		defer close(batches)

		for _, file := range filenames {
			data := LoadData(file)
			batch := CreateBatch(data, batchSize)
			batches <- batch
		}
	}()

	return batches
}

// Training loop
for batch := range AsyncDataLoader(files, 32) {
	model.Train(batch) // GPU computes while CPU loads next batch
}
```

**Tensor Pooling:**
```go
// Reuse tensor allocations to reduce GC pressure
type TensorPool struct {
	pools map[string]*sync.Pool // Key: shape signature
}

func (tp *TensorPool) Get(dims ...int) *Tensor {
	key := shapeKey(dims)
	pool, exists := tp.pools[key]
	if !exists {
		pool = &sync.Pool{
			New: func() interface{} {
				return NewTensor(dims...)
			},
		}
		tp.pools[key] = pool
	}

	return pool.Get().(*Tensor)
}

func (tp *TensorPool) Put(t *Tensor) {
	key := shapeKey(t.dims)
	if pool, exists := tp.pools[key]; exists {
		// Clear data before returning to pool
		for i := range t.data {
			t.data[i] = 0
		}
		pool.Put(t)
	}
}

// Results: 50-90% reduction in allocations, ~20% speedup
```

**Parallelization:**
```go
// Parallelize independent operations
func (t *Transformer) ForwardParallel(x *Tensor) *Tensor {
	numLayers := len(t.Layers)

	// Option 1: Pipeline parallelism (process different samples in different layers)
	pipeline := make(chan *Tensor, numLayers)

	go func() {
		current := x
		for _, layer := range t.Layers {
			current = layer.Forward(current)
		}
		pipeline <- current
	}()

	// Option 2: Data parallelism (split batch across multiple workers)
	batchSize := x.dims[0]
	chunkSize := batchSize / runtime.NumCPU()
	results := make([]*Tensor, runtime.NumCPU())

	var wg sync.WaitGroup
	for i := 0; i < runtime.NumCPU(); i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			start := workerID * chunkSize
			end := min(start+chunkSize, batchSize)
			chunk := x.Slice(start, end, 0) // Slice along batch dimension
			results[workerID] = t.ForwardSingle(chunk)
		}(i)
	}
	wg.Wait()

	// Concatenate results
	return Concatenate(results, 0)
}
```

### 3. Hardware-Specific Optimizations

**CPU SIMD:**
```go
// Use SIMD intrinsics for vectorized operations
// Example: AVX-512 (512-bit vectors = 8 doubles or 16 floats)
func DotProductSIMD(a, b []float64) float64 {
	n := len(a)
	sum := 0.0

	// Process 8 elements at a time
	i := 0
	for ; i+8 <= n; i += 8 {
		// Load 8 elements into vector registers
		va := loadVector(&a[i])
		vb := loadVector(&b[i])

		// Multiply and accumulate
		vprod := mulVector(va, vb)
		sum += sumVector(vprod)
	}

	// Handle remaining elements
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}

	return sum
}

// Speedup: 8x theoretical (often 4-6x in practice due to overhead)
```

**GPU Kernel Fusion:**
```cuda
// Fuse LayerNorm + GELU activation
__global__ void LayerNormGELU(float* input, float* output, float* gamma, float* beta,
                               int N, int D) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	// LayerNorm
	float mean = 0.0f, var = 0.0f;
	for (int i = 0; i < D; i++) {
		mean += input[idx * D + i];
	}
	mean /= D;

	for (int i = 0; i < D; i++) {
		float diff = input[idx * D + i] - mean;
		var += diff * diff;
	}
	var = sqrt(var / D + 1e-5);

	// Normalize + GELU
	for (int i = 0; i < D; i++) {
		float normalized = (input[idx * D + i] - mean) / var;
		normalized = gamma[i] * normalized + beta[i];

		// GELU activation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
		float x = normalized;
		float x3 = x * x * x;
		float tanh_arg = 0.7978845608f * (x + 0.044715f * x3);
		float tanh_val = tanhf(tanh_arg);
		output[idx * D + i] = x * 0.5f * (1.0f + tanh_val);
	}
}

// Speedup: 2-3x vs. separate LayerNorm and GELU kernels
// (saves 1 memory roundtrip)
```

---

## Production Monitoring

### Metrics to Track

```go
type PerformanceMetrics struct {
	// Throughput
	TokensPerSecond      float64
	SamplesPerSecond     float64
	BatchesPerSecond     float64

	// Latency (percentiles)
	P50Latency           time.Duration
	P95Latency           time.Duration
	P99Latency           time.Duration

	// Resource utilization
	CPUUtilization       float64 // 0-100%
	GPUUtilization       float64 // 0-100%
	MemoryUsageBytes     int64
	GPUMemoryUsageBytes  int64

	// Model metrics
	ModelFLOPSUtilization float64 // MFU
	CacheHitRate          float64
}

func MonitorPerformance(model *Transformer, window time.Duration) *PerformanceMetrics {
	metrics := &PerformanceMetrics{}

	// Track latencies
	latencies := make([]time.Duration, 0, 1000)
	ticker := time.NewTicker(window)
	defer ticker.Stop()

	for range ticker.C {
		// Collect latencies from last window
		sort.Slice(latencies, func(i, j int) bool {
			return latencies[i] < latencies[j]
		})

		if len(latencies) > 0 {
			metrics.P50Latency = latencies[len(latencies)/2]
			metrics.P95Latency = latencies[len(latencies)*95/100]
			metrics.P99Latency = latencies[len(latencies)*99/100]
		}

		// Calculate throughput
		metrics.TokensPerSecond = float64(len(latencies)) / window.Seconds()

		// Get resource utilization
		metrics.CPUUtilization = getCPUUtilization()
		metrics.GPUUtilization = getGPUUtilization()
		metrics.MemoryUsageBytes = getMemoryUsage()

		// Log metrics
		logMetrics(metrics)

		// Reset for next window
		latencies = latencies[:0]
	}

	return metrics
}
```

### Alerting Thresholds

```go
type PerformanceAlert struct {
	MetricName  string
	Threshold   float64
	CurrentValue float64
	Severity    string // "warning", "critical"
}

func CheckPerformance(metrics *PerformanceMetrics) []PerformanceAlert {
	alerts := make([]PerformanceAlert, 0)

	// Latency alerts
	if metrics.P99Latency > 500*time.Millisecond {
		alerts = append(alerts, PerformanceAlert{
			MetricName:   "P99Latency",
			Threshold:    500,
			CurrentValue: float64(metrics.P99Latency.Milliseconds()),
			Severity:     "critical",
		})
	}

	// Throughput alerts
	if metrics.TokensPerSecond < 100 {
		alerts = append(alerts, PerformanceAlert{
			MetricName:   "TokensPerSecond",
			Threshold:    100,
			CurrentValue: metrics.TokensPerSecond,
			Severity:     "warning",
		})
	}

	// GPU utilization alerts (low utilization = inefficiency)
	if metrics.GPUUtilization < 50 {
		alerts = append(alerts, PerformanceAlert{
			MetricName:   "GPUUtilization",
			Threshold:    50,
			CurrentValue: metrics.GPUUtilization,
			Severity:     "warning",
		})
	}

	// Memory alerts
	maxMemory := int64(32 * 1024 * 1024 * 1024) // 32 GB
	if metrics.MemoryUsageBytes > int64(float64(maxMemory)*0.9) {
		alerts = append(alerts, PerformanceAlert{
			MetricName:   "MemoryUsage",
			Threshold:    float64(maxMemory) * 0.9,
			CurrentValue: float64(metrics.MemoryUsageBytes),
			Severity:     "critical",
		})
	}

	return alerts
}
```

---

## Appendix: Complete Benchmark Suite

### Running the Full Suite

```bash
# Run all benchmarks with detailed output
go test -v -bench=. -benchmem -benchtime=10s \
  -cpuprofile=cpu.prof \
  -memprofile=mem.prof \
  -trace=trace.out \
  | tee benchmark_results.txt

# Generate comparison report
benchstat baseline.txt benchmark_results.txt > comparison.txt

# Profile analysis
go tool pprof -http=:8080 cpu.prof

# Trace analysis
go tool trace trace.out
```

### Benchmark Results Template

```
=== Benchmark Results ===
Date: 2025-01-15
Hardware: Intel Core i9-12900K, 32GB RAM, NVIDIA A100-40GB
Go Version: 1.21.5

--- ATTENTION BENCHMARKS ---
BenchmarkStandardAttention/B1_H8_S128_D64-16    5000    245 µs/op    0 allocs/op    ~41 GFLOPS
BenchmarkStandardAttention/B1_H8_S512_D64-16    1000   2340 µs/op    0 allocs/op    ~22 GFLOPS
BenchmarkStandardAttention/B1_H8_S1024_D64-16    200  18720 µs/op    0 allocs/op    ~11 GFLOPS

BenchmarkFlashAttention/B1_H8_S128_D64_Block32-16   10000   120 µs/op    0 allocs/op    ~84 GFLOPS
BenchmarkFlashAttention/B1_H8_S512_D64_Block64-16    2000   890 µs/op    0 allocs/op    ~58 GFLOPS
BenchmarkFlashAttention/B1_H8_S1024_D64_Block64-16    500  5200 µs/op    0 allocs/op    ~39 GFLOPS

Speedup: Flash Attention is 2.0x - 3.6x faster than standard attention

--- MATRIX MULTIPLICATION BENCHMARKS ---
BenchmarkMatMul/Naive_1024-16               50   38720 µs/op      ~5.5 GFLOPS
BenchmarkMatMul/Blocked_1024-16            200    9450 µs/op     ~22.4 GFLOPS
BenchmarkMatMul/SIMD_1024-16               500    3180 µs/op     ~66.7 GFLOPS
BenchmarkMatMul/BLAS_1024-16              1000    1640 µs/op    ~129.3 GFLOPS

Speedup: BLAS is 23.6x faster than naive implementation

--- END-TO-END MODEL BENCHMARKS ---
BenchmarkTransformer/GPT2-small/forward-16         20   52340 µs/op   468 MB memory
BenchmarkTransformer/GPT2-small/forward_backward-16  5  187920 µs/op  1872 MB memory
BenchmarkTransformer/GPT2-medium/forward-16         8  124560 µs/op  1380 MB memory

Throughput: ~153 samples/sec (forward only), ~26 samples/sec (training)

--- INFERENCE BENCHMARKS ---
BenchmarkInference/single_token-16        5000    1.89 ms/op     529 tokens/sec
BenchmarkInference/batch_8-16             1000    9.23 ms/op    4345 tokens/sec
BenchmarkInference/batch_32-16             300   28.71 ms/op    5653 tokens/sec

--- MEMORY ANALYSIS ---
Peak memory usage: 3.2 GB (training), 1.1 GB (inference)
Allocations per forward pass: 0 (using tensor pooling)
GC pauses: <5ms (negligible impact)

=== CONCLUSIONS ===
1. Flash Attention provides 2-3.6x speedup over standard attention
2. BLAS-optimized matmul is critical (23x faster than naive)
3. Batch size 32 provides best throughput (5653 tokens/sec)
4. Memory usage scales linearly with model size as expected
5. MFU: ~42% (good for CPU, GPU would achieve 50-60%)

=== RECOMMENDATIONS ===
1. Use Flash Attention for all sequences >512 tokens
2. Link against MKL or OpenBLAS for optimal matmul performance
3. Use batch size 16-32 for inference (balance latency/throughput)
4. Enable gradient checkpointing for training models >1B parameters
5. Consider mixed precision training for 2-3x speedup on GPU
```

---

## Summary

Systematic benchmarking and performance analysis are essential for developing efficient transformer models. Key takeaways:

1. **Measure First**: Always profile before optimizing
2. **Focus on Hotspots**: Optimize the top 2-3 bottlenecks (80/20 rule)
3. **Use Specialized Tools**: Go benchmarks, pprof, traces, GPU profilers
4. **Validate Improvements**: Compare before/after with statistical rigor
5. **Monitor Production**: Track latency, throughput, resource utilization
6. **Hardware Matters**: GPU vs. CPU, SIMD, cache optimization
7. **Algorithmic Wins**: Flash Attention, kernel fusion, mixed precision

**Performance Checklist:**
- ✅ Use optimized BLAS libraries (OpenBLAS, MKL, cuBLAS)
- ✅ Implement Flash Attention for long sequences
- ✅ Enable mixed precision (FP16/BF16) on GPUs
- ✅ Use gradient checkpointing for large models
- ✅ Implement KV caching for inference
- ✅ Profile and eliminate allocations in hot paths
- ✅ Batch operations for GPU efficiency
- ✅ Monitor MFU and aim for >30% (training) or >10% (inference)

With these techniques, you can achieve near-optimal performance and scale transformer models efficiently from prototypes to production.
