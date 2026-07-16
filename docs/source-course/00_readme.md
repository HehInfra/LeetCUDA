# LeetCUDA 源码课程总规划

## 0. 这份文件是什么

这是一套面向 CUDA 初学者的 LeetCUDA 源码阅读课程规划。它不是仓库原始 `README.md` 的复述，也不是把三十个算子目录依次列一遍，而是根据源码之间的知识依赖，重新安排一条可以真正走通的学习路线。

课程的核心目标是让学习者从“只能大致看懂 Python 测试脚本”，逐步走到能够解释 CUDA Kernel 的线程映射、内存访问、并行归约、矩阵分块、Tensor Core 和 FlashAttention 实现。课程既会讲必要的 C++、CUDA C++ 和 Python 语法，也会持续把这些语法放回 LeetCUDA 的真实源码中使用。

本文件只负责课程设计，不展开第一课正文。后续严格遵守一次只写一课的节奏：每完成一篇课程文件，先进行内容和格式自检，再由学习者确认是否进入下一课。

## 1. 课程面向谁

这套课程首先面向会一点 Python、了解矩阵和张量概念，但没有系统学习过 C++ 与 CUDA 的读者。即使读者只写过 PyTorch 模型，也应当能够从课程开头进入，而不需要先去完成一整套通用 C++ 教材。

课程也适合已经写过简单 CUDA Kernel，却难以阅读优化代码的学习者。LeetCUDA 的后半部分涉及向量化访存、Warp Reduce、Shared Memory、流水线、WMMA、MMA PTX、CuTe 与 FlashAttention，这些内容需要一条比“先看所有 CUDA 文档”更聚焦的路线。

课程默认读者具备基本的编程概念，例如变量、函数、循环和条件判断。如果这些概念也不熟悉，课程仍会解释源码中的具体写法，但不会把通用程序设计从零完整重讲一遍。

## 2. 学完后应该获得什么能力

完成基础阶段后，读者应当能独立阅读 `kernels/elementwise/elementwise.py` 与 `elementwise.cu`，说清 Python 如何动态编译 CUDA 扩展、Tensor 如何变成裸指针、绑定函数如何计算 Grid 与 Block，以及 GPU 上每个线程如何找到自己的元素。

完成并行协作阶段后，读者应当能解释 Warp、Block 和 Grid 的区别，理解 `__shfl_xor_sync`、`__shared__`、`__syncthreads()` 与 `atomicAdd` 在 Reduce 中分别承担什么职责，并能判断一个 Kernel 大致是内存受限还是计算受限。

完成大模型算子阶段后，读者应当能把 Softmax、LayerNorm、RMSNorm 与 RoPE 的数学公式映射到并行程序，理解为什么低精度输入常配合 FP32 累加，以及为什么融合、向量化和在线算法会影响性能与显存访问。

完成 GEMM 与 Attention 阶段后，读者应当能沿着 naive、Tile、Shared Memory、双缓冲、异步拷贝、WMMA、MMA PTX 和 Swizzle 的演进路线阅读代码，并能说明 FlashAttention 如何组合矩阵乘、在线 Softmax、分块与片上存储。

最终目标不是记住仓库中每个宏和每个模板参数，而是建立一套可迁移的源码阅读方法。以后面对其他 CUDA 项目时，读者应当知道先找入口、再画数据形状、再找线程映射、再检查存储层次，最后分析同步、精度与性能。

## 3. 本课程不追求什么

课程不会把 C++ 标准库、面向对象设计、模板元编程或构建系统讲成完整教材。只要某个知识点不是阅读 LeetCUDA 主线所必需，就不会在基础阶段扩张它的范围。

课程不会宣称仓库里的教学实现总是优于成熟库。LeetCUDA 的价值主要在于展示优化演进；生产环境中仍需要结合 cuBLAS、官方 FlashAttention、CUTLASS、Triton 或框架原生算子进行选择。

课程也不会把性能数字当作跨设备结论。仓库中的结果来自特定 GPU、CUDA、PyTorch、形状和编译参数，课程会学习测量方法与分析思路，而不是机械复述某个设备上的 TFLOPS。

## 4. 仓库在课程中的地图

```text
LeetCUDA/
├── README.md                    项目总索引、算子清单与性能展示
├── kernels/                     课程的主要源码范围
│   ├── elementwise/             第一条完整调用链
│   ├── relu/ ...                简单逐元素算子
│   ├── reduce/                  Warp 与 Block 协作的核心入口
│   ├── softmax/                 归约与数值稳定性
│   ├── mat-transpose/           二维索引与 Shared Memory
│   ├── layer-norm/              归约、广播和融合
│   ├── rms-norm/                大模型归一化算子
│   ├── rope/                    大模型位置编码算子
│   ├── sgemv/ 与 hgemv/         矩阵向量乘
│   ├── sgemm/ 与 hgemm/         矩阵乘优化主线
│   ├── flash-attn/              全课程综合专题
│   ├── openai-triton/           CUDA 与 Triton 的对照
│   ├── cutlass/                 CuTe/CUTLASS 扩展阅读
│   ├── nvidia-nsight/           性能分析练习
│   └── interview/               复习与面试型综合入口
├── others/                      分布式通信与 TensorRT 支线
├── slides/                      参考资料，不作为线性必读内容
├── HGEMM/                       可阅读的独立 Toy-HGEMM 库
├── ffpa-attn/                   可阅读的工程化 FFPA Attention 库
└── third-party/cutlass/         未初始化的 CUTLASS 子模块
```

`kernels/` 是课程主线。`slides/` 体积很大，但它更适合在需要硬件背景、PTX 指令或 CUTLASS 概念时按需查阅，不适合从第一份 PDF 开始线性阅读。

当前工作区已经能直接读取 `HGEMM/` 与 `ffpa-attn/` 的源码，尽管 Git 的 submodule 状态仍显示它们没有按标准子模块流程初始化。课程以工作区中实际可读内容为准：`HGEMM/` 用于对照 LeetCUDA 内置 HGEMM 专题与独立 Python 库的组织方式，`ffpa-attn/` 用于学习教学原型如何演化为包含正式 API、多后端、反向传播、自动调优和测试体系的工程项目。`third-party/cutlass/` 目前仍无可读源码，因此涉及 CUTLASS 内部实现时会明确外部依赖边界。

### 4.1 `HGEMM/` 与 `kernels/hgemm/` 的关系

`HGEMM/` 不是一套与 LeetCUDA 毫无关联的新算法。它从 LeetCUDA 的 HGEMM 代码独立出来，保留了 naive CUDA Core、WMMA、MMA PTX、CuTe、cuBLAS 对照、PyBind、Python benchmark 与 C++ benchmark 等结构。不过两个目录当前并非逐文件相同，构建脚本、benchmark、若干 MMA 文件和实现细节已经存在差异，主仓库还额外包含 WGMMA 路径。

因此课程不会把两份代码从头重复讲两遍。`kernels/hgemm/` 继续承担优化演进主线；`HGEMM/kernels/hgemm/` 则用于“同一专题如何打包成独立库”的对照阅读，并在关键实现有差异时做版本比较。

### 4.2 `ffpa-attn/` 与 `kernels/flash-attn/` 的关系

`kernels/flash-attn/` 更适合观察 Split KV、Split Q、Shared KV/QKV、细粒度 Tiling 和 MMA Swizzle 等教学型演进。`ffpa-attn/` 则是一套显著更工程化的 Attention 库，公开入口为 `ffpa_attn_func` 与 `ffpa_attn_varlen_func`，并通过 `CUDABackend`、`TritonBackend`、`CuTeDSLBackend` 和 `SDPABackend` 表达运行后端。

该库还包含 `torch.autograd.Function` 前后向桥接、CUDA C++ 扩展、Triton 与 CuTe DSL 实现、Ampere/Hopper 分支、GQA/MQA、causal、attention mask、dropout、decode/prefill、自动调优、Ray 并行调优、CLI、pytest 测试和生成的 Head Dimension 专用 CUDA 编译单元。它适合作为课程最后的工程化阶段，而不适合替代早期的教学 Kernel。

## 5. 一条算子在仓库中的典型运行链

LeetCUDA 的许多入门目录都遵循相似结构。理解这条结构，可以避免把 `.py` 与 `.cu` 当成两个无关程序。

```text
运行 Python 文件
    ↓
torch.utils.cpp_extension.load
    ↓
NVCC 编译 .cu 文件并生成 Python 扩展模块
    ↓
Python 调用 lib.xxx(...)
    ↓
C++ 包装函数检查 Tensor 并计算 launch 参数
    ↓
kernel<<<grid, block>>>(...)
    ↓
GPU 线程从 Tensor 指针读取数据并写回输出
    ↓
Python 同步、校验结果、统计时间
```

`kernels/elementwise` 把以上步骤都放在一个 Python 文件和一个 CUDA 文件中，适合作为第一条完整主线。`kernels/hgemm` 与 `kernels/flash-attn` 的规模更大，因此把绑定声明、多个 CUDA 实现、工具脚本和安装逻辑拆到了不同文件。

课程会多次回到这条调用链。早期重点是“它能运行起来”；中期重点是“线程如何协作”；后期重点则变成“同样的数学计算为什么有许多性能不同的实现”。

## 6. 需要掌握的语言与工具边界

### 6.1 C++ 的最小范围

课程会讲基本类型、作用域、函数、数组、指针、引用、`const`、结构体、头文件、宏、类型转换和模板的阅读方法。这些知识会直接绑定到 `float *a`、`torch::Tensor`、`reinterpret_cast`、`template <const int ...>` 与各种绑定宏。

还会讲 `static`、`inline`、`constexpr`、`__forceinline__` 等常见修饰符，但重点是它们在设备函数和编译期参数中的作用。课程不安排传统 C++ 面向对象体系的长篇铺垫，因为主线源码并不依赖复杂的类层次。

### 6.2 CUDA C++ 的最小范围

课程会覆盖 `.cu` 文件、Host/Device 区分、`__global__`、`__device__`、Kernel Launch 语法、内建索引变量、`dim3`、Shared Memory、同步、Warp Shuffle、Atomic、CUDA 数据类型与错误检查。

后续会逐步加入 `half2`、`float4`、`cp.async`、WMMA 与内联 PTX。高级语法不会在第一阶段一次讲完，而是在相关算子中首次出现时建立足够精确的模型。

### 6.3 Python 与 PyTorch 的最小范围

Python 部分重点是 import、函数、类型标注、列表推导、循环、格式化字符串、`functools.partial` 和简单的性能测试结构。所有语法都会对应到仓库中的测试脚本，不会单独设计与 CUDA 无关的小练习。

PyTorch 部分会讲 Tensor 的 shape、dtype、device、contiguous、数据创建、类型转换、输出预分配、同步和正确性比较。`torch.utils.cpp_extension.load` 是核心内容，因为它解释了 Python 如何获得 `lib.elementwise_add_f32` 这样的可调用对象。

### 6.4 PyBind 与扩展构建

课程会解释 `PYBIND11_MODULE`、`m.def`、`TORCH_EXTENSION_NAME`、`torch::Tensor` 和 `data_ptr()`。这部分是 Python 与 CUDA 之间的桥梁，不能以“框架自动完成”一笔带过。

仓库既有单文件动态编译，也有 `setup.py`、独立 `pybind/*.cc` 和 Makefile。课程会先学习最简单的 `load()` 路径，再在 HGEMM 和 FlashAttention 阶段解释大型扩展为何拆分构建单元。

### 6.5 暂不列为主线的内容

Triton 是一条对照支线，适合在掌握 CUDA 线程与内存模型之后学习。PyTorch Distributed 与 TensorRT 也有价值，但它们不属于理解核心 Kernel 的必要前置，因此不会打断主课程。

## 7. GPU 基础知识的组织方式

GPU 基础不会被写成一段与源码脱节的硬件百科。每个概念都要在一个实际 Kernel 中落地，例如用 `elementwise_add_f32_kernel` 解释线程索引，用 `block_all_reduce_sum_f32_f32_kernel` 解释 Warp 和 Shared Memory，用矩阵转置解释 Bank Conflict。

课程会先建立 Grid、Block、Warp、Thread 的执行层次，再建立寄存器、Shared Memory、L2 与 Global Memory 的存储层次。执行层次回答“谁在算”，存储层次回答“数据放在哪”，两者一起构成分析 Kernel 的基本坐标系。

性能部分会围绕内存带宽、计算吞吐、延迟隐藏、Occupancy、寄存器压力、合并访存、分支发散与算术强度展开。每个概念只讲到足以解释当前源码的程度，并在后续算子中逐步加深。

## 8. 课程总体分期

课程计划分为十个阶段。阶段之间存在依赖，但同一阶段内也会选择代表算子深入讲解，其余算子用于横向迁移，而不是对每个目录机械重复一套说明。

| 阶段 | 主题 | 主要源码 | 核心产出 |
| --- | --- | --- | --- |
| A | 项目与最小语言基础 | `elementwise.py/.cu` | 看懂跨语言调用链 |
| B | GPU 编程模型 | elementwise、Nsight 示例 | 建立执行与存储模型 |
| C | 逐元素算子 | 激活函数目录 | 掌握一线程多元素与向量化 |
| D | 并行协作算子 | reduce、softmax、transpose | 掌握 Warp、Shared Memory 与同步 |
| E | 大模型基础算子 | norm、RoPE、embedding | 从公式映射到并行实现 |
| F | GEMV/GEMM | sgemv、hgemv、sgemm | 掌握分块与数据复用 |
| G | Tensor Core | hgemm、swizzle、CuTe | 理解 WMMA、MMA 与流水线 |
| H | FlashAttention | flash-attn | 综合矩阵乘与在线 Softmax |
| I | FFPA 工程化 | `ffpa-attn` | 从教学 Kernel 走向多后端算子库 |
| J | 分析与拓展 | Nsight、Triton、interview | 建立独立阅读和验证能力 |

## 9. 阶段 A：项目与最小语言基础

### 第 01 课：从 Python 到 GPU 的第一条调用链

对应源码：`kernels/elementwise/elementwise.py`、`kernels/elementwise/elementwise.cu`。

本课先不追求优化，而是回答程序到底如何运行。学习者将从 Python 的 `load()` 开始，追踪 `lib.elementwise_add_f32`，找到 C++ 包装函数、PyBind 注册点和最终的 `elementwise_add_f32_kernel`。

语言知识包括 Python import、函数调用、Tensor 对象、C++ 函数声明、指针参数和 CUDA Kernel Launch。完成后应能手画一张从 Python 到 GPU 再回到 Python 的调用图。

### 第 02 课：读 LeetCUDA 所需的 C++ 最小语法

对应源码：仍以 `elementwise.cu` 为主，辅以 `reduce/block_all_reduce.cu`。

本课讲数组与指针、地址、`const`、作用域、循环、条件判断、函数和基本类型转换。源码中的 `float *a`、`a[idx]`、`const int N` 和 `reinterpret_cast<element_type *>` 将成为实际例子。

模板与宏只讲阅读入口。学习者需要先知道它们如何生成多个相似函数，而不是一开始掌握复杂模板元编程。

### 第 03 课：宏、模板与低精度类型的阅读方法

对应源码：`elementwise.cu` 的绑定宏、`reduce/block_all_reduce.cu` 的函数模板。

本课拆解 `#define`、字符串化、Token 拼接、条件编译、模板非类型参数和 `constexpr`。这些语法在仓库中用于批量生成 FP32、FP16、BF16、FP8 与 INT8 版本，如果跳过它们，后续会误以为代码中存在大量“看不见的函数”。

本课也建立 `float`、`half`、`half2`、`__nv_bfloat16` 和 FP8 的初步精度模型。重点是存储类型、计算类型与累加类型可以不同。

### 第 04 课：Python 测试脚本的最小知识

对应源码：`elementwise.py`、`reduce/block_all_reduce.py`。

本课讲列表推导、循环、函数参数、类型标注、可选参数、`partial`、格式化输出和基准测试函数。学习者会理解为什么脚本需要 warmup，为什么计时前后要 `torch.cuda.synchronize()`，以及输出 Tensor 为什么要提前分配。

课程还会区分“Python 调用返回时”与“GPU 真正完成时”。这是理解异步执行和正确计时的第一步。

### 第 05 课：PyTorch CUDA Extension 与构建参数

对应源码：`elementwise.py` 的 `load()`，以及 `hgemm/setup.py`、`hgemm/pybind/hgemm.cc` 的结构对照。

本课解释 sources、`extra_cuda_cflags`、`extra_cflags`、C++17、快速数学和 CUDA Half 相关宏。学习者会看到简单目录如何即时编译单个 `.cu`，大型目录又为何把 PyBind 声明、实现和安装流程拆开。

本课只建立构建心智模型，不要求记住所有 NVCC 参数。最终应能判断一次报错来自 Python、C++ 编译、CUDA 编译、链接还是运行期。

## 10. 阶段 B：GPU 编程模型

### 第 06 课：Grid、Block、Warp 与 Thread

对应源码：`elementwise_add_f32_kernel` 及其包装函数。

本课从 `blockIdx.x * blockDim.x + threadIdx.x` 出发，解释一维问题如何分给 GPU 线程。随后结合 `dim3 block(...)` 和 `dim3 grid(...)` 说明 Host 端 launch 参数如何决定 Device 端可见的内建变量。

Warp 将作为真实调度单位引入，但暂不深入 Shuffle。学习者需要区分编程层面的 Thread、协作边界 Block 和硬件调度层面的 Warp。

### 第 07 课：内存层次与一次元素加法的真实成本

对应源码：`elementwise_add_f32_kernel`、`elementwise_add_f32x4_kernel`。

本课建立寄存器、Local Memory、Shared Memory、L2 与 Global Memory 的层次。通过 `c[idx] = a[idx] + b[idx]` 分析两次读取、一次写入和一次加法，理解为什么元素加法通常是内存带宽受限。

`float4` 版本用于说明访存指令、对齐、合并访问和每线程处理多个元素。课程会明确区分“向量化访存”与“自动让四个 GPU 核心并行计算”的误解。

### 第 08 课：边界、分支与向量化尾部

对应源码：`elementwise_add_f32x4_kernel` 与 FP16 多元素版本。

本课分析 `if ((idx + 3) < N)` 和尾部循环，解释为什么向量宽度增加后边界判断也必须改变。还会讨论 Warp 内线程走不同分支时的分支发散，以及尾部路径为什么通常不是主要性能瓶颈。

学习者将练习为任意长度输入设计正确的 Grid，并判断源码在哪些形状假设下需要更严格的对齐检查。

### 第 09 课：异步执行、计时与正确性验证

对应源码：各 Python 脚本中的 `run_benchmark`。

本课区分 Kernel Launch 延迟、GPU 执行时间、端到端时间和平均迭代时间。通过 warmup、同步和重复迭代，说明一个看似简单的 Python 计时函数为什么包含多层实验设计。

正确性部分会讨论 `torch.allclose`、最大误差、绝对误差、相对误差和低精度累积。课程不会把“结果看起来差不多”当作充分验证。

### 第 10 课：性能模型与优化问题清单

对应源码：elementwise、reduce、SGEMM 的不同实现版本。

本课建立 Memory-bound、Compute-bound、Arithmetic Intensity、Occupancy 和延迟隐藏的初步模型。目的不是计算精确 Roofline，而是让学习者在看到优化代码时先问：它减少了哪种成本？

课程会形成一张通用检查表：线程映射是否合理、访问是否连续、数据是否复用、同步是否过多、寄存器是否过多、计算精度是否满足要求、基准是否公平。

## 11. 阶段 C：逐元素算子

### 第 11 课：激活函数 Kernel 的共同骨架

对应源码：`relu`、`sigmoid`、`elu`、`hardshrink`、`hardswish`、`swish`、`gelu`。

本课不逐个重复所有文件，而是抽取“读输入—计算标量函数—写输出”的共同骨架。ReLU 用于理解分支，Sigmoid 用于理解指数函数，GELU 或 Swish 用于观察计算量增加后性能特征的变化。

完成后，学习者应能快速浏览一个新激活目录，定位数学表达式、数据类型版本、向量化策略和 PyTorch 对照实现。

### 第 12 课：FP16、Half2 与 Pack 计算

对应源码：`elementwise`、`relu`、`gelu` 中的低精度版本。

本课比较 `half`、`half2` 和 128-bit Pack 的角色。`half2` 既影响访存表达，也可能调用成对算术指令；Pack 数组则常被重新解释为更宽类型完成 Load/Store。

课程会强调对齐、别名和边界条件，并解释低精度存储不意味着所有中间计算都必须使用低精度。

### 第 13 课：从重复代码理解宏生成接口

对应源码：多个激活目录中的 `TORCH_BINDING_*` 宏。

本课从一个具体宏展开结果，恢复出实际 C++ 包装函数和 PyBind 注册语句。学习者会练习在脑中区分预处理阶段、模板实例化阶段和程序运行阶段。

这一能力会直接服务于 Reduce 与 HGEMM，因为后者的函数族更大。如果看不懂生成逻辑，很难确定 Python 中某个 `lib.xxx` 最终对应哪个 Kernel。

## 12. 阶段 D：并行协作算子

### 第 14 课：点积与 Reduce 问题的诞生

对应源码：`dot-product` 与 `reduce`。

本课从数学求和说明为什么多个线程需要合并局部结果。每个线程先在寄存器里得到局部值，然后 Warp 内归约，再在 Block 内归约，最后可能通过 Atomic 合并多个 Block。

点积还展示乘法和求和的组合，为 GEMV、GEMM 与 Attention 中的累加做好准备。

### 第 15 课：Warp Shuffle 归约

对应源码：`warp_reduce_sum_f32` 及其他类型版本。

本课逐轮解释 `mask = 16, 8, 4, 2, 1` 时每个 Lane 获得什么值。重点是寄存器值如何在 Warp 内交换，而不是把 `__shfl_xor_sync` 当作黑盒求和函数。

课程也会说明 active mask、Warp 宽度和部分 Warp 的风险，并对比 Shuffle 与 Shared Memory 归约。

### 第 16 课：Block Reduce、Shared Memory 与 Atomic

对应源码：`block_all_reduce_sum_f32_f32_kernel`。

本课沿源码的两个归约层级展开：先让每个 Warp 得到一个和，由 Lane 0 写入 `reduce_smem`；同步后，再让第一个 Warp 归约这些 Warp 结果；最后由一个线程执行 `atomicAdd`。

学习者需要能说明每一次同步保护了什么依赖，也要理解 Atomic 为什么正确但可能成为争用点。

### 第 17 课：多数据类型归约与累加精度

对应源码：`reduce/block_all_reduce.cu` 的 FP16、BF16、FP8 与 INT8 实现。

本课比较输入类型、局部累加类型和输出类型。课程会用误差传播解释为什么 FP16 输入常转换到 FP32 累加，也会观察 INT8 为什么通常需要更宽整数保存和。

不同版本不是简单的接口排列，而是数值稳定性、吞吐和带宽之间的选择。

### 第 18 课：Softmax 与数值稳定性

对应源码：`softmax/softmax.cu`、`softmax/softmax.py`。

本课从 naive Softmax 的溢出风险出发，推导减去最大值的 Safe Softmax，再引入 Online Softmax。每一步数学变化都会对应到 Reduce 次数、全局内存遍历次数和片上状态。

学习者应能解释最大值归约、指数和归约、归一化三个阶段如何分配给线程，并理解算法等价不代表浮点执行结果逐位相同。

### 第 19 课：矩阵转置与 Shared Memory Bank Conflict

对应源码：`mat-transpose` 与 `nvidia-nsight/bank_conflicts.md`。

本课从二维索引、行主序地址计算和 Tile 搬运开始。Shared Memory 允许把 Global Memory 的读写都组织成较连续的访问，但转置后的访问模式可能在 Shared Memory 中产生 Bank Conflict。

课程将解释 Padding 和 Swizzle 如何改变地址映射，并安排 Nsight 指标作为证据，而不是只凭代码注释判断优化是否有效。

### 第 20 课：Histogram、Embedding 与 NMS 的不规则性

对应源码：`histogram`、`embedding`、`nms`。

这三类算子分别展示写冲突、间接索引和复杂控制流。它们不完全遵循连续数组的一线程一元素模板，因此适合用于扩展 GPU 编程模型。

课程会以主路径为主，不在 NMS 的所有工程细节上过早展开。目标是学会识别 Atomic、随机访问与分支发散带来的问题。

## 13. 阶段 E：大模型基础算子

### 第 21 课：LayerNorm 的公式、两次统计与融合

对应源码：`layer-norm/layer_norm.cu` 与 Python 测试。

本课把均值、方差、归一化和仿射变换映射到并行程序。重点分析一行数据如何交给一个 Block 或 Warp，统计量如何归约，最终结果如何广播给参与线程。

课程还会讨论 `epsilon`、FP32 累加、向量化 Load/Store，以及融合为什么能减少中间 Tensor 的 Global Memory 往返。

### 第 22 课：RMSNorm 与 LayerNorm 的源码对照

对应源码：`rms-norm/rms_norm.cu` 与 `layer-norm/layer_norm.cu`。

RMSNorm 去掉均值中心化，只保留平方均值和缩放。本课通过并排调用链和数据流分析，找出两者能够复用的 Reduce 技术以及不同的数学路径。

学习者将练习从公式直接预测 Kernel 需要的全局读取、归约变量与输出计算。

### 第 23 课：RoPE 的索引、配对与数据布局

对应源码：`rope/rope.cu` 与 `rope/rope.py`。

本课解释旋转位置编码如何把相邻或分半的维度组成二维旋转。源码阅读重点是多维 Tensor 如何展平成地址、线程如何定位 batch、head、position 和 hidden dimension。

课程会把索引公式写回真实 shape，避免只记住 `cos` 和 `sin` 的表面表达式。

### 第 24 课：Transformer 算子图中的位置

对应源码：`transformer` 目录，并回顾 embedding、norm、RoPE、softmax。

本课把此前的独立 Kernel 放回 Transformer 推理数据流，解释哪些算子常是 Memory-bound，哪些环节会进入 GEMM 或 Attention，哪些融合具有现实意义。

这是一节阶段复盘课，不会脱离仓库扩写完整的大模型理论。

## 14. 阶段 F：GEMV 与 GEMM

### 第 25 课：SGEMV——一行输出就是一次并行点积

对应源码：`sgemv/sgemv.cu` 与测试脚本。

本课把 `y = A x` 拆成多行独立点积。学习者将分析一个 Warp 或 Block 如何负责一行，线程怎样跨步读取 K 维，局部乘加结果怎样归约。

不同 K 大小的实现选择会帮助理解“固定模板参数为什么能换取更好的编译优化”。

### 第 26 课：HGEMV 与低精度吞吐

对应源码：`hgemv/hgemv.cu`、`hgemv_cute.cu`。

本课在 SGEMV 基础上考察 FP16 读取、成对处理和累加精度。CuTe 版本先作为高级布局抽象的预览，不要求立即掌握其全部类型系统。

学习者要能够区分算法没有改变、数据类型改变和布局表达方式改变这三个层次。

### 第 27 课：Naive SGEMM 与三层循环的并行化

对应源码：`sgemm/sgemm.cu` 的基础版本。

本课从 CPU 三层循环开始，把 M、N 输出坐标分给 GPU 线程，并让每个线程沿 K 维累加。Naive 实现将作为所有后续优化的基线。

课程会计算重复读取量，说明为什么大量线程会反复从 Global Memory 读取相同的 A、B 元素。

### 第 28 课：Block Tile 与 Shared Memory 数据复用

对应源码：SGEMM 的 sliced-K 与 Tile 版本。

本课把输出矩阵分成 Block Tile，把 K 维分段装入 Shared Memory。一次加载的数据由多个线程复用，从而提高 Arithmetic Intensity。

学习者需要画出 Block Tile、Thread Tile 和 K Tile，并能从下标判断每个线程加载什么、计算什么、写回什么。

### 第 29 课：Thread Tile、向量化与寄存器累加

对应源码：SGEMM 的 8×8 Thread Tile 等版本。

本课解释一个线程为什么可以负责多个输出元素，寄存器中如何保存小型累加块，以及 `float4` Load/Store 如何配合布局。

课程同时讨论寄存器复用与寄存器压力，说明优化不是让每个线程无限承担更多工作。

### 第 30 课：双缓冲、异步拷贝与软件流水线

对应源码：`sgemm_async.cu` 与 SGEMM/HGEMM 中的多阶段版本。

本课将一次 K Tile 的加载与上一次 K Tile 的计算重叠。学习者会追踪预加载、主循环、最后收尾三个阶段，理解为什么循环边界看起来比普通版本复杂。

`cp.async` 将被放到 Ampere 及之后的硬件背景中解释，并明确异步拷贝仍然需要正确的等待与可见性控制。

## 15. 阶段 G：Tensor Core 与 HGEMM

### 第 31 课：Tensor Core、WMMA 与 MMA 形状

对应源码：`hgemm/wmma` 与基础 MMA 文件。

本课解释 CUDA Core 与 Tensor Core 的区别，理解 `m16n16k16`、`m16n8k16` 等名称表示的矩阵片段。WMMA 提供较高层 API，MMA PTX 则暴露更细的寄存器和布局控制。

课程会先读最小实现，再进入多 Warp、多 Tile 版本，避免从最长函数反推全部概念。

### 第 32 课：从 WMMA 到 MMA PTX

对应源码：`hgemm/mma/basic/hgemm_mma.cu` 等基础文件。

本课学习内联 PTX 的最小阅读方法：输入输出操作数、约束、寄存器片段和指令形状。目标不是立即手写所有 PTX，而是能把一条 MMA 指令放回外层 Tile 循环。

课程会反复区分单条 MMA、一个 Warp 的累加块和整个 Thread Block 的输出 Tile。

### 第 33 课：多级 Tiling 与 Warp 布局

对应源码：HGEMM 的 `mma2x4`、`warp4x4` 等版本。

本课解释命名中 MMA Tile、Warp Tile 和 Block Tile 的层次。学习者将用矩阵图标注 Warp 负责的区域，并追踪输入片段如何被多个 MMA 指令消费。

这一步是理解源码中大量编译期常量和地址公式的关键。

### 第 34 课：Shared Memory Swizzle 与布局代数

对应源码：`hgemm/mma/swizzle`、`swizzle` 目录及工具脚本。

本课从 Bank Conflict 复习开始，解释 XOR 型 Swizzle 如何改变逻辑坐标到物理地址的映射。Python 打印布局工具会帮助把抽象公式变成可观察的二维表。

课程不会把 Swizzle 描述为神秘的性能开关，而会逐项验证访问线程、Bank 编号和向量对齐。

### 第 35 课：多阶段 HGEMM 与接近 cuBLAS 的路径

对应源码：HGEMM stage、double buffer、collective store 与 benchmark 文件。

本课把此前的 Tile、异步拷贝、Tensor Core、Swizzle 和寄存器复用组合起来。重点是解释优化版本相对基线增加了什么状态、消除了什么等待，以及付出了什么资源代价。

仓库性能声明会作为实验对象。学习者要检查矩阵形状、布局、精度、GPU 型号、warmup 和基准函数，而不是只记住百分比。

本课还会对照 `HGEMM/kernels/hgemm` 与 `kernels/hgemm`，检查同名 Kernel、PyBind 列表、构建源文件选择和 benchmark 参数的差异。独立 HGEMM 库的 Python wheel 与 C++ 二进制测试会帮助区分 Kernel 性能、扩展调用开销和打包形态。

### 第 36 课：CuTe、CUTLASS 与 WGMMA 导读

对应源码：`kernels/cutlass`、HGEMM CuTe 和 WGMMA 文件。

本课建立 Layout、Shape、Stride、Atom 与 Tiled MMA 的最低阅读框架。由于 `third-party/cutlass` 当前未初始化，课程会清楚区分现有示例代码与外部依赖源码。

WGMMA 与 Hopper 专属路径作为拓展，不会替代前面的 MMA 主线。

## 16. 阶段 H：FlashAttention 综合专题

### 第 37 课：从普通 Attention 到 IO 瓶颈

对应源码：`flash-attn/flash_attn_mma.py` 的参考与测试部分。

本课复习 `QK^T`、缩放、Softmax 和 `PV`，计算中间注意力矩阵的大小。重点是理解 FlashAttention 优化的是精确 Attention 的 IO 路径，而不是把算法简单替换成近似计算。

学习者会把前面的 GEMM、Reduce 和 Softmax 知识汇合成完整数据流。

### 第 38 课：Online Softmax 状态如何跨 Tile 合并

对应源码：FlashAttention 基础实现中的行最大值、行和与输出累加状态。

本课推导当新的 K/V Tile 到来时，旧最大值、旧归一化因子和旧输出怎样重新缩放。只有真正理解这一步，才能读懂 FlashAttention 主循环中的状态更新。

课程会特别讨论指数、缩放与累加精度。

### 第 39 课：Split KV 与 Split Q

对应源码：`flash_attn_mma_split_kv.cu`、`flash_attn_mma_split_q.cu`。

本课比较两种 Warp 工作划分。重点分析 Q、K、V 哪些数据被共享，哪些中间结果需要跨 Warp 通信，以及布局选择怎样影响 Shared Memory 与同步开销。

学习者将用统一的 Br、Bc、D 符号画出两种策略，而不是依赖函数名猜性能。

### 第 40 课：Shared KV 与 Shared QKV

对应源码：`flash_attn_mma_share_kv.cu`、`flash_attn_mma_share_qkv.cu`。

本课分析复用同一片 Shared Memory 的生命周期。共享存储可以降低占用并改善 Occupancy，但要求严格区分每个阶段何时不再使用旧数据。

课程会追踪 Q 从 Shared Memory 预取到寄存器、K/V 从 Global Memory 进入 Shared Memory 的路径。

### 第 41 课：QK 与 QKV 细粒度 Tiling

对应源码：`flash_attn_mma_tiling_qk.cu`、`flash_attn_mma_tiling_qkv.cu`。

本课解释为什么更细的 MMA 级 Tiling 能改变 Shared Memory 复杂度，并支持更大的 Head Dimension。学习者将区分算法矩阵形状、Block Tile 和 MMA Atom 三个尺度。

课程也会讨论更复杂索引与更少片上存储之间的工程权衡。

### 第 42 课：FlashAttention Swizzle、流水线与输出写回

对应源码：`flash-attn/mma/swizzle`、`mma/others` 和工具脚本。

本课汇总 Q/K/V 的 Swizzle 组合、两阶段预取、Collective Store 与寄存器复用。不同变体会按一个基线逐项比较，避免一次面对所有宏和模板参数。

完成后，学习者应能选择一个变体，画出完整的 Global→Shared→Register→MMA→Global 数据路线。

## 17. 阶段 I：从教学 FlashAttention 到工程化 FFPA

### 第 43 课：FFPA 的包结构、公开 API 与输入契约

对应源码：`ffpa-attn/src/ffpa_attn/__init__.py`、`ffpa_attn_interface.py`、`functional.py`、`pyproject.toml` 与 `setup.py`。

本课先从用户真正导入的 `ffpa_attn_func` 与 `ffpa_attn_varlen_func` 出发，梳理公开 API、普通 Dense Attention 与 Varlen Attention 的区别。学习者会检查 Q/K/V 的 BHND 布局、dtype、device、Head Dimension、GQA/MQA、causal、attention mask、dropout 和 scale 等输入契约。

语言部分会补充 Python 包的 `src/` 布局、`__all__`、dataclass、类型联合、关键字参数和异常路径。课程会把“API 层负责什么”与“Kernel 层负责什么”明确分开，避免直接从公开函数跳进最长的 CUDA 模板。

### 第 44 课：多后端调度与回退策略

对应源码：`ffpa-attn/src/ffpa_attn/functional.py`、`cuda/_ffpa_fwd.py`、`triton/_ffpa_fwd.py` 与 `cute/_ffpa_fwd_sm80.py`、`_ffpa_fwd_sm90.py`。

本课沿 `Backend`、`CUDABackend`、`TritonBackend`、`CuTeDSLBackend`、`SDPABackend` 和 `FFPAAttnMeta` 追踪后端选择。重点分析显式 `forward_backend`/`backward_backend`、统一 `backend` 简写、默认 Triton 路径，以及不支持形状或功能时如何回退到 ATen SDPA。

学习者会把运行时分派画成决策树，并区分硬件能力、Head Dimension、序列长度、mask、dropout 与后端功能边界。这个调度层展示了工程库为什么不能只保留一个“最快 Kernel”。

### 第 45 课：Autograd 前向、反向与保存状态

对应源码：`functional.py` 中的 `_FFPAAttnFunc`、`FFPAAttnFunc`，以及 `aten`、`triton`、`cute` 下的 backward 实现与 `tests/test_ffpa_bwd.py`。

本课讲 `torch.autograd.Function` 的 `forward`/`backward` 契约，分析前向需要保存哪些 Tensor、Softmax LSE、随机数状态或元数据，反向如何得到 dQ、dK 与 dV。课程还会解释为什么前向后端与反向后端可以分别选择，以及 CuTe DSL 路径为何要求成对配置。

这部分把课程从推理型 Kernel 扩展到训练型算子。学习者需要理解数学梯度、PyTorch Autograd 接口与底层后端调度是三个不同层次。

### 第 46 课：CUDA 扩展 API、模板 Launch 与生成式实例化

对应源码：`ffpa-attn/csrc/cuffpa/ffpa_attn_api.cc`、`launch_templates.cuh`、`ffpa_attn_fwd.cuh`、`prefill.cuh` 与 `csrc/cuffpa/generated/`。

本课从 `PYBIND11_MODULE` 注册的 `ffpa_attn_forward` 开始，追踪 dtype/acc 分派、输入检查、Head Dimension 专用入口、动态 Shared Memory 属性和最终 Kernel Launch。生成目录中的大量 `.cu` 文件将用于解释模板实例化为什么要从一个巨大编译单元拆开。

核心 Kernel 在教学版基础上增加了非整除 Nq/Nkv、GQA/MQA、causal、mask、dropout、decode/prefill 与多种 Head Dimension。课程会选择一条普通 prefill 主路径深入阅读，再把其余功能作为受控分支，避免把罕见路径误当主流程。

### 第 47 课：自动调优、配置持久化与测试体系

对应源码：`autotune.py`、`triton/_persistent_autotune.py`、`triton/configs/*.json`、`ray/`、`cli/` 与 `tests/`。

本课解释一个高性能库如何为不同 GPU、dtype、序列长度、Head Dimension 和前后向方向选择配置。学习者会追踪 TuneTask、候选配置、计时、结果记录、JSON 持久化和 Ray 多 GPU 并行搜索，而不是把 autotune 理解为单个装饰器。

测试部分会区分正确性参数化、dispatch smoke test、非法输入、后端回退、编译测试、Autograd backward 和性能测试。课程最终要求学习者为新增后端或新 Head Dimension 设计最小但完整的验证矩阵。

## 18. 阶段 J：分析、对照与独立阅读

### 第 48 课：用 Nsight 验证代码假设

对应源码：`nvidia-nsight` 目录及各 benchmark。

本课把带宽、Occupancy、Warp Stall、Bank Conflict 和 Tensor Core 利用率连接到工具指标。课程重点是从一个问题出发选择指标，而不是浏览所有报告页面。

学习者将为 Elementwise、Reduce 和 GEMM 各设计一个可验证的性能假设。

### 第 49 课：Triton 与 CUDA 的编程模型对照

对应源码：`openai-triton/vector-add`、`fused-softmax`、`merge-attn-states`。

本课用相同算子比较 CUDA Thread/Block 和 Triton Program/Block 的表达方式。`tl.arange`、mask、`tl.load`、`tl.store` 和 JIT 元参数将与已掌握的 CUDA 概念一一对应。

Triton 不被描述为“无需理解 GPU”，而是另一种表达和自动优化层次。

### 第 50 课：面试综合文件与全课程复盘

对应源码：`interview/notes-v2.cu` 与 `notes-v2.pdf`。

本课用综合验证文件回顾 BlockReduce、Elementwise、Histogram、Softmax、Norm、RoPE、GEMV、GEMM 与 FlashAttention。学习者将练习从一个大型文件中快速划分模块和依赖。

最终会形成个人源码阅读清单、性能实验模板和后续专题路线。

## 19. 为什么采用这个顺序

Elementwise 被放在最前面，是因为它已经包含 Python、PyTorch Extension、C++ 包装、Kernel Launch、线程索引、低精度和向量化，却不需要线程协作。它以最低的算法负担展示了最完整的工程路径。

Reduce 与 Softmax 位于 GEMM 之前，是因为它们先解决 Warp 通信、Shared Memory 和同步。没有这些基础，LayerNorm 和 FlashAttention 中的归约部分很容易沦为照抄模板。

GEMV 位于 GEMM 之前，是因为矩阵向量乘保留了点积和归约，却暂时避免二维输出 Tile 的复杂性。SGEMM 再逐步引入 Block Tile、Thread Tile、数据复用和流水线。

HGEMM 与 Tensor Core 放在 SGEMM 之后，是为了确保学习者先理解“为什么要分块”，再学习硬件提供的矩阵乘原语。否则 WMMA 和 MMA 的形状容易被误解成孤立 API。

FlashAttention 最后出现，是因为它同时依赖 GEMM、Softmax、在线归约、低精度、Shared Memory、流水线与 Tensor Core。它不是一个适合从函数第一行硬啃的入门目录。

`ffpa-attn` 被安排在教学版 FlashAttention 之后，是因为它引入的问题已经超出单个 Kernel：公开 API 必须稳定，多后端要按能力分派，前向与反向要接入 Autograd，编译时间需要通过生成式实例化控制，性能配置还需要自动搜索和持久化。先理解算法原型，再阅读工程约束，能够把复杂度拆成两层。

## 20. 每篇课程文件的写作约定

每课都会在开头说明本课解决的问题、前置知识和对应源码范围。正文先补充本课真正需要的语法或数学，再沿源码主路径讲运行顺序，不会先堆一章与源码无关的通用理论。

重要结论会尽量指向实际文件、函数、宏、变量或执行阶段。遇到同一算子的多个版本时，先选择最小基线，再逐项增加优化，不把所有版本混在一张巨大代码表中。

课程中的图示主要服务于调用链、线程布局、内存路线、数据形状和状态更新。代码片段会控制长度，保留理解逻辑所需的上下文，同时明确原始文件位置。

每课末尾会包含总结、少量高质量练习、参考分析和下一课预告。练习用于检查是否能迁移知识，不通过重复问定义来凑篇幅。

## 21. 一次只写一课的确认流程

第一步，写作前重新阅读本课对应的源码、README、测试入口和必要的外部调用。若源码已变化，以当前工作区内容为准。

第二步，只创建或修改当前课程文件。即使下一课依赖清晰，也不会在同一轮顺手生成。

第三步，完成后检查行数、语言、段落、源码引用、开头与结尾。若发现主体是一句话一段、泛泛而谈或缺少调用链，会先重写再交付。

第四步，向学习者报告课程文件路径和检查结果，并等待意见。学习者可以要求调整难度、增加图示、替换示例或重新安排后续课程。

第五步，只有学习者确认继续，才开始下一课。

## 22. 建议的学习方法

阅读每课前，先打开对应的 Python 与 CUDA 文件，不要求立即理解。课程讲解后，再从 Python 入口沿函数名重新走一遍，检查自己能否说出每一层的输入和输出。

对 Kernel 不要只逐行翻译。建议固定回答六个问题：输入输出形状是什么、一个线程负责什么、一个 Block 负责什么、数据经过哪些存储层、线程在哪里同步、优化版本减少了什么成本。

遇到复杂地址公式时，先用很小的形状手算。例如把 N 设为 10、Block 设为 4，列出每个线程的 idx；学习 Tile 时，用 8×8 矩阵代替真实大矩阵画图。

性能实验应同时保留正确性基线。任何更快的结果都要先确认形状、dtype、布局、累加精度和同步条件一致，再讨论优化收益。

## 23. 环境与可复现性约定

课程会区分阅读任务和运行任务。没有 NVIDIA GPU 时仍可完成大部分源码阅读、索引推导和调用链分析，但无法诚实验证 CUDA 编译、Kernel 正确性和性能。

运行实验时需要记录 GPU 型号、Compute Capability、CUDA Toolkit、驱动、PyTorch 版本和编译参数。仓库中的部分代码针对 Ampere、Ada 或 Hopper，不能假设所有设备都支持 FP8、`cp.async` 或 WGMMA。

动态扩展编译可能把产物写入 PyTorch 缓存，而不是当前目录。课程在排查构建问题时会明确编译阶段和缓存位置，但不会把生成物作为课程源码提交。

## 24. 课程中的正确性标准

逐元素 FP32 算子可以期待较严格的误差，但 Reduce、Softmax、GEMM 与 Attention 会受到计算顺序和低精度累加影响。课程会根据 dtype 与算法选择容差，不使用一套固定阈值覆盖所有场景。

正确性至少要覆盖常见形状、非整除边界、不同 dtype 和必要的特殊值。对于依赖向量对齐的实现，还要确认源码是否提供尾部路径，或者显式要求输入尺寸整除。

性能优化不能以越界访问、未同步数据或过度放宽误差为代价。出现偶发错误时，应优先检查竞争条件、同步和生命周期，而不是简单增加容差。

## 25. 课程中的性能标准

性能比较需要 warmup、足够迭代、显式同步和相同输入条件。对于极短 Kernel，还应意识到 Python 循环与 Launch 开销可能影响端到端测量。

带宽型算子适合报告有效 GB/s，矩阵乘适合报告 FLOPS/TFLOPS，但公式必须对应实际运算量。只报告毫秒数往往不足以跨形状比较。

与 PyTorch、cuBLAS 或 Triton 对比时，要确认布局、精度模式、Tensor Core 允许情况和输出分配是否一致。课程会把“基准是否公平”作为每个高级专题的固定检查项。

## 26. 常见误区预警

第一，CUDA Thread 不是 CPU Thread 的小号版本。GPU 依靠大量线程和 Warp 调度隐藏延迟，单个线程的独立能力不是理解性能的中心。

第二，Shared Memory 更快不代表应该把所有数据都放进去。容量、同步、Bank Conflict 与 Occupancy 都会限制收益。

第三，向量化不等于改变数学公式。它主要改变每个线程的访问宽度和指令组织，同时带来对齐与边界要求。

第四，FP16 输入不代表必须 FP16 累加。很多数值敏感算子会用 FP32 保存中间统计量。

第五，更多 Warp、更多 Stage 或更大 Tile 不保证更快。它们可能增加寄存器和 Shared Memory 使用，降低可驻留 Block 数。

第六，代码里出现 Tensor Core 指令不意味着整个 Kernel 已达到高利用率。数据搬运、同步、布局和尾部处理仍可能是瓶颈。

第七，FlashAttention 不是单个矩阵乘 Kernel 的别名。它的关键是分块执行 Attention，并在线维护 Softmax 与输出状态以减少显存流量。

## 27. 支线内容的处理

`others/pytorch/distributed` 会在主课程完成后作为多 GPU 通信支线。它涉及 Process Group、Rank、Collective 与 NCCL，和单 GPU Kernel 的线程协作属于不同层次。

`others/tensorrt` 可以作为部署与图优化支线。它不会成为理解 `kernels/` 的前置要求。

`slides/` 中的 CUDA Guide、PTX ISA、架构白皮书、CUTLASS 和 NCCL 材料会按需引用。课程不会要求学习者按文件名顺序通读约 250 MB 的资料。

## 28. 阶段性验收点

完成第 05 课后，学习者应能解释一个 `.py` 和一个 `.cu` 如何组成可调用扩展，并能定位常见构建错误属于哪一层。

完成第 10 课后，学习者应能画出 Grid/Block/Warp/Thread 与存储层次，并能初步判断 Elementwise 的主要瓶颈。

完成第 20 课后，学习者应能独立讲解 Warp Reduce、Block Reduce、Safe Softmax 和矩阵转置中的同步与地址变化。

完成第 24 课后，学习者应能把常见大模型算子的数学步骤映射到线程协作和数据访问。

完成第 30 课后，学习者应能从 Naive GEMM 推导出 Shared Memory Tile，并解释双缓冲的主循环结构。

完成第 36 课后，学习者应能区分 CUDA Core、WMMA、MMA PTX、CuTe 与 WGMMA 的抽象层次。

完成第 42 课后，学习者应能完整说明一个 FlashAttention 变体的数据流、在线状态和片上存储策略。

完成第 47 课后，学习者应能从 `ffpa_attn_func` 追到选定后端，解释 Autograd 前后向、CUDA 模板实例化、自动调优和回退策略，并能区分教学原型与工程库的设计目标。

完成第 50 课后，学习者应具备独立选择源码入口、建立基线、验证正确性和分析性能的能力。

## 29. 第一篇正文将如何开始

下一篇课程文件计划为 `01_python_to_cuda_call_chain.md`。它只聚焦 `kernels/elementwise`，从运行 `elementwise.py` 开始，直到 GPU 执行 `elementwise_add_f32_kernel` 并由 Python 打印结果。

该课会同时讲最小 Python 调用语法、C++ 函数和指针、CUDA Kernel Launch 与 PyBind 注册，但不会提前深入 Reduce、Shared Memory 或 Tensor Core。这样可以先得到一条完整、可验证的主路径。

第一课完成后会立即停下，由学习者检查讲解速度、语言密度、图示方式和源码引用是否合适，再决定第二课是否按当前规划继续。

## 30. 本规划的结论

LeetCUDA 的合理学习顺序不是按目录字母排列，也不是先读完一本 C++、一本 CUDA 和一本 GPU 架构教材。更有效的路线是先掌握读懂项目所需的最小语言，通过 Elementwise 建立跨语言调用链，再用 Reduce、Softmax 和转置建立线程协作与存储模型。

在这些基础之上，Norm 与 RoPE 用来连接大模型领域知识，GEMV/GEMM 用来建立分块、复用和流水线，Tensor Core 用来进入硬件矩阵原语，教学版 FlashAttention 完成算法综合，`ffpa-attn` 再展示多后端、Autograd、生成式编译与自动调优的工程化落地。Nsight 与 Triton 最后帮助学习者从“能读懂”走向“能验证、能比较、能迁移”。

本规划将作为后续所有课程文件的路线基准。如果实际源码阅读发现某一课范围过大，会拆分课程；如果两个算子高度重复，会选择代表实现深入讲解，并将其余版本作为迁移练习。任何调整都应保持“最小语言知识服务于真实源码主线”的原则。
