# 第 01 课：从 Python 到 GPU 的第一条调用链

## 1. 本课要解决的问题

第一次打开 LeetCUDA，最容易产生的困惑不是“加法怎么算”，而是“这些代码到底从哪里开始运行”。`elementwise.py` 是 Python，`elementwise.cu` 同时包含 C++、CUDA C++、宏和 PyBind；运行命令却只有一句 `python3 elementwise.py`。如果没有一张调用链地图，读者很容易在两个文件之间来回跳，却不知道每一层何时执行。

本课只解决一条最小而完整的路径：Python 调用 `lib.elementwise_add_f32` 后，程序如何进入 C++ 包装函数，如何启动 `elementwise_add_f32_kernel`，GPU 线程如何完成 `c[idx] = a[idx] + b[idx]`，最后 Python 又如何同步并打印结果。

我们暂时不深入 Shared Memory、Warp Shuffle、Tensor Core、MMA PTX 或 FlashAttention。第一课的目标不是展示最多的 CUDA 技巧，而是建立一个以后可以反复复用的跨语言执行模型。

## 2. 对应源码

本课只需要三个文件：

- `kernels/elementwise/README.md`
- `kernels/elementwise/elementwise.py`
- `kernels/elementwise/elementwise.cu`

其中，README 给出目录功能、运行命令和历史性能输出；Python 文件负责构建扩展、准备 Tensor、调用算子和计时；CUDA 文件负责 Kernel、Host 包装函数和 Python 绑定。

建议阅读时同时打开 `elementwise.py` 与 `elementwise.cu`。不要先把一个文件从头背到尾，而要沿着函数名跨文件追踪。

## 3. 学习目标

完成本课后，你应当能够回答以下问题：

1. 为什么运行的是 Python 文件，却会编译 `.cu` 文件？
2. `lib` 是什么，`lib.elementwise_add_f32` 从哪里出现？
3. Python 的 `torch.Tensor` 怎样交给 C++？
4. C++ 包装函数与 CUDA Kernel 为什么不是同一个函数？
5. `<<<grid, block>>>` 表示什么？
6. GPU 上的一个线程怎样得到自己的 `idx`？
7. 为什么 Kernel Launch 后通常不会立即等待 GPU？
8. `torch.cuda.synchronize()` 为什么对计时很重要？
9. 为什么输出 Tensor `c` 要在 Python 中提前创建？
10. FP32 标量版本与 FP32x4、FP16 版本共享了哪条主调用链？

如果你能不看讲义，画出“Python → 扩展模块 → C++ 包装函数 → CUDA Kernel → 输出 Tensor → Python”的流程，本课的核心目标就达成了。

## 4. 先看全局，不急着看语法

这个示例的数学任务极其简单：

```text
给定两个形状相同的 Tensor a 和 b
输出 Tensor c
对每个合法下标 i：c[i] = a[i] + b[i]
```

Python 测试中使用二维 Tensor，形状记为 `(S, K)`。但最基础的 CUDA Kernel 并不关心二维坐标，它把整个 Tensor 看成长度为 `N = S * K` 的连续一维数组。

完整执行链可以先概括为：

```text
python3 elementwise.py
        │
        ├─ import torch
        │
        ├─ load(name="elementwise_lib", sources=["elementwise.cu"])
        │      │
        │      ├─ 调用 C++/CUDA 构建工具
        │      ├─ 编译 elementwise.cu
        │      ├─ 生成 Python 可加载扩展
        │      └─ 返回模块对象 lib
        │
        ├─ 创建 CUDA Tensor a、b、c
        │
        ├─ lib.elementwise_add_f32(a, b, c)
        │      │
        │      ├─ 进入 C++ 包装函数 elementwise_add_f32
        │      ├─ 检查 dtype
        │      ├─ 计算 N、grid、block
        │      ├─ 取得 a、b、c 的设备数据指针
        │      └─ 启动 elementwise_add_f32_kernel<<<grid, block>>>
        │               │
        │               ├─ 每个 GPU 线程计算 idx
        │               ├─ 检查 idx < N
        │               └─ 执行 c[idx] = a[idx] + b[idx]
        │
        ├─ torch.cuda.synchronize()
        └─ 将结果的一小部分复制到 CPU 后打印
```

后续所有细节都应当能放回这张图。如果某个语法暂时无法理解，先问它属于构建、绑定、Host 包装、Kernel 还是测试层。

## 5. 三种语言角色不要混在一起

`elementwise.py` 使用 Python。它适合组织实验、构造输入、调用 PyTorch、重复运行和打印结果。Python 本身不负责执行 `c[idx] = a[idx] + b[idx]` 这一版自定义 GPU 计算。

`elementwise.cu` 中既有普通 C++，也有 CUDA C++。普通 C++ 部分运行在 CPU，也称 Host 端；带 `__global__` 的 Kernel 运行在 GPU，也称 Device 端。

PyBind 是连接 Python 与 C++ 的绑定工具。它让编译后的 C++ 函数表现得像 Python 模块中的函数，所以 Python 才能写出 `lib.elementwise_add_f32(...)`。

可以把职责概括为：

| 层次 | 主要代码 | 运行位置 | 职责 |
| --- | --- | --- | --- |
| Python 测试层 | `elementwise.py` | CPU 上的 Python 解释器 | 构建、准备输入、调用、计时、打印 |
| C++ 绑定层 | `PYBIND11_MODULE` | CPU | 将函数注册到 Python 模块 |
| C++ 包装层 | `elementwise_add_f32` | CPU | 检查 Tensor、计算 Launch 参数、取指针 |
| CUDA Kernel | `elementwise_add_f32_kernel` | GPU | 逐元素读取、相加、写回 |

“CPU 上运行的代码”不等于“只能处理 CPU Tensor”。Host 端包装函数可以持有 CUDA Tensor 对象，并取得指向 GPU 显存的数据指针，再把这些指针作为 Kernel 参数传给 GPU。

## 6. 从运行命令开始

README 给出的主要运行方式是：

```bash
export TORCH_CUDA_ARCH_LIST=Ada
python3 elementwise.py
```

这条命令应当在 `kernels/elementwise` 目录中执行，因为 Python 中写的是相对路径：

```python
sources=["elementwise.cu"]
```

相对路径通常相对于当前工作目录解析。如果从仓库根目录直接运行 `python3 kernels/elementwise/elementwise.py`，当前源码写法可能找不到 `elementwise.cu`。这是阅读构建脚本时必须留意的运行上下文。

`TORCH_CUDA_ARCH_LIST=Ada` 用于限制 PyTorch 扩展编译的目标 GPU 架构。README 的注释说明，如果不指定，可能为多个架构编译，耗时更长。它不改变 Python 代码的算法，只影响编译产物包含哪些 GPU 目标代码。

本课不实际运行编译，因为运行需要合适的 NVIDIA GPU、CUDA Toolkit、驱动与 PyTorch CUDA 环境。阅读课程时要区分“源码调用链已确认”和“当前机器上已成功执行”。

## 7. Python 文件的导入部分

文件开头是：

```python
import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load
```

`import time` 用于取得计时前后的时间戳。这里使用的是 Python 的墙钟时间，因此必须配合 GPU 同步才能测量 GPU 实际完成一批工作的时间。

`partial` 可以预先固定一个函数的部分参数。后面会看到 `partial(torch.add, out=c)`，它把 PyTorch 的 `out` 参数固定为 `c`，从而生成一个调用形式类似 `perf_func(a, b)` 的新可调用对象。

`Optional[torch.Tensor]` 是类型标注，表示参数可以是 `torch.Tensor`，也可以是 `None`。Python 运行时不会仅凭这个标注自动检查类型，它主要帮助读者和工具理解接口。

`torch` 是 PyTorch 包，而 `load` 是 PyTorch 提供的 C++ Extension 构建入口。第一课最关键的导入就是这一行：

```python
from torch.utils.cpp_extension import load
```

## 8. 为什么关闭自动求导

源码接着写：

```python
torch.set_grad_enabled(False)
```

这个 Elementwise 示例只做前向数值与性能测试，不需要 PyTorch 构建反向传播图。关闭梯度可以避免测试代码引入不必要的 Autograd 状态。

这并不会改变自定义 CUDA Kernel 的实现。Kernel 本来就只接受 `a`、`b`、`c`，没有定义 backward；这里是在 Python 全局上下文中说明“本脚本只关心前向”。

后续学习 `ffpa-attn` 时会看到正式的 `torch.autograd.Function` 前向与反向接口。当前示例刻意保持简单。

## 9. `load()`：跨语言调用链的起点

源码调用：

```python
lib = load(
    name="elementwise_lib",
    sources=["elementwise.cu"],
    extra_cuda_cflags=[...],
    extra_cflags=["-std=c++17"],
)
```

`load()` 做的事情远多于普通 Python import。它需要为 C++/CUDA 扩展准备构建描述，调用编译器，将源码编译和链接成动态库，再把动态库加载进当前 Python 进程。

返回值赋给 `lib`。因此 `lib` 不是 `elementwise.cu` 的文本内容，也不是某个 Tensor；它是已经加载的扩展模块对象。

`name="elementwise_lib"` 给扩展指定构建与加载名称。名称还会参与缓存和模块标识，但它不等于之后暴露的函数名。

`sources=["elementwise.cu"]` 告诉构建系统，本扩展只有这一个显式源码文件。虽然文件后缀是 `.cu`，它同时包含 Host C++ 和 Device CUDA 代码，NVCC 会协调两部分编译。

## 10. 编译参数先认识，不要求背诵

CUDA 编译参数中首先出现：

```text
-O3
```

它请求较高级别优化。性能测试项目通常会开启优化，否则测到的可能是调试或未优化代码表现。

随后几项 `-U__CUDA_NO_HALF_*` 用于取消 CUDA 头文件中可能禁用 Half 类型运算、转换或 `half2` 运算符的宏。它们主要服务于文件后半部分的 FP16 版本。

```text
--expt-relaxed-constexpr
--expt-extended-lambda
```

这两项启用 CUDA 的扩展 constexpr 和 Lambda 支持。当前最小 FP32 Kernel 并不依赖复杂 Lambda，但仓库采用了一套较统一的编译参数。

```text
--use_fast_math
```

它允许编译器使用速度更快、精度或 IEEE 行为可能不同的数学实现。单纯 FP32 加法看不出大部分影响，但在 Sigmoid、GELU、Softmax 等涉及指数或除法的算子中需要认真评估。

```python
extra_cflags=["-std=c++17"]
```

它要求 Host C++ 部分按 C++17 标准编译。现阶段只要知道它决定可使用的语言版本，不需要先学习 C++17 的全部特性。

## 11. `load()` 在什么时候执行

Python 文件的顶层语句会在脚本加载时按顺序执行。`lib = load(...)` 不在某个函数内部，因此运行脚本后，在进入 benchmark 循环前就会发生扩展构建和加载。

第一次运行时通常会看到明显的编译等待。后续运行如果源码、参数和环境没有变化，PyTorch Extension 可能复用缓存，速度会快很多。

所以“第一次运行耗时很长”不代表 Kernel 本身很慢。构建时间和 Kernel 执行时间必须分开讨论，README 中微秒或毫秒级的结果只针对编译完成后的重复执行。

## 12. C++/CUDA 文件的头文件

`elementwise.cu` 开头包含：

```cpp
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>
```

`#include` 是 C/C++ 预处理指令，可以把头文件提供的声明引入当前编译单元。`cuda_runtime.h` 提供 CUDA Runtime 相关声明，FP16/BF16/FP8 头文件提供低精度类型与操作。

`torch/extension.h` 与 `torch/types.h` 让 C++ 代码能够使用 `torch::Tensor` 和扩展绑定能力。没有这些声明，编译器不知道 `torch::Tensor` 是什么。

文件还包含 `<algorithm>`、`<vector>`、`<stdio.h>` 等头文件，其中并非每一项都在最小 FP32 路径中使用。阅读源码时不必因为看到一个头文件就假设主路径一定依赖它。

## 13. 暂时跳过一组宏

头文件之后定义了：

```cpp
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
```

这些宏用于向量化与重新解释数据。它们在 FP32 标量主路径中没有被使用，所以第一次阅读时可以先做标记，然后继续向下寻找 `elementwise_add_f32_kernel`。

这是源码阅读的重要技巧：不是从第一行开始平均分配注意力，而是先确定本次要追踪的函数，只阅读它直接依赖的代码。

宏、`reinterpret_cast`、对齐与别名会在后续课程详细拆解。第一课只需要知道 FP32x4 等优化版本会用到它们。

## 14. 第一个 CUDA Kernel

核心代码只有几行：

```cpp
__global__ void elementwise_add_f32_kernel(
    float *a, float *b, float *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = a[idx] + b[idx];
}
```

`void` 表示这个函数不通过普通返回值返回结果。输出会直接写入 `c` 指向的显存。

`float *a` 表示 `a` 是一个指向 `float` 数据的指针。这里的地址指向 GPU 显存，因为它来自 CUDA Tensor 的 `data_ptr()`。

`int N` 是元素总数。Kernel 不需要知道原始 Tensor 是 `(S, K)`，只需要知道合法线性下标范围是 `[0, N)`。

## 15. `__global__` 的意义

`__global__` 是 CUDA C++ 的函数修饰符。它说明这个函数是一个 Kernel：由 Host 端发起调用，在 Device 端由大量 GPU 线程执行。

普通 C++ 函数调用写成：

```cpp
func(arg1, arg2);
```

CUDA Kernel Launch 写成：

```cpp
kernel<<<grid, block>>>(arg1, arg2);
```

三尖括号中的配置不是普通函数参数。它描述要启动多少个 Thread Block，以及每个 Block 中有多少线程。

Kernel 中不能假设只有一次函数执行。相同函数体会由许多 GPU 线程执行，每个线程通过内建索引判断自己负责哪一份数据。

## 16. `idx`：线程与数据的连接点

代码：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

`threadIdx.x` 是线程在线程块内部的 x 坐标。若一个 Block 有 256 个线程，它通常取 0 到 255。

`blockIdx.x` 是当前 Block 在 Grid 中的 x 坐标。`blockDim.x` 是每个 Block 的 x 维线程数。

因此，`blockIdx.x * blockDim.x` 给出当前 Block 覆盖的起始线性位置，再加上 `threadIdx.x`，就得到全局线性线程编号。

假设 `blockDim.x = 4`，则前几个线程的 idx 为：

| `blockIdx.x` | `threadIdx.x` | `idx` |
| ---: | ---: | ---: |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 0 | 2 | 2 |
| 0 | 3 | 3 |
| 1 | 0 | 4 |
| 1 | 1 | 5 |
| 2 | 3 | 11 |

这种映射让相邻线程访问相邻元素，是最基本的一维 CUDA 映射。

## 17. 为什么必须检查 `idx < N`

Grid 通常按向上取整方式创建，因此启动的线程总数可能大于 N。多出来的线程不能访问 `a[idx]`、`b[idx]` 或 `c[idx]`，否则会越界。

源码用：

```cpp
if (idx < N)
  c[idx] = a[idx] + b[idx];
```

只有合法线程执行访存和加法。若 N 恰好能被 Block 覆盖长度整除，所有线程都合法；若不能整除，最后一个 Block 的一部分线程会跳过计算。

边界检查是正确性条件，不是可有可无的装饰。后面的向量化版本每线程处理多个元素，边界条件也会相应变成 `idx + 3 < N` 或 `idx + 7 < N`。

## 18. 数学计算发生在哪里

真正的逐元素加法只有：

```cpp
c[idx] = a[idx] + b[idx];
```

右侧读取两个 `float`，做一次浮点加法，左侧写回一个 `float`。所有合法线程执行相同指令，但使用不同的 idx。

这体现 CUDA 的数据并行：不是一个线程写循环遍历 N 个元素，而是启动足够多的线程，每个线程处理一个元素。

当前代码没有线程间依赖。线程 idx=0 的结果不依赖 idx=1，因此不需要 `__syncthreads()`、Shared Memory 或 Atomic。

## 19. Kernel 不能直接被 Python 绑定

Python 最终调用的是：

```python
lib.elementwise_add_f32(a, b, c)
```

注意名字里没有 `_kernel`。这是因为被 PyBind 暴露的是 Host 端包装函数 `elementwise_add_f32`，不是 Device Kernel `elementwise_add_f32_kernel`。

包装函数负责理解 `torch::Tensor`，而 Kernel 接收的是裸指针和整数。两者接口层次不同：

```text
Python/C++ 边界：torch::Tensor
C++/CUDA 边界：float* 与 int
```

这个区分会贯穿整个项目。复杂算子也常有“面向框架的包装函数”和“面向 GPU 的 Kernel 模板”。

## 20. 包装函数是由宏生成的

源码没有直接写出完整的：

```cpp
void elementwise_add_f32(torch::Tensor a, ...)
```

而是定义宏 `TORCH_BINDING_ELEM_ADD`，再调用：

```cpp
TORCH_BINDING_ELEM_ADD(f32, torch::kFloat32, float, 1)
```

预处理器会把宏参数代入宏体，生成名为 `elementwise_add_f32` 的 C++ 函数。第一课不要求掌握所有 `##` Token 拼接规则，但必须知道这个函数确实在编译前被生成。

把本次宏参数记下来：

| 宏参数 | FP32 标量路径的值 | 含义 |
| --- | --- | --- |
| `packed_type` | `f32` | 用于拼接函数和 Kernel 名 |
| `th_type` | `torch::kFloat32` | PyTorch dtype |
| `element_type` | `float` | CUDA/C++ 元素类型 |
| `n_elements` | `1` | 每个线程处理的元素数 |

## 21. 展开后的 FP32 包装函数

忽略宏的反斜杠和拼接细节，FP32 标量版本可以近似理解为：

```cpp
void elementwise_add_f32(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)

  const int ndim = a.dim();
  // 根据 ndim 和 K 选择 grid、block
  // 取得 Tensor 指针
  // 启动 elementwise_add_f32_kernel
}
```

这个函数运行在 CPU。它的工作不是逐元素相加，而是检查和调度。

以后遇到宏生成函数时，可以先手动展开一个具体实例。理解一个实例后，再回头看宏如何批量生成六个版本。

## 22. `torch::Tensor` 是框架对象

包装函数参数是按值传入的 `torch::Tensor` 句柄：

```cpp
torch::Tensor a
```

它包含 dtype、shape、device、stride 和底层存储等元数据。复制 Tensor 句柄不等于把整块 GPU 数据复制到 CPU。

包装函数可以调用：

```cpp
a.dim()
a.size(i)
a.options().dtype()
a.data_ptr()
```

这些方法分别用于查询维数、某维大小、dtype 和底层数据地址。

这也是为什么 Kernel 不直接接收 `torch::Tensor`：Kernel 需要尽量简单、可在设备端使用的数据，而框架对象由 Host 包装层处理。

## 23. dtype 检查

宏生成函数首先检查 a、b、c：

```cpp
CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
```

若 dtype 不匹配，源码打印 Tensor options，然后抛出 `std::runtime_error`。异常会跨越扩展边界，最终在 Python 侧表现为调用失败。

这能阻止把 FP16 Tensor 的数据地址错误解释成 `float*`。如果类型宽度与解释方式不一致，Kernel 会读取错误的字节组合。

不过当前包装层主要检查 dtype，并没有在这里完整检查 shape 相等、CUDA device、contiguous 或当前 Stream 等条件。学习源码时要区分“这个示例做了哪些检查”和“生产级扩展通常还应做哪些检查”。

## 24. 计算 Tensor 元素总数

包装函数先取得：

```cpp
const int ndim = a.dim();
```

如果 `ndim != 2`，它将所有维大小相乘：

```cpp
int N = 1;
for (int i = 0; i < ndim; ++i) {
  N *= a.size(i);
}
```

如果 Tensor 是二维，则直接写：

```cpp
const int S = a.size(0);
const int K = a.size(1);
const int N = S * K;
```

无论哪条分支，Kernel 最终都只看到 N。二维分支的特殊之处主要在于 Launch 配置可能让一个 Block 对应一行。

## 25. `dim3`：描述 Grid 和 Block

CUDA Runtime 提供 `dim3` 类型表示三维尺寸。源码虽然只传一个数字，但本质上使用 x 维，y、z 维采用默认值 1。

对于普通非二维分支，FP32 标量版近似为：

```cpp
dim3 block(256);
dim3 grid((N + 256 - 1) / 256);
```

`block(256)` 表示每个 Block 有 256 个线程。因为 `n_elements=1`，每个线程处理一个元素。

Grid 使用整数向上取整公式。对正整数 a、b，常用：

```text
ceil(a / b) = (a + b - 1) / b
```

所以 `(N + 255) / 256` 能保证线程总覆盖范围不少于 N。

## 26. 二维输入的特殊 Launch 路径

Python 测试中的 K 取 1024、2048 或 4096。对于 FP32 标量版，条件是：

```cpp
if ((K / 1) <= 1024)
```

当 K=1024 时，源码设置：

```cpp
dim3 block(K);
dim3 grid(S);
```

这意味着一个 Block 对应 Tensor 的一行，Block 中 K 个线程各处理一个元素。由于每个 Block 最多支持的线程数通常是 1024，这条路径检查了 `K <= 1024`。

当 K=2048 或 4096 时，不能创建 2048 或 4096 线程的 Block，于是回退到一维扁平路径：Block 256 线程，Grid 覆盖整个 N。

二维路径并没有改变 Kernel 的 idx 公式。只要 Grid 和 Block 的总线性编号覆盖 `[0, N)`，同一个 Kernel 就能运行。

## 27. 从 Tensor 句柄取得设备指针

Kernel Launch 中出现：

```cpp
reinterpret_cast<float *>(a.data_ptr())
```

`a.data_ptr()` 返回底层数据地址。由于包装函数已经确认 dtype 是 `torch::kFloat32`，这里把地址解释成 `float*`。

这个地址仍然指向 GPU 显存，不会因为调用 `data_ptr()` 就把数据复制到 CPU。Host 代码只把地址值作为 Kernel 参数交给 CUDA Runtime。

对 b、c 也做同样转换。Kernel 得到三个设备指针后，便可以按 idx 访问显存。

后续会更严格讨论 `reinterpret_cast`、const 正确性和 Tensor contiguous。第一课只需建立“Tensor 对象 → 数据指针”的桥梁。

## 28. 真正的 Kernel Launch

FP32 标量实例最终形成：

```cpp
elementwise_add_f32_kernel<<<grid, block>>>(
    reinterpret_cast<float *>(a.data_ptr()),
    reinterpret_cast<float *>(b.data_ptr()),
    reinterpret_cast<float *>(c.data_ptr()),
    N);
```

函数名前缀与宏实例相互对应：包装函数 `elementwise_add_f32` 启动 Kernel `elementwise_add_f32_kernel`。

Launch 时，CUDA Runtime 将参数和执行配置提交给 GPU。CPU 线程通常不会等待所有 GPU 线程执行结束，而是继续向下运行。

包装函数没有普通返回值，因为结果原地写入 Python 传入的 c。Python 与 C++ 持有的 Tensor 句柄指向同一底层存储，所以 Python 随后能看到 GPU 写入的内容。

## 29. PyBind 注册发生在哪里

文件末尾是：

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8_pack)
}
```

`PYBIND11_MODULE` 定义扩展模块初始化入口。`TORCH_EXTENSION_NAME` 由 PyTorch 构建系统提供，它会与 `load()` 的模块构建名称协调。

`m` 可以理解为正在构造的 Python 模块对象。源码中的注册宏近似展开为：

```cpp
m.def(
    "elementwise_add_f32",
    &elementwise_add_f32,
    "elementwise_add_f32");
```

第一项是 Python 可见名称，第二项是 C++ 函数地址，第三项是简短文档字符串。

正是这次注册让 `lib` 获得 `elementwise_add_f32` 属性。若包装函数存在但没有注册，Python 仍无法通过 `lib.xxx` 调用它。

## 30. 现在把编译期和运行期分开

跨语言项目常见的误解，是把所有代码当成同一时刻执行。实际上至少有几个阶段：

| 阶段 | 发生的事情 |
| --- | --- |
| Python 解释阶段 | 执行 import、调用 `load()` |
| C/C++ 预处理 | 展开 `#include` 与宏 |
| 编译与链接 | 生成可加载扩展模块 |
| 模块初始化 | 执行 PyBind 注册，创建 Python 可见函数 |
| Python 测试运行 | 创建 Tensor，调用 `lib.elementwise_add_f32` |
| Host 包装运行 | 检查参数、计算 Launch 配置、提交 Kernel |
| Device 执行 | 大量 GPU 线程完成元素加法 |
| 同步与打印 | Python 等待 GPU，读取一小部分结果 |

宏生成函数发生在编译之前，PyBind 注册发生在模块加载时，而每次 benchmark 的 Kernel Launch 发生在运行期。把这三件事混在一起，会很难理解源码中的“函数到底在哪里”。

## 31. 回到 Python：准备形状组合

源码创建：

```python
Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]
```

最后一行是列表推导，相当于两层循环，共生成 3×3=9 个形状组合。

这些形状既测试不同数据规模，也触发包装函数中不同的 Launch 分支。例如 FP32 标量版在 K=1024 时可以使用一行一个 Block，而 K>1024 时需要扁平化覆盖。

测试脚本没有包含很小或奇数维度，因此不能仅凭这组 benchmark 证明所有尾部边界都已经充分覆盖。

## 32. 创建 CUDA Tensor

对每个 `(S, K)`，源码执行：

```python
a = torch.randn((S, K)).cuda().float().contiguous()
b = torch.randn((S, K)).cuda().float().contiguous()
c = torch.zeros_like(a).cuda().float().contiguous()
```

`torch.randn((S, K))` 默认先创建随机 Tensor。`.cuda()` 把 Tensor 放到默认 CUDA 设备，`.float()` 确保 dtype 是 FP32，`.contiguous()` 确保内存布局连续。

`torch.zeros_like(a)` 创建形状、dtype 和 device 等属性类似 a 的全零 Tensor。后续重复 `.cuda().float().contiguous()` 有一定冗余，但明确了期望状态。

连续性很重要，因为 CUDA Kernel 把 Tensor 当成扁平连续数组访问，没有使用 stride。若输入是非连续视图，线性 `a[idx]` 不一定对应逻辑 Tensor 的预期元素顺序。

## 33. 输出为什么由 Python 预分配

自定义包装函数签名接受 a、b、c：

```python
lib.elementwise_add_f32(a, b, c)
```

它不会在 C++ 中创建并返回新 Tensor，而是把结果写入已有的 c。这种接口称为 out 参数或原地写输出缓冲区。

优点是 benchmark 可重复使用同一块输出显存，避免每次迭代都分配 Tensor。缺点是调用者必须保证 c 的 shape、dtype、device 和容量正确。

当前 C++ 代码只显式检查 dtype，没有完整检查 a、b、c 的 shape 是否一致。这是教学实现与生产级算子接口之间的一个差距。

## 34. `run_benchmark` 的函数接口

源码定义：

```python
def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
```

`perf_func` 是待测试的可调用对象，可以是扩展函数，也可以是通过 `partial` 包装后的 PyTorch 函数。Python 的函数可以像普通对象一样作为参数传递。

`tag` 用于格式化输出，不参与计算。`warmup` 和 `iters` 提供默认值，调用时可以省略。

`out` 默认是 `None`，说明这个 benchmark 试图同时支持“函数写入外部输出”和“函数返回新输出”两种接口。

## 35. Warmup 为什么存在

如果 out 存在，源码先清零：

```python
out.fill_(0)
```

随后运行 warmup 次：

```python
for i in range(warmup):
    perf_func(a, b, out)
```

Warmup 可以让首次调用中的动态加载、上下文初始化、缓存和其他一次性成本不直接进入正式平均时间。具体有哪些一次性成本取决于环境和算子。

Warmup 后执行：

```python
torch.cuda.synchronize()
```

它确保预热提交的 GPU 工作已经完成，正式计时不会把未完成的 warmup 混进去。

## 36. 为什么普通 Python 计时会出错

CUDA Kernel Launch 通常是异步的。CPU 调用包装函数并提交 Kernel 后，可能很快返回，而 GPU 仍在执行。

如果写成：

```python
start = time.time()
perf_func(a, b, out)
end = time.time()
```

测到的可能主要是 Python、PyBind 和 Kernel Launch 开销，不是 GPU 完成计算的全部时间。

源码在正式循环前后同步：

```python
torch.cuda.synchronize()
start = time.time()
for i in range(iters):
    perf_func(a, b, out)
torch.cuda.synchronize()
end = time.time()
```

前一次同步建立干净起点，后一次同步保证所有迭代都结束后才读取终点。

## 37. 平均时间是如何得到的

源码计算：

```python
total_time = (end - start) * 1000
mean_time = total_time / iters
```

`time.time()` 的差值单位是秒，乘 1000 转换成毫秒。再除以迭代次数，得到每次调用的平均毫秒数。

这种测量包含 Python 循环、扩展调用和 Kernel Launch 开销，但通过大量迭代减少了单次测量噪声。它不是纯粹的 Kernel 内部硬件时间。

更专业的 CUDA Event 或 profiler 会在后续性能课程中讨论。第一课只需要确认源码为什么必须同步。

## 38. 如何打印结果

源码只打印前两个元素：

```python
out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
```

`flatten()` 把 Tensor 视作一维；`detach()` 断开 Autograd 关系；`.cpu()` 把数据复制到 CPU；`.numpy()` 转成 NumPy 数组；`.tolist()` 转成 Python 列表；`[:2]` 只取前两项。

`.cpu()` 隐含了需要获得 GPU 结果，因此会涉及数据传输和必要的完成条件。不过打印发生在计时同步之后，不计入前面的 mean_time。

只看两个元素不能证明整个 Tensor 正确。该脚本主要展示与 benchmark，严格测试应使用全 Tensor 比较或误差统计。

## 39. 第一次实际调用

Python 调用：

```python
run_benchmark(lib.elementwise_add_f32, a, b, "f32", c)
```

这里没有立即写括号调用 `lib.elementwise_add_f32(...)`，而是把这个函数对象作为 `perf_func` 参数传入。

进入 `run_benchmark` 后，实际调用变为：

```python
perf_func(a, b, out)
```

此时 `perf_func` 就是 PyBind 暴露的 `elementwise_add_f32`，`out` 就是 c。调用跨越 Python/C++ 边界。

## 40. 一次调用的完整时序

现在可以按时间顺序描述一次 benchmark 迭代：

1. Python 执行 `perf_func(a, b, c)`。
2. PyBind 把 Python Tensor 参数转换为 C++ 可接收的 `torch::Tensor` 句柄。
3. C++ 包装函数检查 a、b、c 的 dtype。
4. 包装函数读取 a 的维数和形状。
5. 包装函数计算 N、Grid 和 Block。
6. 包装函数取得三个 Tensor 的设备数据指针。
7. 包装函数通过 `<<<grid, block>>>` 提交 Kernel。
8. 包装函数返回，控制权回到 Python。
9. GPU 调度多个 Block 与线程执行 Kernel。
10. 每个合法线程读取 a[idx]、b[idx]，写入 c[idx]。
11. benchmark 循环继续提交下一次调用。
12. 循环后的 `torch.cuda.synchronize()` 等待全部工作完成。

步骤 8 与步骤 9 可能在时间上重叠，因为 Kernel Launch 是异步的。这正是理解 GPU 程序时与普通同步 CPU 函数最大的差异之一。

## 41. PyTorch 基线为什么使用 `partial`

源码还测试：

```python
run_benchmark(partial(torch.add, out=c), a, b, "f32_th")
```

`torch.add` 常见调用形式是：

```python
torch.add(a, b, out=c)
```

但 benchmark 在 `out is None` 分支中只会调用 `perf_func(a, b)`。`partial(torch.add, out=c)` 生成一个已固定 `out=c` 的函数，所以只传 a、b 也可以把结果写入 c。

这里有一个值得观察的接口差异：自定义扩展函数接收三个位置参数，因此调用 benchmark 时把 c 作为 `out` 参数传入；PyTorch partial 返回结果句柄，benchmark 走 `out is None` 的分支，但底层仍写入预先指定的 c。

这段通用 benchmark 代码可以工作，但接口设计略显绕。阅读性能结果时要确认两条路径的分配和写回行为是否可比。

## 42. FP32x4 是同一主链的变体

第二个自定义调用是：

```python
lib.elementwise_add_f32x4(a, b, c)
```

它仍经过同一个扩展模块、PyBind、C++ 包装和 Kernel Launch 结构。区别是宏实例的 `n_elements=4`，每个 GPU 线程负责四个连续 FP32 元素。

Kernel 的起始索引变为：

```cpp
int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
```

它使用 `float4` 完成较宽的读取和写回，并逐分量计算 x、y、z、w。尾部不足四个元素时回退到标量循环。

第一课只需要看清“接口链没变，线程工作粒度变了”。向量化的对齐、访存事务和性能影响会在后续单独展开。

## 43. FP16 路径也是同一主链

Python 将 FP32 输入转换为：

```python
a_f16 = a.half().contiguous()
b_f16 = b.half().contiguous()
c_f16 = c.half().contiguous()
```

随后调用 `elementwise_add_f16`、`f16x2`、`f16x8` 与 `f16x8_pack`。这些函数都由同一个绑定宏族生成，并在 PyBind 模块中注册。

FP16 标量 Kernel 参数类型从 `float*` 变为 `half*`，加法使用 `__hadd`。向量版本进一步使用 `half2` 和 128-bit Pack。

这里再次体现课程主线：语言和数据类型变化很多，但“Python Tensor → C++ 包装 → 数据指针 → Kernel”没有变化。

## 44. 用一个小例子手算 Launch

为了避免被 1024×1024 的大数字遮住，假设有 N=10 个 FP32 元素，并采用 Block 大小 4。

Grid 大小按向上取整：

```text
grid = (10 + 4 - 1) / 4 = 13 / 4 = 3
```

C++ 整数除法舍去小数，所以结果是 3。总共启动 3×4=12 个线程。

线程 idx=0 到 9 合法，执行加法；idx=10、11 不满足 `idx < N`，不访问内存。

若 a 与 b 是：

```text
a = [0,1,2,3,4,5,6,7,8,9]
b = [10,10,10,10,10,10,10,10,10,10]
```

最终 c 为：

```text
c = [10,11,12,13,14,15,16,17,18,19]
```

三个 Block 之间无需通信，每个线程只写唯一的 c[idx]，所以不存在写竞争。

## 45. shape 与线性地址

Python 的 a 形状是 `(S, K)`，但连续存储时可以按行主序理解为：

```text
a[0,0], a[0,1], ..., a[0,K-1], a[1,0], ...
```

二维坐标 `(row, col)` 对应线性下标：

```text
idx = row * K + col
```

当前 Kernel 直接使用 idx，所以它不会显式恢复 row 和 col。对于逐元素加法，这完全足够，因为每个位置使用相同运算。

如果算子需要按行归约、转置或矩阵乘，就必须理解二维或更高维坐标。后面的 Reduce、Softmax 和 GEMM 课程会逐步进入这些情况。

## 46. 当前实现依赖 contiguous

PyTorch Tensor 可以是非连续视图。例如矩阵转置后，逻辑相邻元素在物理内存中不一定相邻，Tensor 会通过 stride 描述布局。

当前 C++ 包装函数没有把 stride 传给 Kernel，Kernel 只执行 `a[idx]`。因此它按连续一维存储解释输入。

Python 测试明确调用 `.contiguous()`，满足了这个假设。若将来把该函数当成通用库接口，应在 C++ 中检查 contiguous，或者实现 stride-aware Kernel。

阅读 CUDA 扩展时，必须把 Python 端准备条件与 C++ 端假设合起来看。单看 Kernel 往往看不到输入是如何被规范化的。

## 47. 当前实现还假设 shape 一致

包装函数用 a 的维度和形状计算 N，却没有验证 b 与 c 具有相同元素数。正常测试中，b 与 a 都是 `(S,K)`，c 由 `zeros_like(a)` 创建，因此条件成立。

如果 b 更小，Kernel 可能越界读取；如果 c 更小，可能越界写入。dtype 检查无法防止这种错误。

这不是说教学代码没有价值，而是提醒读者区分算法核心与接口防护。生产级算子通常需要 device、dtype、shape、stride、alignment 和当前 CUDA device/stream 等检查。

后续阅读 `ffpa-attn` 时会看到更加系统的输入契约与回退逻辑。

## 48. 当前 Launch 使用哪个 CUDA Stream

源码的 Kernel Launch 没有显式写第四个配置参数：

```cpp
kernel<<<grid, block>>>(...)
```

这意味着它使用默认 Stream 语义，而不是显式取得 PyTorch 当前 CUDA Stream。教学脚本在简单同步场景下可以运行，但与复杂 PyTorch Stream 环境集成时，需要更加谨慎。

正式 PyTorch CUDA 扩展通常应考虑当前 Stream，使自定义算子与框架已经排队的工作保持正确顺序。第一课只标记这个工程问题，不展开 Stream API。

## 49. 输出示例应该怎样读

README 输出类似：

```text
out_f32:   [...], time: ... ms
out_f32x4: [...], time: ... ms
out_f32_th:[...], time: ... ms
```

前两个打印值相同，说明所展示位置的结果一致。不同版本时间接近或向量版更快，反映特定 GPU、形状和环境下的观测。

不能仅凭 README 历史结果断言在任何 GPU 上 FP32x4 都更快。对于极小 Kernel，Launch 开销、缓存、编译参数和测量方法都可能影响结果。

同理，FP16 更快也不仅因为“数字位数少”。硬件吞吐、内存带宽、指令类型、向量宽度和 Tensor 规模都会参与。

## 50. 如何自己重新追踪一遍

第一遍从 Python 找入口：搜索 `elementwise_add_f32`，先看到 benchmark 调用。

第二步在 `.cu` 中搜索同名文本。你会看到宏实例、PyBind 注册，以及带 `_kernel` 后缀的 Kernel 名。

第三步手动展开宏实例的四个参数，确认包装函数的名字、dtype、指针类型和每线程元素数。

第四步从包装函数读取 N、grid、block 和 `data_ptr()`，再进入 Kernel 解释 idx。

第五步回到 Python，找到同步与输出读取。这样就完成一次闭环，而不是停在 Kernel 的最后一行。

## 51. 调用链中的数据形态变化

同一份数据在不同层次有不同表示：

| 层次 | a 的表示 |
| --- | --- |
| Python | `torch.Tensor` 对象 |
| PyBind/C++ | `torch::Tensor` 句柄 |
| 包装函数内部 | 含 shape、dtype、device 的框架对象 |
| Kernel 参数 | `float*` 设备指针 |
| 单个线程 | `a[idx]` 这一项 FP32 值 |

数据本身通常没有在每一层复制。变化的是接口看到的抽象：Python 看到 Tensor，Kernel 看到指针和标量。

理解抽象边界后，`data_ptr()` 就不再神秘。它只是从框架对象进入低层内存访问的出口。

## 52. 调用链中的控制权变化

控制权也在不同执行者之间流动：

```text
Python 解释器
   ↓ 调用扩展函数
C++ Host 包装函数
   ↓ 提交 Kernel
CUDA Runtime / GPU 调度器
   ↓ 启动线程
GPU Device Kernel
   ↓ 写入 c
Python 在同步点等待并继续
```

CPU 负责组织和提交，GPU 负责大规模并行计算。GPU 不是在 Python 里逐行执行 `.cu` 文本，而是执行已经编译的 Device 代码。

异步提交使 CPU 可以继续工作，但也要求程序在读取结果、计时或释放资源前建立正确同步。

## 53. 编译错误与运行错误怎样区分

如果 `elementwise.cu` 语法错误、头文件缺失或 GPU 架构参数不受支持，问题通常出现在 `load()` 阶段。这时 benchmark 循环还没有开始。

如果扩展成功加载，但 Python 写错函数名，例如 `lib.unknown_func`，会出现模块属性错误。这通常说明 PyBind 没有注册对应名称。

如果 dtype 不匹配，会进入 C++ 包装函数后抛出 runtime error。若 shape 或指针假设错误，则可能出现 CUDA 非法内存访问，并可能延迟到同步点才报告。

如果计时数字异常但没有报错，需要检查同步、warmup、迭代、输出分配和环境噪声。不同错误属于不同层，排查时不要一律归因于 Kernel。

## 54. 第一课暂时不深挖的内容

我们没有完整展开 `TORCH_BINDING_ELEM_ADD` 的所有预处理细节。第 02、03 课会系统讲 C++ 指针、宏、Token 拼接、模板与低精度类型。

我们没有证明 `float4` Load/Store 一定生成哪条机器指令。第 07、08、12 课会结合内存层次、对齐和向量化讨论。

我们没有分析 Occupancy、Warp 合并访存和 Roofline。阶段 B 会把这些概念放回本算子。

我们也没有把 README 中的性能结果在当前环境重跑。源码阅读结论与硬件实验结论必须分开。

## 55. 关键概念对照表

| 概念 | 在本项目中的实例 | 一句话解释 |
| --- | --- | --- |
| Python 入口 | `elementwise.py` | 组织构建和实验 |
| 扩展构建 | `load(...)` | 编译并加载 C++/CUDA 模块 |
| 扩展模块 | `lib` | Python 可调用的已加载模块 |
| PyBind 注册 | `PYBIND11_MODULE` | 将 C++ 函数暴露给 Python |
| Host 包装函数 | `elementwise_add_f32` | 检查 Tensor 并启动 Kernel |
| Device Kernel | `elementwise_add_f32_kernel` | 在 GPU 上执行元素加法 |
| Launch 配置 | `<<<grid, block>>>` | 指定 Block 数和每 Block 线程数 |
| 设备指针 | `float*` | 指向 CUDA Tensor 数据的地址 |
| 全局线性索引 | `idx` | 当前线程负责的数据位置 |
| 边界检查 | `idx < N` | 防止最后部分线程越界 |
| 异步执行 | Kernel Launch 后 Host 可继续 | CPU 提交不等于 GPU 已完成 |
| 同步 | `torch.cuda.synchronize()` | 等待已提交 CUDA 工作完成 |
| 输出缓冲 | `c` | 预分配并由 Kernel 原地写入 |

## 56. 阅读检查：你能否解释这些名字

`elementwise_lib` 是传给 `load()` 的扩展名称。

`lib` 是 Python 变量，引用加载完成的扩展模块。

`elementwise_add_f32` 是 Python 可见名称，也是宏生成的 C++ Host 包装函数名称。

`elementwise_add_f32_kernel` 是 GPU Device Kernel 名称。

`f32` 是宏实例中的标识片段，用于生成函数名，并不是 C++ 内建类型。

`torch::kFloat32` 是 PyTorch C++ dtype 标识。

`float` 是 C++/CUDA 元素类型。

把这些名称区分开，能避免在搜索源码时混淆“类型标签、包装函数、Kernel 和模块”。

## 57. 练习一：手动画调用链

不看前面的流程图，从以下 Python 调用开始：

```python
lib.elementwise_add_f32(a, b, c)
```

画出至少六个节点，并在每条边上写出数据表示。你的图应包含 Python Tensor、C++ Tensor、设备指针、Kernel、GPU 线程和输出 c。

### 参考分析

一种合格答案是：Python 将三个 Tensor 传给 PyBind；PyBind 调用 C++ 包装函数；包装函数把 Tensor 检查后取出 `float*`；CUDA Runtime 以 grid/block 启动 Kernel；每个线程用 idx 访问指针；Kernel 写入 c 的底层存储；同步后 Python 通过同一个 c 句柄读取结果。

重点不是图形样式，而是不能遗漏包装函数。Python 不会直接以 `torch.Tensor` 参数启动这个裸指针 Kernel。

## 58. 练习二：计算线程索引

假设 `blockDim.x=8`，请计算：

1. `blockIdx.x=0, threadIdx.x=5` 时的 idx。
2. `blockIdx.x=3, threadIdx.x=2` 时的 idx。
3. N=25 时，第二个线程是否执行加法？

### 参考答案

第一个 idx 是 `0*8+5=5`。第二个 idx 是 `3*8+2=26`。当 N=25 时，合法下标是 0 到 24，因此 idx=26 不满足边界条件，不执行访存和加法。

## 59. 练习三：计算 Grid

若 N=1000，Block 大小为 256，Grid x 维应该是多少？总共启动多少线程？有多少线程会被边界条件挡住？

### 参考答案

Grid 为 `(1000+256-1)/256 = 1255/256 = 4`。总线程数为 `4*256=1024`，其中 1000 个线程处理合法元素，24 个线程跳过。

## 60. 练习四：找出隐含假设

根据本课源码，列出至少四个由 Python 测试满足、但 C++ 包装函数没有完整验证的条件。

### 参考分析

可以包括：a、b、c shape 相同；三者在 CUDA device；三者 contiguous；元素数不会使 `int N` 溢出；数据指针对向足够存储；当前 Stream 语义满足调用上下文；向量化版本满足必要对齐。第一课不要求修复它们，但要能识别接口契约。

## 61. 练习五：解释为什么需要同步

如果删除正式计时循环后的 `torch.cuda.synchronize()`，平均时间可能代表什么？为什么结果打印时程序仍可能得到正确数值？

### 参考分析

缺少同步时，计时终点可能在 GPU 完成之前记录，所以主要测到提交开销。随后 `.cpu()` 读取输出需要获得 GPU 结果，相关操作可能隐式等待，因此打印仍可能正确；只是等待时间被移出了计时区间。

## 62. 练习六：比较标量和向量版本

FP32 标量版每线程处理一个元素，FP32x4 每线程处理四个元素。若 N=1024，它们在宏的普通路径中分别使用多少线程一个 Block，Grid 设计希望覆盖多少元素？

### 参考分析

标量版 `block=256/1=256`，每 Block 覆盖 256 个元素。x4 版 `block=256/4=64`，每个线程负责四项，每 Block仍希望覆盖 256 个元素。Grid 的分母保持 256，体现宏把“每 Block 覆盖元素数”与“每线程处理元素数”分开。

## 63. 一份可复用的源码阅读模板

以后打开任意 LeetCUDA 算子目录，可以按以下顺序阅读：

1. README：确认算子、运行命令和版本列表。
2. Python：找 `load()`、import 或已安装扩展模块。
3. Python：找第一次实际调用的 `lib.xxx`。
4. C++/CUDA：搜索同名包装函数或宏实例。
5. C++/CUDA：搜索 PyBind 注册。
6. 包装函数：记录输入检查、shape、dtype、grid、block、stream。
7. Kernel：记录一个线程、一个 Warp、一个 Block各负责什么。
8. Kernel：记录 Global、Shared、Register 的数据路线。
9. Python：检查基线、误差、warmup、同步和计时。
10. 最后才比较优化版本。

对于 Elementwise，步骤 7 很简单：一个线程负责一个或几个元素。到了 Reduce 和 GEMM，这一步将成为课程的核心。

## 64. 本课总结

`elementwise.py` 是入口，但自定义加法不在 Python 中执行。顶层 `load()` 编译并加载 `elementwise.cu`，返回扩展模块 `lib`。

`PYBIND11_MODULE` 把 C++ Host 包装函数注册成 Python 可见的 `lib.elementwise_add_f32`。包装函数接收 `torch::Tensor`，检查 dtype、读取 shape、计算 N 与 Launch 配置，再取得 GPU 数据指针。

`elementwise_add_f32_kernel<<<grid, block>>>` 在 GPU 上启动许多线程。每个线程用 `blockIdx.x * blockDim.x + threadIdx.x` 得到 idx，通过 `idx < N` 防止越界，并执行 `c[idx] = a[idx] + b[idx]`。

Kernel Launch 通常异步返回，所以 benchmark 在正式计时前后调用 `torch.cuda.synchronize()`。输出 c 在 Python 中预先分配，C++ 与 Python 通过同一底层 CUDA 存储观察结果。

FP32x4 和 FP16 版本改变了元素类型、向量宽度与每线程工作量，但仍复用“Python → PyBind → C++ 包装 → CUDA Kernel”的同一条主调用链。

## 65. 下一课预告

下一课将系统补齐阅读 LeetCUDA 所需的最小 C++ 语法。我们会继续使用 `elementwise.cu` 和 `block_all_reduce.cu` 的真实代码，重点讲指针、数组、`const`、作用域、函数、类型转换和地址，而不会脱离项目写一套通用 C++ 教程。

在继续之前，请先确认你是否能独立说出 `lib.elementwise_add_f32`、`elementwise_add_f32` 与 `elementwise_add_f32_kernel` 三个名字分别属于哪一层，以及 a 从 Python Tensor 变成 Kernel 中 `float*` 的过程。
