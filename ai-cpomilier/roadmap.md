# AI编译器学习路线

## 📚 第一阶段：基础知识准备（1-2个月）

### 1. 编译原理基础
- **词法分析、语法分析**
- **中间表示（IR）**
- **代码优化技术**
- **代码生成**
- 推荐资源：《编译原理》（龙书）、Stanford CS143

### 2. 深度学习基础
- **神经网络基本概念**（CNN、RNN、Transformer等）
- **常见算子**（Conv、MatMul、Pooling、Attention等）
- **训练与推理流程**
- 推荐框架：PyTorch、TensorFlow

### 3. 编程能力
- **C++**（性能关键代码）
- **Python**（前端接口）
- **CUDA/OpenCL**（GPU编程基础）

## 🔧 第二阶段：核心技术学习（2-3个月）

### 1. 计算图与IR
- **计算图表示**
- **静态图 vs 动态图**
- **多层IR设计**（High-level IR → Low-level IR）
- **SSA形式**

### 2. 图优化技术
- **算子融合**（Operator Fusion）
- **常量折叠**（Constant Folding）
- **公共子表达式消除**（CSE）
- **死代码消除**（DCE）
- **布局转换优化**

### 3. 自动调优
- **AutoTVM**
- **AutoScheduler**
- **机器学习驱动的优化**
- **搜索空间设计**

## 🎯 第三阶段：主流框架实践（3-4个月）

### 1. TVM（重点推荐）
```
核心模块：
├── Relay IR（高层IR）
├── TIR（张量IR）
├── AutoTVM/AutoScheduler
├── BYOC（Bring Your Own Codegen）
└── Runtime系统
```
- 官方教程：https://tvm.apache.org/docs/
- 动手实践：编写自定义Pass、算子

### 2. MLIR（Multi-Level IR）
- **Dialect系统**
- **Pass管理**
- **Pattern Rewriting**
- **与TVM的集成**
- 官方文档：https://mlir.llvm.org/

### 3. XLA（Accelerated Linear Algebra）
- **HLO IR**
- **融合策略**
- **布局优化**
- TensorFlow/JAX后端

### 4. 其他框架了解
- **TensorRT**（NVIDIA推理优化）
- **OpenVINO**（Intel）
- **ONNX Runtime**
- **Glow**（Facebook）

## 🚀 第四阶段：硬件适配（2-3个月）

### 1. GPU优化
- **CUDA编程深入**
- **Tensor Core利用**
- **内存层次优化**
- **Kernel融合技术**

### 2. 其他硬件平台
- **CPU优化**（SIMD、多线程）
- **移动端**（ARM、NPU）
- **专用AI芯片**（TPU、昇腾等）

### 3. 量化与压缩
- **INT8/FP16量化**
- **模型剪枝**
- **知识蒸馏**

## 🔬 第五阶段：进阶与专题（持续学习）

### 1. 前沿技术
- **动态shape处理**
- **稀疏计算优化**
- **混合精度训练**
- **分布式编译**

### 2. 论文阅读
- TVM: An Automated End-to-End Optimizing Compiler for Deep Learning
- MLIR: A Compiler Infrastructure for the End of Moore's Law
- XLA: TensorFlow, Compiled
- Ansor: Generating High-Performance Tensor Programs for Deep Learning

### 3. 开源贡献
- 参与TVM、MLIR等社区
- 提交PR、修复Bug
- 实现新特性

## 📖 推荐学习资源

### 在线课程
- **CMU 15-745**: Optimizing Compilers
- **Stanford CS243**: Program Analysis and Optimization
- **MIT 6.S965**: TinyML and Efficient Deep Learning

### 书籍
- 《深度学习编译器原理与实践》
- 《LLVM Cookbook》
- 《Programming Massively Parallel Processors》

### 博客与社区
- TVM Discuss论坛
- MLIR Discourse
- 知乎AI编译器话题
- 各大公司技术博客

## 🎓 实践项目建议

1. **入门项目**：用TVM编译一个简单的CNN模型到不同硬件
2. **进阶项目**：实现一个自定义的图优化Pass
3. **高级项目**：为新硬件后端添加支持
4. **综合项目**：构建一个端到端的模型部署pipeline



