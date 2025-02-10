## Phase 01 Plan

### **Week 1-2 Plan: CUDA Basics**  
🔹 **Goal**: Learn CUDA architecture, write basic kernels, and understand memory hierarchy.  
🔹 **Tools**:  
- Jetson Orin (or another CUDA-capable GPU)  
- `nvcc` compiler (comes with CUDA Toolkit)  
- Nsight Systems (`sudo apt install nsight-systems`)  

---

### **📚 Step 1: Understand CUDA Architecture (Day 1-2)**  
- Learn about **threads, warps, blocks, and grids**.  
- Understand **memory hierarchy**: Global, shared, local, and constant memory.  

**📖 Reading & Videos**:  
- [CUDA Programming Guide (Chapters 1-3)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)  
- [Intro to CUDA Video](https://developer.nvidia.com/cuda-training-series)  

**🛠️ Task**:  
- Write a simple C++ program to check your **GPU properties** using `cudaGetDeviceProperties()`.  

---

### **📌 Step 2: Write Your First CUDA Program (Day 3-4)**  
- Learn **kernel launches** and **thread indexing**.  
- Understand **basic memory transfers** (host ↔ device).  

**📖 Reading**:  
- [CUDA by Example - First Kernel](https://developer.download.nvidia.com/books/cuda-by-example/cuda-by-example-sample.pdf)  

**🛠️ Task**:  
🔹 **Project 1: Parallel Vector Addition**  
- Write a CUDA program that **adds two large arrays** in parallel.  
- Compare CPU vs GPU execution times.  

---

### **📌 Step 3: Matrix Multiplication with Shared Memory (Day 5-7)**  
- Learn **shared memory and synchronization** (`__syncthreads()`).  
- Optimize matrix multiplication by **minimizing global memory accesses**.  

**🛠️ Task**:  
🔹 **Project 2: Matrix Multiplication**  
- Implement **naïve matrix multiplication** (each thread computes one element).  
- Optimize using **shared memory** and compare performance.  

---

### **Week 3-4 Plan: Memory Optimization & Debugging**  
🔹 **Goal**: Optimize memory usage, avoid performance bottlenecks, and learn CUDA debugging tools.  

### **📌 Step 4: Memory Coalescing & Bank Conflicts (Day 8-10)**  
- Learn how **memory access patterns** affect performance.  
- Understand **coalesced memory access** for global memory.  

**🛠️ Task**:  
- Modify the **matrix multiplication kernel** to use **coalesced memory access**.  
- Use **Nsight Systems** to profile memory usage.  

---

### **📌 Step 5: Debugging & Profiling (Day 11-12)**  
- Learn how to use **Nsight Systems (`nsys`) and Nsight Compute (`nv-nsight-cu-cli`)**.  
- Debug race conditions using **cuda-memcheck**.  

**🛠️ Task**:  
- Profile the **vector addition** and **matrix multiplication** kernels.  
- Identify **bottlenecks** and optimize execution configuration (grid/block sizes).  

---

### **📌 Step 6: Image Processing in CUDA (Day 13-14)**  
- Learn about **image storage in GPU memory**.  
- Implement **parallel convolution** for basic image filtering.  

**🛠️ Task**:  
🔹 **Project 3: CUDA Edge Detection**  
- Implement a **Sobel filter** in CUDA.  
- Compare execution time with OpenCV’s CPU implementation.  

---

### **🎯 Milestone 1: Review & Compare CPU vs GPU Performance**  
By the end of Phase 1, you should:  
- Understand CUDA memory hierarchy & parallelism.  
- Optimize memory access in simple kernels.  
- Use **Nsight Systems** for profiling.  
- Implement **basic CUDA perception tasks** (vector math, matrix ops, Sobel filtering).  
