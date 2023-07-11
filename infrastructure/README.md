# Infrastructure code for workloads

Common SYCL/CUDA functions and utilities for the workloads used by VelocityBench 

Current workloads that use this infrastructure code:
- apriori
- easywave
- kingsoft-nlm
- parboil-sad
- shoc-bfs

Note: When using the file handler code, please compile your workload with the following flags

```
-lstdc++
-std=c++17
-lstdc++fs
```


