Many computational scenarios for data-parallel applications exist today. They use and target:

•	Varying heterogeneous platforms leveraging CPUs, GPUs, FPGAs, ASICs, and other specialized accelerators.
•	Multiple programming languages that support these platforms—some are based on open standards and others are proprietary, vendor-specific language extensions and solutions.   
•	Differing programming infrastructures such as libraries, optimization tools, analyzers, and debuggers.
•	Many compute domains such as ML/DL, data analytics, video/imaging, high-performance computing (HPC), healthcare, science, enterprise, and other industries.

For each compute domain, every platform and programming model combination may perform differently. 

This makes it very difficult for developers to assess and characterize device performance for their software workloads and determine which platform or language is the best target for application development and deployment. 

Most GPU vendors have their own set of tools that report specific performance information. Often the data is collected and computed using different methodologies. Even worse, sometimes they report metrics having similar names but with different interpretations. 

# Benchmarking Workloads across Platforms

Numerous benchmarks are available in the open-source community, academia, and vendor-specific repositories. They usually assess the performance of one specific platform and infrastructure combination. This makes an apples-to-apples comparison for a specific use case challenging.

If you search for a good reference benchmark that is representative of your codebase, you may find differing versions for each environment. The most likely scenario is that these versions use different algorithms, optimization levels, datasets, and measurement mechanisms such as timers and iterations. 

Identifying a set of benchmark workloads applicable to multiple GPUs and accelerators is the goal the Velocity Bench GitHub project is setting for itself.

We are driving towards a suite that helps to enable fair comparisons. One that:
•	Provides objective GPU offload performance data across compute domains, environments, and hosts as well as target compute architectures.
•	Has representative applications or workloads covering multiple domains like HPC, ML/DL, and data analytics.
•	Is targeted for multiple parallel programming models (e.g. SYCL*, CUDA*, HIP*).
To achieve this, the suite must be capable of running on most environments, using similar program structure, algorithms, datasets, timing mechanisms, and levels of optimization.  

# Velocity Bench: Simplifying GPU Performance Assessment

This benchmark suite of optimized workloads helps solve the problem of benchmark portability and applicability across different platform configurations. The suite has 15 workloads; each is available in SYCL, HIP, and CUDA to allow for runs on Intel, AMD, and Nvidia GPUs using the different programming models. Additionally, with SYCL’s open backend, workloads can be extended to support other types of accelerators moving forward. Thus, we can look at platform performance using native platform programming languages as well as multiarchitecture programming models (e.g., SYCL vs. HIP on AMD GPU and SYCL vs. CUDA on Nvidia GPU).

Ensuring that all versions of the individual benchmark workloads are optimized to the same degree is a key focus of ongoing Velocity Bench development. This includes the use of equivalent algorithms, libraries, and input data types. Of course, further opportunities for changes and optimizations always exist. 

Some workloads in Velocity Bench measure time, while others measure throughput or other metrics. For compute and execution time measurements, a consistent methodology (begin-to-end time) is used. Time for external I/O and data-verification is excluded from the performance data collection because it could vary depending on the available hardware device and device drivers. 

 
# Workloads Included in Velocity Bench

These benchmark workloads cover different use case scenarios and exercise different aspects of the underlying hardware.
1.	**Reverse Time Migration (RTM)**: Domain: HPC. A 2D seismic wave-propagation workload. RTM is a seismic imaging method to map the surface (oil, gas, etc.) reflectivity using recorded seismic waveforms. 
2.	**HashTable**: Domains: Compute, HPC. This workload provides hash table search implementation using a new and efficient lock-free algorithm that utilizes linear probing. Atomic operations are used to insert key/value pairs into the hash table on multiple GPU threads.
3.	**QuickSilver**: Domain: HPC. QuickSilver represents the key elements of the Mercury workload by solving a simpliﬁed, dynamic Monte Carlo particle-transport problem. It attempts to replicate the memory-access patterns, communication patterns, and the branching or divergence of Mercury for problems using multigroup cross sections. 
4.	**easyWave**: Domain: HPC. A tsunami wave simulator tool used for researching tsunami generation and wave propagation. 
5.	**LC0**: Domains: Deep learning, neural networks. Leela Chess Zero is a reinforced learning-based chess engine. As of December 2022, LC0 has played more than 1.5 billion games against itself, playing around 1 million games every day.  
6.	**DL-CIFAR**: Domains: Deep learning, neural networks. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes (bird, cat, airplane, etc.), with 6,000 images per class. There are 50,000 training images and 10,000 test images. This workload takes input image and classifies it using neural networks. CIFAR is one of the most popular databases. 
7.	**DL-MNIST**: Domains: Deep learning, neural networks. This implementation provides digits recognition based on the MNIST digits database, which is one of the oldest and most popular databases of digits. Workload uses neural networks to identify digits. 
8.	**SAM**: Domain: HPC. This workload performs acoustic wave propagation of multiple source locations for a 2D sub-surface earth model using finite difference time domain modeling.
9.	**CudaSift**: Domains: Vision, compute. SIFT (Scale Invariant Feature Transform) algorithm implementation is a popular workload in the open source community. It implements an algorithm to detect, describe, and match local features in images. 
10.	**TSNE**: Domains: Big data, cloud. t-SNE (t-distributed stochastic neighbor embedding) is an unsupervised, non-linear technique primarily used for data exploration and visualizing high-dimensional data by doing data dimension reduction.
11.	**BitCracker**: Domains: Security, cryptography. It is a password-cracking application for memory units encrypted with BitLocker*. By means of a dictionary attack, BitCracker tries to find the correct User Password or Recovery Password to decrypt the encrypted storage device.
12.	**ETHMiner**: Domain: Cryptography. This is a bitcoin-mining workload. ETHMiner is an Ethash GPU mining worker to mine every coin which relies on an Ethash Proof of Work, including Ethereum bitcoins.
13.	**SVM**: Domain: Classical machine learning. Support Vector Machine is one of the most popular classical machine learning techniques. SVMs are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis. 
14.	**Sobel Filter**: Domains: Imaging, compute. Sobel filter is a popular and widely used RGB-to-grayscale image conversion (2D to 3D image conversion) technique, which applies a gaussian filter to reduce edge artifacts. 
15.	**HP LINPACK**: Domains: Compute, system. This is not the most performant LINPACK tuned by various companies for which results are quoted, but it mainly uses libraries to calculate a device’s rate of execution. It uses GEMM calls to solve dense system of linear equations. 

The Velocity Bench suite is a collection of workloads, some of which are developed and optimized by us. Others originated from the open source community. For the latter, we created/ported and optimized comparable code versions in the two other languages. For example, if CUDA was the originating code, we developed the SYCL and AMD versions. See the detailed workload descriptions and links to the source code origins in the Velocity Bench repository.

# Understand Multiplatform Application Performance

Further optimization and other modifications are welcome from community members using standard GitHub processes and comments. Our intent is to update this repository continuously with further optimizations and changes as well as for inclusion of new workloads and to deprecate others when no longer needed. 

Take the workloads for a spin on your targeted platform setups.

# Contribute to the Community

We look forward to hearing about your experience with various configurations.
•	How does your offload compute perform on different GPUs? 
•	What type of workload is missing from Velocity Bench?

Additionally, we look forward to your feedback, repository contributions, and optimization ideas.



# Notices and Disclaimers

Performance varies by use, configuration and other factors. Learn more at www.Intel.com/PerformanceIndex. Results may vary.
Performance results are based on testing as of dates shown in configurations and may not reflect all publicly available updates. No product or component can be absolutely secure. 
Your costs and results may vary. 
Intel technologies may require enabled hardware, software or service activation.
Intel does not control or audit third-party data. You should consult other sources to evaluate accuracy. 
© Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.  
*Other names and brands may be claimed as the property of others.  


