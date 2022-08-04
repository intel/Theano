DISCONTINUATION OF PROJECT.

This project will no longer be maintained by Intel.

Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project. 

Intel no longer accepts patches to this project.

If you have an ongoing need to use this project, are interested in independently developing it, or would like to maintain patches for the open source software community, please create your own fork of this project. 
# Intel Software Optimization for Theano*
---

This repo is dedicated to improving Theano performance on CPU, especially in Intel® Xeon® and Intel® Xeon Phi™ processors.

**Key Features**
  * New backend of Intel® MKL (version >= 2017.0 which includes neural network primitives)
  * Advanced graph optimizations
  * CPU friendly OPs
  * Switch to Intel® MKL backend automatically in Intel®  Architecture
  * Out-of-box performance improvements for legacy models  
  * Transparently supports for Keras (Lasagne, etc.) workloads 

**Benchmark**
  * Hardwares
    - Intel® Xeon® CPU E5-2699 v4 @ 2.20GHz, 128GB RAM
    - Intel® Xeon Phi™  CPU 7250F @ 1.40GHz, 98G RAM
  * Softwares
    - Script: **[convnet-benchmarks](https://github.com/soumith/convnet-benchmarks/blob/master/theano/benchmark_imagenet.py)**
    - **[Stock Theano](https://github.com/theano/theano)**, commit ID: 2fa3ce
    - **[Intel Theano](https://github.com/intel/theano)**, commit ID: e3f7b4, ver-1.1
  * Terminologies
    - FWD, forward for inference
    - FWD+BWD, forward and backward for training
  * Results
    
| FWD:sec/batch | Stock Theano/Xeon | Intel Theano/Xeon | Intel Theano/Xeon Phi |
|---------------|-------------------|-------------------|-----------------------|
| AlexNet       | 1.045             | 0.077             | 0.054                 |
| GoogLeNet     | 2.228             | 0.280             | 0.169                 |
| VGG           | 5.089             | 0.836             | 0.570                 |
| OverFeat      | 6.105             | 0.273             | 0.185                 |
   
--
 
| FWD+BWD: sec/batch | Stock Theano/Xeon | Intel Theano/Xeon | Intel Theano/Xeon Phi |
|---------------|-------------------|-------------------|-----------------------|
| AlexNet       | 2.333             | 0.239             | 0.186                 |
| GoogLeNet     | 5.866             | 0.860             | 0.568                 |
| VGG           | 12.783            | 2.699             | 1.902                 |
| OverFeat      | 13.202            | 0.865             | 0.636                 |

**Performance Tips**
  * Add bias after convolution to archieve high performance since this sub-graph can be replaced with MKL Op
  * Use group convolution OP, [AbstractConvGroup](https://github.com/intel/Theano/blob/master/theano/sandbox/mkl/mkl_conv.py)
  * Use New MKL OP: [LRN](https://github.com/intel/Theano/blob/master/theano/tensor/nnet/lrn.py)

**Branch Information**
  * master, stable and fully tested version based on 0.9dev2 with Intel® MKL backend
  * nomkl-optimized, based on 0.9.0dev1 with generic optimizations
  * others, experimental codes for different applications which may be merged into master and/or deleted soon

**Installation**
    
  * Environment Setting
  
    Set Intel MKL library path into both LD_LIBRARY_PATH and LIBRARY_PATH
    
  * Quick Commands

    ```
    git clone https://github.com/intel/theano.git intel-theano
    cd intel-theano
    python setup.py build
    python setup.py install --user [--mkl]   # Note: using 'mkl' option will check and download MKL if it is not available
    cp intel-theano/theanorc_icc_mkl ~/.theanorc
    ```

   * Check Intel MKL 
   
    ```
    python theano/theano/sandbox/mkl/tests/test_mkl.py
         WARNING (theano.gof.cmodule): WARNING: your Theano flags `gcc.cxxflags` specify an `-march=X` flags.
         It is better to let Theano/g++ find it automatically, but we don't do it now
         mkl_available: True
         mkl_version: 20170209
         .
         ----------------------------------------------------------------------
         Ran 1 test in 2.213s

         OK
    ```
    

   * Run Benchmark
    
    ```
    python democase/alexnet/benchmark.py
    ```

  * Install Guide (recommend to go througth this document and set up optimized softwares)
    https://github.com/intel/theano/blob/master/Install_Guide.pdf


**Other Optimized Software**
  * Self-contained MKL in [here](https://github.com/01org/mkl-dnn/releases)
  * Optimized Numpy in [here](https://github.com/pcs-theano/numpy)

---
>\* Other names and trademarks may be claimed as the property of others.
