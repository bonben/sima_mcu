# Simple Attention for Simple MCUs

This repository provides the open-source implementation for the paper **"Simple Attention for Simple MCUs"**, demonstrating the efficiency of replacing the standard Multi-Head Self-Attention (MHSA) softmax operation with an $L_1$ normalization (Simple Attention) to enable high-performance, low-latency inference on microcontrollers.

## Repository Structure

The codebase is cleanly separated into two main parts:

### 1. `eeg_conformer/`
Contains the Python/PyTorch codebase for training and evaluating the EEGConformer model on the BNCI2014_001 dataset using both Standard Attention and Simple Attention.

* **`attention.py`**: Implementation of PyTorch modules for Standard Multi-Head Attention and Simple Attention.
* **`eegconformer.py`**: The EEGConformer model architecture.
* **`train.py`**: Training script to run experiments across different subjects and attention configurations.
* **`train.sh`** & **`train_K.sh`**: Shell scripts for running training jobs.
* **`plot_accuracy.py`**: Script to visualize the results.
* **`requirements.txt`**: Python dependencies.

### 2. `stm32_mcu_c_code/`
Contains the minimal C implementation of the attention mechanisms optimized for STM32 microcontrollers (specifically targeted and tested on STM32H747).

* **`Src/Operations.c`** & **`Inc/Operations.h`**: The core C kernels for standard scaled dot-product attention (`attention_engine`) and softmax-free simple attention (`simpleAttention_engine`), along with necessary matrix multiplication routines (including the $O(Nd^2)$ associativity trick).
* **`Src/utils.c`** & **`Inc/utils.h`**: Utility functions, including initialization for the DWT cycle counter used for precise hardware benchmarking.
* **`Inc/sima_data.h`**: Data definitions and configurations for the test benchmarks.

## Getting Started

### EEGConformer Training
To reproduce the training results, navigate to the `eeg_conformer` directory, install the requirements, and run the training script:
```bash
cd eeg_conformer
pip install -r requirements.txt
python train.py --subject 1 --attentions simpleattention multiheadattention --n-seeds 5
```

### MCU Deployment
The code in `stm32_mcu_c_code/` can be easily integrated into any STM32CubeIDE project. 
1. Copy the contents of `Src/` and `Inc/` into your project's respective directories.
2. Ensure your Cortex-M7 DWT cycle counter is enabled if you wish to run the cycle profiling benchmarks.
3. Call `SimMHAttention` from `Operations.h` to execute the optimized Simple Attention layer.

## License
[Specify License Here]
