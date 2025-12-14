# Quantization Techniques Reference (For Benchmarking)

This document is a concise reference of quantization techniques to be used during Step 2 (quantizing [obadx/muaalem-model-v3_0](https://huggingface.co/obadx/muaalem-model-v3_0)) evaluation and comparison.  

---

## 1. Post-Training Quantization (PTQ)

### 1.1 Dynamic Quantization

**Type:** Post-Training  
**Weights:** INT8  
**Activations:** FP16/FP32 (runtime dynamic)

**Speed Impact:** Medium  
**Size Reduction:** Medium  
**Accuracy Impact (ASR/Wav2Vec2-BERT):** Low–Medium  

**Frameworks / Tools:**
- PyTorch `quantize_dynamic`
- ONNX Runtime (Dynamic)

**Usability Status:** Stable  
**Benchmark Priority:** High

---

### 1.2 Static Quantization

**Type:** Post-Training  
**Weights:** INT8  
**Activations:** INT8  

**Speed Impact:** High  
**Size Reduction:** High  
**Accuracy Impact (ASR/Wav2Vec2-BERT):** Medium  

**Requirements:**
- Calibration dataset

**Frameworks / Tools:**
- PyTorch Static Quantization
- ONNX Runtime (Static)
- Intel Neural Compressor

**Usability Status:** Stable  
**Benchmark Priority:** High

---

### 1.3 Weight-Only Quantization

**Type:** Post-Training  
**Weights:** INT8 / INT4  
**Activations:** FP16/FP32  

**Speed Impact:** Low–Medium  
**Size Reduction:** High  
**Accuracy Impact (ASR/Wav2Vec2-BERT):** Low  

**Frameworks / Tools:**
- bitsandbytes
- Optimum (weight-only backends)

**Usability Status:** Stable  
**Benchmark Priority:** High

---

## 2. Quantization-Aware Training (QAT)

### 2.1 Full QAT

**Type:** Training-Time  
**Weights:** INT8 / INT4 (simulated during training)  
**Activations:** Simulated  

**Speed Impact:** High  
**Size Reduction:** High  
**Accuracy Impact (ASR/Wav2Vec2-BERT):** Very Low  

**Frameworks / Tools:**
- PyTorch QAT
- TensorFlow Model Optimization

**Usability Status:** Stable  
**Benchmark Priority:** Medium

---

### 2.2 Partial (Hybrid) QAT

**Type:** Training-Time  
**Weights:** Mixed  
**Activations:** Mixed  

**Typical FP Layers:**
- Embeddings
- LayerNorm

**Quantized Layers:**
- Attention
- Feed-Forward

**Usability Status:** Research/Production  
**Benchmark Priority:** Medium

---

## 3. Ultra Low-Bit Quantization

### 3.1 GPTQ

**Type:** Post-Training (Layer-wise Hessian-aware)  
**Weights:** INT4  
**Activations:** FP16/FP32  

**Speed Impact:** Medium  
**Size Reduction:** Very High  
**Accuracy Impact (ASR/Wav2Vec2-BERT):** Medium  

**Frameworks / Tools:**
- auto-gptq

**Usability Status:** Production/Research  
**Benchmark Priority:** High

---

### 3.2 AWQ (Activation-Aware Weight Quantization)

**Type:** Post-Training (Activation-scaled)  
**Weights:** INT4  
**Activations:** FP16/FP32  

**Speed Impact:** Medium  
**Size Reduction:** Very High  
**Accuracy Impact (ASR/Wav2Vec2-BERT):** Low–Medium  

**Frameworks / Tools:**
- awq

**Usability Status:** Production  
**Benchmark Priority:** High

---

### 3.3 INT2 Quantization

**Type:** Research  
**Weights:** INT2  
**Activations:** FP16/FP32  

**Speed Impact:** Very High  
**Size Reduction:** Extreme  
**Accuracy Impact (ASR):** High  

**Frameworks / Tools:**
- Custom kernels

**Usability Status:** Experimental  
**Benchmark Priority:** Low

---

## 4. Mixed-Precision Quantization

**Type:** Layer-wise Mixed Precision  
**Weights:** INT8 / INT4 / FP16  
**Activations:** Mixed  

**Layer Precision Map (Typical):**
- Embeddings → FP16
- LayerNorm → FP16
- Attention → INT8
- FFN → INT8

**Speed Impact:** High  
**Size Reduction:** High  
**Accuracy Impact (ASR):** Low  

**Frameworks / Tools:**
- PyTorch AMP + Quant
- TensorRT

**Usability Status:** Production-Ready  
**Benchmark Priority:** High

---

## 5. Hardware-Aware Quantization

### 5.1 CPU-Oriented

**Target:** x86 / ARM  
**Instruction Sets:**
- AVX512-VNNI
- ARM NEON

**Backends:**
- FBGEMM
- QNNPACK

**Benchmark Priority:** Medium

---

### 5.2 GPU-Oriented

**Target:** NVIDIA GPUs  
**Precisions:**
- FP16
- INT8
- FP8

**Backends:**
- TensorRT
- CUDA Kernels

**Benchmark Priority:** High

---

## 6. Technique Comparison Table

| Technique          | Weights | Activations | Speed Gain | Size Reduction | Accuracy Risk | Priority | Used |
|-------------------|--------|------------|------------|----------------|--------------|----------|----------|
| Dynamic PTQ        | INT8   | FP32       | Medium     | Medium         | Low–Medium   | High     |❌       |
| Static PTQ         | INT8   | INT8       | High       | High           | Medium       | High     |❌       |
| Weight-Only        | INT8/4 | FP16/32    | Low–Medium | High           | Low          | High     |❌       |
| QAT Full           | INT8/4 | Simulated  | High       | High           | Very Low     | Medium   |❌       |
| QAT Partial        | Mixed  | Mixed      | High       | High           | Low          | Medium   |❌       |
| GPTQ               | INT4   | FP16/32    | Medium     | Very High      | Medium       | High     |❌       |
| AWQ                | INT4   | FP16/32    | Medium     | Very High      | Low–Medium   | High     |❌       |
| INT2               | INT2   | FP16/32    | Very High  | Extreme        | High         | Low      |❌       |
| Mixed Precision    | Mixed  | Mixed      | High       | High           | Low          | High     |❌       |

---

## 7. Benchmarking Notes (For Step 2)

Track for each technique:

- Model load time
- Real-Time Factor (RTF)
- GPU/CPU memory peak
- WER/CER deltas
- Throughput (samples/sec)

