# Attention Against Recurrence

### Prediction
Learned positions: Will fail to extrapolate robustly to length 32 and 48 because the learned embedding table lacks parameters for indices beyond the trained range.   

Sinusoidal positions: Will generalize better to longer sequence lengths (32 and 48) because trigonometric functions are continuous and unbounded.

### Results
#### 1. Verification & Parameter CountsPyTorch Verification: All outputs matched PyTorch's reference implementation across self-attention, causal masking, and cross-attention.
Parameter Counts:
1 Head: 4,224 parameters
2 Heads: 4,224 parameters
4 Heads: 4,224 parameters
8 Heads: 4,224 parameters
Explanation: Multi-head attention parameter counts remain invariant because splitting $d_{model}$ into $n$ heads divides the subspace dimensions without changing total projection matrix sizes.

#### 2. Delayed Recall (Attention vs. LSTM)
Attention Readout: Maintained 1.000 accuracy across lengths 30, 100, and 400 for all seeds, consistently placing high weight on step 0
LSTM: Performed well at length 30 (1.000), but degraded significantly to chance level at lengths 100 and 400 due to gradient decay and memory bottlenecks over long sequences.

#### 3. Positional Encodings & Mixed-Length Training
##### Standard Training (Len 16):
None: Test@16 = 0.507, Test@32 = 0.495
Learned: Test@16 = 1.000, Test@32 = 0.610.
Sinusoidal: Test@16 = 1.000, Test@32 = 0.680.
##### Mixed-Length Training (8, 16, 24):
Learned: Test@16 = 0.990 | Test@32 = 0.704 | Test@48 = 0.538   
Sinusoidal: Test@16 = 1.000 | Test@32 = 0.908 | Test@48 = 0.643

#### 4. Normalization Stacks & Cost
##### 12-Block Stack Accuracy@16:
Post-norm, No Warmup: 0.517 (failed to converge properly)
Post-norm, 100 Warmup: 1.000
Pre-norm, No Warmup: 1.000 (trained stably without warmup)
##### Forward / Backward Timing (ms):
Length 100: LSTM = 6.32 ms | Attention = 14.47 ms   
Length 200: LSTM = 14.03 ms | Attention = 33.87 ms   
Length 400: LSTM = 29.14 ms | Attention = 123.83 ms 

### Figures
##### 1. Recall Performance vs. Sequence Length
Figure 1: Validation accuracy comparing Attention Readout and LSTM across lengths 30, 100, and 400.   
##### 2. Computational Cost Scaling
Figure 2: Execution time scaling showing linear recurrence vs. quadratic attention compute costs.

### Conclusion
Attention successfully removes the distance problem of recurrence through direct token-to-token routing, achieving perfect recall over long contexts. However, this comes at the cost of quadratic computational scaling. Furthermore, pre-normalization enables stable deep stack training without learning rate warmup, and sinusoidal encodings extrapolate better to unseen sequence lengths than learned tables
