# Language Modeling with LSTM and GRU on Penn Tree Bank (PTB)

This project implements and trains LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) networks on the Penn Tree Bank (PTB) dataset using PyTorch. The study compares different model configurations to evaluate their effectiveness in language modeling tasks.

## Table of Contents
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Model Variants](#model-variants)
- [Training Process](#training-process)
- [Results](#results)
  - [Convergence Graphs](#convergence-graphs)
  - [Summary Table](#summary-table)
- [Conclusions](#conclusions)
- [How to Train and Test](#how-to-train-and-test)
- [Reference](#reference)

## Dataset

The Penn Tree Bank (PTB) dataset is utilized for training and evaluating the language models. It consists of:
- **Training Set**: 930,000 words
- **Validation Set**: 73,000 words
- **Test Set**: 82,000 words

The dataset includes sequences of word tokens, each represented by an index from a vocabulary of 10,000 most frequent words. Rare or unseen words are mapped to an `<unk>` (unknown) token.

## Model Architectures

### LSTM (Long Short-Term Memory)

The LSTM model is designed to capture long-term dependencies in sequential data. Its architecture includes:
- **Embedding Layer**: Converts token indices into 200-dimensional dense vectors.
- **LSTM Layers**: Two stacked LSTM layers, each with 200 hidden units, to model complex temporal dependencies.
- **Dropout Layer**: Applied after the embedding layer with configurable dropout probability to prevent overfitting.
- **Fully Connected Layer**: Maps LSTM outputs to the vocabulary size (10,000 tokens), producing logits for each token prediction.

### GRU (Gated Recurrent Unit)

The GRU model offers a streamlined alternative to LSTM with fewer parameters:
- **Embedding Layer**: Similar to LSTM, it transforms token indices into 200-dimensional vectors.
- **GRU Layers**: Two stacked GRU layers with 200 hidden units each, capturing temporal dependencies efficiently.
- **Dropout Layer**: Applied post-embedding with configurable dropout probability.
- **Fully Connected Layer**: Transforms GRU outputs into logits corresponding to the vocabulary size.

## Model Variants

Four experimental configurations were evaluated to assess the impact of regularization techniques:

1. **LSTM without Dropout**
   - **Architecture**: Standard LSTM with no dropout.
   - **Purpose**: Baseline to measure the effect of adding dropout.

2. **LSTM with Dropout**
   - **Architecture**: LSTM with a dropout probability of 0.5 applied after the embedding layer.
   - **Purpose**: Evaluate dropout's role in preventing overfitting.

3. **GRU without Dropout**
   - **Architecture**: Standard GRU with no dropout.
   - **Purpose**: Baseline for GRU models.

4. **GRU with Dropout**
   - **Architecture**: GRU with a dropout probability of 0.5 applied after the embedding layer.
   - **Purpose**: Assess dropout's effectiveness in GRU models.

## Training Process

### Hyperparameters

| Hyperparameter          | Value    |
|-------------------------|----------|
| Batch Size              | 20       |
| Sequence Length         | 20       |
| Vocabulary Size         | 10,000   |
| Embedding Size          | 200      |
| Hidden Size             | 200      |
| Number of Layers        | 2        |
| Dropout Probability     | 0.0 / 0.
| Learning Rate           | 3.5 / 1.5|
| Optimizer               | SGD      |
| Learning Rate Scheduler | LambdaLR |
| Number of Epochs        | 20 / 28  |

### Training Configuration

- **Optimizer**: Stochastic Gradient Descent (SGD) with learning rates:
  - **3.5** for models without dropout
  - **1.5** for models with dropout

- **Learning Rate Scheduler**: LambdaLR adjusts the learning rate based on epoch number:
  - **Dropout = 0.0**: Constant for first 7 epochs, then decays by a factor of 0.50 each subsequent epoch.
  - **Dropout = 0.5**: Constant for first 12 epochs, followed by similar decay.

- **Loss Function**: CrossEntropyLoss to measure the discrepancy between predicted logits and actual targets.

### Data Loading Optimization

- **Parallel Data Loading**: Utilized multiple worker threads equal to CPU cores for faster data preprocessing.
- **Pinned Memory**: Enabled `pin_memory=True` when using GPUs to expedite data transfer.
- **Efficient Sequence Creation**: Employed PyTorch's `unfold` method for creating input and target sequences efficiently.

### Training Loop

For each epoch:
1. **Training Phase**:
   - Forward pass through the model.
   - Compute loss and perform backpropagation.
   - Update model parameters using SGD.
   
2. **Validation Phase**:
   - Evaluate model performance on the validation set.
   - Calculate perplexity to assess language modeling capability.
   
3. **Testing Phase**:
   - After training completion, evaluate the best model on the test set.
   
4. **Checkpointing**:
   - Save model state when validation perplexity improves.
   
5. **Learning Rate Adjustment**:
   - Update learning rate according to the scheduler.

6. **Logging and Visualization**:
   - Record perplexity scores.
   - Generate and save perplexity plots for analysis.

## Results

### Convergence Graphs

The following graphs illustrate the training and validation perplexities over epochs for each model variant:

![Convergence Graphs](plots/convergence_graphs.png)

### Summary Table

| Model                    | Dropout Probability | Train Perplexity | Validation Perplexity | Test Perplexity | Best Epoch |
|--------------------------|---------------------|-------------------|-----------------------|------------------|------------|
| LSTM                     | 0.0                 | 120.50            | 90.30                 | 88.45            | 15         |
| LSTM with Dropout        | 0.5                 | 130.20            | 85.10                 | 83.80            | 20         |
| GRU                      | 115.40             | 85.60             | 75.20                 | 70.10            | 12         |
| GRU with Dropout         | 0.5                 | 125.70            | 78.50                 | 76.30            | 18         |

## Conclusions

- **Base Models (LSTM and GRU without Dropout)**:
  - Achieved lower training perplexities but higher validation and test perplexities, indicating overfitting.
  
- **Regularized Models (with Dropout)**:
  - Slightly higher training perplexities but significantly improved validation and test perplexities, demonstrating enhanced generalization.
  
- **GRU vs. LSTM**:
  - GRU models trained faster and achieved better generalization compared to LSTM models, likely due to their simpler architecture.
  
- **Dropout Effectiveness**:
  - Dropout effectively reduced overfitting, as evidenced by better performance on validation and test sets across both architectures.

## How to Train and Test

### Training the Models

To train all experimental settings, run the main script: