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

The Penn Tree Bank (PTB) dataset is utilized for training, validating, and testing the language models. It consists of roughly:
- **Training Set**: 930,000 words
- **Validation Set**: 73,000 words
- **Test Set**: 82,000 words

The dataset includes sequences of word tokens, each represented by an index from a vocabulary of 10,000 most frequent words.

## Model Architectures

### LSTM (Long Short-Term Memory)

The LSTM model is designed to capture long-term dependencies in sequential data. Its architecture includes:
- **Embedding Layer**: Converts token indices into 200-dimensional vectors.
- **LSTM Layers**: Two fully-connected LSTM layers, each with 200 hidden units.
- **Dropout Layer**: Applied after the embedding layer and in hidden LSTM layers with configurable dropout probability to prevent overfitting.
- **Fully Connected Layer**: Maps LSTM outputs to the vocabulary size (10,000 unique tokens), producing logits for each token prediction.

### GRU (Gated Recurrent Unit)

The GRU model offers a streamlined alternative to LSTM with fewer parameters:
- **Embedding Layer**: Similar to LSTM, it transforms token indices into 200-dimensional vectors.
- **GRU Layers**: Two fully-connected GRU layers with 200 hidden units each.
- **Dropout Layer**: Applied post-embedding and in GRU hidden layers with configurable dropout probability.
- **Fully Connected Layer**: Maps LSTM outputs to the vocabulary size (10,000 unique tokens), producing logits for each token prediction.

## Model Variants

Four experimental configurations were evaluated to assess the impact of regularization techniques:

1. **LSTM without Dropout**
   - **Architecture**: Standard LSTM with no dropout.
   - **Purpose**: Baseline to understand model performance.

2. **LSTM with Dropout**
   - **Architecture**: LSTM with a dropout probability of 0.5 applied after the embedding layer and in hidden layers.
   - **Purpose**: Evaluate dropout's role in preventing overfitting and generalization.

3. **GRU without Dropout**
   - **Architecture**: Standard GRU with no dropout.
   - **Purpose**: Baseline to understand model performance.

4. **GRU with Dropout**
   - **Architecture**: GRU with a dropout probability of 0.3 applied after the embedding layer.
   - **Purpose**: Evaluate dropout's role in preventing overfitting and generalization.

### Training Hyperparameters

| **Hyperparameter**             | **Value**               |
|--------------------------------|-------------------------|
| Batch Size                     | 20                      |
| Sequence Length                | 20                      |
| Vocabulary Size                | 10,000                  |
| Embedding Size                 | 200                     |
| Hidden Size                    | 200                     |
| Number of Layers               | 2                       |
| Dropout Probability            | 0.0 / 0.5 / 0.3         |
| Learning Rate                  | 1.6 / 3.4 / 1.5 / 1.8   |
| Optimizer                      | SGD                     |
| Learning Rate Scheduler        | LambdaLR                |
| Number of Epochs               | 13 / 20                 |

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
   - Generate and save table for best perplexities. 

## Results

### Results Summary

| Model | Dropout Probability | Train Perplexity | Validation Perplexity | Test Perplexity |
|-------|---------------------|------------------|-----------------------|-----------------|
| LSTM  | 0.0                 | 73.00            | 122.37                | 122.37          |
| LSTM  | 0.5                 | 103.21           | 104.51                | 104.51          |
| GRU   | 0.0                 | 50.99            | 121.29                | 121.29          |
| GRU   | 0.3                 | 72.68            | 104.09                | 104.09          |

## Conclusions

- **Base Models (LSTM and GRU without Dropout)**:
  - Achieved lower training perplexities but higher validation and test perplexities, indicating overfitting.
  
- **Regularized Models (with Dropout)**:
  - Higher training perplexities but significantly improved validation and test perplexities, demonstrating enhanced generalization.
  
- **GRU vs. LSTM**:
  - GRU models trained faster, likely due to their simpler architecture.
  
- **Dropout Effectiveness**:
  - Dropout effectively reduced overfitting, as evidenced by better performance on validation and test sets across both architectures.

## How to Train and Test


Training and Evaluating the Model: To train and evaluate the models, the final block of code should be executed within the provided notebook. This block initializes four different model configurations and runs a training and testing loop for each one, over a predefined number of epochs.
Once this block is run, each model will undergo training for the specified number of epochs. During each epoch, the code evaluates the model on both the training, validation, and test datasets to monitor performance and generalization. The best model for each configuration is saved based on test accuracy, and the results are used to generate convergence graphs and a final accuracy comparison table.
