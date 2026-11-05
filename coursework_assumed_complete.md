# Coursework: 02456 Deep Learning (draft assuming all required exercises are finished)

**Status: planning draft, not a record of completed work.** The notebooks do not currently show much of what is described here. Before using a line on a CV, finish that work, replace each `[TODO]` with the real result, and delete anything you did not do. `coursework.md` remains the version based on the current notebooks.

Number sources: "notebook output" = already in the saved notebook. `[TODO]` = no result exists yet, so none has been entered.

### 02456 Deep Learning | Week 1 | Feedforward networks: pen-and-paper derivations
- **What this week covered**: Mathematical basis of feedforward networks: forward equations, loss functions from maximum likelihood, gradient descent, and backpropagation.
- **What I did myself**:
  - Derived the forward equations for one, two, and L layers (from scratch).
  - Derived sum-of-squares and cross-entropy losses from maximum likelihood (with guidance).
  - Derived the softmax cross-entropy gradient and the general backpropagation rule (with guidance).
- **Concepts I can explain**:
  - How forward propagation can be written as a recursion over layers.
  - Why Gaussian noise leads to squared error, and a categorical output leads to cross-entropy.
  - Why taking the negative log of the likelihood does not move its optimum.
  - How the backward recursion for the delta terms reuses values stored in the forward pass.
- **Tech / methods**: Languages: mathematical notation, LaTeX; Libraries: none; Methods: maximum likelihood, chain rule, backpropagation; Tools: Jupyter
- **Results (quantified, only if available)**: None; this is a derivation exercise.
- **Difficulties & how I solved them**: `[TODO]`
- **Possible CV bullet**: Combine with other weeks

### 02456 Deep Learning | Week 2 | Neural network from scratch with Nanograd
- **What this week covered**: Building, testing, and training a neural network with a minimal autodiff framework instead of PyTorch.
- **What I did myself**:
  - Traced `backward()` by hand and explained repeated runs and gradient zeroing (with guidance).
  - Implemented a finite-difference gradient check (with guidance).
  - Implemented the dense-layer forward pass, activation functions, and cross-entropy loss (with guidance).
  - Implemented Glorot and He initialization (with guidance).
  - Wrote a forward-pass unit test and a parameter printing function (from scratch).
  - Trained regression networks and varied their architecture to show underfitting and overfitting (with guidance).
- **Concepts I can explain**:
  - How reverse-mode autodiff stores local derivatives and accumulates gradients.
  - How finite differences can check an autodiff implementation.
  - Why Glorot and He scale initial weights to control signal variance.
  - How to recognize underfitting and overfitting from train, validation, and test loss.
- **Tech / methods**: Languages: Python; Libraries: Nanograd, NumPy, NetworkX, Matplotlib; Methods: automatic differentiation, feedforward networks, gradient descent, Glorot and He initialization, finite differences; Tools: Jupyter
- **Results (quantified, only if available)**: Regression test loss -> `[TODO]`. Underfitted, overfitted, and balanced settings -> `[TODO]`.
- **Difficulties & how I solved them**: `[TODO]`
- **Possible CV bullet**: Implemented and trained a neural network in the Nanograd autodiff framework, including activations, Glorot and He initialization, and finite-difference gradient checks (test loss `[TODO]`).

### 02456 Deep Learning | Week 3 | PyTorch basics and feedforward networks
- **What this week covered**: PyTorch tensors, autograd, and feedforward classifiers on 2D half-moon data and MNIST.
- **What I did myself**:
  - Completed the tensor, NumPy, and autograd assignments (with a library).
  - Extended a half-moon logistic classifier into a neural network and tested depth, width, and optimizers (with guidance).
  - Modified an MNIST classifier: He initialization, a second hidden layer, ReLU, momentum, and Adam (with guidance).
  - Added L2 or L1 regularization, dropout, and batchnorm (with guidance).
- **Concepts I can explain**:
  - Why logistic regression gives a linear boundary, and how a hidden layer removes that limit.
  - How `.grad`, leaf tensors, and `.retain_grad()` behave in autograd.
  - How SGD, momentum, and Adam differ.
  - How weight decay, dropout, and batchnorm affect training, and why `train()` and `eval()` matter.
- **Tech / methods**: Languages: Python; Libraries: PyTorch, torchvision, NumPy, scikit-learn, Matplotlib; Methods: feedforward networks, SGD, momentum, Adam, weight decay, dropout, batchnorm; Tools: Jupyter
- **Results (quantified, only if available)**:
  - MNIST subset, one-hidden-layer baseline -> two hidden layers, ReLU, He, Adam: validation accuracy 0.860 -> 0.884; training accuracy 0.950 -> 1.000 (notebook output).
  - Half-moon logistic regression -> neural network: test accuracy `[TODO]`.
  - MNIST with regularization: validation accuracy `[TODO]`.
- **Difficulties & how I solved them**: `[TODO]`
- **Possible CV bullet**: Tuned a PyTorch MNIST classifier across initialization, depth, optimizer, and regularization, raising validation accuracy from 0.860 to `[TODO]`.

### 02456 Deep Learning | Week 4 | Convolutional networks and transfer learning
- **What this week covered**: Convolution arithmetic, CNN design, CIFAR-10 classification, and transfer learning from ImageNet to Caltech101.
- **What I did myself**:
  - Calculated convolutions, pooling, and output dimensions by hand (with guidance).
  - Tested stacked convolutions, pooling or stride, dropout, and batchnorm on MNIST (with guidance).
  - Designed and trained a CNN for CIFAR-10 and analyzed its confusion matrix and per-class accuracy (with guidance).
  - Adapted a CNN to 224x224 Caltech101 images (with guidance).
  - Fine-tuned an ImageNet-pretrained ResNet34-D, comparing head-only and full-network training (with a library).
- **Concepts I can explain**:
  - How kernel size, stride, and padding determine output dimensions.
  - Why CNNs generally use fewer parameters than dense layers on images.
  - How pooling, dropout, and batchnorm change a network's behavior.
  - What a confusion matrix shows that overall accuracy hides.
  - When to train only the head and when to fine-tune the whole pretrained network.
- **Tech / methods**: Languages: Python; Libraries: PyTorch, torchvision, timm, scikit-learn, seaborn; Methods: CNNs, max pooling, dropout, batchnorm, data augmentation, transfer learning, fine-tuning; Tools: Jupyter
- **Results (quantified, only if available)**:
  - MNIST, feedforward vs. one conv layer (one run each): validation accuracy 0.972 vs. 0.977; test accuracy 0.977 vs. 0.974 (notebook output).
  - CIFAR-10 test accuracy -> `[TODO]`.
  - Caltech101 test accuracy: custom CNN `[TODO]`; head-only ResNet34-D `[TODO]`; fully fine-tuned ResNet34-D `[TODO]`.
- **Difficulties & how I solved them**: `[TODO]`
- **Possible CV bullet**: Compared a custom CNN with head-only and fully fine-tuned ResNet34-D models on Caltech101, reaching `[TODO]` test accuracy, after training a CIFAR-10 CNN to `[TODO]`.

### 02456 Deep Learning | Week 5 | Recurrent networks and LSTMs
- **What this week covered**: Sequence representation and next-token prediction with RNNs and LSTMs, in Nanograd and then PyTorch.
- **What I did myself**:
  - Implemented vocabulary maps and an `RNNLayer` in Nanograd (with guidance).
  - Wrote forward and backward unit tests (from scratch).
  - Implemented He initialization and a sequence loss (with guidance).
  - Completed the RNN training loop and tested text generation on longer sequences (with guidance).
  - Implemented an `LSTMLayer` in Nanograd and compared its learning curve with the RNN (with guidance).
  - Built and trained an LSTM in PyTorch and compared it with the Nanograd version (with a library).
- **Concepts I can explain**:
  - How weights are shared across time steps and how the hidden state carries context.
  - How backpropagation through time works.
  - Why vanilla RNNs suffer from vanishing gradients.
  - What the LSTM's input, forget, and output gates and memory cell each do.
  - Why PyTorch trains far faster than Nanograd.
- **Tech / methods**: Languages: Python; Libraries: Nanograd, NumPy, PyTorch; Methods: RNNs, LSTMs, backpropagation through time, one-hot encoding, unit testing; Tools: Jupyter
- **Results (quantified, only if available)**:
  - Forward unit test -> `[1.0, 3.0, 6.0]`, matching expected values (notebook output).
  - Validation NLL: RNN `[TODO]`; Nanograd LSTM `[TODO]`; PyTorch LSTM `[TODO]`.
- **Difficulties & how I solved them**: `[TODO]`
- **Possible CV bullet**: Implemented RNN and LSTM layers from their equations in a minimal autodiff framework, and compared them with a PyTorch LSTM on next-token prediction (validation NLL `[TODO]`).

### 02456 Deep Learning | Week 6 | Transformers and language models
- **What this week covered**: Tokenization, embeddings, language models, attention, the Transformer architecture, and pre-trained model applications.
- **What I did myself**:
  - Compared four tokenizers on one sentence (ran and analyzed).
  - Ran GloVe nearest-neighbor and word-analogy experiments (ran and analyzed).
  - Implemented the loss for an LSTM language model (with guidance).
  - Implemented left-to-right, bidirectional, and right-to-left attention masks (with guidance).
  - Analyzed attention temperature and self-attention maps (ran and analyzed).
  - Tested GPT-2 generation and prompting, and DPR retrieval (with a library).
- **Concepts I can explain**:
  - The trade-off between vocabulary size and sequence length across tokenizers.
  - Why language-model inputs are shifted right, and how future information leakage is prevented.
  - How queries, keys, and values produce attention weights, and how temperature changes them.
  - How attention masks enforce the dependency structure of a language model.
  - Why Transformers need positional encodings.
- **Tech / methods**: Languages: Python; Libraries: PyTorch, Hugging Face Transformers, Tokenizers, Datasets; Methods: tokenization, GloVe embeddings, LSTM language models, scaled dot-product attention, masked attention, Transformers, GPT-2, DPR; Tools: Jupyter
- **Results (quantified, only if available)**:
  - Tokens for one sentence (vocabulary size): word-level 11 (400,004); WordPiece 15 (28,996); BPE 17 (50,257); character-level 57 (256) (notebook output).
  - LSTM language-model loss -> `[TODO]`. DPR correct retrievals out of 15 questions -> `[TODO]`.
- **Difficulties & how I solved them**: `[TODO]`
- **Possible CV bullet**: Implemented attention masks and an LSTM language-model loss in PyTorch, and evaluated pre-trained GPT-2 and DPR models with Hugging Face (DPR `[TODO]`/15 correct).

## Course summary (rewritten on every run)
### Course summary: 02456 Deep Learning (assuming completion)

**Topics**
- Week 1: Feedforward network math and backpropagation
- Week 2: Neural networks from scratch with Nanograd
- Week 3: PyTorch, optimizers, and regularization
- Week 4: CNNs, CIFAR-10, and transfer learning
- Week 5: RNNs and LSTMs
- Week 6: Attention, Transformers, and pre-trained language models

**Skills and methods by depth**
- **Implemented from scratch**: Feedforward equations on paper; unit tests for the network and recurrent layers. *Reasoning: these exercises start from an empty cell or a task with little or no scaffold.*
- **Implemented with guidance**: Backpropagation derivations; Nanograd network, RNN, and LSTM; CNNs for MNIST, CIFAR-10, and Caltech101; language-model loss; attention masks. *Reasoning: you complete key parts of a provided template.*
- **Applied with a library or ran and analyzed**: PyTorch tensors and autograd; PyTorch LSTM; ResNet34-D fine-tuning; tokenizers; GloVe; GPT-2; DPR. *Reasoning: these rely mainly on existing library models or provided code.*

**Consolidated CV bullets: ML / computer vision internships**
- Trained a custom CNN on CIFAR-10 (`[TODO]` test accuracy) and compared a custom CNN with head-only and fully fine-tuned ResNet34-D models on Caltech101 (`[TODO]` best test accuracy).
- Tuned a PyTorch MNIST classifier across initialization, depth, optimizer, and regularization, raising validation accuracy from 0.860 to `[TODO]`.
- Implemented feedforward, RNN, and LSTM layers in a minimal autodiff framework, then trained an LSTM in PyTorch for next-token prediction (validation NLL `[TODO]`).
- Implemented attention masks and an LSTM language-model loss, and evaluated pre-trained GPT-2 and DPR models with Hugging Face.

**Consolidated CV bullets: research assistant roles**
- Derived backpropagation and common loss functions from first principles, then checked an autodiff implementation with finite differences and unit tests.
- Analyzed CNN errors with confusion matrices and per-class accuracy, and compared training strategies for transfer learning (`[TODO]` results).
- Compared RNN and LSTM learning curves, and two LSTM implementations, using negative log-likelihood and training speed (`[TODO]` results).
- Examined self-attention maps and the effect of temperature on attention weights using GloVe word vectors.

**Gaps: what is still needed before using this**
- **Results:** Complete the work and replace every `[TODO]`; no bullet with a `[TODO]` is usable yet.
- **Depth labels:** Check each label against what you actually wrote.
- **Reproducibility:** Set seeds and repeat key experiments to report a mean and spread.
- **Interview stories:** Add one or two real debugging stories to the "Difficulties" fields.
- **Final project:** Add Project 1 (PINNs) once you have its method, contribution, and metrics.
- **Optional exercises:** Week 1 optional proofs, Nanograd classification and `NeuralNetwork` class, GRU, harder RNN tasks, and other `timm` models are not included.
