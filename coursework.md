# Coursework: 02456 Deep Learning

Source key: "notebook output" = printed in the saved notebook; "provided by me" = stated by me, not visible in the notebook.
Unfinished or unconfirmed work is marked "TODO: to be filled in" and is not used in the CV bullets.

### 02456 Deep Learning | Week 1 | Feedforward networks: pen-and-paper derivations
- **What this week covered**: Mathematical basis of feedforward networks: forward equations, loss functions from maximum likelihood, gradient descent, and backpropagation.
- **What I did myself**: TODO: to be filled in. The notebook's exercise placeholders are unfilled, so no work is confirmed.
- **Concepts I can explain**: TODO: to be filled in once the derivations are confirmed.
- **Tech / methods**: Languages: TODO; Libraries: TODO; Methods: TODO; Tools: Jupyter
- **Results (quantified, only if available)**: None; this was a derivation exercise.
- **Difficulties & how I solved them**: TODO: to be filled in
- **Possible CV bullet**: TODO: to be filled in

### 02456 Deep Learning | Week 2 | Automatic differentiation with Nanograd
- **What this week covered**: How a minimal autodiff framework stores a computation graph and computes gradients, as a basis for building a network without PyTorch.
- **What I did myself**:
  - Traced `backward()` by hand for a product expression, listing each object's values and the sequence of `backprop` calls (with guidance).
  - Explained why a variable used twice in `f = ab + ad` receives the gradient `b + d = 14` (with guidance).
  - Added derivative labels to two computation-graph plots (with guidance).
  - TODO: to be filled in: forward pass, activations, initialization, cross-entropy, training, and the overfitting exercise are blank in the notebook.
- **Concepts I can explain**:
  - Each `Var` stores its value, its gradient, and its parents with the local derivative for each.
  - `backward()` starts at 1.0 and multiplies by local derivatives along each path to the inputs.
  - Gradients add up when a variable reaches the output through several paths.
- **Tech / methods**: Languages: Python; Libraries: Nanograd, NetworkX, Matplotlib; Methods: reverse-mode automatic differentiation, computation graphs; Tools: Jupyter
- **Results (quantified, only if available)**: Hand-explained gradient `df/da = 14` -> matches the printed `grad=14.0000` (notebook output). No trained model.
- **Difficulties & how I solved them**: TODO: to be filled in
- **Possible CV bullet**: Combine with other weeks

### 02456 Deep Learning | Week 3 | PyTorch basics and feedforward networks
- **What this week covered**: PyTorch tensors, autograd, and feedforward classifiers on 2D half-moon data and MNIST.
- **What I did myself**:
  - Created and converted tensors between PyTorch and NumPy, and observed that they share memory (with a library).
  - Derived the 4.5 autograd gradient by hand with the chain rule (with guidance).
  - Compared `.grad` on leaf and non-leaf tensors, before and after `.retain_grad()` (with a library).
  - Modified an MNIST classifier: He initialization, a second hidden layer, ReLU, and Adam (with guidance).
  - Annotated the network code, including how the weight matrix shape follows from input and hidden sizes.
  - TODO: to be filled in: 3.3 Half Moon was not modified or run; momentum, weight decay, dropout, and batchnorm in 3.4 are not done.
- **Concepts I can explain**:
  - Chain rule behind `d(out)/dx = 4.5` for the autograd example.
  - Why non-leaf tensors have no `.grad` by default, and what `.retain_grad()` changes.
  - Why `W` has `num_hidden` rows and `num_features` columns in a dense layer.
  - Why training accuracy of 1.000 with validation accuracy of 0.884 points to overfitting.
- **Tech / methods**: Languages: Python; Libraries: PyTorch, torchvision, NumPy, scikit-learn, Matplotlib; Methods: feedforward networks, autograd, Xavier and He initialization, SGD, Adam, cross-entropy loss; Tools: Jupyter
- **Results (quantified, only if available)**:
  - MNIST subset (1,000 train, 500 validation), epoch 191 of 200: baseline (one hidden layer, ELU, Xavier, SGD) -> modified (two hidden layers, ReLU, He, Adam). Validation accuracy 0.860 -> 0.884; training accuracy 0.950 -> 1.000 (notebook output).
  - All changes were applied together in a single run, so the effect of each change is unknown. No test accuracy was computed.
- **Difficulties & how I solved them**: Found that `b.grad` and `c.grad` were `None` after `backward()`, and used `.retain_grad()` to keep the gradient on `b` (notebook code and printed note).
- **Possible CV bullet**: Modified a PyTorch MNIST classifier (He initialization, extra hidden layer, ReLU, Adam), changing validation accuracy from 0.860 to 0.884 on a 1,000-image training subset while training accuracy reached 1.000.

### 02456 Deep Learning | Week 4 | Convolutional networks
- **What this week covered**: Convolution arithmetic, adding convolutional layers to a classifier, then CIFAR-10 classification and transfer learning on Caltech101.
- **What I did myself**:
  - Calculated a convolution and a 2x2 max pooling by hand (with guidance).
  - Calculated output dimensions for 5 convolution configurations (with guidance).
  - Added a 32-filter 5x5 convolution with padding to an MNIST classifier, trained it, and compared it with the feedforward baseline (with guidance).
  - TODO: to be filled in: stacked convolutions, pooling, dropout, and batchnorm (4.1 exercises 2.2-2.6), plus all of 4.2 CIFAR-10 and 4.3 transfer learning, are blank or not run in the notebooks.
- **Concepts I can explain**:
  - How kernel size, stride, and padding determine the output height and width.
  - Why the number of filters sets the number of output channels.
  - How padding=2 keeps a 28x28 input at 28x28 with a 5x5 kernel, and how this sets the input size of the following linear layer.
  - Why one run each with a 0.003 test difference does not show that one model generalizes better.
- **Tech / methods**: Languages: Python; Libraries: PyTorch, torchvision, scikit-learn, Matplotlib, seaborn; Methods: 2D convolution, max pooling, Adam, cross-entropy loss, train/validation/test evaluation; Tools: Jupyter
- **Results (quantified, only if available)**: MNIST, feedforward -> one conv layer added (one run each): validation accuracy 0.972 -> 0.977; test accuracy 0.977 -> 0.974 (notebook output).
- **Difficulties & how I solved them**: TODO: to be filled in
- **Possible CV bullet**: Compared feedforward and convolutional MNIST classifiers in PyTorch on held-out data (validation accuracy 0.972 vs. 0.977; test accuracy 0.977 vs. 0.974; one run each).

### 02456 Deep Learning | Week 5 | Recurrent networks
- **What this week covered**: Representing token sequences and building recurrent models for next-token prediction, first in Nanograd and then in PyTorch.
- **What I did myself**:
  - Completed `sequences_to_dicts`, building token-to-index and index-to-token maps (with guidance).
  - Implemented `RNNLayer` in Nanograd by filling in the template's blanks, with the layer definitions and step-by-step hidden-state passing (with guidance; provided by me).
  - Wrote a forward-pass unit test with constant weights and identity activation (from scratch).
  - Annotated the data, layer, and one-hot encoding code to explain how each part works.
  - TODO: to be filled in: He initialization, sequence loss, RNN training, LSTM, PyTorch LSTM, and the backward unit test are blank in the notebook.
- **Concepts I can explain**:
  - How one input layer and one recurrent layer are reused at every time step, so weights are shared across time.
  - The difference between `forward_step` (one time step) and `forward_sequence` (loops over the sequence and passes the hidden state along).
  - Why the recurrent layer has no bias: the input layer already provides one.
  - Why the test expects `[1, 3, 6]`: with unit weights and identity activations, the hidden state is a running sum of the inputs.
- **Tech / methods**: Languages: Python; Libraries: Nanograd, NumPy, PyTorch `Dataset`; Methods: RNNs, one-hot encoding, unit testing; Tools: Jupyter
- **Results (quantified, only if available)**: Forward unit test -> output `[1.0, 3.0, 6.0]`, matching the expected values (notebook output). No trained model.
- **Difficulties & how I solved them**: TODO: to be filled in
- **Possible CV bullet**: Implemented a recurrent layer in the Nanograd autodiff framework and verified its forward pass with a unit test against hand-calculated values.

### 02456 Deep Learning | Week 6 | Transformers and language models
- **What this week covered**: Tokenization, embeddings, language models, attention, the Transformer architecture, and pre-trained model applications.
- **What I did myself**:
  - Ran and compared four tokenizers on one sentence (ran and analyzed).
  - Ran the embedding lookup example and annotated it (ran and analyzed).
  - TODO: to be filled in: exercises 1-12, including the language-model loss and attention masks, are unanswered, and later cells were not run.
- **Concepts I can explain**:
  - The trade-off between vocabulary size and sequence length across word, sub-word, and character tokenizers.
  - Why a word-level tokenizer maps out-of-vocabulary words to an unknown token, while sub-word tokenizers split them.
  - How an embedding lookup equals multiplying a one-hot token vector by the embedding matrix.
- **Tech / methods**: Languages: Python; Libraries: PyTorch, Hugging Face Transformers, Tokenizers, GloVe vectors; Methods: word-level, WordPiece, BPE, and character-level tokenization, embeddings; Tools: Jupyter
- **Results (quantified, only if available)**: Same sentence, tokens produced (vocabulary size): word-level 11, including 2 unknown (400,004); WordPiece 15 (28,996); BPE 17 (50,257); character-level 57 (256) (notebook output).
- **Difficulties & how I solved them**: TODO: to be filled in
- **Possible CV bullet**: Combine with other weeks

## Course summary (rewritten on every run)
### Course summary: 02456 Deep Learning

**Topics so far**
- Week 1: Feedforward network math and backpropagation (no work confirmed yet)
- Week 2: Automatic differentiation with Nanograd
- Week 3: PyTorch tensors and autograd; feedforward classifiers on MNIST
- Week 4: Convolution arithmetic; feedforward vs. convolutional comparison on MNIST
- Week 5: Recurrent layer implementation in Nanograd
- Week 6: Tokenization and embeddings

**Skills and methods by depth**
- **Implemented from scratch**: Unit test for the RNN forward pass. *Reasoning: the notebook gave an empty cell and a task description, so it is the only code with no scaffold.*
- **Implemented with guidance**: `RNNLayer` and vocabulary maps in Nanograd; hand-traced backpropagation; convolution arithmetic; MNIST classifier modifications; adding a convolutional layer. *Reasoning: you completed or changed key parts of a provided template or exercise.*
- **Applied with a library or ran and analyzed**: PyTorch tensors and autograd; tokenizer comparison; embedding lookup. *Reasoning: you used existing APIs or ran provided code and studied the output.*

**Consolidated CV bullets: ML / computer vision internships**
- Implemented a recurrent layer in the Nanograd autodiff framework and verified its forward pass with a unit test against hand-calculated values.
- Modified a PyTorch MNIST classifier (He initialization, extra hidden layer, ReLU, Adam), changing validation accuracy from 0.860 to 0.884 on a 1,000-image training subset while training accuracy reached 1.000.
- Compared feedforward and convolutional MNIST classifiers in PyTorch on held-out data (validation accuracy 0.972 vs. 0.977; test accuracy 0.977 vs. 0.974; one run each).
- Compared word-level, WordPiece, BPE, and character-level tokenizers on one sentence, measuring 11 to 57 tokens across vocabularies of 256 to 400,004 entries.

**Consolidated CV bullets: research assistant roles**
- Traced reverse-mode automatic differentiation by hand through computation graphs and checked an analytic chain-rule gradient against PyTorch autograd output.
- Evaluated a feedforward and a convolutional MNIST classifier on separate validation and test sets, where validation accuracy rose (0.972 to 0.977) and test accuracy fell (0.977 to 0.974) across one run each.
- Designed a deterministic unit test (constant weights, identity activation) that checks an RNN forward pass against hand-calculated values.

**Gaps: what would make this material stronger**
- **Trained models:** Finish RNN or LSTM training and record the loss curves; this is the largest gap in Week 5.
- **CIFAR-10 and transfer learning:** Complete 4.2 and 4.3. Test accuracy and head-only vs. full fine-tuning would be the most relevant results for computer vision roles.
- **Better experiments:** Change one factor at a time in the MNIST work, and repeat runs with several seeds to report a mean and spread. No seeds were set in those experiments, so reproducibility can't be claimed yet.
- **Regularization:** Test dropout, weight decay, and batchnorm in 3.4 or 4.1, since the 3.4 model reached 1.000 training accuracy.
- **Attention:** Implement the attention masks and language-model loss in 6.1.
- **Week 1:** Confirm which derivations you completed, if any exist on paper.
- **Final project:** Project 1 (PINNs, per the brief's file name), covering all parts of the work (provided by me). No project code or results are in this folder, so it is not in the bullets yet. Add the problem, method, your contribution, and final metrics when available.
- **Interview stories:** Add one or two real debugging stories; most "Difficulties" fields are still TODO.
