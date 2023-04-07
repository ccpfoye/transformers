# Transformer Architecture Adventrues (TAA?)

## Understand the basics:
a. Read the original Transformer paper: "Attention Is All You Need" by Vaswani et al. https://arxiv.org/abs/1706.03762 Blog: https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html
b. Understand key concepts: self-attention mechanism, positional encoding, multi-head attention, and feed-forward neural networks.
- Self Attention
    Self attention has three steps:
    1. Calculate Query, Key and Value vectors
    - Query vector **represents** the current token the model is processing
    - Key vector **represents** the unique characteristics of each token in the sentence
    - Value vector **represents** the value of the current token.
    2. Scoring and Scaling
    - We calculate dot products between the query vector of the current token and all key vectors to calcualte attention
    3. Softmax and Aggregation
    - We score by multiplying the value vectors by the normalized Q*K scores
- 

c. Learn about the overall architecture: encoder-decoder structure, layer normalization, and residual connections.

## Get familiar with TensorFlow:
a. Install TensorFlow and go through the official TensorFlow tutorials.
b. Learn about TensorFlow's core concepts: tensors, variables, computation graphs, and gradient descent.
c. Get hands-on experience with TensorFlow APIs like Keras.

Installing TensorRT: https://docs.nvidia.com/drive/drive-os-5.2.0.0L/drive-os/index.html#page/DRIVE_OS_Linux_SDK_Development_Guide/install_tensorrt.html

## Implement basic building blocks:
a. Implement scaled dot-product attention.
b. Implement multi-head attention.
c. Implement position-wise feed-forward networks.
d. Implement positional encoding.
e. Implement layer normalization.

## Build the Transformer model:
a. Implement the encoder layer and stack multiple encoder layers.
b. Implement the decoder layer and stack multiple decoder layers.
c. Combine the encoder and decoder to create the full Transformer model.

## Training and testing:
a. Choose a dataset for a natural language processing task (e.g., machine translation, text summarization, or sentiment analysis).
b. Preprocess the data: tokenize and create a vocabulary.
c. Implement the training loop, using the appropriate loss function and optimizer.
d. Implement a validation loop to monitor progress and prevent overfitting.
e. Train the model and test its performance on the chosen dataset.

## Experiment with complex features and improvements:
a. Use techniques like learning rate schedules, dropout, and label smoothing to improve the model's performance.
b. Implement BERT, a bidirectional Transformer, and fine-tune it for specific tasks.
c. Implement other variants of the Transformer architecture, such as GPT and T5.
d. Explore advanced applications of Transformers in fields like computer vision and reinforcement learning.

## Stay up-to-date with research:
a. Follow recent publications in natural language processing and Transformer-based models.
b. Participate in online forums, discussions, and conferences to learn from the community.
c. Collaborate with others on open-source projects and contribute to the TensorFlow ecosystem.

## Learn Modern tools
a. Vertex AI w/ Google:
https://www.cloudskillsboost.google/paths/17
