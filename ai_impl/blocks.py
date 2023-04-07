"""
Basic Building Blocks for the Neural Network

a. Implement scaled dot-product attention.
b. Implement multi-head attention.
c. Implement position-wise feed-forward networks.
d. Implement positional encoding.
e. Implement layer normalization.

"""

import tensorflow as tf



"""
Scaled Dot-Product Attention

a. Compute the dot products of the query matrix Q and key matrix K, and divide each by the square root of the dimensionality of K.
b. Apply the softmax function to the scaled dot-product to obtain the attention weights.
c. Multiply the attention weights by the value matrix V to obtain the output of the attention layer.

"""
class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def call(self, Q, K, V, mask):
        """
        Q: (..., seq_len_q, d_k)
        K: (..., seq_len_k, d_k)
        V: (..., seq_len_v, d_v)
        mask: (..., seq_len_q, seq_len_k)
        """
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        scores.masked_fill(mask == 0, -1e9)
        attention = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(attention, V)
        return context, attention
    

"""
Multi-Head Attention

a. Split the d_model dimension into multiple heads.
b. Apply scaled dot-product attention to each head.
c. Concatenate the heads and apply a final linear layer.

"""
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.WQ = tf.keras.layers.Dense(d_model)
        self.WK = tf.keras.layers.Dense(d_model)
        self.WV = tf.keras.layers.Dense(d_model)
        self.linear = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """
        x: (batch_size, seq_len, d_model)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, Q, K, V, mask):
        """
        Q: (batch_size, seq_len_q, d_model)
        K: (batch_size, seq_len_k, d_model)
        V: (batch_size, seq_len_v, d_model)
        mask: (batch_size, 1, 1, seq_len_q)
        """
        batch_size = tf.shape(Q)[0]
        Q = self.WQ(Q)
        K = self.WK(K)
        V = self.WV(V)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        context, attention = ScaledDotProductAttention(self.d_k)(Q, K, V, mask)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.d_model))
        output = self.linear(context)
        return output, attention

"""
Position-wise Feed-Forward Networks

a. Two linear transformations with a ReLU activation in between.
b. Position-wise means that each position in the sequence is treated independently and the same way.

"""
class PositionWiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.linear_1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.linear_2 = tf.keras.layers.Dense(d_model)
        
    def call(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        return self.linear_2(self.linear_1(x))
    

"""
Positional Encoding

a. Add positional encodings to the input embeddings at the bottoms of the encoder and decoder stacks.
b. The positional encodings have the same dimension as the embeddings, so that the two can be summed.
c. The positional encoding vector is added to the embedding vector.
d. PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
e. PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
f. The positional encodings have the same dimension as the embeddings, so that the two can be summed.
g. The positional encoding vector is added to the embedding vector.

"""
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pos_encoding = self.positional_encoding(max_len, d_model)
        
    def get_angles(self, pos, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(tf.range(position)[:, tf.newaxis], tf.range(d_model)[tf.newaxis, :], d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]


"""
Layer Normalization

a. Normalize the activations of the previous layer at each position.
b. i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
c. It is implemented as a layer rather than a function.
d. It is initialized with two new parameters, gamma and beta, which are learned.
e. gamma is initialized to 1 and beta is initialized to 0.
f. The output of the layer normalization is gamma * x + beta.

"""
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6):
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon
        
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=tf.ones_initializer(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=tf.zeros_initializer(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
        
    def call(self, x):
        mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta
    
    def get_config(self):
        config = super(LayerNormalization, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)