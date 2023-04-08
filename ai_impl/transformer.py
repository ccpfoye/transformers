"""
## Build the Transformer model:
a. Implement the encoder layer and stack multiple encoder layers.
b. Implement the decoder layer and stack multiple decoder layers.
c. Combine the encoder and decoder to create the full Transformer model.
"""

import tensorflow as tf

from blocks import MultiHeadAttention, PositionWiseFeedForward, PositionalEncoding, LayerNormalization


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, target_vocab_size, dropout_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """
        inp: (batch_size, input_seq_len)
        tar: (batch_size, target_seq_len)
        enc_padding_mask: (batch_size, 1, 1, input_seq_len)
        look_ahead_mask: (batch_size, 1, target_seq_len, target_seq_len)
        dec_padding_mask: (batch_size, 1, 1, input_seq_len)
        """
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights


"""
Encoders
"""

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm_1 = LayerNormalization(epsilon=1e-6)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm_2 = LayerNormalization(epsilon=1e-6)
        
    def call(self, x, mask, training):
        """
        x: (batch_size, input_seq_len, d_model)
        mask: (batch_size, 1, 1, input_seq_len)
        """
        attn_output, _ = self.attention(x, x, x, mask)
        attn_output = self.dropout_1(attn_output, training=training)
        out1 = self.layer_norm_1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout_2(ffn_output, training=training)
        return self.layer_norm_2(out1 + ffn_output)
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout_rate)
        self.enc_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, x, training, mask):
        """
        x: (batch_size, input_seq_len)
        mask: (batch_size, 1, 1, input_seq_len)
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training)
        return x


"""
Decoders 
"""

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.ln_1 = LayerNormalization(epsilon=1e-6)
        self.ln_2 = LayerNormalization(epsilon=1e-6)
        self.ln_3 = LayerNormalization(epsilon=1e-6)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.dropout_3 = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, x, enc_output, look_ahead_mask, padding_mask, training):
        """
        x: (batch_size, target_seq_len, d_model)
        enc_output: (batch_size, input_seq_len, d_model)
        look_ahead_mask: (batch_size, 1, target_seq_len, target_seq_len)
        padding_mask: (batch_size, 1, 1, input_seq_len)
        """
        attn1, attn_weights_block1 = self.self_attention(x, x, x, look_ahead_mask)
        attn1 = self.dropout_1(attn1, training=training)
        out1 = self.ln_1(attn1 + x)
        attn2, attn_weights_block2 = self.attention(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout_2(attn2, training=training)
        out2 = self.ln_2(attn2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout_3(ffn_output, training=training)
        out3 = self.ln_3(ffn_output + out2)
        return out3, attn_weights_block1, attn_weights_block2
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, target_vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout_rate)
        self.dec_layers = [DecoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        x: (batch_size, target_seq_len)
        enc_output: (batch_size, input_seq_len, d_model)
        look_ahead_mask: (batch_size, 1, target_seq_len, target_seq_len)
        padding_mask: (batch_size, 1, 1, input_seq_len)
        """
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask, training)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        return x, attention_weights


