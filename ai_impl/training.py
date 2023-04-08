from transformer import Transformer
import tensorflow as tf
from dataset import load_and_preprocess_dataset
from sklearn.model_selection import train_test_split
import sys
sys.path.append("/home/cfoye/transformers/ai_impl/datsaet")
from race_dataset import fit_tokenizer_on_race_data
from race_dataset import load_and_preprocess_dataset


tokenizer = tf.keras.preprocessing.text.Tokenizer()

fit_tokenizer_on_race_data()

# Load and preprocess your dataset
input_sequences, target_sequences = load_and_preprocess_dataset() # Implement this function

# Split your dataset into training and validation sets
input_train, input_val, target_train, target_val = train_test_split(input_sequences, target_sequences, test_size=0.1)

# Configure the training
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, dropout_rate=0.1):
model = Transformer(num_layers=2, d_model=512, num_heads=8, d_ff=2048, input_vocab_size=len(tokenizer.word_index) + 1, target_vocab_size=len(tokenizer.word_index) + 1, dropout_rate=0.1)

# Compile the model
model.compile(optimizer=optimizer, loss=loss_function, metrics=[train_accuracy])

# Set batch size and number of epochs
batch_size = 64
epochs = 20

# Train the model
history = model.fit(x=input_train, y=target_train, batch_size=batch_size, epochs=epochs, validation_data=(input_val, target_val))


