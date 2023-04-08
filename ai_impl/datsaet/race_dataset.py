from datasets import load_dataset
import tensorflow as tf

# Load the RACE dataset with the "middle" configuration
race_dataset = load_dataset("race", "middle")

# Define the tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()

def fit_tokenizer_on_race_data():
    all_texts = []

    for example in race_dataset['train']:
        context = example['article']
        question = example['question']
        answer = example['answer']

        input_text = f"{context} {question}"
        target_text = answer

        all_texts.append(input_text)
        all_texts.append(target_text)

    # Fit the tokenizer on the text data
    tokenizer.fit_on_texts(all_texts)

# Call the function to fit the tokenizer on the RACE dataset
fit_tokenizer_on_race_data()

def load_and_preprocess_dataset():
    input_sequences = []
    target_sequences = []

    for example in race_dataset['train']:
        context = example['article']
        question = example['question']
        answer = example['answer']

        input_text = f"{context} {question}"
        target_text = answer

        # Tokenize the input and target text
        input_tokens = tokenizer.texts_to_sequences([input_text])[0]
        target_tokens = tokenizer.texts_to_sequences([target_text])[0]

        input_sequences.append(input_tokens)
        target_sequences.append(target_tokens)

    return input_sequences, target_sequences

# Call the function to load and preprocess the dataset
input_sequences, target_sequences = load_and_preprocess_dataset()

# Pad input and target sequences to the same length
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, padding='post')
target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, padding='post')
