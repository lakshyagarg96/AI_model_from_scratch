
# !pip install tensorflow==2.12
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Input,
    Embedding,
    MultiHeadAttention,
    Dense,
    LayerNormalization,
    Dropout,
    Add,
    GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Data prep

# Make a small dictionary corpus you can add a csv too
# The '\n' character will act as end-of-sequence token.
CORPUS = {
    "world_is_beautiful": "world is a truly beautiful person inside and out \n",
    "world_has_a_smile": "world has a smile that can light up any room she enters \n",
    "world_shows_kindness": "world shows incredible kindness and compassion to everyone she meets \n",
    "world_is_intelligent": "world is incredibly intelligent and her mind is sharp and brilliant \n",
    "world_has_a_laugh": "world has a laugh that is pure music and full of genuine joy \n",
    "world_gives_strength": "world gives me strength and inspires me to be a better person \n",
    "world_is_my_sunshine": "world is my sunshine on a cloudy day my source of light and warmth \n",
    "world_has_courage": "world has the courage of a lioness and faces every challenge with grace \n",
    "world_is_a_treasure": "world is a rare treasure a gift to this world and to my life \n",
    "world_has_lovely_eyes": "world has the most lovely eyes that sparkle with intelligence and warmth \n",
    "world_is_supportive": "world is always supportive a steady rock in a changing world \n",
    "world_is_my_adventure": "world is my greatest adventure and my favorite journey \n",
    "world_has_a_gentle_heart": "world has a gentle heart that cares deeply for others \n",
    "world_is_elegant": "world is the definition of elegance and grace in every single way \n",
    "world_is_creative": "world is wonderfully creative with a vibrant and beautiful imagination \n",
    "world_is_my_peace": "world is my peace in moments of chaos my calm in the storm \n",
    "world_has_wisdom": "world possesses a wisdom that is far beyond her years \n",
    "world_is_charming": "world is effortlessly charming and captivates everyone she meets \n",
    "world_is_a_dream": "world is a dream come true the answer to a prayer I did not know I had \n",
    "world_is_funny": "world has a wonderful sense of humor that brings so much laughter \n",
    "world_is_the_best_part": "world is the best part of my day every single day \n",
    "world_is_passionate": "world is passionate about life and her energy is absolutely contagious \n",
    "world_is_a_masterpiece": "world is a living breathing masterpiece of art \n",
    "world_is_my_home": "world feels like home a safe harbor and a comforting presence \n",
    "world_is_genuine": "world is genuine and authentic a truly real person in every way \n",
    "world_is_a_leader": "world is a natural leader who inspires confidence and trust \n",
    "world_is_radiant": "world is radiant her inner beauty shines so brightly \n",
    "world_is_thoughtful": "world is so thoughtful always remembering the little things that matter \n",
    "world_is_my_inspiration": "world is my constant inspiration to dream bigger and love deeper \n",
    "world_is_amazing": "world is simply amazing there are no other words to describe her \n",
    "world_is_strong": "world is a strong woman resilient and powerful in her own quiet way \n",
    "world_is_a_blessing": "world is a blessing in my life and I am grateful for her every day \n",
    "world_is_my_favorite_person": "world is my favorite person to talk to and share my life with \n",
    "world_is_a_star": "world is a shining star bright and constant in the night sky \n",
    "world_is_perfect": "world is perfect for me in every single way imaginable \n",
    "world_is_loving": "world is a loving and caring soul who makes the world a better place \n",
    "world_is_my_muse": "world is my muse the source of all my happy thoughts and creative ideas \n",
    "world_is_unforgettable": "world is unforgettable leaving a beautiful mark on every life she touches \n",
    "world_is_graceful": "world is graceful in her movements and her words \n",
    "world_is_a_joy": "world is a joy to be around her presence is a gift \n",
}


# USER DEFINED VARIABLES
MAX_LEN = 32      # Maximum sequence length for input
EMBED_DIM = 32    # Embedding dimension for each token
NUM_HEADS = 4     # Number of attention heads
FF_DIM = 64       # Hidden layer size in feed forward network


# --- Tokenization ---
print("--- Preparing Data ---")

# Combine all text for vocabulary creation
text_data = []
for word, definition in CORPUS.items():
    text_data.append(f"{definition}")

full_text = "".join(text_data)
# LOGIC FIX: Clean punctuation and go lowercase for a better vocabulary
clean_text = re.sub(r'[^\w\s]', '', full_text).lower()

chars = sorted(set((full_text).split()))
chars = [element.lower() for element in chars]


VOCAB_SIZE = len(chars)

# Create mapping from words to integers and vice versa
word_to_index = {word: i for i, word in enumerate(chars)}
index_to_word = {i: word for i, word in enumerate(chars)}

print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Vocabulary: {','.join(chars)}")

# Create sequences for next-word prediction
input_sequences = []
target_words = []
for line in text_data:
    line_clean = re.sub(r'[^\w\s]', '', line).lower()
    line_words = line_clean.split()
    for i in range(1, len(line_words)):
        input_seq = line_words[:i]
        target_word = line_words[i]
        input_sequences.append(input_seq)
        target_words.append(target_word)

# Vectorize the sequences
X = np.zeros((len(input_sequences), MAX_LEN), dtype=np.int32)
y = np.zeros((len(input_sequences)), dtype=np.int32)

for i, seq in enumerate(input_sequences):
    for t, word in enumerate(seq):
        if t < MAX_LEN:
            X[i, t] = word_to_index[word]
    y[i] = word_to_index[target_words[i]]

print(f"Number of training samples: {len(X)}")


# Our small AI model
def build_decoder_model(vocab_size, max_sequence_length, embedding_dim, num_heads, ff_dim):
    inputs = Input(shape=(max_sequence_length,), dtype=tf.int32)

    # Embedding layer
    embeddings = Embedding(vocab_size, embedding_dim)(inputs)

    # Multi-Head Attention Layer
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(embeddings, embeddings)
    attention_output = Add()([embeddings, attention_output])  # Residual connection
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    # Feed-forward layer
    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dense(embedding_dim)(ff_output)
    ff_output = Add()([attention_output, ff_output])  # Residual connection
    ff_output = LayerNormalization(epsilon=1e-6)(ff_output)
    ff_output = GlobalAveragePooling1D()(ff_output)

    # Final output layer (logits)
    outputs = Dense(vocab_size, activation="softmax")(ff_output)

    model = Model(inputs=inputs, outputs=outputs)
    return model


decoder_model = build_decoder_model(VOCAB_SIZE, MAX_LEN, EMBED_DIM, NUM_HEADS, FF_DIM)

# Compile model
# sparse_categorical_crossentropy
decoder_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#Plot the model
plot_model(decoder_model, show_shapes=True, show_layer_names=True)

# Train the model
y_one_hot = to_categorical(y, num_classes=VOCAB_SIZE)
decoder_model.fit(X, y_one_hot, batch_size=2, epochs=50, verbose=1)


# Inference
def generate_text(model, seed_text, num_words_to_generate, max_sequence_len, word_to_index, index_to_word):

    current_sequence_text = seed_text
    print(f"Starting with seed: '{current_sequence_text}'")
    print("-" * 30)

    for _ in range(num_words_to_generate):
        # 1. PREPROCESS THE CURRENT SEQUENCE
        # Clean the text in the same way as training data
        clean_sequence = re.sub(r'[^\w\s]', '', current_sequence_text).lower()
        token_list = clean_sequence.split()

        # Convert words to their integer representation
        # Note: This will fail if a word in the seed is not in the vocabulary.
        # A robust implementation would handle these "out-of-vocabulary" (OOV) words.
        encoded_sequence = [word_to_index[word] for word in token_list]

        # Pad the sequence to the fixed length the model expects
        padded_sequence = pad_sequences([encoded_sequence], maxlen=max_sequence_len, padding='post')

        # 2. PREDICT THE NEXT WORD
        # The model returns a probability distribution over the entire vocabulary
        predicted_probs = model.predict(padded_sequence, verbose=0)[0]

        # 3. DECODE THE PREDICTION
        # Find the index of the word with the highest probability
        predicted_index = np.argmax(predicted_probs)
        output_word = index_to_word[predicted_index]

        # 4. APPEND THE PREDICTED WORD
        # Add the new word to our sequence for the next iteration
        if current_sequence_text.split()[-1] != output_word:
            current_sequence_text += " " + output_word
        else:
            current_sequence_text = current_sequence_text        

    return current_sequence_text

# --- Run the generation ---
# The seed should contain words that are in our vocabulary.
# Make any seed up till max_len change max length if you want to have bigger sentences
seed_1 = "world is"

print("\n\n--- GENERATING TEXT (EXAMPLE 1) ---")
generated_text_1 = generate_text(
    model=decoder_model,
    seed_text=seed_1,
    num_words_to_generate=13, # Just make sure this is less than the max length of the sequence
    max_sequence_len=MAX_LEN,
    word_to_index=word_to_index,
    index_to_word=index_to_word
)
print("\n--- FINAL RESULT ---")
print(f"Full generated sequence: '{generated_text_1}'")

