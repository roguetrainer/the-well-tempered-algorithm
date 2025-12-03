import music21
import numpy as np
import tensorflow as tf
# We use the 'transformers' library, the standard for modern AI
from transformers import TFGPT2LMHeadModel, GPT2Config

# ==========================================
# MODULE 3: THE TRANSFORMER (GPT-2 Style)
# ==========================================
# Instead of an LSTM, we use a Transformer.
# Transformers don't read left-to-right; they use "Attention"
# to look at all notes at once and understand relationships.

# We treat music exactly like a language.
# Notes = Words.
# Phrases = Sentences.

SEQ_LENGTH = 32  # Transformers can handle much longer contexts
BATCH_SIZE = 16
EPOCHS = 10      # Transformers learn faster but need more epochs for style

def load_bach_as_text():
    """
    Step 1: Music to Text
    Transformers expect text (tokens). We convert Bach into a string format.
    Example: "C4 1.0 D4 0.5 E4 0.5"
    """
    print("--- Step 1: Loading & Tokenizing Bach ---")
    chorales = music21.corpus.getComposer('bach')[:10] # Use 10 chorales
    
    # We will build a vocabulary (our "Dictionary" of valid notes)
    # We use a set to ensure unique items
    vocab = set()
    dataset_strings = []
    
    for file in chorales:
        score = music21.corpus.parse(file)
        melody = score.parts[0].flatten().notes
        
        # Convert melody to a single string of tokens
        # We combine Pitch and Duration into one token for simplicity
        # e.g., "C4_1.0"
        tokens = []
        for n in melody:
            if isinstance(n, music21.note.Note):
                tok = f"{n.nameWithOctave}_{n.quarterLength}"
                tokens.append(tok)
                vocab.add(tok)
            elif isinstance(n, music21.chord.Chord):
                # Pick top note
                tok = f"{n.notes[-1].nameWithOctave}_{n.quarterLength}"
                tokens.append(tok)
                vocab.add(tok)
        
        dataset_strings.append(tokens)
        
    # Add special tokens for the model
    vocab.add("[PAD]") # Padding (filler)
    vocab.add("[BOS]") # Beginning of Song
    
    # Create mapping: Token -> ID
    vocab_list = sorted(list(vocab))
    token_to_id = {t: i for i, t in enumerate(vocab_list)}
    id_to_token = {i: t for i, t in enumerate(vocab_list)}
    
    print(f"Vocabulary Size: {len(vocab)} unique Note_Duration pairs.")
    return dataset_strings, token_to_id, id_to_token

def prepare_tensors(dataset_strings, token_to_id):
    """
    Step 2: Preparing Tensors for Hugging Face
    We convert our lists of strings into the format GPT-2 expects.
    """
    input_ids = []
    labels = []
    
    # Create sliding windows
    for song in dataset_strings:
        for i in range(len(song) - SEQ_LENGTH):
            window = song[i : i + SEQ_LENGTH]
            target_token = song[i + 1 : i + SEQ_LENGTH + 1] # Next token prediction
            
            # Convert to IDs
            win_ids = [token_to_id[t] for t in window]
            label_ids = [token_to_id[t] for t in target_token]
            
            input_ids.append(win_ids)
            labels.append(label_ids)
            
    return np.array(input_ids), np.array(labels)

def build_gpt_model(vocab_size):
    """
    Step 3: The Transformer Architecture
    We configure a tiny GPT-2 model.
    """
    print("--- Step 3: Initializing GPT-2 Config ---")
    
    # Configuration for a small model (NanoGPT)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=SEQ_LENGTH,
        n_ctx=SEQ_LENGTH,
        n_embd=128,      # Embedding dimension (Size of vector for each note)
        n_layer=4,       # Number of Transformer blocks (Depth)
        n_head=4,        # Number of Attention Heads (Parallel focus)
        bos_token_id=0,
        eos_token_id=0
    )
    
    # Initialize the model from the config (Not pre-trained, random weights)
    model = TFGPT2LMHeadModel(config)
    
    # Compile with standard optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer=optimizer, loss=loss)
    return model

def generate_gpt_music(model, start_token_str, token_to_id, id_to_token):
    """
    Step 4: Generation with Temperature
    """
    print(f"--- Step 4: Generating from seed '{start_token_str}' ---")
    
    # Encode seed
    if start_token_str not in token_to_id:
        # Fallback if seed not in vocab
        start_id = list(token_to_id.values())[0]
    else:
        start_id = token_to_id[start_token_str]
        
    input_ids = [start_id]
    
    # Generate 32 tokens
    # We implement a simple loop. Hugging Face has a .generate() method,
    # but doing it manually helps students understand the loop.
    for _ in range(32):
        # Prepare input tensor shape: (1, current_length)
        input_tensor = np.array([input_ids])
        
        # Get predictions
        outputs = model(input_tensor)
        predictions = outputs.logits
        
        # Get the logits for the last token
        last_token_logits = predictions[0, -1, :]
        
        # Apply Softmax to get probabilities
        probs = tf.nn.softmax(last_token_logits).numpy()
        
        # Sample from the distribution (Temperature = 1.0)
        # This adds "Creativity" vs just picking the max
        next_id = np.random.choice(len(probs), p=probs)
        
        input_ids.append(next_id)
        
        # Stop if sequence gets too long (optional)
        if len(input_ids) >= SEQ_LENGTH:
            # Shift window if needed, but for this demo we just stop or keep appending
            # Note: GPT-2 has a fixed context window.
            pass

    # Decode back to notes
    generated_tokens = [id_to_token[i] for i in input_ids]
    return generated_tokens

def save_to_midi(tokens):
    """
    Step 5: Render
    """
    s = music21.stream.Stream()
    for t in tokens:
        try:
            # Token format: "NoteName_Duration"
            parts = t.split('_')
            note_name = parts[0]
            duration_val = float(parts[1])
            
            n = music21.note.Note(note_name)
            n.quarterLength = duration_val
            s.append(n)
        except:
            continue
            
    s.write('midi', fp='bach_gpt_output.mid')
    print("Saved to bach_gpt_output.mid")

def main():
    # 1. Load
    dataset, t2i, i2t = load_bach_as_text()
    
    # 2. Prepare
    X, y = prepare_tensors(dataset, t2i)
    
    # 3. Build
    model = build_gpt_model(len(t2i))
    model.build(input_shape=(None, SEQ_LENGTH))
    model.summary()
    
    # 4. Train
    # Note: Transformers usually need massive data. 
    # With 10 chorales, this will likely overfit (memorize).
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # 5. Generate
    seed_note = list(t2i.keys())[5] # Pick a random note as start
    new_song = generate_gpt_music(model, seed_note, t2i, i2t)
    print("Generated Tokens:", new_song)
    
    # 6. Save
    save_to_midi(new_song)

if __name__ == "__main__":
    main()
