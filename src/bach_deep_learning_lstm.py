import music21
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
import random
import sys

# ==========================================
# MODULE 3: DEEP LEARNING (LSTM)
# ==========================================
# Unlike the Markov Chain, which only looks at the previous note,
# this LSTM (Long Short-Term Memory) network looks at a SEQUENCE
# of previous notes to decide what comes next.

# CONFIGURATION
SEQUENCE_LENGTH = 10  # The AI looks at 10 notes to predict the 11th
EPOCHS = 5            # How many times to practice (In real life, use 50+)
BATCH_SIZE = 128

def get_data_from_corpus():
    """
    Step 1: Load Data
    We load a small collection of Bach chorales to train our neural network.
    """
    print("--- Step 1: Loading Bach Corpus ---")
    # We grab 5 chorales for this demo. In a real project, we'd use 300+.
    chorales = music21.corpus.getComposer('bach')[:5] 
    
    notes = []
    
    for i, file_path in enumerate(chorales):
        print(f"Parsing chorale {i+1}/{len(chorales)}...")
        try:
            score = music21.corpus.parse(file_path)
            # We only use the Soprano line for this simple melody generator
            melody = score.parts[0].flatten().notes
            for note in melody:
                # We store the pitch name (e.g., "C#4")
                # For chords, we'd need more complex logic, so we stick to melody.
                if isinstance(note, music21.note.Note):
                    notes.append(str(note.nameWithOctave))
                elif isinstance(note, music21.chord.Chord):
                    # Simplification: just take the top note of the chord
                    notes.append(str(note.notes[-1].nameWithOctave))
        except:
            continue
            
    print(f"Total notes loaded: {len(notes)}")
    return notes

def prepare_sequences(notes):
    """
    Step 2: Data Preprocessing
    Neural Networks cannot read strings like "C#4". They only read numbers.
    We must create a mapping: "C#4" -> 5, "D4" -> 6, etc.
    """
    print("--- Step 2: Vectorization ---")
    
    # Sort to ensure consistent mapping
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(pitchnames)
    
    # Dictionary to map Note -> Integer
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    
    network_input = []
    network_output = []
    
    # We slide a window across the data:
    # Input: [A, B, C, D] -> Target: E
    for i in range(0, len(notes) - SEQUENCE_LENGTH, 1):
        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        sequence_out = notes[i + SEQUENCE_LENGTH]
        
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
        
    n_patterns = len(network_input)
    
    # Reshape for Keras LSTM layer: (Number of Sequences, Length of Sequence, 1)
    X = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
    
    # Normalize input (neural nets work best with numbers between 0 and 1)
    X = X / float(n_vocab)
    
    # One-hot encode the output (the target note)
    y = tf.keras.utils.to_categorical(network_output)
    
    print(f"Data prepared. {n_patterns} training sequences.")
    print(f"Vocabulary size: {n_vocab} unique notes.")
    
    return X, y, note_to_int, n_vocab, pitchnames

def create_model(input_shape, n_vocab):
    """
    Step 3: The Architecture
    This is the 'Brain'. 
    LSTM layers are special because they have 'memory loops' that allow 
    information to persist, helping capture musical phrasing.
    """
    print("--- Step 3: Building LSTM Model ---")
    model = Sequential()
    
    # LSTM Layer: The core memory unit
    model.add(LSTM(128, input_shape=(input_shape[1], input_shape[2]), return_sequences=False))
    
    # Dense Layer: The decision maker
    model.add(Dense(n_vocab))
    
    # Activation: Softmax turns the output into a probability distribution (Sequence of % chances)
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    return model

def generate_music(model, network_input, note_to_int, pitchnames, n_vocab):
    """
    Step 4: Generation
    1. Pick a random starting sequence.
    2. Ask the AI: "What comes next?"
    3. Add the predicted note to our song.
    4. Slide the window forward and repeat.
    """
    print("--- Step 4: Generating Music ---")
    
    start_index = np.random.randint(0, len(network_input) - 1)
    
    # Mapping back from Integer -> Note
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    
    pattern = network_input[start_index] # This is our "seed" phrase
    prediction_output = []
    
    # Generate 50 notes
    for note_index in range(50):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        
        # The AI predicts probabilities for every possible note
        prediction = model.predict(prediction_input, verbose=0)
        
        # We pick the note with the highest probability
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        
        # Add new note to pattern, remove oldest note (sliding window)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        
    return prediction_output

def create_midi(prediction_output):
    """
    Step 5: Save to file
    """
    offset = 0
    output_notes = []
    
    for pattern in prediction_output:
        # Create Note object
        new_note = music21.note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = music21.instrument.Piano()
        output_notes.append(new_note)
        
        # For simplicity, every note is a quarter note
        offset += 0.5
        
    midi_stream = music21.stream.Stream(output_notes)
    midi_stream.write('midi', fp='bach_lstm_output.mid')
    print("Music saved to 'bach_lstm_output.mid'")

def main():
    # 1. Load Data
    raw_notes = get_data_from_corpus()
    
    if len(raw_notes) < 50:
        print("Not enough data to train. Exiting.")
        return

    # 2. Prepare Data
    X, y, note_map, n_vocab, pitchnames = prepare_sequences(raw_notes)
    
    # 3. Build Model
    model = create_model(X.shape, n_vocab)
    model.summary()
    
    # 4. Train (In a classroom, run for 1-5 epochs to show it working. Real results need 50+.)
    print(f"Training for {EPOCHS} epochs...")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # 5. Generate
    # We need the raw network_input list (ints) for the seed
    # We reconstruct it quickly from X for simplicity
    network_input_list = []
    for sequence in X:
        # Un-normalize back to integers
        seq_ints = [int(x[0] * n_vocab) for x in sequence]
        network_input_list.append(seq_ints)

    generated_notes = generate_music(model, network_input_list, note_map, pitchnames, n_vocab)
    
    # 6. Save
    create_midi(generated_notes)

if __name__ == "__main__":
    main()
