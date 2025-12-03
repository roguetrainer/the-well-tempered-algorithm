import music21
import random
import sys

# ==========================================
# EDUCATIONAL CODE: BACH STYLE GENERATOR
# ==========================================
# This script demonstrates the "Markov Chain" approach to style generation.
# It teaches the AI to write music by calculating the probability of 
# one note following another, based entirely on real Bach data.

def get_bach_corpus():
    """
    Step 1: The Dataset.
    We load a specific Bach chorale from the music21 corpus.
    In a real AI training scenario, we would load hundreds of these.
    """
    print("--- Step 1: Loading Bach Data ---")
    try:
        # BWV 66.6 is a standard four-part chorale (SATB)
        bach_score = music21.corpus.parse('bach/bwv66.6')
        print(f"Successfully loaded: {bach_score.metadata.title}")
        return bach_score
    except Exception as e:
        print(f"Error loading corpus: {e}")
        sys.exit(1)

def analyze_style(score):
    """
    Step 2: Analysis (Training).
    We 'read' the Soprano line (the melody) and build a dictionary
    of transitions. This is the 'Brain' of our simple AI.
    
    The Brain looks like this:
    {
        'C4': ['D4', 'D4', 'E4'],  # If we are on C4, 66% chance of D4, 33% of E4
        'D4': ['E4', 'G4']
    }
    """
    print("--- Step 2: Analyzing Style (Training) ---")
    
    # Extract the Soprano part (Part 0)
    soprano_part = score.parts[0]
    
    # Flatten the score to a simple list of notes (ignoring rests/chords for simplicity)
    notes = soprano_part.flatten().notes
    
    transition_dict = {}
    
    # We iterate through the melody, looking at pairs of notes (Current -> Next)
    # This is a "1st Order Markov Chain". 
    # PEDAGOGY NOTE: To make the AI smarter, we would look at triplets (Prev, Current -> Next).
    for i in range(len(notes) - 1):
        current_note = notes[i]
        next_note = notes[i+1]
        
        # We use the note name and octave (e.g., "C#4") as our data token
        current_name = current_note.nameWithOctave
        next_name = next_note.nameWithOctave
        
        if current_name not in transition_dict:
            transition_dict[current_name] = []
        
        transition_dict[current_name].append(next_name)
        
    print(f"Training complete. Learned transitions for {len(transition_dict)} unique notes.")
    return transition_dict

def generate_music(transition_dict, start_note='A4', length=20):
    """
    Step 3: Generation (Inference).
    We start with a seed note, then roll the dice based on our dictionary
    to pick the next note. We repeat this 'length' times.
    """
    print(f"--- Step 3: Generating New Melody ({length} notes) ---")
    
    current_note = start_note
    generated_melody = [current_note]
    
    for _ in range(length):
        if current_note in transition_dict:
            # PROBABILITY IN ACTION:
            # If 'C4' appears 10 times in the dict, and 9 times it goes to 'D4',
            # random.choice naturally picks 'D4' 90% of the time.
            next_note = random.choice(transition_dict[current_note])
            generated_melody.append(next_note)
            current_note = next_note
        else:
            # Dead end: The AI reached a note it never saw in training.
            # Fallback: Pick a random note from the known keys to restart.
            current_note = random.choice(list(transition_dict.keys()))
            generated_melody.append(current_note)
            
    return generated_melody

def save_and_show(melody_list):
    """
    Step 4: Realization.
    Convert our list of text strings back into a Music21 Stream 
    so we can hear it or see the sheet music.
    """
    print("--- Step 4: Creating Score ---")
    stream = music21.stream.Stream()
    stream.metadata = music21.metadata.Metadata()
    stream.metadata.title = "AI Generated Bach Motif"
    stream.metadata.composer = "Python & J.S. Bach"
    
    for note_name in melody_list:
        n = music21.note.Note(note_name)
        # We give every note a quarter length for simplicity
        n.quarterLength = 1.0 
        stream.append(n)
        
    print("Done! Resulting Melody:")
    print(melody_list)
    
    # This will generate a MIDI file in your directory
    output_file = 'bach_generated.mid'
    stream.write('midi', fp=output_file)
    print(f"Saved to {output_file}")
    
    # If you have MuseScore or Finale installed, this command would open it:
    # stream.show() 

def main():
    print("Welcome to the Bach Style Learner.")
    
    # 1. Load Data
    score = get_bach_corpus()
    
    # 2. Train Model
    brain = analyze_style(score)
    
    # 3. Generate
    # We try to start on a note that we know exists in the key (e.g., 'A4' for A major/minor)
    # If the starting note isn't in the specific chorale, we pick a random start.
    start_seed = list(brain.keys())[0] 
    new_melody = generate_music(brain, start_note=start_seed, length=32)
    
    # 4. Output
    save_and_show(new_melody)

if __name__ == "__main__":
    main()
