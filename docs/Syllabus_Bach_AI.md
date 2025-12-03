# Syllabus: Algorithmic Composition in the Style of J.S. Bach

This curriculum moves from classical music theory into statistical modeling and finally deep learning.

## Module 1: The Data (What the AI Sees)
Before an AI can compose, it must understand the representation of music. Bach's music is polyphonic (multiple voices moving independently), typically in four parts: Soprano, Alto, Tenor, Bass (SATB).

### Concepts:
* **Tokenization:** Converting a note (Pitch + Duration) into a number.
* **Vertical vs. Horizontal:**
    * *Horizontal:* The melody of a single voice (e.g., Soprano line).
    * *Vertical:* The harmony formed by all four voices at a specific tick (Chord).
* **Piano Roll:** The matrix format used by most AI models (Time x Pitch).

## Module 2: Probability & Markov Chains (The Code Example)
This is the most effective way to learn *style*. A Markov chain looks at the current note and asks: "Based on all of Bach's history, what is the % chance the *next* note is a C#?"

* **N-Grams:** Looking at sequences. A "1-gram" looks only at the previous note. A "3-gram" looks at the previous 3 notes to decide the 4th.
* **The "Bach" Signature:** The model naturally learns that a "Leading Tone" (e.g., B natural in C major) almost always resolves to the "Tonic" (C), simply because Bach did that 99% of the time.

## Module 3: Deep Learning (DeepBach Framework)
Once you master Markov chains, you move to Neural Networks (NNs). The **DeepBach** framework (developed by Sony CSL) uses a specific architecture called a **Dependency Network**.

### Architecture:
1.  **Bi-Directional LSTM:** Unlike a human who reads left-to-right, DeepBach looks at the *future* notes (generated so far) and the *past* notes to infer the middle note.
2.  **Harmonization:** You can feed it a melody, and it fills in the other 3 voices.

## Module 4: Attention & Transformers (State of the Art)
Modern models (like Music Transformer or MuseNet) use **Self-Attention**.
* **Long-Term Structure:** Markov chains forget what happened 2 bars ago. Transformers can "attend" to a motif played at the beginning of the piece and repeat it at the end, mimicking Bach's fugal structures.

---

### Recommended Frameworks for Students
1.  **music21 (Python):** Best for parsing scores, analyzing intervals, and building datasets. (Used in the attached code).
2.  **Magenta (TensorFlow):** Best for using pre-trained Transformer models.
3.  **Keras/PyTorch:** Best for building your own LSTM from scratch.
