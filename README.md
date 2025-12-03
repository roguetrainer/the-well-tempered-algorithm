# The Well-Tempered Algorithm
### Algorithmic Composition Workshop

This repository contains an interactive workshop (`Bach_AI_Workshop.ipynb`) designed to teach Algorithmic Composition in the style of J.S. Bach.

## Project Structure
The workshop progresses through three "eras" of AI music history:

1.  **The Probabilistic Era:** Using **Markov Chains** to understand style as a game of dice.
2.  **The Deep Learning Era:** Using **LSTMs** (Recurrent Neural Networks) to gain "memory."
3.  **The Transformer Era:** Using **GPT-2** architectures to treat music as a language.

## How to use this file

### Option 1: Google Colab (Recommended)
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Click **File > Upload Notebook**.
3. Upload the `Bach_AI_Workshop.ipynb` file.
4. Run the cells in order. The environment setup cell will install all necessary libraries (`music21`, `transformers`, etc.) for you.

### Option 2: Local Jupyter Lab
1. Ensure you have Python installed.
2. Install the requirements:
   `pip install notebook music21 tensorflow transformers numpy`
3. Launch Jupyter:
   `jupyter notebook`
4. Open `Bach_AI_Workshop.ipynb`.

## Note on Music21
The code is configured to output text representations of notes (e.g., `['C#4', 'D4', 'E4']`) and MIDI files. To see actual sheet music notation in a notebook, you generally need to install **MuseScore** on your local machine and configure the path. In Google Colab, this is difficult, so we stick to MIDI/Text output for reliability.
