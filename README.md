# Ultrakill RL Framework

This framework allows you to train an AI to play Ultrakill using Reinforcement Learning (PPO) and Imitation Learning (Teaching Mode).

## Prerequisites

1.  **Python 3.8+**: Ensure you have Python installed on your Windows machine.
2.  **Tesseract OCR**:
    - Download and install Tesseract for Windows from [here](https://github.com/UB-Mannheim/tesseract/wiki).
    - Note the installation path (usually `C:\Program Files\Tesseract-OCR\tesseract.exe`).
3.  **Ultrakill**: Run the game in **Windowed Mode** at **1920x1080** resolution.

## Installation

1.  Clone this repository or download the files.
2.  Open a terminal (cmd or PowerShell) in the folder.
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

- `inputs.py`: Handles all keyboard and mouse simulations.
- `vision.py`: Screen capture and OCR for health/score/enemy detection.
- `environment.py`: The Gymnasium environment that connects the AI to the game.
- `teacher.py`: Script to record your gameplay for the AI to learn from.
- `main.py`: The main training loop.

## How to Use

### 1. Training the AI
Run `main.py` while Ultrakill is running and you are in a level.
```bash
python main.py
```
The AI will take control of your mouse and keyboard. **Press 'Ctrl+C' in the terminal to stop.**

### 2. Teaching the AI (Imitation Learning)
If the AI is struggling, you can teach it by recording your own gameplay and then running behavioral cloning.
1. Run `python teacher.py`.
2. Press **'y'** to start recording.
3. Play the game normally.
4. Press **'y'** again to stop recording and save the data to the `recordings/` folder.
5. Run `python imitation_learning.py` to train the AI on your demonstrations.

## Allowed Actions
The AI can use the following keys:
- Movement: `W`, `A`, `S`, `D`, `Space` (Jump), `Shift` (Dash), `Ctrl` (Slide)
- Weapons: `1`, `2`, `3`, `4`, `5`, `6`
- Abilities: `E` (Punch), `Q` (Change Weapon), `R` (Whiplash)
- Combat: `Left Click` (Fire), `Right Click` (Alt-Fire) - Supports holding down.
- Camera: Free mouse movement.

## Warning
The AI takes control of your inputs. Make sure you can easily switch focus or have a way to stop the script (like a hotkey or Ctrl+C in the terminal).
