# Random Forest Classifier

## Description
This project is about a Pacman agent operating in a deterministic world.

The Pacman code used was developed at *UC Berkeley* for their AI course and made available to everyone. The homepage for the Berkeley AI Pacman projects is [here](http://ai.berkeley.edu/).

<br>
<br>

## Running the program

#### Basic command
To run the program, navigate to the directory containing the project and run

    python pacman.py

The command will automatically open a GUI to see Pacman operating in the world. 

#### Random Forest Classifier
To run the Random Forest Classifier run

    python pacman.py --pacman ClassifierAgent

#### Training Set
A training set is provided in the file `good-moves.txt`. Each line in the file is a vector identifying a state-action pair:
- All the characters but the last one are the features describing the world's state (note that they can assume value {0, 1});
- The last character indicates the action taken from the state described by the features. The available actions are:
  - 0 = *North*
  - 1 = *East*
  - 2 = *South*
  - 3 = *West*


#### Overriding the Training Set
The provided training set may be overridden using the command

    python pacman.py --p TraceAgent

This command will allow to control Pacman using the keyboard. 

The various states of the game (together with the action performed) will be registered and saved as a vector in `good-moves.txt`. This data is written to moves.txt. 

> **IMPORTANT:** This will not append new text to the `good-moves.txt` file, rather, it will override it !!

<br>
<br>

## Limitations
The program uses **Python 2.7**, running this program using **Python 3.x** will result in an error.