CS 370: Pirate Intelligent Agent

Trevor Bescherer | CS 370 Current and Emerging Trends in Computer Science | Southern New Hampshire University

This repository contains TreasureHuntGame.ipynb, a deep Q-learning agent that trains a pirate non-player character (NPC) to find treasure in an 8x8 maze before the human player does.

Project Overview
What code was I given?

The course provided most of the environment and the neural network:

TreasureMaze.py: the maze environment. It tracks the pirate's position, lists valid moves, applies actions, and returns rewards (a large reward for the treasure, small penalties for each step, revisiting cells, and hitting walls).
GameExperience.py: the experience replay class that stores past moves and returns training batches, including target Q-values from a separate target network.
The model and helpers in the notebook: the neural network (two hidden layers of 64 neurons with PReLU activations, a linear output layer for the four directions, Adam optimizer, mean squared error loss), the train_step() function, the maze definition, and the play_game() and completion_check() functions used to test the agent.
What code did I create?

I implemented the qtrain() function, the deep Q-learning training loop. For each epoch, it:

Resets the pirate to a random free cell so the agent learns the whole maze, not one route.
Picks moves with an epsilon-greedy strategy: a random valid move with probability epsilon (exploration), otherwise the move with the highest predicted Q-value (exploitation).
Stores each move (state, action, reward, next state, game over) in a replay memory of 512 episodes.
Samples a random batch of 32 episodes after each move and trains the network with train_step(), using a discount factor of 0.95.
Syncs the target network every 50 epochs, tracks the win rate over the last 32 games, and lowers exploration to 5% once the win rate passes 90%.
Stops when the win rate reaches 100% and completion_check() confirms the pirate can win from every free cell.

One bug was worth catching: the starter pseudocode suggested storing the game status string in each episode, but the replay code treats any non-empty value as the end of a game. Storing 'not_over' would have marked every move as final and stopped the agent from learning future rewards, so I stored a true/false game-over flag instead.

Result: the agent reached a 100% win rate at epoch 305 (about 345 seconds of training) and passed the completion check from every starting cell.

Reflection
What do computer scientists do and why does it matter?

Computer scientists turn real problems into something a computer can solve, then build, test, and improve the solution. In this project, "make an NPC that finds treasure" had to become states, actions, rewards, and a neural network that could estimate which move was best. That matters because the same techniques reach well beyond games. Reinforcement learning is used in robotics, logistics, recommendation systems, and resource scheduling, so the ability to frame a problem this way has real impact on products people use every day.

How do I approach a problem as a computer scientist?

I start by understanding what I'm working with before writing code. For this project, that meant reading the starter files closely, since my course version included a target network and a custom train_step() that changed how the training loop had to work. Then I break the problem into pieces (action selection, memory, training, evaluation), build them one at a time, and test against clear success criteria. The game-over flag bug is a good example: the code ran without errors, but it would never have learned. Checking how each piece actually behaved, instead of trusting the pseudocode, is what made it work. When results are off, I trace the data flow and adjust hyperparameters one at a time, rather than guessing.

What are my ethical responsibilities to the end user and the organization?

To the end user, I'm responsible for building systems that are fair, safe, and respectful of their data. This course covered how AI can pick up hidden bias from training data and how personalization can collide with privacy laws like GDPR. That means collecting only the data I need, being transparent about how it is used, and testing for unequal results across different groups. Even in a game, an agent should make the experience fun and fair, not frustrating or exploitative. To the organization, I'm responsible for writing reliable, well-documented code, being honest about a system's limits and costs (deep Q-learning took hundreds of epochs and real compute time here), and raising concerns early when a design could create legal, safety, or reputational risk.
