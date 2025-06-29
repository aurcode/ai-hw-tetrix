import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


class TetrisAgent:
    def __init__(
        self, grid_width, grid_height, load_model=False, model_path="tetris_dqn.pth"
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=2000)
        self.model_path = model_path

        # The number of state features
        self.n_state_features = 4
        # The number of possible actions will be dynamic based on the piece
        self.n_actions = 1

        self.q_network = DeepQNetwork(self.n_state_features, self.n_actions)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        if load_model:
            self.load_weights()

    def get_state_features(self, board):
        # Aggregate Height
        heights = np.max(
            np.where(
                board != 0, self.grid_height - np.arange(self.grid_height)[:, None], 0
            ),
            axis=0,
        )
        agg_height = np.sum(heights)

        # Number of Holes
        holes = 0
        for col in range(self.grid_width):
            col_holes = 0
            occupied = False
            for row in range(self.grid_height):
                if board[row, col] != 0:
                    occupied = True
                elif occupied:
                    col_holes += 1
            holes += col_holes

        # Bumpiness
        bumpiness = np.sum(np.abs(heights[:-1] - heights[1:]))

        # Lines Cleared (This will be passed in from the game environment after a move)
        # For now, we'll just use a placeholder
        lines_cleared = 0

        return np.array([agg_height, holes, bumpiness, lines_cleared])

    def get_possible_actions(self, piece, board):
        possible_actions = []
        for rotation in range(4):
            rotated_piece = piece.clone()
            for _ in range(rotation):
                rotated_piece.rotate(
                    None
                )  # The group argument is not needed here as we are not checking for collisions yet

            for x in range(self.grid_width - rotated_piece.struct.shape[1] + 1):
                # Create a copy of the piece for simulation
                sim_piece = rotated_piece.clone()
                sim_piece.x = x
                sim_piece.y = 0

                # Simulate hard drop
                while not self._check_collision(sim_piece, board):
                    sim_piece.y += 1
                sim_piece.y -= 1  # Go back to the last valid position

                final_board = board.copy()
                for r, row in enumerate(sim_piece.struct):
                    for c, cell in enumerate(row):
                        if cell:
                            final_board[sim_piece.y + r, sim_piece.x + c] = 1

                possible_actions.append(
                    {"rotation": rotation, "x": x, "final_board": final_board}
                )
        return possible_actions

    def _check_collision(self, piece, board):
        for r, row in enumerate(piece.struct):
            for c, cell in enumerate(row):
                if cell:
                    if (
                        piece.y + r >= self.grid_height
                        or piece.x + c < 0
                        or piece.x + c >= self.grid_width
                        or board[piece.y + r, piece.x + c]
                    ):
                        return True
        return False

    def choose_action(self, state, piece, board):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.get_possible_actions(piece, board))

        possible_actions = self.get_possible_actions(piece, board)
        q_values = []
        for action in possible_actions:
            next_state_features = self.get_state_features(action["final_board"])
            state_tensor = torch.FloatTensor(next_state_features).unsqueeze(0)
            q_values.append(self.q_network(state_tensor).item())

        best_action_index = np.argmax(q_values)
        return possible_actions[best_action_index]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target = (
                    reward
                    + self.gamma * torch.max(self.q_network(next_state_tensor)).item()
                )

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            q_values[0][
                0
            ] = target  # The action is implicit in the state, so we update the single output

            self.optimizer.zero_grad()
            loss = self.criterion(q_values, torch.FloatTensor([[target]]))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_weights(self):
        try:
            self.q_network.load_state_dict(torch.load(self.model_path))
            print("Loaded pre-trained weights.")
        except FileNotFoundError:
            print("No pre-trained weights found, starting from scratch.")

    def save_weights(self):
        torch.save(self.q_network.state_dict(), self.model_path)
        print(f"Saved weights to {self.model_path}")

    def train(self):
        pass


if __name__ == "__main__":
    from src.game.blocks_group import BlocksGroup, TopReached
    from src.common.constants import BOARD_WIDTH_TILES, BOARD_HEIGHT_TILES
    import matplotlib.pyplot as plt

    def train_agent():
        episodes = 1000
        batch_size = 32
        agent = TetrisAgent(BOARD_WIDTH_TILES, BOARD_HEIGHT_TILES)
        scores = []

        for e in range(episodes):
            game = BlocksGroup()
            board = game.get_board_state_array()
            current_piece = game.current_block
            done = False
            score = 0

            while not done:
                state = agent.get_state_features(board)
                action = agent.choose_action(state, current_piece, board)

                # Simulate action
                game.current_block.rotate(game, n_rotations=action["rotation"])
                game.current_block.x = action["x"]
                try:
                    game.hard_drop_current_block()
                except TopReached:
                    done = True

                # Get new state and reward
                new_board = game.get_board_state_array()
                lines_cleared = np.sum(np.all(new_board == 1, axis=1))
                reward = 1 + lines_cleared**2
                if done:
                    reward = -10

                next_state = agent.get_state_features(new_board)
                agent.remember(state, action, reward, next_state, done)

                board = new_board
                current_piece = game.current_block
                score += reward

                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

            scores.append(score)
            print(
                f"Episode {e+1}/{episodes}, Score: {score}, Epsilon: {agent.epsilon:.2f}"
            )

            if e % 50 == 0:
                agent.save_weights()

        plt.plot(scores)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title("Training Progress")
        plt.savefig("tetris_training_progress.png")
        plt.show()

    train_agent()
