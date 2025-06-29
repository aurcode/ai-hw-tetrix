import numpy as np
import os
import pickle
import random
import pygame

from src.agent.BaseAIPlayer import BaseAIPlayer
from src.common.constants import BOARD_WIDTH_TILES, BOARD_HEIGHT_TILES
from src.game.blocks_group import BlocksGroup
from src.game.block import TopReached


class QLearningAIPlayer(BaseAIPlayer):
    def __init__(
        self,
        grid_width,
        grid_height,
        load_q_table=False,
        q_table_path="models/tetris_q_table.pkl",
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.q_table_path = q_table_path
        self.q_table = self.load_q_table() if load_q_table else {}

        # Hyperparameters
        self.learning_rate = 0.1  # Alpha
        self.discount_factor = 0.95  # Gamma
        self.exploration_rate = 0.1  # Epsilon

        # Actions: 0=left, 1=right, 2=rotate, 3=hard_drop
        self.actions = [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]
        self.num_actions = len(self.actions)

    def get_state(self, game_board_state):
        # Simplified state representation
        column_heights = np.zeros(self.grid_width)
        for col in range(self.grid_width):
            for row in range(self.grid_height):
                if game_board_state[row, col] == 1:
                    column_heights[col] = self.grid_height - row
                    break

        holes = 0
        for col in range(self.grid_width):
            col_has_block = False
            for row in range(self.grid_height):
                if game_board_state[row, col] == 1:
                    col_has_block = True
                elif col_has_block and game_board_state[row, col] == 0:
                    holes += 1

        bumpiness = 0
        for i in range(self.grid_width - 1):
            bumpiness += abs(column_heights[i] - column_heights[i + 1])

        lines_cleared = 0
        for row in range(self.grid_height):
            if np.all(game_board_state[row, :] == 1):
                lines_cleared += 1

        return (tuple(column_heights), holes, bumpiness, lines_cleared)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(self.actions)  # Explore
        else:
            q_values = [self.q_table.get((state, action), 0) for action in self.actions]
            return self.actions[np.argmax(q_values)]  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table.get((state, action), 0)
        next_max = np.max([self.q_table.get((next_state, a), 0) for a in self.actions])

        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
            reward + self.discount_factor * next_max
        )
        self.q_table[(state, action)] = new_value

    def get_next_move(self, game_board_state, current_block, next_block):
        state = self.get_state(game_board_state)
        action = self.choose_action(state)
        return [action]

    def game_ended(self):
        self.save_q_table()

    def save_q_table(self):
        os.makedirs(os.path.dirname(self.q_table_path), exist_ok=True)
        with open(self.q_table_path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {self.q_table_path}")

    def load_q_table(self):
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, "rb") as f:
                return pickle.load(f)
        return {}


def train_q_learning_agent(episodes=1000):
    agent = QLearningAIPlayer(
        grid_width=BOARD_WIDTH_TILES, grid_height=BOARD_HEIGHT_TILES
    )
    max_score = -1

    for episode in range(episodes):
        blocks = BlocksGroup()
        game_over = False
        score = 0

        while not game_over:
            state = agent.get_state(blocks.get_board_state_array())
            action = agent.choose_action(state)

            prev_score = blocks.score
            try:
                if action == pygame.K_LEFT:
                    blocks.start_moving_current_block(pygame.K_LEFT)
                    blocks.move_current_block()
                    blocks.stop_moving_current_block()
                elif action == pygame.K_RIGHT:
                    blocks.start_moving_current_block(pygame.K_RIGHT)
                    blocks.move_current_block()
                    blocks.stop_moving_current_block()
                elif action == pygame.K_UP:
                    blocks.rotate_current_block()
                elif action == pygame.K_DOWN:
                    blocks.hard_drop_current_block()

                # Simulate block falling
                blocks.update_current_block()

            except TopReached:
                game_over = True

            next_state = agent.get_state(blocks.get_board_state_array())
            reward = blocks.score - prev_score
            if game_over:
                reward = -10

            agent.update_q_table(state, action, reward, next_state)
            score = blocks.score

        if score > max_score:
            max_score = score
            agent.save_q_table()
            print(f"Episode {episode + 1}: New max score: {max_score}")

    print("Training finished.")


if __name__ == "__main__":
    train_q_learning_agent()
