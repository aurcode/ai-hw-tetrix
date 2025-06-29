import pygame
from src.agent.BaseAIPlayer import BaseAIPlayer
from src.agent.deep_q_learning import TetrisAgent


class DeepQNAIPlayer(BaseAIPlayer):
    def __init__(self, grid_width, grid_height, load_model=False):
        super().__init__(grid_width, grid_height)
        self.agent = TetrisAgent(grid_width, grid_height, load_model=load_model)

    def get_next_move(self, game_board_state, current_block_obj, next_block_obj):
        state = self.agent.get_state_features(game_board_state)
        action = self.agent.choose_action(state, current_block_obj, game_board_state)

        actions = []
        for _ in range(action["rotation"]):
            actions.append(pygame.K_UP)

        current_x = current_block_obj.x
        target_x = action["x"]

        if target_x > current_x:
            for _ in range(target_x - current_x):
                actions.append(pygame.K_RIGHT)
        elif target_x < current_x:
            for _ in range(current_x - target_x):
                actions.append(pygame.K_LEFT)

        actions.append(pygame.K_DOWN)  # Hard drop

        return actions
