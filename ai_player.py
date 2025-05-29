import pygame
import numpy as np
import random # Added for random moves

class RandomAIPlayer: # Renamed from SimpleAI
    def __init__(self, grid_width, grid_height):
        """
        Initializes the Random AI player.
        Args:
            grid_width (int): The width of the game grid.
            grid_height (int): The height of the game grid.
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.action_cooldown = 0 # Counter to limit action frequency
        self.action_interval = 3 # Perform an action roughly every 5 calls
        print(f"RandomAIPlayer initialized for a {grid_width}x{grid_height} grid.")

    def get_next_move(self, game_board_state, current_block, next_block):
        """
        Determines the next move for the AI.
        This is a placeholder and should be implemented with actual AI logic.

        Args:
            game_board_state (np.array): A 2D numpy array representing the current state of the game board.
                                         0 for empty, 1 (or other non-zero) for filled.
            current_block (Block): The current falling block object.
            next_block (Block): The next block object.

        Returns:
            list: A list of actions (e.g., [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN])
                  An empty list means no action or let the block fall.
                  pygame.K_UP for rotate.
                  pygame.K_LEFT/RIGHT for horizontal movement.
                  pygame.K_DOWN for soft drop (or hard drop if implemented).
        """
        actions = []
        self.action_cooldown -= 1

        if self.action_cooldown <= 0 and current_block:
            self.action_cooldown = self.action_interval # Reset cooldown

            possible_moves = [
                pygame.K_LEFT, 
                pygame.K_RIGHT, 
                pygame.K_UP, # Rotate
                pygame.K_DOWN, # Soft drop one step
                None, None, None # Increase probability of doing nothing
            ]
            
            chosen_move = random.choice(possible_moves)
            
            if chosen_move is not None:
                actions.append(chosen_move)
                # print(f"RandomAI chose: {chosen_move}")

        return actions

    def update_game_state(self, game_board_state): # Renamed from update_game_state_on_landing for consistency
        """
        Optional method for the AI to update its internal representation or strategy
        based on the new game state after a block has landed.
        """
        # print("AI received updated game state.")
        pass

# Example of how game_board_state might be created in main.py
# This is just for conceptual understanding, the actual implementation will be in BlocksGroup or main.py
def get_game_board_array_example(blocks_group, grid_width, grid_height):
    board = np.zeros((grid_height, grid_width), dtype=int)
    for block_sprite in blocks_group.sprites():
        if not block_sprite.is_current: # Only consider landed blocks for the board state
            for r_idx, row in enumerate(block_sprite.shape):
                for c_idx, cell in enumerate(row):
                    if cell:
                        # Convert block's local grid coordinates to board's global grid coordinates
                        board_x = block_sprite.rect.x // block_sprite.tile_size + c_idx
                        board_y = block_sprite.rect.y // block_sprite.tile_size + r_idx
                        if 0 <= board_y < grid_height and 0 <= board_x < grid_width:
                            board[board_y, board_x] = 1 # Or block_sprite.shape_name or some ID
    return board
