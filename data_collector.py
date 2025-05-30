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

    def update_game_state(self, game_board_state):
        """
        Optional method for the AI to update its internal representation or strategy
        based on the new game state after a block has landed.
        """
        # print("AI received updated game state.")
        pass


class DataCollector:
    def __init__(self, grid_width, grid_height):
        """
        Initializes the Data Collector.
        Args:
            grid_width (int): The width of the game grid.
            grid_height (int): The height of the game grid.
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.collected_data = [] # List to store (game_state, action) pairs
        print(f"DataCollector initialized for a {grid_width}x{grid_height} grid. Data collection active.")

    def get_next_move(self, game_board_state, current_block, next_block):
        """
        For DataCollector, this method returns an empty list as the human player
        will be making the moves. The purpose is to collect data.
        """
        return []

    def log_action(self, game_board_state, current_block, next_block, action_key, score):
        """
        Logs the current game state, the action taken by the human player, and the current score.
        Args:
            game_board_state (np.array): The 2D numpy array of the game board.
            current_block (Block): The current falling block.
            next_block (Block): The next block.
            action_key (int): The pygame.K_* key representing the action taken.
            score (int): The current score of the player.
        """
        # Convert numpy array to a list for JSON serialization if needed later
        board_list = game_board_state.tolist() if isinstance(game_board_state, np.ndarray) else game_board_state
        
        # Store relevant info. You might want to store more details about blocks.
        data_point = {
            "board_state": board_list,
            "current_block_shape": current_block.shape_name if current_block else None,
            "current_block_pos": (current_block.x, current_block.y) if current_block else None,
            "current_block_rotation": current_block.rotation if current_block else None, # Added rotation
            "current_block_points": current_block.get_points() if current_block else None, # Added points
            "next_block_shape": next_block.shape_name if next_block else None,
            "action": action_key, # Store the raw key for now
            "score": score # Added current score
        }
        self.collected_data.append(data_point)
        # print(f"Logged action: {action_key}")

    def update_game_state(self, game_board_state):
        """
        Called when a block lands or game state changes.
        """
        pass # No specific action needed for data collection on state update

    def save_data(self, filename="tetris_training_data.json"):
        """
        Saves the collected data to a file.
        """
        import json
        try:
            with open(filename, 'w') as f:
                json.dump(self.collected_data, f, indent=4)
            print(f"Collected data saved to {filename}. Total data points: {len(self.collected_data)}")
        except Exception as e:
            print(f"Error saving data: {e}")

# Helper function (not part of the class, but useful for context)
# Example of how game_board_state might be created in main.py
# This is just for conceptual understanding, the actual implementation is in BlocksGroup.get_board_state_array()
# def get_game_board_array_example(blocks_group, grid_width, grid_height):
#     board = np.zeros((grid_height, grid_width), dtype=int)
#     for block_sprite in blocks_group.sprites():
#         if not block_sprite.is_current: # Only consider landed blocks for the board state
#             for r_idx, row in enumerate(block_sprite.shape):
#                 for c_idx, cell in enumerate(row):
#                     if cell:
#                         # Convert block's local grid coordinates to board's global grid coordinates
#                         board_x = block_sprite.rect.x // block_sprite.tile_size + c_idx
#                         board_y = block_sprite.rect.y // block_sprite.tile_size + r_idx
#                         if 0 <= board_y < grid_height and 0 <= board_x < grid_width:
#                             board[board_y, board_x] = 1 # Or block_sprite.shape_name or some ID
#     return board
