import pygame
import numpy as np

class SimpleAI:
    def __init__(self, grid_width, grid_height):
        """
        Initializes the AI player.
        Args:
            grid_width (int): The width of the game grid.
            grid_height (int): The height of the game grid.
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        print(f"SimpleAI initialized for a {grid_width}x{grid_height} grid.")

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
        # Placeholder logic:
        # For now, the AI does nothing and lets the block fall.
        # In a real AI, you would analyze the game_board_state, current_block, and next_block
        # to decide the optimal sequence of moves.

        # Example: print the state for debugging
        # print("Current board state:\n", game_board_state)
        # if current_block:
        #     print("Current block:", current_block.shape_name, "at", current_block.rect.topleft)
        # if next_block:
        #     print("Next block:", next_block.shape_name)

        actions = []

        # --- AI Logic would go here ---
        # 1. Evaluate all possible placements for the current_block.
        #    - This involves trying different rotations and horizontal positions.
        # 2. For each possible placement, simulate dropping the block.
        # 3. Score each resulting board state based on heuristics (e.g., height, holes, completed lines).
        # 4. Choose the placement that results in the best score.
        # 5. Determine the sequence of moves (rotations, left/right shifts) to achieve that placement.
        # 6. Return the first move in that sequence, or a series of moves.

        # Simplistic example: move left if possible, just to show it can return an action
        # if current_block and current_block.rect.x > 0:
        #     can_move_left = True
        #     test_rect = current_block.rect.copy()
        #     test_rect.x -= current_block.tile_size
        #     for row_idx, row in enumerate(current_block.shape):
        #         for col_idx, cell in enumerate(row):
        #             if cell: # if part of the block
        #                 board_x = test_rect.x // current_block.tile_size + col_idx
        #                 board_y = test_rect.y // current_block.tile_size + row_idx
        #                 if board_x < 0 or game_board_state[board_y, board_x] != 0:
        #                     can_move_left = False
        #                     break
        #         if not can_move_left:
        #             break
        #     if can_move_left:
        #         actions.append(pygame.K_LEFT)


        return actions

    def update_game_state(self, game_board_state):
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
