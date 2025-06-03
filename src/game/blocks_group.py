import pygame
import numpy as np
import random
from collections import OrderedDict

from src.game.block import Block, SquareBlock, TBlock, LineBlock, LBlock, ZBlock, BottomReached, TopReached
from src.common.constants import TILE_SIZE, BOARD_WIDTH_TILES, BOARD_HEIGHT_TILES # Added TILE_SIZE, BOARD_WIDTH_TILES, BOARD_HEIGHT_TILES
from src.common.utils import remove_empty_columns

class BlocksGroup(pygame.sprite.OrderedUpdates):

    @staticmethod
    def get_random_block():
        return random.choice((SquareBlock, TBlock, LineBlock, LBlock, ZBlock))()

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self._reset_grid()
        self._ignore_next_stop = False
        self.score = 0
        self.next_block = None
        # Not really moving, just to initialize the attribute.
        self.stop_moving_current_block()
        # The first block.
        self._create_new_block()

    def _check_line_completion(self):
        """
        Check each line of the grid and remove the ones that
        are complete.
        """
        # Start checking from the bottom.
        for i, row in enumerate(self.grid[::-1]):
            if all(row):
                self.score += 5
                # Get the blocks affected by the line deletion and
                # remove duplicates.
                affected_blocks = list(OrderedDict.fromkeys(self.grid[-1 - i]))

                for block, y_offset in affected_blocks:
                    # Remove the block tiles which belong to the
                    # completed line.
                    block.struct = np.delete(block.struct, y_offset, 0)
                    if block.struct.any():
                        # Once removed, check if we have empty columns
                        # since they need to be dropped.
                        block.struct, x_offset = remove_empty_columns(block.struct)
                        # Compensate the space gone with the columns to
                        # keep the block's original position.
                        block.x += x_offset
                        # Force update.
                        block.redraw()
                    else:
                        # If the struct is empty then the block is gone.
                        self.remove(block)

                # Instead of checking which blocks need to be moved
                # once a line was completed, just try to move all of
                # them.
                for block in self:
                    # Except the current block.
                    if block.current:
                        continue
                    # Pull down each block until it reaches the
                    # bottom or collides with another block.
                    while True:
                        try:
                            block.move_down(self)
                        except BottomReached:
                            break

                self.update_grid()
                # Since we've updated the grid, now the i counter
                # is no longer valid, so call the function again
                # to check if there're other completed lines in the
                # new grid.
                self._check_line_completion()
                break

    def _reset_grid(self):
        self.grid = [[0 for _ in range(10)] for _ in range(20)]

    def _create_new_block(self):
        new_block = self.next_block or BlocksGroup.get_random_block()
        if Block.collide(new_block, self):
            raise TopReached
        self.add(new_block)
        self.next_block = BlocksGroup.get_random_block()
        self.update_grid()
        self._check_line_completion()

    def update_grid(self):
        self._reset_grid()
        for block in self:
            for y_offset, row in enumerate(block.struct):
                for x_offset, digit in enumerate(row):
                    # Prevent replacing previous blocks.
                    if digit == 0:
                        continue
                    rowid = block.y + y_offset
                    colid = block.x + x_offset
                    self.grid[rowid][colid] = (block, y_offset)

    @property
    def current_block(self):
        return self.sprites()[-1]

    def update_current_block(self, force_move=False): # Added force_move parameter
        # The 'force_move' parameter is primarily for AI to make a deliberate downward step
        # outside the normal EVENT_UPDATE_CURRENT_BLOCK timer.
        try:
            # If forced, or if it's a regular update, move down.
            # The distinction of 'force_move' is mainly for the caller's intent;
            # the move_down method itself handles collision.
            self.current_block.move_down(self)
        except BottomReached:
            self.stop_moving_current_block()
            self._create_new_block()
        else:
            self.update_grid()

    def move_current_block(self):
        # First check if there's something to move.
        if self._current_block_movement_heading is None:
            return
        action = {
            pygame.K_DOWN: self.current_block.move_down,
            pygame.K_LEFT: self.current_block.move_left,
            pygame.K_RIGHT: self.current_block.move_right,
        }
        try:
            # Each function requires the group as the first argument
            # to check any possible collision.
            action[self._current_block_movement_heading](self)
        except BottomReached:
            self.stop_moving_current_block()
            self._create_new_block()
        else:
            self.update_grid()

    def start_moving_current_block(self, key):
        if self._current_block_movement_heading is not None:
            self._ignore_next_stop = True
        self._current_block_movement_heading = key

    def stop_moving_current_block(self):
        if self._ignore_next_stop:
            self._ignore_next_stop = False
        else:
            self._current_block_movement_heading = None

    def rotate_current_block(self):
        # Prevent SquareBlocks rotation.
        if not isinstance(self.current_block, SquareBlock):
            self.current_block.rotate(self)
            self.update_grid()

    def get_board_state_array(self):
        """
        Creates a 2D numpy array representing the current state of the game board.
        0 for empty, 1 for filled.
        This is used to pass the state to the AI.
        """
        board_array = np.zeros((BOARD_HEIGHT_TILES, BOARD_WIDTH_TILES), dtype=int)
        
        for block_sprite in self: # Iterate over all blocks in the group
            if not block_sprite.current: # Only consider landed blocks
                for r_idx, row in enumerate(block_sprite.struct):
                    for c_idx, cell in enumerate(row):
                        if cell: # If this part of the block's shape is solid
                            # Convert block's local grid coordinates (block.x, block.y)
                            # and its internal shape coordinates (c_idx, r_idx)
                            # to the main board's grid coordinates.
                            board_x = block_sprite.x + c_idx
                            board_y = block_sprite.y + r_idx
                            
                            # Ensure the coordinates are within the board bounds
                            if 0 <= board_y < BOARD_HEIGHT_TILES and 0 <= board_x < BOARD_WIDTH_TILES:
                                board_array[board_y, board_x] = 1 # Mark as filled
                            # else:
                                # This case (part of a landed block being out of bounds)
                                # ideally shouldn't happen if collision detection is correct.
                                # print(f"Warning: Part of landed block {block_sprite} at ({board_x},{board_y}) is out of bounds.")
        return board_array
