import pygame
import numpy as np
import random
from collections import OrderedDict

from src.game.block import (
    Block,
    SquareBlock,
    TBlock,
    LineBlock,
    LBlock,
    JBlock,
    SBlock,
    ZBlock,
    BottomReached,
    TopReached,
)
from src.common.constants import (
    TILE_SIZE,
    BOARD_WIDTH_TILES,
    BOARD_HEIGHT_TILES,
)  # Added TILE_SIZE, BOARD_WIDTH_TILES, BOARD_HEIGHT_TILES
from src.common.utils import remove_empty_columns


class BlocksGroup(pygame.sprite.OrderedUpdates):

    @staticmethod
    def get_random_block():
        # Initial position for new blocks
        initial_x = 4
        initial_y = 0
        return random.choice(
            (SquareBlock, TBlock, LineBlock, LBlock, JBlock, SBlock, ZBlock)
        )(initial_x, initial_y)

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
                self.score += 1  # Increase score by 1 for each line cleared
                cleared_row_y = BOARD_HEIGHT_TILES - 1 - i

                # Iterate through all blocks to update their structures
                for block in self:
                    new_struct = []
                    for sx, sy in block.struct:
                        global_y = block.y + sy
                        if global_y == cleared_row_y:
                            # This part of the block is on the cleared line, remove it
                            continue
                        elif global_y < cleared_row_y:
                            # This part is above the cleared line, it falls down
                            new_struct.append((sx, sy + 1))
                        else:
                            # This part is below the cleared line, no change in relative y
                            new_struct.append((sx, sy))

                    block.struct = new_struct
                    if not block.struct:
                        # If the block is empty after clearing lines, remove it
                        self.remove(block)
                    else:
                        # Redraw the block with its new structure
                        block.redraw()

                # After updating all blocks, re-evaluate the grid and check for more completions
                self.update_grid()
                self._check_line_completion()  # Recursively check for more lines
                break  # Exit after finding and processing one completed line

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
            for sx, sy in block.struct:
                rowid = block.y + sy
                colid = block.x + sx
                # Ensure coordinates are within grid bounds before assigning
                if 0 <= rowid < BOARD_HEIGHT_TILES and 0 <= colid < BOARD_WIDTH_TILES:
                    self.grid[rowid][colid] = (block, sy)
                else:
                    # This case should ideally not happen if collision detection is robust
                    # but it's good to have a safeguard or log for debugging.
                    print(
                        f"Warning: Block part at ({colid}, {rowid}) is out of grid bounds."
                    )

    @property
    def current_block(self):
        return self.sprites()[-1]

    def update_current_block(self, force_move=False):  # Added force_move parameter
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

    def hard_drop_current_block(self):
        """
        Moves the current block all the way down until it lands.
        """
        while True:
            try:
                self.current_block.move_down(self)
            except BottomReached:
                self.stop_moving_current_block()
                self._create_new_block()
                break  # Exit the loop once the block has landed
            else:
                self.update_grid()  # Update grid after each step down during hard drop

    def get_board_state_array(self):
        """
        Creates a 2D numpy array representing the current state of the game board.
        0 for empty, 1 for filled.
        This is used to pass the state to the AI.
        """
        board_array = np.zeros((BOARD_HEIGHT_TILES, BOARD_WIDTH_TILES), dtype=int)

        for block_sprite in self:  # Iterate over all blocks in the group
            if not block_sprite.current:  # Only consider landed blocks
                for sx, sy in block_sprite.struct:
                    # Convert block's local grid coordinates (block.x, block.y)
                    # and its internal shape coordinates (sx, sy)
                    # to the main board's grid coordinates.
                    board_x = block_sprite.x + sx
                    board_y = block_sprite.y + sy

                    # Ensure the coordinates are within the board bounds
                    if (
                        0 <= board_y < BOARD_HEIGHT_TILES
                        and 0 <= board_x < BOARD_WIDTH_TILES
                    ):
                        board_array[board_y, board_x] = 1  # Mark as filled
                    # else:
                    # This case (part of a landed block being out of bounds)
                    # ideally shouldn't happen if collision detection is correct.
                    # print(f"Warning: Part of landed block {block_sprite} at ({board_x},{board_y}) is out of bounds.")
        return board_array

    def get_ghost_coords(self):
        """
        Calculates the final landing position (x, y) of the current block
        if it were to hard drop.
        Returns (x, y) tuple of the ghost block's top-left corner.
        """
        ghost_block = self.current_block.clone()  # Create a copy to simulate movement

        final_y = ghost_block.y
        while True:
            try:
                # Pass the actual current_block as ignore_block to prevent self-collision
                ghost_block.move_down(self, ignore_block=self.current_block)
                final_y = ghost_block.y  # Update final_y as it moves down
            except BottomReached:
                break

        return ghost_block.x, final_y
