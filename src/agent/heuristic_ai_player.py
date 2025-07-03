import pygame
import numpy as np

from src.common.constants import BOARD_WIDTH_TILES, BOARD_HEIGHT_TILES


class HeuristicAIPlayer:
    def __init__(self, weights=None):
        self.grid_width_tiles = BOARD_WIDTH_TILES
        self.grid_height_tiles = BOARD_HEIGHT_TILES

        if weights:
            self.weights = weights
        else:
            # Default weights (GA will optimize these)
            self.weights = {
                "lines_cleared": 50.0,
                "aggregate_height": -2.0,
                "holes": -8.0,
                "bumpiness": -1.0,
                "game_over_penalty": -1000.0,
            }

        self.target_rotation_idx = None
        self.target_x_col = None
        self.current_block_processed = False

    def _get_rotated_shape(self, block_instance, rotation):
        return block_instance.shapes[rotation]

    def _check_collision(self, board_state_np, shape, x_col, y_row):
        for dx, dy in shape:
            board_r, board_c = y_row + dy, x_col + dx
            if not (
                0 <= board_r < self.grid_height_tiles
                and 0 <= board_c < self.grid_width_tiles
            ):
                return True
            if board_state_np[board_r, board_c] == 1:
                return True
        return False

    def _simulate_drop_and_place(self, board_state_np, shape, target_x_col):
        min_y = 0
        landed_y = -1

        for y_row_test in range(min_y, self.grid_height_tiles):
            if self._check_collision(board_state_np, shape, target_x_col, y_row_test):
                landed_y = y_row_test - 1
                break
        else:
            landed_y = self.grid_height_tiles - max(p[1] for p in shape) - 1

        if landed_y < 0:
            return None, None, 0, True

        new_board = board_state_np.copy()
        is_game_over_move = False
        for dx, dy in shape:
            place_r, place_c = landed_y + dy, target_x_col + dx
            if not (
                0 <= place_r < self.grid_height_tiles
                and 0 <= place_c < self.grid_width_tiles
            ):
                is_game_over_move = True
                break
            new_board[place_r, place_c] = 1

        if is_game_over_move:
            return None, None, 0, True

        lines_cleared = 0
        rows_to_keep = []
        for r in range(self.grid_height_tiles):
            if not np.all(new_board[r, :]):
                rows_to_keep.append(new_board[r, :])
            else:
                lines_cleared += 1

        final_board_state = np.zeros_like(new_board)
        num_kept_rows = len(rows_to_keep)
        if num_kept_rows > 0:
            final_board_state[self.grid_height_tiles - num_kept_rows :, :] = np.array(
                rows_to_keep
            )

        return landed_y, final_board_state, lines_cleared, False

    def _calculate_board_features(self, board_state_np):
        heights = np.zeros(self.grid_width_tiles, dtype=int)
        for c in range(self.grid_width_tiles):
            for r in range(self.grid_height_tiles):
                if board_state_np[r, c] == 1:
                    heights[c] = self.grid_height_tiles - r
                    break

        aggregate_height = np.sum(heights)
        holes = 0
        for c in range(self.grid_width_tiles):
            col_has_block_above_hole = False
            for r in range(self.grid_height_tiles):
                if board_state_np[r, c] == 1:
                    col_has_block_above_hole = True
                elif board_state_np[r, c] == 0 and col_has_block_above_hole:
                    holes += 1
        bumpiness = 0
        for i in range(self.grid_width_tiles - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return aggregate_height, holes, bumpiness

    def _find_best_placement_target(self, game_board_state_np, current_block_instance):
        best_score = -float("inf")
        best_target_rotation = 0
        best_target_x_col = 0

        for rotation in current_block_instance.shapes:
            shape = self._get_rotated_shape(current_block_instance, rotation)
            min_x = min(p[0] for p in shape)
            max_x = max(p[0] for p in shape)
            for x_col in range(-min_x, self.grid_width_tiles - max_x):
                _landed_y, next_board_state, lines_cleared, is_game_over = (
                    self._simulate_drop_and_place(game_board_state_np, shape, x_col)
                )

                score = 0
                if is_game_over:
                    score = self.weights["game_over_penalty"]
                elif next_board_state is not None:
                    agg_h, num_holes, bump = self._calculate_board_features(
                        next_board_state
                    )
                    score = (
                        self.weights["lines_cleared"] * lines_cleared
                        + self.weights["aggregate_height"] * agg_h
                        + self.weights["holes"] * num_holes
                        + self.weights["bumpiness"] * bump
                    )
                else:
                    score = -float("inf")

                if score > best_score:
                    best_score = score
                    best_target_rotation = rotation
                    best_target_x_col = x_col

        if best_score == -float("inf"):
            return 0, current_block_instance.x
        return best_target_rotation, best_target_x_col

    def get_next_move(
        self, game_board_state_np, current_block_instance, next_block_instance=None
    ):
        if current_block_instance is None:
            self.current_block_processed = False
            return []

        if not self.current_block_processed:
            self.target_rotation, self.target_x_col = self._find_best_placement_target(
                game_board_state_np, current_block_instance
            )
            self.current_block_processed = True

        if current_block_instance.rotation != self.target_rotation:
            return [pygame.K_UP]

        if current_block_instance.x < self.target_x_col:
            return [pygame.K_RIGHT]
        elif current_block_instance.x > self.target_x_col:
            return [pygame.K_LEFT]

        self.current_block_processed = False
        return [pygame.K_SPACE]

    def game_ended(self):
        """Called when the game is over to reset AI state."""
        self.current_block_processed = False
        self.target_rotation_idx = None
        self.target_x_col = None

    def update_game_state(self, game_board_state_np):
        pass  # Placeholder
