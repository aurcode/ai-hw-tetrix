import pygame
import numpy as np
import random

from src.common.constants import BOARD_WIDTH_TILES, BOARD_HEIGHT_TILES
# Import block classes to access their canonical structs
from src.game.block import SquareBlock, TBlock, LineBlock, LBlock, ZBlock # Add JBlock, SBlock if they exist

# Define canonical shapes based on your block.py definitions
CANONICAL_BLOCK_SHAPES = {
    SquareBlock.__name__: np.array(SquareBlock.struct),
    TBlock.__name__: np.array(TBlock.struct),
    LineBlock.__name__: np.array(LineBlock.struct),
    LBlock.__name__: np.array(LBlock.struct),
    ZBlock.__name__: np.array(ZBlock.struct),
    # Add other block types here if they exist in your game
    # e.g., "JBlock": np.array(JBlock.struct), "SBlock": np.array(SBlock.struct)
}

class HeuristicAIPlayer:
    def __init__(self, weights=None):
        self.grid_width_tiles = BOARD_WIDTH_TILES
        self.grid_height_tiles = BOARD_HEIGHT_TILES

        if weights:
            self.weights = weights
        else:
            # Default weights (GA will optimize these)
            self.weights = {
                'lines_cleared': 50.0,
                'aggregate_height': -2.0,
                'holes': -8.0,
                'bumpiness': -1.0,
                'game_over_penalty': -1000.0
            }
        
        self.target_rotation_idx = None 
        self.target_x_col = None      
        self.current_block_processed = False 

    def _get_rotated_struct(self, shape_name, rotation_idx):
        canonical_struct = CANONICAL_BLOCK_SHAPES.get(shape_name)
        if canonical_struct is None:
            print(f"Warning (HeuristicAI): Unknown shape_name '{shape_name}'. Using SquareBlock.")
            canonical_struct = CANONICAL_BLOCK_SHAPES.get(SquareBlock.__name__, np.array([[1,1],[1,1]])) # Fallback
        return np.rot90(canonical_struct, k=rotation_idx)

    def _check_collision(self, board_state_np, block_struct, x_col, y_row):
        block_h, block_w = block_struct.shape
        for r_idx, row_data in enumerate(block_struct):
            for c_idx, cell in enumerate(row_data):
                if cell: 
                    board_r, board_c = y_row + r_idx, x_col + c_idx
                    if not (0 <= board_r < self.grid_height_tiles and \
                            0 <= board_c < self.grid_width_tiles):
                        return True 
                    if board_state_np[board_r, board_c] == 1:
                        return True 
        return False

    def _simulate_drop_and_place(self, board_state_np, block_struct, target_x_col):
        block_h, block_w = block_struct.shape

        if not (0 <= target_x_col <= self.grid_width_tiles - block_w):
            return None, None, 0, True # board, lines_cleared, is_game_over

        initial_y_test = -block_h 
        landed_y = -1 

        for y_row_test in range(initial_y_test, self.grid_height_tiles - block_h + 1):
            current_y_for_collision_check = max(0, y_row_test) 
            if self._check_collision(board_state_np, block_struct, target_x_col, current_y_for_collision_check):
                landed_y = current_y_for_collision_check - 1
                break
            if y_row_test == self.grid_height_tiles - block_h:
                landed_y = y_row_test
        
        if landed_y < 0:
            if self._check_collision(board_state_np, block_struct, target_x_col, 0): # Collision at spawn
                return None, None, 0, True 

        new_board = board_state_np.copy()
        is_game_over_move = False
        for r_idx, row_data in enumerate(block_struct):
            for c_idx, cell in enumerate(row_data):
                if cell:
                    place_r, place_c = landed_y + r_idx, target_x_col + c_idx
                    if not (0 <= place_r < self.grid_height_tiles and \
                            0 <= place_c < self.grid_width_tiles):
                        is_game_over_move = True; break
                    new_board[place_r, place_c] = 1
            if is_game_over_move: break
        
        if is_game_over_move: return None, None, 0, True

        lines_cleared = 0; rows_to_keep = []
        for r in range(self.grid_height_tiles):
            if not np.all(new_board[r, :]): rows_to_keep.append(new_board[r, :])
            else: lines_cleared += 1
        
        final_board_state = np.zeros_like(new_board)
        num_kept_rows = len(rows_to_keep)
        if num_kept_rows > 0:
            final_board_state[self.grid_height_tiles - num_kept_rows:, :] = np.array(rows_to_keep)
            
        return landed_y, final_board_state, lines_cleared, False

    def _calculate_board_features(self, board_state_np):
        heights = np.zeros(self.grid_width_tiles, dtype=int)
        for c in range(self.grid_width_tiles):
            for r in range(self.grid_height_tiles):
                if board_state_np[r, c] == 1:
                    heights[c] = self.grid_height_tiles - r; break
        
        aggregate_height = np.sum(heights)
        holes = 0
        for c in range(self.grid_width_tiles):
            col_has_block_above_hole = False
            for r in range(self.grid_height_tiles): 
                if board_state_np[r,c] == 1: col_has_block_above_hole = True 
                elif board_state_np[r,c] == 0 and col_has_block_above_hole: holes += 1 
        bumpiness = 0
        for i in range(self.grid_width_tiles - 1):
            bumpiness += abs(heights[i] - heights[i+1])
        return aggregate_height, holes, bumpiness

    def _find_best_placement_target(self, game_board_state_np, current_block_shape_name):
        best_score = -float('inf')
        best_target_rotation_idx = 0
        best_target_x_col = 0

        for rot_idx in range(4): 
            rotated_struct = self._get_rotated_struct(current_block_shape_name, rot_idx)
            _block_h_tiles, block_w_tiles = rotated_struct.shape
            for x_col in range(self.grid_width_tiles - block_w_tiles + 1):
                _landed_y, next_board_state, lines_cleared, is_game_over = \
                    self._simulate_drop_and_place(game_board_state_np, rotated_struct, x_col)
                
                score = 0
                if is_game_over: score = self.weights['game_over_penalty']
                elif next_board_state is not None:
                    agg_h, num_holes, bump = self._calculate_board_features(next_board_state)
                    score = (self.weights['lines_cleared'] * lines_cleared +
                             self.weights['aggregate_height'] * agg_h + 
                             self.weights['holes'] * num_holes +        
                             self.weights['bumpiness'] * bump)
                else: score = -float('inf') 
                    
                if score > best_score:
                    best_score = score
                    best_target_rotation_idx = rot_idx
                    best_target_x_col = x_col
        
        if best_score == -float('inf'): 
            default_struct = self._get_rotated_struct(current_block_shape_name, 0)
            default_x = self.grid_width_tiles // 2 - default_struct.shape[1] // 2
            default_x = max(0, min(default_x, self.grid_width_tiles - default_struct.shape[1]))
            return 0, default_x
        return best_target_rotation_idx, best_target_x_col

    def get_next_move(self, game_board_state_np, current_block_instance, next_block_instance=None):
        if current_block_instance is None:
            self.current_block_processed = False; return []
        
        # Ensure block has necessary attributes
        required_attrs = ['shape_name', 'rotation', 'x']
        for attr in required_attrs:
            if not hasattr(current_block_instance, attr):
                print(f"Error (HeuristicAI): current_block_instance missing attribute '{attr}'. Cannot plan.")
                self.current_block_processed = False # Allow re-planning if block changes
                return [] # No action if block is malformed

        if not self.current_block_processed:
            self.target_rotation_idx, self.target_x_col = self._find_best_placement_target(
                game_board_state_np, current_block_instance.shape_name
            )
            self.current_block_processed = True

        current_rot_idx = current_block_instance.rotation // 90
        if current_rot_idx != self.target_rotation_idx: return [pygame.K_UP] 

        current_block_x_tile = current_block_instance.x 
        if current_block_x_tile < self.target_x_col: return [pygame.K_RIGHT]
        elif current_block_x_tile > self.target_x_col: return [pygame.K_LEFT]

        if current_block_instance.shape_name == SquareBlock.__name__:
            # For SquareBlock, always hard drop immediately from current position.
            # No need to calculate target rotation or x, as it won't move or rotate.
            if hasattr(self, 'is_current_square_move') and self.is_current_square_move: # Check if debug flag exists
                 print(f"[DEBUG] SquareBlock detected. Forcing immediate hard drop.")
            self.current_block_processed = False # Reset for the next block
            return [pygame.K_SPACE]       
        
        self.current_block_processed = False 
        return [pygame.K_SPACE] 

    def game_ended(self):
        """Called when the game is over to reset AI state."""
        self.current_block_processed = False
        self.target_rotation_idx = None
        self.target_x_col = None

    def update_game_state(self, game_board_state_np):
        pass # Placeholder

class EvolutionaryOptimizer:
    def __init__(self, population_size=50, mutation_rate=0.1, mutation_strength=0.2, crossover_rate=0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.population = self._initialize_population()

    def _initialize_population(self):
        population = []
        for _ in range(self.population_size):
            weights = {
                'lines_cleared': random.uniform(10, 100),
                'aggregate_height': random.uniform(-10, -1),
                'holes': random.uniform(-20, -5),
                'bumpiness': random.uniform(-10, -1),
                'game_over_penalty': -1000.0 
            }
            population.append(HeuristicAIPlayer(weights=weights))
        return population

    def _select_parents(self, fitness_scores):
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            # All individuals have fitness 0, select randomly
            return random.choices(self.population, k=2)

        selection_probs = [score / total_fitness for score in fitness_scores]
        return random.choices(self.population, weights=selection_probs, k=2)

    def _crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1.weights.copy(), parent2.weights.copy()

        child1_weights, child2_weights = {}, {}
        for key in parent1.weights:
            if random.random() < 0.5:
                child1_weights[key] = parent1.weights[key]
                child2_weights[key] = parent2.weights[key]
            else:
                child1_weights[key] = parent2.weights[key]
                child2_weights[key] = parent1.weights[key]
        return child1_weights, child2_weights

    def _mutate(self, weights):
        mutated_weights = weights.copy()
        for key in mutated_weights:
            if key != 'game_over_penalty' and random.random() < self.mutation_rate:
                change = random.uniform(-self.mutation_strength, self.mutation_strength)
                mutated_weights[key] *= (1 + change)
        return mutated_weights

    def evolve(self, fitness_scores):
        new_population = []
        
        # Elitism: Keep the best individual
        best_individual_idx = np.argmax(fitness_scores)
        best_individual = self.population[best_individual_idx]
        new_population.append(HeuristicAIPlayer(weights=best_individual.weights.copy()))

        while len(new_population) < self.population_size:
            parent1, parent2 = self._select_parents(fitness_scores)
            
            child1_weights, child2_weights = self._crossover(parent1, parent2)
            
            mutated_child1_weights = self._mutate(child1_weights)
            mutated_child2_weights = self._mutate(child2_weights)

            new_population.append(HeuristicAIPlayer(weights=mutated_child1_weights))
            if len(new_population) < self.population_size:
                new_population.append(HeuristicAIPlayer(weights=mutated_child2_weights))
        
        self.population = new_population