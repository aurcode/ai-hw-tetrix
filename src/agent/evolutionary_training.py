import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import json
import pygame

from src.game.block import TopReached
from src.game.blocks_group import BlocksGroup
from src.agent.evolutionary_optimizer import EvolutionaryOptimizer

def run_heuristic_training():
    """Runs the evolutionary algorithm to train the Heuristic AI without a GUI."""
    print("Starting Heuristic AI Training...")
    # Initialize pygame for game logic, but without creating a window
    pygame.init()

    num_generations = 10
    population_size = 20
    optimizer = EvolutionaryOptimizer(population_size=population_size)
    best_overall_fitness = -float('inf')
    best_weights = None

    for gen in range(num_generations):
        fitness_scores = []
        print(f"--- Generation {gen + 1}/{num_generations} ---")

        for i, ai_player in enumerate(optimizer.population):
            # --- Run a single game for the current AI ---
            blocks = BlocksGroup()
            game_over = False
            # The game loop doesn't need event handling for the GUI
            while not game_over:
                if blocks.current_block:
                    game_board_state = blocks.get_board_state_array()
                    current_block_obj = blocks.current_block
                    next_block_obj = blocks.next_block

                    ai_actions = ai_player.get_next_move(game_board_state, current_block_obj, next_block_obj)

                    for action in ai_actions:
                        try:
                            if action == pygame.K_UP:
                                blocks.rotate_current_block()
                            elif action == pygame.K_LEFT:
                                # Simulate a quick move
                                blocks.start_moving_current_block(pygame.K_LEFT)
                                blocks.move_current_block()
                                blocks.stop_moving_current_block()
                            elif action == pygame.K_RIGHT:
                                # Simulate a quick move
                                blocks.start_moving_current_block(pygame.K_RIGHT)
                                blocks.move_current_block()
                                blocks.stop_moving_current_block()
                            elif action == pygame.K_DOWN: # Using K_DOWN for hard drop
                                blocks.hard_drop_current_block()
                        except TopReached:
                            game_over = True
                            break
                else: # No current block
                    game_over = True

                # In training, we just need to update the block state until it lands.
                # The AI's hard drop action effectively handles this, so we don't need a separate fall timer.

            # Game is over for this AI, record fitness
            fitness = blocks.score
            fitness_scores.append(fitness)
            if fitness > best_overall_fitness:
                best_overall_fitness = fitness
                best_weights = ai_player.weights.copy()
            
            # Optional: Print progress for each individual
            # print(f"  Individual {i+1}/{population_size}, Score: {fitness}")

        print(f"Generation {gen + 1}: Best Score = {max(fitness_scores)}, Overall Best = {best_overall_fitness}")
        optimizer.evolve(fitness_scores)

    print("\nTraining finished!")
    if best_weights:
        with open('best_heuristic_weights.json', 'w') as f:
            json.dump(best_weights, f, indent=4)
        print("Saved best weights to best_heuristic_weights.json")

    pygame.quit()

if __name__ == '__main__':
    # This allows the script to be run directly from the command line
    run_heuristic_training()
