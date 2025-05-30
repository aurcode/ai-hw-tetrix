# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pygame",
# ]
# ///

import pygame
import sys # Added for sys.exit

from constants import WINDOW_WIDTH, WINDOW_HEIGHT, TILE_SIZE, GRID_WIDTH, GRID_HEIGHT, BOARD_WIDTH_TILES, BOARD_HEIGHT_TILES
from utils import draw_grid, draw_centered_surface
from block import TopReached
from blocks_group import BlocksGroup
from data_collector import RandomAIPlayer, DataCollector # Import RandomAIPlayer and DataCollector
from ai_player import NNAIPlayer # Import the new NNAIPlayer


def main():
    pygame.init()
    pygame.display.set_caption("Tetris with PyGame")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    
    # Game states
    run = True
    menu_active = True
    paused = False
    game_over = False
    player_type = None # "human", "ai1", "ai2", etc.
    ai_instance = None # To hold the AI object
    
    # Create background.
    background = pygame.Surface(screen.get_size())
    bgcolor = (0, 0, 0)
    background.fill(bgcolor)
    # Draw the grid on top of the background.
    draw_grid(background, TILE_SIZE, GRID_WIDTH, GRID_HEIGHT)
    # This makes blitting faster.
    background = background.convert()

    try:
        font = pygame.font.Font("Roboto-Regular.ttf", 20)
        menu_font = pygame.font.Font("Roboto-Regular.ttf", 30) # Larger font for menu
    except OSError:
        # If the font file is not available, the default will be used.
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        menu_font = pygame.font.Font(pygame.font.get_default_font(), 30)

    # --- Menu Setup ---
    menu_options = [
        "Human Player", 
        "Random AI Player", 
        "Data Collection Player", # Renamed for clarity
        "NN AI Player (Gameplay)", # New option for the actual NN AI
        "AI Player 3 (Placeholder)", 
        "AI Player 4 (Placeholder)", 
        "AI Player 5 (Placeholder)", 
        "AI Player 6 (Placeholder)"
    ]
    selected_option_index = 0
    menu_title_text = menu_font.render("Select Player Mode", True, (255, 255, 255), bgcolor)
    option_texts = [menu_font.render(option, True, (255, 255, 255), bgcolor) for option in menu_options]
    selected_option_color = (255, 220, 0) # Yellow for selected

    # --- In-Game Text ---
    next_block_text = font.render("Next figure:", True, (255, 255, 255), bgcolor)
    score_msg_text = font.render("Score:", True, (255, 255, 255), bgcolor)
    game_over_text = font.render("¡Game over!", True, (255, 220, 0), bgcolor)

    # Event constants.
    MOVEMENT_KEYS = pygame.K_LEFT, pygame.K_RIGHT, pygame.K_DOWN
    EVENT_UPDATE_CURRENT_BLOCK = pygame.USEREVENT + 1
    EVENT_MOVE_CURRENT_BLOCK = pygame.USEREVENT + 2
    pygame.time.set_timer(EVENT_UPDATE_CURRENT_BLOCK, 1000)
    pygame.time.set_timer(EVENT_MOVE_CURRENT_BLOCK, 100)

    blocks = BlocksGroup() # Initialize blocks here, might be reset after menu

    while run:
        if menu_active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selected_option_index = (selected_option_index - 1) % len(menu_options)
                    elif event.key == pygame.K_DOWN:
                        selected_option_index = (selected_option_index + 1) % len(menu_options)
                    elif event.key == pygame.K_RETURN:
                        if selected_option_index == 0:
                            player_type = "human"
                        elif selected_option_index == 1:
                            player_type = "random_ai" # Internal name for Random AI
                            ai_instance = RandomAIPlayer(grid_width=BOARD_WIDTH_TILES, grid_height=BOARD_HEIGHT_TILES)
                        elif selected_option_index == 2:
                            player_type = "data_collection" # Internal name for DataCollector
                            ai_instance = DataCollector(grid_width=BOARD_WIDTH_TILES, grid_height=BOARD_HEIGHT_TILES)
                        elif selected_option_index == 3:
                            player_type = "nn_ai_gameplay" # Internal name for NN AI gameplay
                            ai_instance = NNAIPlayer(grid_width=BOARD_WIDTH_TILES, grid_height=BOARD_HEIGHT_TILES)
                        elif selected_option_index == 4: # Shifted index for placeholders
                            player_type = "ai3" # Placeholder
                            print("AI Player 3 selected - using RandomAIPlayer as placeholder for now.")
                            ai_instance = RandomAIPlayer(grid_width=BOARD_WIDTH_TILES, grid_height=BOARD_HEIGHT_TILES) # Placeholder
                        elif selected_option_index == 5:
                            player_type = "ai4" # Placeholder
                            print("AI Player 4 selected - using RandomAIPlayer as placeholder for now.")
                            ai_instance = RandomAIPlayer(grid_width=BOARD_WIDTH_TILES, grid_height=BOARD_HEIGHT_TILES) # Placeholder
                        elif selected_option_index == 6:
                            player_type = "ai5" # Placeholder
                            print("AI Player 5 selected - using RandomAIPlayer as placeholder for now.")
                            ai_instance = RandomAIPlayer(grid_width=BOARD_WIDTH_TILES, grid_height=BOARD_HEIGHT_TILES) # Placeholder
                        elif selected_option_index == 7: # New index for last placeholder
                            player_type = "ai6" # Placeholder
                            print("AI Player 6 selected - using RandomAIPlayer as placeholder for now.")
                            ai_instance = RandomAIPlayer(grid_width=BOARD_WIDTH_TILES, grid_height=BOARD_HEIGHT_TILES) # Placeholder
                        
                        menu_active = False
                        # Reset game state for a new game
                        game_over = False
                        paused = False
                        blocks = BlocksGroup() # Re-initialize blocks for the selected mode
                        # Ensure timers are set correctly if they were stopped or changed
                        pygame.time.set_timer(EVENT_UPDATE_CURRENT_BLOCK, 1000)
                        pygame.time.set_timer(EVENT_MOVE_CURRENT_BLOCK, 100)


            # Draw Menu
            screen.blit(background, (0, 0)) # Clear screen with background (grid might be visible)
            
            title_rect = menu_title_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 4))
            screen.blit(menu_title_text, title_rect)

            for i, option_text_surface in enumerate(option_texts):
                color = selected_option_color if i == selected_option_index else (255, 255, 255)
                # Re-render text with current color to show selection
                rendered_text = menu_font.render(menu_options[i], True, color, bgcolor)
                text_rect = rendered_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + i * 50))
                screen.blit(rendered_text, text_rect)
            
            pygame.display.flip()
            continue # Skip game logic while menu is active

        # --- Main Game Loop (when menu_active is False) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            elif event.type == pygame.KEYUP:
                if not paused and not game_over:
                    if player_type == "human" or player_type == "data_collection": # Process human input
                        if event.key in MOVEMENT_KEYS:
                            blocks.stop_moving_current_block()
                            if player_type == "data_collection":
                                game_board_state = blocks.get_board_state_array()
                                current_block_obj = blocks.current_block
                                next_block_obj = blocks.next_block
                                ai_instance.log_action(game_board_state, current_block_obj, next_block_obj, event.key)
                        elif event.key == pygame.K_UP:
                            blocks.rotate_current_block()
                            if player_type == "data_collection":
                                game_board_state = blocks.get_board_state_array()
                                current_block_obj = blocks.current_block
                                next_block_obj = blocks.next_block
                                ai_instance.log_action(game_board_state, current_block_obj, next_block_obj, event.key)
                if event.key == pygame.K_p:
                    paused = not paused
                if game_over and event.key == pygame.K_RETURN: # Restart game or go to menu
                    if player_type == "data_collection" and ai_instance:
                        ai_instance.save_data() # Save data when game ends and returning to menu
                    menu_active = True # Go back to menu
                    # Reset game state variables
                    game_over = False
                    paused = False
                    # blocks will be re-initialized when menu selection is made
                    # ai_instance will be (re-)initialized if an AI mode is selected

            # Stop moving blocks if the game is over or paused.
            if game_over or paused:
                # If AI is playing and game over, it might want to know
                if game_over and ai_instance and hasattr(ai_instance, 'game_ended'):
                    ai_instance.game_ended() # Optional: notify AI game ended
                continue

            # --- AI Control Logic (for actual AI players) ---
            if player_type in ["random_ai", "nn_ai_gameplay", "ai3", "ai4", "ai5", "ai6"] and ai_instance and not paused and not game_over:
                # 1. Get game state from blocks
                game_board_state = blocks.get_board_state_array()
                current_block_obj = blocks.current_block
                next_block_obj = blocks.next_block

                # 2. Get AI's next move
                ai_actions = ai_instance.get_next_move(game_board_state, current_block_obj, next_block_obj)

                # 3. Execute AI actions
                for action in ai_actions:
                    if action == pygame.K_LEFT:
                        blocks.start_moving_current_block(pygame.K_LEFT)
                        blocks.move_current_block()
                        blocks.stop_moving_current_block()
                    elif action == pygame.K_RIGHT:
                        blocks.start_moving_current_block(pygame.K_RIGHT)
                        blocks.move_current_block()
                        blocks.stop_moving_current_block()
                    elif action == pygame.K_UP: # Rotate
                        blocks.rotate_current_block()
                    elif action == pygame.K_DOWN: # Soft drop one step
                        blocks.update_current_block(force_move=True)
                    # Add more actions like hard drop if needed

            # --- Human Input Processing (for human and Data Collection) ---
            if player_type == "human" or player_type == "data_collection":
                if event.type == pygame.KEYDOWN:
                    if event.key in MOVEMENT_KEYS:
                        blocks.start_moving_current_block(event.key)
                        if player_type == "data_collection":
                            # Log KEYDOWN for continuous movement start
                            game_board_state = blocks.get_board_state_array()
                            current_block_obj = blocks.current_block
                            next_block_obj = blocks.next_block
                            ai_instance.log_action(game_board_state, current_block_obj, next_block_obj, event.key)
                    elif event.key == pygame.K_UP: # Rotation is a KEYDOWN event
                        if player_type == "data_collection":
                            game_board_state = blocks.get_board_state_array()
                            current_block_obj = blocks.current_block
                            next_block_obj = blocks.next_block
                            ai_instance.log_action(game_board_state, current_block_obj, next_block_obj, event.key)
            
            # --- Game Logic (Movement, Updates) ---
            try:
                if event.type == EVENT_UPDATE_CURRENT_BLOCK: # Natural fall
                    blocks.update_current_block()
                elif event.type == EVENT_MOVE_CURRENT_BLOCK: # Continuous movement for human
                    if player_type == "human" or player_type == "data_collection":
                        blocks.move_current_block()
                    # AI movement is handled above based on ai_actions, not this timer directly.

            except TopReached:
                game_over = True
                if ai_instance and hasattr(ai_instance, 'update_game_state'):
                    final_board_state = blocks.get_board_state_array()
                    ai_instance.update_game_state(final_board_state)
                if player_type == "data_collection" and ai_instance:
                    ai_instance.save_data() # Save data if game over
        # Draw background and grid.
        screen.blit(background, (0, 0))
        # Blocks.
        blocks.draw(screen)
        # Sidebar with misc. information.
        draw_centered_surface(screen, next_block_text, 50)
        draw_centered_surface(screen, blocks.next_block.image, 100)
        draw_centered_surface(screen, score_msg_text, 240)
        score_text = font.render(str(blocks.score), True, (255, 255, 255), bgcolor)
        draw_centered_surface(screen, score_text, 270)
        if game_over:
            draw_centered_surface(screen, game_over_text, 360)
            restart_text = font.render("Press ENTER to return to Menu", True, (255, 255, 255), bgcolor)
            draw_centered_surface(screen, restart_text, 400)
        elif paused:
            pause_text = menu_font.render("PAUSED", True, (255,220,0), bgcolor)
            draw_centered_surface(screen, pause_text, WINDOW_HEIGHT // 2)


        # Update.
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
