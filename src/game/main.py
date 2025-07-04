import sys
import pygame
import json
from src.common.constants import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    TILE_SIZE,
    GRID_WIDTH,
    GRID_HEIGHT,
    NORMAL_FALL_SPEED_MS,
    NORMAL_MOVE_SPEED_MS,
)
from src.common.utils import draw_grid, draw_centered_surface
from src.game.block import TopReached
from src.game.blocks_group import BlocksGroup
from src.game.menu import Menu


def main():
    pygame.init()
    pygame.display.set_caption("Tetris with PyGame")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    # Game states
    run = True
    menu_active = True
    paused = False
    game_over = False
    player_type = None
    ai_instance = None

    # Create background
    background = pygame.Surface(screen.get_size())
    bgcolor = (0, 0, 0)
    background.fill(bgcolor)
    draw_grid(background, TILE_SIZE, GRID_WIDTH, GRID_HEIGHT)
    background = background.convert()

    try:
        font = pygame.font.Font("Roboto-Regular.ttf", 20)
        menu_font = pygame.font.Font("Roboto-Regular.ttf", 30)
    except OSError:
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        menu_font = pygame.font.Font(pygame.font.get_default_font(), 30)

    menu = Menu(screen)
    blocks = BlocksGroup()

    # --- Event Constants ---
    MOVEMENT_KEYS = (pygame.K_LEFT, pygame.K_RIGHT)
    HARD_DROP_KEY = pygame.K_DOWN
    EVENT_UPDATE_CURRENT_BLOCK = pygame.USEREVENT + 1
    EVENT_MOVE_CURRENT_BLOCK = pygame.USEREVENT + 2
    pygame.time.set_timer(EVENT_UPDATE_CURRENT_BLOCK, NORMAL_FALL_SPEED_MS)
    pygame.time.set_timer(EVENT_MOVE_CURRENT_BLOCK, NORMAL_MOVE_SPEED_MS)

    # --- In-Game Text ---
    next_block_text = font.render("Next figure:", True, (255, 255, 255), bgcolor)
    score_msg_text = font.render("Score:", True, (255, 255, 255), bgcolor)
    game_over_text = font.render("¡Game over!", True, (255, 220, 0), bgcolor)

    while run:
        if menu_active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                    sys.exit()

                player_type, ai_instance = menu.handle_input(event)
                if player_type:
                    menu_active = False
                    game_over = False
                    paused = False
                    blocks = BlocksGroup()

            menu.draw_menu()
            pygame.display.flip()
            continue

        # --- Main Game Loop ---
        if (
            player_type == "data_collection"
            and ai_instance
            and not game_over
            and not paused
        ):
            game_board_state = blocks.get_board_state_array()
            current_block_obj = blocks.current_block
            next_block_obj = blocks.next_block
            ai_instance.log_state(
                game_board_state,
                current_block_obj,
                next_block_obj,
                blocks.score,
                action_key=None,
            )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

            if event.type == pygame.KEYUP:
                if (
                    not paused
                    and not game_over
                    and (player_type == "human" or player_type == "data_collection")
                ):
                    if event.key in MOVEMENT_KEYS:
                        blocks.stop_moving_current_block()
                        if player_type == "data_collection":
                            game_board_state = blocks.get_board_state_array()
                            current_block_obj = blocks.current_block
                            next_block_obj = blocks.next_block
                            ai_instance.log_state(
                                game_board_state,
                                current_block_obj,
                                next_block_obj,
                                blocks.score,
                                event.key,
                            )
                if event.key == pygame.K_p:
                    paused = not paused
                if game_over and event.key == pygame.K_RETURN:
                    if player_type == "data_collection" and ai_instance:
                        ai_instance.save_data()
                    menu_active = True
                    game_over = False
                    paused = False

            if game_over or paused:
                if game_over and ai_instance and hasattr(ai_instance, "game_ended"):
                    ai_instance.game_ended()
                continue

            # --- AI Control Logic ---
            if (
                player_type
                in [
                    "random_ai",
                    "heuristic",
                    "evolutionary_ai",
                    "nn_ai",
                    "deep_qn_ai",
                    # "ai6",
                ]
                and ai_instance
                and not paused
                and not game_over
            ):
                game_board_state = blocks.get_board_state_array()
                current_block_obj = blocks.current_block
                next_block_obj = blocks.next_block
                ai_actions = ai_instance.get_next_move(
                    game_board_state, current_block_obj, next_block_obj
                )
                for action in ai_actions:
                    if action == pygame.K_LEFT:
                        blocks.start_moving_current_block(pygame.K_LEFT)
                        blocks.move_current_block()
                        blocks.stop_moving_current_block()
                    elif action == pygame.K_RIGHT:
                        blocks.start_moving_current_block(pygame.K_RIGHT)
                        blocks.move_current_block()
                        blocks.stop_moving_current_block()
                    elif action == pygame.K_UP:
                        blocks.rotate_current_block()
                    elif action == HARD_DROP_KEY:
                        try:
                            blocks.hard_drop_current_block()
                        except TopReached:
                            game_over = True
                            if ai_instance and hasattr(
                                ai_instance, "update_game_state"
                            ):
                                final_board_state = blocks.get_board_state_array()
                                ai_instance.update_game_state(final_board_state)
                            if player_type == "data_collection" and ai_instance:
                                ai_instance.save_data()

            # --- Human Input Processing ---
            if player_type == "human" or player_type == "data_collection":
                if event.type == pygame.KEYDOWN:
                    if event.key in MOVEMENT_KEYS:
                        blocks.start_moving_current_block(event.key)
                    elif event.key == pygame.K_UP:
                        blocks.rotate_current_block()
                        if player_type == "data_collection":
                            game_board_state = blocks.get_board_state_array()
                            current_block_obj = blocks.current_block
                            next_block_obj = blocks.next_block
                            ai_instance.log_state(
                                game_board_state,
                                current_block_obj,
                                next_block_obj,
                                blocks.score,
                                event.key,
                            )
                    elif event.key == HARD_DROP_KEY:
                        try:
                            blocks.hard_drop_current_block()
                            if player_type == "data_collection":
                                game_board_state = blocks.get_board_state_array()
                                current_block_obj = blocks.current_block
                                next_block_obj = blocks.next_block
                                ai_instance.log_state(
                                    game_board_state,
                                    current_block_obj,
                                    next_block_obj,
                                    blocks.score,
                                    event.key,
                                )
                        except TopReached:
                            game_over = True
                            if ai_instance and hasattr(
                                ai_instance, "update_game_state"
                            ):
                                final_board_state = blocks.get_board_state_array()
                                ai_instance.update_game_state(final_board_state)
                            if player_type == "data_collection" and ai_instance:
                                ai_instance.save_data()
                elif event.type == pygame.KEYUP:
                    if not paused and not game_over and event.key in MOVEMENT_KEYS:
                        blocks.stop_moving_current_block()
                        if player_type == "data_collection":
                            game_board_state = blocks.get_board_state_array()
                            current_block_obj = blocks.current_block
                            next_block_obj = blocks.next_block
                            ai_instance.log_state(
                                game_board_state,
                                current_block_obj,
                                next_block_obj,
                                blocks.score,
                                event.key,
                            )

            # --- Game Logic ---
            try:
                if event.type == EVENT_UPDATE_CURRENT_BLOCK:
                    blocks.update_current_block()
                elif event.type == EVENT_MOVE_CURRENT_BLOCK and (
                    player_type == "human" or player_type == "data_collection"
                ):
                    blocks.move_current_block()
            except TopReached:
                game_over = True
                if ai_instance and hasattr(ai_instance, "update_game_state"):
                    final_board_state = blocks.get_board_state_array()
                    ai_instance.update_game_state(final_board_state)
                if player_type == "data_collection" and ai_instance:
                    ai_instance.save_data()

        # --- Drawing ---
        screen.blit(background, (0, 0))

        if blocks.current_block:
            ghost_x, ghost_y = blocks.get_ghost_coords()
            ghost_image = blocks.current_block.image.copy()
            ghost_image.set_alpha(90)
            screen.blit(ghost_image, (ghost_x * TILE_SIZE, ghost_y * TILE_SIZE))

        blocks.draw(screen)
        draw_centered_surface(screen, next_block_text, 50)
        draw_centered_surface(screen, blocks.next_block.image, 100)
        draw_centered_surface(screen, score_msg_text, 240)
        score_text = font.render(str(blocks.score), True, (255, 255, 255), bgcolor)
        draw_centered_surface(screen, score_text, 270)

        if game_over:
            draw_centered_surface(screen, game_over_text, 360)
            restart_text = font.render(
                "Press ENTER to return to Menu", True, (255, 255, 255), bgcolor
            )
            draw_centered_surface(screen, restart_text, 400)
        elif paused:
            pause_text = menu_font.render("PAUSED", True, (255, 220, 0), bgcolor)
            draw_centered_surface(screen, pause_text, WINDOW_HEIGHT // 2)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
