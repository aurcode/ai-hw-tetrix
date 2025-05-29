# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pygame",
# ]
# ///

import pygame

from constants import WINDOW_WIDTH, WINDOW_HEIGHT, TILE_SIZE, GRID_WIDTH, GRID_HEIGHT
from utils import draw_grid, draw_centered_surface
from block import TopReached
from blocks_group import BlocksGroup


def main():
    pygame.init()
    pygame.display.set_caption("Tetris with PyGame")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    run = True
    paused = False
    game_over = False
    # Create background.
    background = pygame.Surface(screen.get_size())
    bgcolor = (0, 0, 0)
    background.fill(bgcolor)
    # Draw the grid on top of the background.
    draw_grid(background)
    # This makes blitting faster.
    background = background.convert()

    try:
        font = pygame.font.Font("Roboto-Regular.ttf", 20)
    except OSError:
        # If the font file is not available, the default will be used.
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
    next_block_text = font.render("Next figure:", True, (255, 255, 255), bgcolor)
    score_msg_text = font.render("Score:", True, (255, 255, 255), bgcolor)
    game_over_text = font.render("¡Game over!", True, (255, 220, 0), bgcolor)

    # Event constants.
    MOVEMENT_KEYS = pygame.K_LEFT, pygame.K_RIGHT, pygame.K_DOWN
    EVENT_UPDATE_CURRENT_BLOCK = pygame.USEREVENT + 1
    EVENT_MOVE_CURRENT_BLOCK = pygame.USEREVENT + 2
    pygame.time.set_timer(EVENT_UPDATE_CURRENT_BLOCK, 1000)
    pygame.time.set_timer(EVENT_MOVE_CURRENT_BLOCK, 100)

    blocks = BlocksGroup()

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            elif event.type == pygame.KEYUP:
                if not paused and not game_over:
                    if event.key in MOVEMENT_KEYS:
                        blocks.stop_moving_current_block()
                    elif event.key == pygame.K_UP:
                        blocks.rotate_current_block()
                if event.key == pygame.K_p:
                    paused = not paused

            # Stop moving blocks if the game is over or paused.
            if game_over or paused:
                continue

            if event.type == pygame.KEYDOWN:
                if event.key in MOVEMENT_KEYS:
                    blocks.start_moving_current_block(event.key)

            try:
                if event.type == EVENT_UPDATE_CURRENT_BLOCK:
                    blocks.update_current_block()
                elif event.type == EVENT_MOVE_CURRENT_BLOCK:
                    blocks.move_current_block()
            except TopReached:
                game_over = True

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
        # Update.
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
