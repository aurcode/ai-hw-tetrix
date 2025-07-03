import pygame
import sys
from src.common.constants import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    NORMAL_FALL_SPEED_MS,
    FAST_FALL_SPEED_MS,
    NORMAL_MOVE_SPEED_MS,
    FAST_MOVE_SPEED_MS,
)
from src.agent import (
    RandomAIPlayer,
    DataCollector,
    NNAIPlayer,
    # QLearningAIPlayer,
    HeuristicAIPlayer,
    # DeepQNAIPlayer,
)
from src.common.constants import BOARD_WIDTH_TILES, BOARD_HEIGHT_TILES


class Menu:
    def __init__(self, screen):
        self.screen = screen
        self.bgcolor = (0, 0, 0)
        try:
            self.menu_font = pygame.font.Font("Roboto-Regular.ttf", 30)
        except OSError:
            self.menu_font = pygame.font.Font(pygame.font.get_default_font(), 30)

        self.menu_options = [
            "Human Player",
            "Random AI Player",
            "Data Collection Player",
            "Heuristic AI Player (Gameplay)",
            "Evolutionary AI Player (Gameplay)",
            "NN AI Player (Gameplay)",
            "DeepQN AI Player (Gameplay)",
        ]
        self.selected_option_index = 0
        self.selected_option_color = (255, 220, 0)  # Yellow for selected

    def draw_menu(self):
        self.screen.fill(self.bgcolor)

        title_text = self.menu_font.render("Select Player Mode", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 4))
        self.screen.blit(title_text, title_rect)

        for i, option in enumerate(self.menu_options):
            color = (
                self.selected_option_color
                if i == self.selected_option_index
                else (255, 255, 255)
            )
            text = self.menu_font.render(option, True, color)
            rect = text.get_rect(
                center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + i * 50)
            )
            self.screen.blit(text, rect)

    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_option_index = (self.selected_option_index - 1) % len(
                    self.menu_options
                )
            elif event.key == pygame.K_DOWN:
                self.selected_option_index = (self.selected_option_index + 1) % len(
                    self.menu_options
                )
            elif event.key == pygame.K_RETURN:
                return self.select_option()
        return None, None

    def select_option(self):
        selected_option = self.menu_options[self.selected_option_index]
        player_type = None
        ai_instance = None

        if selected_option == "Human Player":
            player_type = "human"

        elif selected_option == "Random AI Player":
            player_type = "random_ai"
            ai_instance = RandomAIPlayer(
                grid_width=BOARD_WIDTH_TILES, grid_height=BOARD_HEIGHT_TILES
            )

        elif selected_option == "Data Collection Player":
            player_type = "data_collection"
            ai_instance = DataCollector(
                grid_width=BOARD_WIDTH_TILES, grid_height=BOARD_HEIGHT_TILES
            )

        elif selected_option == "Heuristic AI Player (Gameplay)":
            player_type = "heuristic"
            print("Selected - HeuristicAIPlayer.")
            ai_instance = HeuristicAIPlayer()

        elif selected_option == "Evolutionary AI Player (Gameplay)":
            player_type = "evolutionary_ai"
            print("Selected - EvolutionaryAIPlayer.")
            ai_instance = RandomAIPlayer(
                grid_width=BOARD_WIDTH_TILES, grid_height=BOARD_HEIGHT_TILES
            )

        elif selected_option == "NN AI Player (Gameplay)":
            player_type = "nn_ai"
            ai_instance = NNAIPlayer(
                grid_width=BOARD_WIDTH_TILES, grid_height=BOARD_HEIGHT_TILES
            )
        elif selected_option == "DeepQN AI Player (Gameplay)":
            player_type = "deep_qn_ai"
            ai_instance = RandomAIPlayer(
                grid_width=BOARD_WIDTH_TILES,
                grid_height=BOARD_HEIGHT_TILES,
                # load_model=True,
            )

        if player_type:
            if player_type in [
                "random_ai",
                "heuristic",
                "evolutionary_ai",
                "nn_ai",
                "deep_qn_ai",
                # "ai6",
            ]:
                pygame.time.set_timer(pygame.USEREVENT + 1, FAST_FALL_SPEED_MS)
                pygame.time.set_timer(pygame.USEREVENT + 2, FAST_MOVE_SPEED_MS)
            else:
                pygame.time.set_timer(pygame.USEREVENT + 1, NORMAL_FALL_SPEED_MS)
                pygame.time.set_timer(pygame.USEREVENT + 2, NORMAL_MOVE_SPEED_MS)

        return player_type, ai_instance
