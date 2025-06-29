from abc import ABC, abstractmethod

class BaseAIPlayer(ABC):
    """
    Abstract base class for all AI players in the Tetris game.

    This class defines the essential methods that every AI agent must implement,
    ensuring a consistent interface for the main game loop to interact with.
    """

    @abstractmethod
    def get_next_move(self, game_board_state, current_block, next_block):
        """
        Determines the next sequence of actions for the AI to perform.

        Args:
            game_board_state (np.array): The current state of the game board.
            current_block (Block): The currently falling block.
            next_block (Block): The upcoming block.

        Returns:
            list: A list of pygame keys (e.g., pygame.K_LEFT) representing the
                  sequence of actions to take.
        """
        pass

    def game_ended(self):
        """
        Optional method to notify the AI that the game has ended.
        This can be used for cleanup, logging, or model updates.
        """
        pass

    def update_game_state(self, game_board_state):
        """
        Optional method to update the AI with the latest game state after a
        block has landed.
        """
        pass