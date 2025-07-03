import pygame
import numpy as np
import random
from pygame import Rect
from src.common.constants import TILE_SIZE, GRID_WIDTH, GRID_HEIGHT


class BottomReached(Exception):
    pass


class TopReached(Exception):
    pass


class Block(pygame.sprite.Sprite):

    @staticmethod
    def collide(block, group, ignore_block=None):  # Added ignore_block parameter
        """
        Check if the specified block collides with some other block
        in the group.
        """
        for other_block in group:
            # Ignore the current block which will always collide with itself,
            # or any specified block to ignore.
            if block == other_block or other_block == ignore_block:
                continue
            if pygame.sprite.collide_mask(block, other_block) is not None:
                return True
        return False

    def __init__(self, x, y, shapes):
        super().__init__()
        # Get a random color.
        self.color = random.choice(
            (
                (200, 200, 200),
                (215, 133, 133),
                (30, 145, 255),
                (0, 170, 0),
                (180, 0, 140),
                (200, 200, 0),
            )
        )
        self.current = True
        self.shapes = shapes
        self.shape_name = self.__class__.__name__  # Add shape_name attribute
        self.rotation = 0  # Initialize rotation
        self.struct = self.shapes[self.rotation]  # Initial struct based on 0 rotation
        self._draw(x, y)

    def _draw(self, x=4, y=0):
        # Determine the maximum width and height needed for the current shape
        max_x = 0
        max_y = 0
        for sx, sy in self.struct:
            max_x = max(max_x, sx)
            max_y = max(max_y, sy)

        width = (max_x + 1) * TILE_SIZE
        height = (max_y + 1) * TILE_SIZE

        self.image = pygame.surface.Surface([width, height])
        self.image.set_colorkey((0, 0, 0))
        # Position and size
        self.rect = Rect(0, 0, width, height)
        self.x = x
        self.y = y
        for sx, sy in self.struct:
            pygame.draw.rect(
                self.image,
                self.color,
                Rect(
                    sx * TILE_SIZE + 1,
                    sy * TILE_SIZE + 1,
                    TILE_SIZE - 2,
                    TILE_SIZE - 2,
                ),
            )
        self._create_mask()

    def redraw(self):
        self._draw(self.x, self.y)

    def _create_mask(self):
        """
        Create the mask attribute from the main surface.
        The mask is required to check collisions. This should be called
        after the surface is created or update.
        """
        self.mask = pygame.mask.from_surface(self.image)

    def initial_draw(self):
        raise NotImplementedError

    @property
    def group(self):
        return self.groups()[0]

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self.rect.left = value * TILE_SIZE

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
        self.rect.top = value * TILE_SIZE

    def move_left(self, group, ignore_block=None):
        self.x -= 1
        # Check if we reached the left margin.
        if self.x < 0 or Block.collide(self, group, ignore_block):
            self.x += 1

    def move_right(self, group, ignore_block=None):
        self.x += 1
        # Check if we reached the right margin or collided with another
        # block.
        if self.rect.right > GRID_WIDTH or Block.collide(self, group, ignore_block):
            # Rollback.
            self.x -= 1

    def move_down(self, group, ignore_block=None):
        self.y += 1
        # Check if the block reached the bottom or collided with
        # another one.
        if self.rect.bottom > GRID_HEIGHT or Block.collide(self, group, ignore_block):
            # Rollback to the previous position.
            self.y -= 1
            self.current = False
            raise BottomReached

    def rotate(self, group, ignore_block=None, n_rotations=1):
        original_rotation = self.rotation
        original_struct = self.struct
        original_image = self.image.copy()
        original_rect = self.rect.copy()
        original_mask = self.mask.copy()
        original_x = self.x
        original_y = self.y

        try:
            for _ in range(n_rotations):
                self.rotation = (self.rotation + 90) % 360
                if self.rotation not in self.shapes:
                    # If the new rotation doesn't have a defined shape, try the next one
                    # This handles cases like SquareBlock where 0, 90, 180, 270 all map to the same shape
                    for r in [0, 90, 180, 270]:
                        if (self.rotation + r) % 360 in self.shapes:
                            self.rotation = (self.rotation + r) % 360
                            break
                self.struct = self.shapes[self.rotation]
                self.redraw()  # Redraw to update image, rect, and mask based on new struct

                # Check the new position doesn't exceed the limits or collide
                # with other blocks and adjust it if necessary.
                # Apply wall kicks if necessary (simple implementation for now)
                kick_offsets = [
                    (0, 0),
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ]  # Basic kick attempts
                kick_success = False
                for dx, dy in kick_offsets:
                    self.x += dx
                    self.y += dy
                    if (
                        self.rect.left >= 0
                        and self.rect.right <= GRID_WIDTH
                        and self.rect.bottom <= GRID_HEIGHT
                        and (
                            group is None
                            or not Block.collide(self, group, ignore_block)
                        )
                    ):
                        kick_success = True
                        break
                    else:
                        self.x -= dx  # Rollback
                        self.y -= dy  # Rollback

                if not kick_success:
                    raise ValueError("Rotation failed due to collision/boundary")

        except ValueError:
            # Rollback all changes if rotation fails
            self.rotation = original_rotation
            self.struct = original_struct
            self.image = original_image
            self.rect = original_rect
            self.mask = original_mask
            self.x = original_x
            self.y = original_y
            self.redraw()  # Ensure image/mask/rect are consistent with rolled back state

    def get_points(self):
        """
        Returns a list of (x, y) tuples representing the global grid coordinates
        of each filled tile in the block.
        """
        points = []
        for sx, sy in self.struct:
            points.append((self.x + sx, self.y + sy))
        return points

    def update(self):
        if self.current:
            self.move_down()

    def clone(self):
        """
        Creates a new Block instance with the same properties as the current one.
        Used for ghost piece calculation.
        """
        # Create a new instance of the same class (e.g., SquareBlock, TBlock)
        new_block = type(self)(self.x, self.y)  # Pass only x and y
        new_block.color = self.color
        new_block.current = self.current
        new_block.rotation = self.rotation
        new_block.struct = new_block.shapes[
            new_block.rotation
        ]  # Set struct based on current rotation
        new_block.redraw()  # Redraw to update image and mask
        return new_block


class SquareBlock(Block):
    """The O-block."""

    def __init__(self, x, y):
        shapes = {
            0: [(0, 0), (1, 0), (0, 1), (1, 1)],
        }
        # Since all rotations are the same, we only need one.
        shapes[90] = shapes[180] = shapes[270] = shapes[0]
        super().__init__(x, y, shapes)


class TBlock(Block):
    """The T-block."""

    def __init__(self, x, y):
        shapes = {
            0: [(0, 1), (1, 1), (2, 1), (1, 0)],
            90: [(1, 0), (1, 1), (1, 2), (0, 1)],
            180: [(0, 1), (1, 1), (2, 1), (1, 2)],
            270: [(0, 0), (0, 1), (0, 2), (1, 1)],
        }
        super().__init__(x, y, shapes)


class LineBlock(Block):
    """The I-block."""

    def __init__(self, x, y):
        shapes = {
            0: [(0, 1), (1, 1), (2, 1), (3, 1)],
            90: [(0, 0), (0, 1), (0, 2), (0, 3)],  # Normalized
            180: [(0, 2), (1, 2), (2, 2), (3, 2)],
            270: [(0, 0), (0, 1), (0, 2), (0, 3)],  # Normalized
        }
        super().__init__(x, y, shapes)


class LBlock(Block):
    """The L-block."""

    def __init__(self, x, y):
        shapes = {
            0: [(0, 1), (1, 1), (2, 1), (2, 0)],
            90: [(0, 0), (0, 1), (0, 2), (1, 2)],
            180: [(0, 2), (1, 2), (2, 2), (0, 1)],
            270: [(0, 0), (1, 0), (1, 1), (1, 2)],
        }
        super().__init__(x, y, shapes)


class JBlock(Block):
    """The J-block (mirrored L)."""

    def __init__(self, x, y):
        shapes = {
            0: [(0, 0), (0, 1), (1, 1), (2, 1)],
            90: [(0, 0), (1, 0), (0, 1), (0, 2)],
            180: [(0, 1), (1, 1), (2, 1), (2, 2)],
            270: [(1, 0), (1, 1), (1, 2), (0, 2)],
        }
        super().__init__(x, y, shapes)


class SBlock(Block):
    """The S-block."""

    def __init__(self, x, y):
        shapes = {
            0: [(1, 0), (2, 0), (0, 1), (1, 1)],
            90: [(0, 0), (0, 1), (1, 1), (1, 2)],
            180: [(1, 0), (2, 0), (0, 1), (1, 1)],  # Same as 0
            270: [(0, 0), (0, 1), (1, 1), (1, 2)],  # Same as 90
        }
        super().__init__(x, y, shapes)


class ZBlock(Block):
    """The Z-block (mirrored S)."""

    def __init__(self, x, y):
        shapes = {
            0: [(0, 0), (1, 0), (1, 1), (2, 1)],
            90: [(1, 0), (0, 1), (1, 1), (0, 2)],  # Normalized
            180: [(0, 0), (1, 0), (1, 1), (2, 1)],  # Same as 0
            270: [(1, 0), (0, 1), (1, 1), (0, 2)],  # Normalized
        }
        super().__init__(x, y, shapes)
