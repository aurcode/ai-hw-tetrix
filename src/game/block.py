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

    def __init__(self):
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
        self.struct = np.array(self.struct)
        self.shape_name = self.__class__.__name__  # Add shape_name attribute
        # Initial random rotation and flip.
        self.rotation = 0  # Initialize rotation
        if random.randint(0, 1):
            self.struct = np.rot90(self.struct)
            self.rotation = (self.rotation + 90) % 360
        if random.randint(0, 1):
            # Flip in the X axis.
            self.struct = np.flip(self.struct, 0)
        self._draw()

    def _draw(self, x=4, y=0):
        width = len(self.struct[0]) * TILE_SIZE
        height = len(self.struct) * TILE_SIZE
        self.image = pygame.surface.Surface([width, height])
        self.image.set_colorkey((0, 0, 0))
        # Position and size
        self.rect = Rect(0, 0, width, height)
        self.x = x
        self.y = y
        for y_pos, row in enumerate(self.struct):
            for x_pos, col in enumerate(row):
                if col:
                    pygame.draw.rect(
                        self.image,
                        self.color,
                        Rect(
                            x_pos * TILE_SIZE + 1,
                            y_pos * TILE_SIZE + 1,
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
        for _ in range(n_rotations):
            self.image = pygame.transform.rotate(self.image, 90)
            # Once rotated we need to update the size and position.
            self.rect.width = self.image.get_width()
            self.rect.height = self.image.get_height()
            self._create_mask()
            # Check the new position doesn't exceed the limits or collide
            # with other blocks and adjust it if necessary.
            while self.rect.right > GRID_WIDTH:
                self.x -= 1
            while self.rect.left < 0:
                self.x += 1
            while self.rect.bottom > GRID_HEIGHT:
                self.y -= 1
            while True:
                if group is None or not Block.collide(self, group, ignore_block):
                    break
                self.y -= 1
            self.struct = np.rot90(self.struct)
            self.rotation = (self.rotation + 90) % 360

    def get_points(self):
        """
        Returns a list of (x, y) tuples representing the global grid coordinates
        of each filled tile in the block.
        """
        points = []
        for y_offset, row in enumerate(self.struct):
            for x_offset, cell in enumerate(row):
                if cell:
                    points.append((self.x + x_offset, self.y + y_offset))
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
        new_block = type(self)()
        new_block.x = self.x
        new_block.y = self.y
        new_block.color = self.color
        new_block.struct = np.copy(self.struct)  # Use np.copy for numpy arrays
        new_block.current = self.current
        new_block.rotation = self.rotation
        new_block.redraw()  # Redraw to update image and mask
        return new_block


class SquareBlock(Block):
    struct = ((1, 1), (1, 1))


class TBlock(Block):
    struct = ((1, 1, 1), (0, 1, 0))


class LineBlock(Block):
    struct = ((1,), (1,), (1,), (1,))


class LBlock(Block):
    struct = (
        (1, 1),
        (1, 0),
        (1, 0),
    )


class ZBlock(Block):
    struct = (
        (0, 1),
        (1, 1),
        (1, 0),
    )
