import os
import numpy as np
import pygame
from core.bird import Bird
from config.settings import PIPE_IMAGE_PATH, PIPE_GAP, PIPE_VELOCITY

# Pipe class
class Pipe:
    """
    Represents the pipes in the game, both the top and bottom parts.
    Pipes move from right to left and create a gap for the bird to pass through.
    """
    def __init__(self, x: int, rng: np.random.Generator):
        """
        Initializes the pipe object with its position and random height.

        Args:
            x (int): The x-coordinate of the pipe's initial position.
            rng (np.random.Generator): Random number generator for setting the pipe's height.
        """
        self.x = x          # Horizontal position of the pipe
        self.rng = rng      # Random number generator for height

        # Load the pipe image
        raw_image = pygame.image.load(PIPE_IMAGE_PATH)
        self.pipe_image = raw_image  # Base pipe image

        # Gap between the top and bottom pipes
        self.gap = PIPE_GAP
        self.velocity = PIPE_VELOCITY  # Speed at which the pipes move to the left

        # Pipe positions
        self.height = 0
        self.top = 0
        self.bottom = 0

        # Flipped and non-flipped images for top and bottom pipes
        self.pipe_top = pygame.transform.flip(self.pipe_image, False, True)     # Flipped for the top pipe
        self.pipe_bottom = self.pipe_image                                      # Regular for the bottom pipe

        self.passed = False     # Tracks if the pipe has been passed by the bird (for scoring)
        self.set_height()       # Set the initial height of the pipes

    def set_height(self) -> None:
        """
        Sets the height of the top and bottom pipes, ensuring they stay within the screen.
        """
        # Minimum and maximum height constraints for the top pipe
        min_height = 50                     # Minimum height of the top pipe
        max_height = 480 - self.gap - 50    # Maximum height to ensure space for the bottom pipe

        # Randomly determine the height of the pipes within the allowed range
        self.height = self.rng.integers(low=min_height, high=max_height)

        # Calculate positions for the top and bottom pipes
        self.top = self.height - self.pipe_top.get_height()     # Position of the top pipe
        self.bottom = self.height + self.gap                    # Position of the bottom pipe

    def draw(self, surface: pygame.Surface) -> None:
        """
        Draws the top and bottom pipes on the screen.

        Args:
            surface (pygame.Surface): The surface to draw the pipes on.
        """
        surface.blit(self.pipe_top, (self.x, self.top))             # Draw top pipe
        surface.blit(self.pipe_bottom, (self.x, self.bottom))       # Draw bottom pipe

    def move(self) -> None:
        """
        Moves the pipes to the left by their velocity.
        """
        self.x -= self.velocity  # Decrease the x-coordinate to move the pipes left

    def collide(self, bird: Bird) -> bool:
        """
        Checks if the bird collides with the top or bottom pipes.

        Args:
            bird (Bird): The bird object to check for collisions.

        Returns:
            bool: True if the bird collides with either pipe, False otherwise.
        """
        # Create masks for collision detection
        bird_mask = bird.get_mask()                                 # Mask for the bird
        top_mask = pygame.mask.from_surface(self.pipe_top)          # Mask for the top pipe
        bottom_mask = pygame.mask.from_surface(self.pipe_bottom)    # Mask for the bottom pipe

        # Calculate offsets for masks based on the pipe's and bird's positions
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        # Check for overlaps between the bird and the pipes
        top_collision = bird_mask.overlap(top_mask, top_offset)
        bottom_collision = bird_mask.overlap(bottom_mask, bottom_offset)

        # Return True if there is a collision, otherwise False
        return bool(top_collision or bottom_collision)
