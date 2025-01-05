import os, pygame
from config.settings import BASE_IMAGE_PATH 

# Base class
class Base:
    def __init__(self, y: int):
        """
        Initialize the Base object (the ground).

        Args:
            y (int): The vertical position of the base (ground level).
        """
        self.y = y  # Set the vertical position of the base.

        # Load the base image
        raw_image = pygame.image.load(BASE_IMAGE_PATH)
        self.base_image = raw_image  # The raw base image.

        self.velocity = 5                           # Speed at which the base moves left.
        self.width = self.base_image.get_width()    # Width of the base image.
        self.image = self.base_image                # Store the base image for use.

        # Define two x-coordinates to handle the scrolling effect.
        self.x1 = 0             # Position of the first base image. 
        self.x2 = self.width    # Position of the second base image (right after the first one).

    def draw(self, surface: pygame.Surface) -> None:
        """
        Draw the base on the screen.

        Args:
            surface (pygame.Surface): The game window or surface to draw on.
        """
        # Draw the first base image.
        surface.blit(self.image, (self.x1, self.y))
        # Draw the second base image.
        surface.blit(self.image, (self.x2, self.y))

    def move(self) -> None:
        """
        Move the base to create a scrolling effect.
        """
        # Move both base images to the left by the set velocity.
        self.x1 -= self.velocity
        self.x2 -= self.velocity

        # If the first base moves completely off-screen, reset its position
        # to the right of the second base.
        if self.x1 + self.width < 0:
            self.x1 = self.x2 + self.width

        # If the second base moves completely off-screen, reset its position
        # to the right of the first base.
        if self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width
