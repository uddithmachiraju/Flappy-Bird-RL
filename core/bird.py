import os
import pygame
from config.settings import UPFLAP_IMAGE_PATH, MIDFLAP_IMAGE_PATH, DOWNFLAP_IMAGE_PATH

# Bird class
class Bird:
    def __init__(self, x, y):
        """
        Initialize the Bird object.

        Args:
            x (int): The x-coordinate of the bird.
            y (int): The y-coordinate of the bird.
        """
        # Bird coordinates
        self.x = x
        self.y = y

        # Load and store the images in sequence for animation
        sprites = [UPFLAP_IMAGE_PATH, MIDFLAP_IMAGE_PATH, DOWNFLAP_IMAGE_PATH]
        raw_images = [pygame.image.load(sprite) for sprite in sprites]
        self.images = raw_images  # Bird animation frames

        # Bird physics and animation parameters
        self.max_rotation = 25              # Maximum upward tilt of the bird
        self.rotation_velocity = 20         # Speed of rotation while falling
        self.animation_time = 2             # Time before switching to the next animation frame

        # Bird state variables
        self.tilt = 0                       # Current tilt angle
        self.tick_count = 0                 # Tracks the number of frames since the last jump
        self.velocity = 0                   # Vertical velocity
        self.height = self.y                # Height where the bird last jumped
        self.image_count = 0                # Tracks frames for animation
        self.image = self.images[0]         # Current image to display

    def jump(self):
        """
        Make the bird jump by resetting its velocity and tick count.
        """
        self.velocity = -5               # Negative velocity to move upward
        self.tick_count = 0                 # Reset frame counter for smooth jump
        self.height = self.y                # Set the current height as the jump start point

    def draw(self, surface):
        """
        Draw the bird on the screen with proper animation and rotation.

        Args:
            surface (pygame.Surface): The surface to draw the bird on.
        """
        # Update animation frame
        self.image_count += 1

        # Cycle through the bird images for flapping animation
        if self.image_count < self.animation_time:
            self.image = self.images[0]             # Upflap
        elif self.image_count < self.animation_time * 2:
            self.image = self.images[1]             # Midflap
        elif self.image_count < self.animation_time * 3:
            self.image = self.images[2]             # Downflap
        elif self.image_count < self.animation_time * 4:
            self.image = self.images[1]             # Midflap
        elif self.image_count < self.animation_time * 4 + 1:
            self.image = self.images[0]             # Reset to Upflap
            self.image_count = 0  # Reset animation counter

        # If the bird is diving (tilt <= -80), freeze animation at midflap
        if self.tilt <= -80:
            self.image = self.images[1]  # Midflap
            self.image_count = self.animation_time * 2

        # Rotate the bird based on its tilt angle
        rotated_image = pygame.transform.rotate(self.image, self.tilt)
        new_rect = rotated_image.get_rect(
            center=self.image.get_rect(topleft=(self.x, self.y)).center
        )

        # Draw the rotated bird on the screen
        surface.blit(rotated_image, new_rect.topleft)

    def move(self):
        """
        Update the bird's position and tilt based on its velocity and gravity.
        """
        self.tick_count += 1  # Increment frame count

        # Calculate displacement using velocity and gravity
        displacement = self.velocity * self.tick_count + 1.2 * self.tick_count ** 2

        # Limit the maximum downward displacement
        if displacement > 6:
            displacement = (displacement / abs(displacement)) * 6

        # Make upward motion slightly more pronounced
        if displacement < -10:
            displacement = -10

        # Update the bird's vertical position
        self.y += displacement

        # Adjust the bird's tilt
        if displacement < 0:  # Bird is moving upwards
            self.tilt = max(self.tilt, self.max_rotation)  # Tilt upwards
        elif self.tilt > -90:  # Bird is falling
            self.tilt -= self.rotation_velocity  # Tilt downwards gradually

    def get_mask(self):
        """
        Get a collision mask for the bird's current image.

        Returns:
            pygame.Mask: A mask of the bird's current image.
        """
        return pygame.mask.from_surface(self.image)
