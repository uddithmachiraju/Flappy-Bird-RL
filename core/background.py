from config.settings import BACKGROUND_IMAGE_PATH, BACKGROUND_SCALE, BACKGROUND_POSITION
import os, pygame 

# Background class
class Background:
    def __init__(self):
        
        # Load background image
        background_image = pygame.image.load(BACKGROUND_IMAGE_PATH)  
        original_width, original_height = background_image.get_size()
        new_width = int(original_width * BACKGROUND_SCALE)
        new_height = int(original_height * BACKGROUND_SCALE)
        self.background_image = pygame.transform.scale(background_image, (new_width, new_height))

    def draw(self, surface):
        """
        Draw the background on the screen.

        Args:
            surface (pygame.Surface): The surface to draw the background on.
        """
        surface.blit(self.background_image, BACKGROUND_POSITION)  