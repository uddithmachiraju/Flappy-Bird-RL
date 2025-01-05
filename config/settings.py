# Game settings
import os 

# Screen Settings
SCREEN_WIDTH = 350
SCREEN_HEIGHT = 480 
FPS = 30

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Store the highscore 
HIGHSCORE_FILE = os.path.join(BASE_DIR, "core", "highscore.json")

# Path to the assets folder
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

# Background Settings
BACKGROUND_IMAGE_PATH = os.path.join("assets", "background.png")  # Path to the background image
BACKGROUND_SCALE = 1.21        # Scale factor for the background image
BACKGROUND_POSITION = (0, -50)    

# Bird Animation Settings
UPFLAP_IMAGE_PATH = os.path.join(ASSETS_DIR, "upflap.png")
MIDFLAP_IMAGE_PATH = os.path.join(ASSETS_DIR, "midflap.png")
DOWNFLAP_IMAGE_PATH = os.path.join(ASSETS_DIR, "downflap.png")
BIRD_INITIAL_X = 100       # Initial X-coordinate of the bird
BIRD_INITIAL_Y = 200       # Initial Y-coordinate of the bird

# Pipe Settings
PIPE_IMAGE_PATH = os.path.join("assets", "pipe.png")  # Path to the pipe image
PIPE_GAP = 150              # Gap between the top and bottom pipes
PIPE_VELOCITY = 5           # Speed of pipe movement
PIPE_SPAWN_COORDINATE = 400 # Spawning Pipes coordinate

# Base Settings
BASE_IMAGE_PATH = os.path.join("assets", "base.png")  # Path to the base image
BASE_HEIGHT = 50            # Height of the base from the bottom

# Fonts
FONT_PATH = os.path.join("assets", "flappy-font.ttf")  # Path to the font file
SCORE_FONT_SIZE = 25        # Font size for the score display
TITLE_FONT_SIZE = 35        # Font size for the title screen
BUTTON_FONT_SIZE = 25       # Font size for the button text

# Colors
WHITE = (255, 255, 255) 
BLACK = (0, 0, 0) 