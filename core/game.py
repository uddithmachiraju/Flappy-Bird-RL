import pygame
import sys
import json 
import numpy as np

from core.bird import Bird
from core.base import Base
from core.pipe import Pipe
from core.background import Background
from config.settings import * 
from core.state_utils import preprocess_state

# Initialize Pygame
pygame.init()

# Scaled fonts for smaller screen
SCORE_FONT = pygame.font.Font('Assets/flappy-font.ttf', SCORE_FONT_SIZE)  # Smaller font for score

# Main game loop
class Game:
    def __init__(self):
        """
        Initializes the game, creating the display, clock, and necessary game objects.
        """
        # Setup
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird")
        self.clock = pygame.time.Clock()

        # Game objects
        self.bird = Bird(BIRD_INITIAL_X, BIRD_INITIAL_Y)
        self.base = Base(SCREEN_HEIGHT - BASE_HEIGHT)
        self.background = Background()
        self.pipes = [Pipe(PIPE_SPAWN_COORDINATE, np.random.default_rng())]

        self.score = 0
        self.reward = 0
        self.running = True
        self.start_screen = True
        self.highscore = self.load_highscore()

    def load_highscore(self):
        """
        Loads the high score from the JSON file.
        """
        try:
            with open(HIGHSCORE_FILE, 'r') as file:
                data = json.load(file)
                return data.get("highscore", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            return 0

    def save_highscore(self, new_highscore):
        """
        Saves the high score to the JSON file if the new score is higher.
        """
        if new_highscore > self.highscore:
            self.highscore = new_highscore
            with open(HIGHSCORE_FILE, 'w') as file:
                json.dump({"highscore": self.highscore}, file)

    def draw_start_screen(self):
        """
        Draws the start screen with a "Start" button and title.
        """
        self.background.draw(self.screen)

        # Title text
        title_font = pygame.font.Font('Assets/flappy-font.ttf', TITLE_FONT_SIZE)  # Smaller font for title
        title_text = title_font.render("Flappy Bird", True, WHITE) 
        self.screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 100))

        # Start button
        button_font = pygame.font.Font('Assets/flappy-font.ttf', BUTTON_FONT_SIZE) 
        start_text = button_font.render("Start", True, WHITE) 
        button_rect = start_text.get_rect(center=(SCREEN_WIDTH // 2, 220))
        self.screen.blit(start_text, button_rect.topleft)

        # High score
        highscore_text = SCORE_FONT.render(f"High Score: {self.highscore}", True, WHITE)
        self.screen.blit(highscore_text, (SCREEN_WIDTH // 2 - highscore_text.get_width() // 2, 300))

        pygame.display.update()
        return button_rect

    def draw_score(self):
        """
        Draws the score in the center of the screen with a pixelated font.
        """
        score_text = SCORE_FONT.render(str(self.score), True, WHITE)
        self.screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 20))

    def reset(self):
        """
        Resets the game to its initial state for a new episode.
        """
        self.score = 0
        self.reward = -5
        self.running = True    
        self.bird = Bird(BIRD_INITIAL_X, BIRD_INITIAL_Y)
        self.base = Base(SCREEN_HEIGHT - BASE_HEIGHT)
        self.background = Background()
        self.pipes = [Pipe(PIPE_SPAWN_COORDINATE, np.random.default_rng())]
        self.start_screen = False

    def handle_events(self, human = False, AI = False, action = None): 
        """
        Handles events like quitting and bird's jump.
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False 
                pygame.quit()
                sys.exit()

            if human:
                if self.start_screen and event.type == pygame.MOUSEBUTTONDOWN:
                    button_rect = self.draw_start_screen()  # Removed self.screen argument
                    if button_rect.collidepoint(event.pos):
                        self.start_screen = False

                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.bird.jump() 
                    pygame.display.update()

            else:
                if self.start_screen and event.type == pygame.MOUSEBUTTONDOWN:
                    button_rect = self.draw_start_screen()  # Removed self.screen argument
                    if button_rect.collidepoint(event.pos):
                        self.start_screen = False

        if action is not None:
            if action == 1: 
                # self.reward = 0.1
                self.bird.jump() 
            # else: 
            #     self.bird.velocity = 0
                    
    def update(self):
        """
        Updates the game state including movement of bird, pipes, and base.
        Adjusts rewards dynamically.
        """
        self.bird.move()
        self.base.move()

        # Initialize frame reward
        self.reward = 0.01  # Reward for survival

        for pipe in self.pipes:
            pipe.move()
            if pipe.x + pipe.pipe_top.get_width() < 0:
                self.pipes.remove(pipe)
                self.pipes.append(Pipe(SCREEN_WIDTH + 10, np.random.default_rng()))

            # Collision penalty
            if pipe.collide(self.bird):
                self.running = False
                self.reward = -10  # Large penalty for collision

            # Passing pipe reward
            if not pipe.passed and pipe.x < self.bird.x:
                pipe.passed = True
                self.score += 1
                difficulty_factor = 1.0 if PIPE_GAP > 145 else 2.0  # Higher reward for harder gaps 
                self.reward = 5 * difficulty_factor

        # Check for ground or top collision
        if self.bird.y + self.bird.image.get_height() >= self.base.y or self.bird.y <= 0:
            self.running = False
            self.reward = -10

        # Additional penalties
        if self.running:
            
            # Distance from pipe center penalty
            pipes = [pipe for pipe in self.pipes if pipe.x + pipe.pipe_top.get_width() > self.bird.x]
            if pipes:
                next_pipe = pipes[0]
                pipe_center = ((next_pipe.bottom - PIPE_GAP) + next_pipe.bottom) / 2
                distance_penalty = -((abs(self.bird.y - pipe_center) / SCREEN_HEIGHT) / 10)
                self.reward += distance_penalty


    def draw(self):
        """
        Draws all game objects on the screen and visualizes debugging information.
        """
        self.background.draw(self.screen)
        for pipe in self.pipes:
            pipe.draw(self.screen)
        self.base.draw(self.screen)
        self.bird.draw(self.screen)
        self.draw_score()

        # Get the state values
        bird_y = self.bird.y
        
        # Find the next pipe for pipe distance, top, and bottom
        pipes = [pipe for pipe in self.pipes if pipe.x + pipe.pipe_top.get_width() > self.bird.x]
        if pipes:
            next_pipe = pipes[0]
            pipe_dist = next_pipe.x - self.bird.x
            pipe_top = next_pipe.bottom - PIPE_GAP
            pipe_bottom = next_pipe.bottom
            pipe_gap_center = (pipe_top + pipe_bottom) / 2
        else:
            pipe_dist, pipe_top, pipe_bottom, pipe_gap_center = 0, 0, 0, 0
        
        # Calculate relative position of the bird
        relative_bird_position = bird_y + 15

        # Draw debug markers for the pipe positions and relative bird position
        if pipes:
            # Mark pipe top and bottom with red circles
            pygame.draw.circle(self.screen, (255, 0, 0), 
                            (next_pipe.x + next_pipe.pipe_top.get_width() // 2, pipe_top), 5)  # Top marker
            pygame.draw.circle(self.screen, (255, 0, 0), 
                            (next_pipe.x + next_pipe.pipe_top.get_width() // 2, pipe_bottom), 5)  # Bottom marker
            
            # Mark the bird's relative position in relation to the pipe gap center with a green circle
            pygame.draw.circle(self.screen, (0, 255, 0), 
                            (self.bird.x, relative_bird_position), 5)  # Bird position marker
            
            # Mark the pipe gap center with a blue circle
            pygame.draw.circle(self.screen, (0, 0, 255), 
                            (next_pipe.x + next_pipe.pipe_top.get_width() // 2, pipe_gap_center), 5)  # Gap center marker


        pygame.display.update()


    def get_state(self):
        """
        Extracts the current state of the game as an input for the RL model.
        Returns:
            np.ndarray: The processed state.
        """
        bird_y = self.bird.y # / SCREEN_HEIGHT
        # bird_velocity = self.bird.velocity / MAX_VELOCITY

        # Find the next pipe
        pipes = [pipe for pipe in self.pipes if pipe.x + pipe.pipe_top.get_width() > self.bird.x]
        # print(pipes) 
        if pipes:
            next_pipe = pipes[0]
            pipe_dist = (next_pipe.x - self.bird.x) # / SCREEN_WIDTH
            pipe_top = (next_pipe.bottom - PIPE_GAP) # / SCREEN_HEIGHT
            pipe_bottom = next_pipe.bottom # / SCREEN_HEIGHT 

            # New features
            pipe_gap_center = (pipe_top + pipe_bottom) / 2
            relative_bird_position = bird_y + 15 # Difference from gap center
        else:
            pipe_dist, pipe_top, pipe_bottom = 0, 0, 0  # Defaults if no pipe
            pipe_gap_center, relative_bird_position = 0, bird_y  # Neutral defaults

        # Construct state vector
        return np.array(
            [bird_y, self.bird.velocity, pipe_dist, pipe_top, pipe_bottom, pipe_gap_center, relative_bird_position],
            dtype=np.float32
        )

    def run(self, human = True, AI = False, action = None):
        """
        Runs the game loop, handling events, updating game state, and drawing objects.
        """
        self.clock.tick(FPS)
        while self.start_screen: 
            self.handle_events(human = human, AI = AI, action = action)
        
        self.handle_events(human = human, AI = AI, action = action) 
        self.update()
        self.draw()

        # Save high score after game ends
        self.save_highscore(self.score) 

        return self.reward, self.running

if __name__ == "__main__":
    game = Game()
    game.run()
