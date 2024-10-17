# %% [markdown]
# # Computer Vision Based Game
# 
# Welcome to our computer vision-based game! In this game, enemy objects are falling from the top of the screen, and the player must use their hand to avoid these objects. The game leverages computer vision technology to track the player's hand movements in real-time.
# 
# ## Features
# 
# - Real-time Hand Tracking: Uses computer vision to detect and track hand movements.
# - Interactive Gameplay: Avoid falling enemy objects using your hand.
# 
# ## Problem Statement: Enemy Dodging Game
# 
# A player-controlled object is moved using hand movements. The objective is to dodge falling objects (blocks) that randomly appear on the screen. The game could rely on OpenCV for hand tracking, and falling objects are randomly generated.
# 
# ## Requirements
# - Python 3.x
# - OpenCV
# - Mediapipe

# %% [markdown]
# # Solution

# %%
## initialize libraries
import cv2
import mediapipe as mp
import numpy as np
import random
import time


# %%
# Initialize Mediapipe and OpenCV modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Constants
WIDTH, HEIGHT = 640, 480  # Dimensions of the game window
PLAYER_RADIUS = 30  # Radius of the player circle
ENEMY_SIZE = 50  # Size of the falling enemy blocks
ENEMY_SPEED = 5  # Speed of falling enemies
FPS = 30  # Frames per second


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)

# %%
# Enemy class defintion to handle falling blocks
class Enemy:
    def __init__(self):
        self.x = random.randint(0, WIDTH - ENEMY_SIZE)
        self.y = 0
        self.speed = ENEMY_SPEED

    def move(self):
        self.y += self.speed

    def draw(self, frame):
        cv2.rectangle(frame, (self.x, self.y), (self.x + ENEMY_SIZE, self.y + ENEMY_SIZE), RED, -1)

    def off_screen(self):
        return self.y > HEIGHT

# %%
# Check for collision between player and enemy
def check_collision(player_pos, enemy):
    px, py = player_pos
    if (px > enemy.x and px < enemy.x + ENEMY_SIZE) or (px + PLAYER_RADIUS > enemy.x and px - PLAYER_RADIUS < enemy.x + ENEMY_SIZE):
        if py + PLAYER_RADIUS > enemy.y and py - PLAYER_RADIUS < enemy.y + ENEMY_SIZE:
            return True
    return False

# %%
# Main game loop
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)  # Set width of the video capture
cap.set(4, HEIGHT)  # Set height of the video capture

# %%
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    enemies = []  # List to store enemies
    score = 0  # Player's score
    start_time = time.time()  # Start time to control enemy generation

    while cap.isOpened():
        # Read a frame from the video capture
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        # Convert the frame to RGB for Mediapipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Default player position if no hand is detected
        player_pos = (WIDTH // 2, HEIGHT - PLAYER_RADIUS * 2)

        # If a hand is detected, update player position based on hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Update player position based on index finger tip position
                player_pos = (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * WIDTH),
                              int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * HEIGHT))

        # Create a new enemy every 2 seconds
        if time.time() - start_time > 2:
            enemies.append(Enemy())  # Add a new enemy to the list
            start_time = time.time()  # Reset the start time

        # Update each enemy's position and draw it on the frame
        for enemy in enemies:
            enemy.move()  # Move enemy down the screen
            enemy.draw(frame)  # Draw enemy on the frame

        # Remove enemies that go off screen and increment score for each removed enemy
        enemies = [enemy for enemy in enemies if not enemy.off_screen() or (score := score + 1)]

        # Check for collisions between player and each enemy
        for enemy in enemies:
            if check_collision(player_pos, enemy):
                # If collision occurs, display "Game Over" and end the game
                cv2.putText(frame, "Game Over", (WIDTH // 2 - 100, HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, RED, 3)
                cv2.imshow('Enemy Dodging Game', frame)
                cv2.waitKey(3000)
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # Draw the player as a white circle
        cv2.circle(frame, player_pos, PLAYER_RADIUS, WHITE, -1)

        # Display the current score
        cv2.putText(frame, f'Score: {score}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

        # Display the frame
        cv2.imshow('Enemy Dodging Game', frame)
        # If 'q' key is pressed, exit the game
        if cv2.waitKey(1000 // FPS) & 0xFF == ord('q'):
            break

# %%
# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


