import pygame
import random
import numpy as np 

# initialize Pygame
pygame.init()

# set the screen size
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# set the game clock
clock = pygame.time.Clock()

# set the game font
font = pygame.font.Font(None, 50)

# set the game variables
ball_x = WIDTH // 2
ball_y = HEIGHT // 2
ball_radius = 10
ball_dx = 5 * random.choice([-1, 1])
ball_dy = 5 * random.choice([-1, 1])

paddle_width = 10
paddle_height = 100
paddle_speed = 5
left_paddle_x = 50
left_paddle_y = HEIGHT // 2 - paddle_height // 2
right_paddle_x = WIDTH - 50 - paddle_width
right_paddle_y = HEIGHT // 2 - paddle_height // 2

left_score = 0
right_score = 0

# define the ball movement function
def move_ball():
    global ball_x, ball_y, ball_dx, ball_dy, left_score, right_score

    # update the ball position
    ball_x += ball_dx
    ball_y += ball_dy

    # check for collision with top or bottom wall
    if ball_y + ball_radius > HEIGHT or ball_y - ball_radius < 0:
        ball_dy *= -1

    # check for collision with left or right wall
    if ball_x + ball_radius > WIDTH:
        left_score += 1
        ball_x = WIDTH // 2
        ball_y = HEIGHT // 2
        ball_dx = 5 * random.choice([-1, 1])
        ball_dy = 5 * random.choice([-1, 1])
    elif ball_x - ball_radius < 0:
        right_score += 1
        ball_x = WIDTH // 2
        ball_y = HEIGHT // 2
        ball_dx = 5 * random.choice([-1, 1])
        ball_dy = 5 * random.choice([-1, 1])

    # check for collision with left or right paddle
    if ball_x - ball_radius < left_paddle_x + paddle_width and left_paddle_y < ball_y < left_paddle_y + paddle_height:
        ball_dx *= -1
    elif ball_x + ball_radius > right_paddle_x and right_paddle_y < ball_y < right_paddle_y + paddle_height:
        ball_dx *= -1

# define the left paddle movement function
def move_left_paddle(direction):
    global left_paddle_y

    if direction == 'up':
        left_paddle_y -= paddle_speed
    elif direction == 'down':
        left_paddle_y += paddle_speed

    # keep the paddle on the screen
    if left_paddle_y < 0:
        left_paddle_y = 0
    elif left_paddle_y + paddle_height > HEIGHT:
        left_paddle_y = HEIGHT - paddle_height

# define the right paddle movement function
def move_right_paddle(direction):
    global right_paddle_y

    if direction == 'up':
        right_paddle_y -= paddle_speed
    elif direction == 'down':
        right_paddle_y += paddle_speed

    # keep the paddle on the screen
    if right_paddle_y < 0:
        right_paddle_y = 0
    elif right_paddle_y + paddle_height > HEIGHT:
        right_paddle_y = HEIGHT -








