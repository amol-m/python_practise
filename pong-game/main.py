from turtle import Turtle , Screen
from paddle import Paddle , Ball
import time

screen = Screen()
screen.setup(width =800, height =600)
screen.title("PONG")
screen.bgcolor("black")


r_paddle = Paddle((350,0))
l_paddle = Paddle((-350,0))

screen.tracer(0)
screen.listen()

ball = Ball((0,0))


def go_up():
    new_y = r_paddle.ycor() + 20
    r_paddle.goto(350, new_y)
    r_paddle.down
    print(r_paddle.xcor(), r_paddle.ycor())


def go_down():
    new_y = l_paddle.ycor() - 20
    l_paddle.goto(-350, new_y)
    l_paddle.down
    print(-350, new_y)

screen.onkey(go_up,"Up")
screen.onkey(go_down,"Down")

screen.onkey( go_up,"w")
screen.onkey( go_down,"s")



game_is_on = True
while game_is_on:
    time.sleep(0.1)
    screen.update()
    ball.move()


screen.exitonclick()
