from turtle import Turtle

#POSITION_TURTLE = [(350,0), (-350,0) ]


class Paddle(Turtle):
    def __init__(self,position):
        super().__init__()
        self.segments = []
        self.position = position
        self.create_paddle()


    def create_paddle(self):

        new_turtle = Turtle(shape='square')
        new_turtle.shapesize(stretch_wid=5, stretch_len=1)

        new_turtle.goto(self.position)
        new_turtle.penup()
        new_turtle.color("white")

class Ball(Turtle):
    def __init__(self,position):
        super().__init__()
        self.position = position
        self.create_ball()


    def create_ball(self):

        new_turtle = Turtle(shape ='circle')
        new_turtle.circle(20)
        new_turtle.color("white")
        new_turtle.goto(self.position)
        new_turtle.penup()

    def move (self):
        new_x = self.xcor() + 10
        new_y = self.ycor() + 10
        self.goto(new_x, new_y)




