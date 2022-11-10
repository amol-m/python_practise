import turtle
import pandas as pd

screen = turtle.Screen()
screen.title("US State Game")
image = "blank_states_img.gif"

screen.addshape(image)
turtle.shape(image)


state_data = pd.read_csv("50_states.csv")
state_data.state = state_data.state.str.lower()
all_state_date = state_data.state.to_list()



list_gussed_state =[]
while len(list_gussed_state) <50:

        guess_state = screen.textinput(title=f"{len(list_gussed_state)}/50", prompt="Your Guess")
        guess_state = guess_state.lower()

        if guess_state =='exit':
            break

        if guess_state in state_data['state'].values :
            state_info = state_data[state_data.state == guess_state]
            t = turtle.Turtle()
            t.hideturtle()
            t.penup()
            t.goto(int(state_info.x),int(state_info.y))
            t.write(guess_state)
            list_gussed_state.append(guess_state)


list_unguessed_state = [state for state in all_state_date if state not in list_gussed_state]
df_list_unguessed_state = pd.DataFrame(list_unguessed_state)
print(df_list_unguessed_state)
df_list_unguessed_state.to_csv("unguessed_state.txt")






