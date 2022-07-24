from unicodedata import decimal
from fastapi import FastAPI
from tensorflow.keras.models import load_model
from pydantic import BaseModel

import model

app=FastAPI()

class Env_variables(BaseModel):
    variables: list[float] = []


agent = model.Agent(state_space=8, action_size=4)
agent.model = load_model('model.h5')

@app.post('/action')
def get_action(state: Env_variables):
    action = agent.act(state)
    return action

if __name__ == "__main__":
    app.run(debug=True)