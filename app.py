import os
from typing import List

from fastapi import FastAPI
from tensorflow import keras
from pydantic import BaseModel

import model_training

app = FastAPI()

class Env_variables(BaseModel):
    variables: list[float] = []


agent = model_training.Agent(state_space=8, action_size=4)
if os.path.exists("saved_model"):
    agent.model = keras.models.load_model('saved_model')

@app.post('/action')
def get_action(state: List[Env_variables]):
    action = agent.act(state)
    return {'action': action}

