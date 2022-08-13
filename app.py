import os
import pickle

from fastapi import FastAPI
from tensorflow import keras
from pydantic import BaseModel
import numpy as np

import model_training

app = FastAPI()
OBSERVATION_SPACE = 8


class EnvVariables(BaseModel):
    variables: list[float]


agent = model_training.Agent(state_space=8, action_size=4, )
if os.path.exists("saved_model"):
    agent.model = keras.models.load_model('saved_model')


@app.post('/action')
def get_action(observation: EnvVariables):
    observation = np.asarray(observation.variables)
    state = np.reshape(observation, [1, OBSERVATION_SPACE])
    action = np.argmax(agent.model.predict(state)[0])
    return int(action)
