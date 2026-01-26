import mesa
from typing import Tuple
import numpy as np

def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.agents]
    x = sorted(agent_wealths)
    n = model.num_agents
    B = sum(x_i * (n-i) for i, x_i in enumerate(x)) / (n * sum(x))
    return 1 + (1/n) - 2*B

class MoneyAgent(mesa.Agent):
    def __init__(self, model, rand_starting_wealth= False):
        super().__init__(model)
        if rand_starting_wealth:
            self.wealth = self.random.randint(0,10)
        else:
            self.wealth = 1

    def exchange(self):
        if self.wealth > 0:
            other_agent = self.random.choice(self.model.agents)
            if other_agent is not None:
                other_agent.wealth += 1
                self.wealth -= 1

    def say_hi(self):
        print(f"I am agent {self.unique_id}, I have {self.wealth} dollars")

class MoneyModel(mesa.Model):
    def __init__(self, n= 10, seed= None):
        super().__init__(seed= seed)
        self.num_agents = n
        self.datacollector = mesa.DataCollector(
            model_reporters= {"Gini": compute_gini}, 
            agent_reporters= {"Wealth": "wealth"}
        )

        MoneyAgent.create_agents(model= self, n= n)

    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("exchange")

    def get_traj(self):
        traj = {}
        traj["wealth"] = np.array([a.wealth for a in self.agents])
        traj["gini"] = self.datacollector.get_model_vars_dataframe()["Gini"].to_numpy()
        return traj

