import os
import numpy as np
from constants import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
import pandas as pd

import AgentTorch.agent_torch
import covid_abm
from populations import pop21009
from AgentTorch.agent_torch.core.executor import Executor
from AgentTorch.agent_torch.core.dataloader import LoadPopulation


class LearnableParams(nn.Module):
    def __init__(self, num_params, device=DEVICE):
        super().__init__()
        self.device = device
        self.num_params = num_params
        self.fc1 = nn.Linear(1, 64).to(self.device)
        self.fc2 = nn.Linear(64, 32).to(self.device)
        self.fc3 = nn.Linear(32, self.num_params + 2).to(self.device)
        self.ReLU = nn.ReLU()
        self.learnable_params = nn.Parameter(torch.rand(num_params + 2, device=self.device))
        
        min_values_list = [0] * num_params + [0.0] + [0.2]
        max_values_list = [5] * num_params + [0.0005] + [1] 
        
        self.min_values = torch.tensor(min_values_list, device=self.device)
        self.max_values = torch.tensor(max_values_list, device=self.device)


        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.to(self.device)
        out = self.ReLU(self.fc1(x))
        out = self.ReLU(self.fc2(out))
        out = self.fc3(out)

        out = self.min_values + (self.max_values -
                                 self.min_values) * self.sigmoid(out)
        return out

def map_and_replace_tensor(input_string):
    parts = input_string.split('.')
    
    function = parts[1]
    index = parts[2]
    sub_func = parts[3]
    arg_type = parts[4]
    var_name = parts[5]

    
    def getter_and_setter(runner, counterfactual=False, new_value=None, mode_calibrate=True):
        substep_type = getattr(runner.initializer, function)
        substep_function = getattr(substep_type[str(index)], sub_func)

        if mode_calibrate:
            current_tensor = getattr(substep_function, 'calibrate_' + var_name)
        else:
            current_tensor = getattr(getattr(substep_function, 'learnable_args'), var_name)
        
        if new_value is not None:
            if not counterfactual:
                assert new_value.requires_grad == current_tensor.requires_grad
            if mode_calibrate:
                setvar_name = 'calibrate_' + var_name
                setattr(substep_function, setvar_name, new_value)
                current_tensor = getattr(substep_function, setvar_name)
            else:
                setvar_name = var_name
                subfunc_param = getattr(substep_function, 'learnable_args')
                setattr(subfunc_param, setvar_name, new_value)
                current_tensor = getattr(subfunc_param, setvar_name)

            return current_tensor
        else:
            return current_tensor

    return getter_and_setter

def execute(sim, runner, Y_actual, epoch, epochs, n_steps=28):
    runner.step(n_steps)

    labels = runner.state_trajectory[-1][-1]['environment']['daily_infected']

    weeks_to_eval = sim.config['simulation_metadata']['NUM_WEEKS_TO_EVAL']

    start = n_steps - weeks_to_eval * 7
    labels = labels[start:]

    Y_sched = labels.to(Y_actual.device)

    date = sim.config['simulation_metadata']['DATE']
    initial_rate = sim.config['simulation_metadata']['INITIAL_INFECTION_RATE']
    exposed_to_infected = sim.config['simulation_metadata']['EXPOSED_TO_INFECTED_TIME']
    infected_to_recovered = sim.config['simulation_metadata']['INFECTED_TO_RECOVERED_TIME']
    population = sim.config['simulation_metadata']['POPULATION']
    generating_counterfactual = sim.config['simulation_metadata']['GENERATING_COUNTERFACTUAL']
    cf_type = sim.config['simulation_metadata']['COUNTERFACTUAL_TYPE']
    with_k = sim.config['simulation_metadata']['WITH_K']
    with_vacc = sim.config['simulation_metadata']['WITH_VACC']

    if generating_counterfactual:
        washtenaw_data = pd.read_csv(f"data/processed_data/{population}/daily_data.csv", parse_dates=['date'])
        washtenaw_cases = torch.tensor(washtenaw_data['cases'].values[:n_steps], dtype=torch.float)
        
        labels_np = labels.cpu().detach().numpy()
        washtenaw_cases_np = washtenaw_cases.cpu().detach().numpy()

        plt.figure(figsize=(8, 6))
        plt.plot(labels_np, marker='o', label='Simulation Cases')
        plt.plot(washtenaw_cases_np, marker='x', label='Actual Cases')
        plt.xlabel('Days')
        plt.ylabel('Number of Cases')
        plt.title(f'Factual vs Counterfactual Data for {population}')
        plt.legend()

        output_dir = f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}'
        os.makedirs(output_dir, exist_ok=True)

        plt.savefig(f'{output_dir}/counterfactual_results{cf_type}.png')
        plt.close()
        
        counterfactual_df = pd.DataFrame({
            'day': range(len(labels_np)),
            'counterfactual_cases': labels_np,
            'actual_cases': washtenaw_cases_np[:len(labels_np)]
        })
        counterfactual_df.to_csv(f'{output_dir}/counterfactual_data{cf_type}.csv', index=False)

        output_dir = f'results/{population}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}'
        os.makedirs(output_dir, exist_ok=True)

        counterfactual_df.to_csv(f'{output_dir}/counterfactual_data{cf_type}.csv', index=False)
        
        return

    if epoch == epochs - 1:
        output_dir = f'results/{population}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}'
        os.makedirs(output_dir, exist_ok=True)

        generated_df = pd.DataFrame({
            "day": list(range(len(Y_sched))),
            "generated_factual_cases": Y_sched.cpu().detach().numpy()
        })

        generated_df.to_csv(f"{output_dir}/generated_factual.csv", index=False)

    under_loss = torch.clamp(Y_actual - Y_sched, min=0)
    under_loss_squared = torch.clamp((Y_actual - Y_sched)**2, min=0)
    total_loss = (under_loss_squared).mean() * len(under_loss)
    return total_loss

def eval_net():
    sim = Executor(covid_abm, pop_loader=LoadPopulation(pop21009))
    test = sim._get_runner
    runner = test(sim.config)
    runner.init()
    learnable_params = [(name, param) for (name, param) in runner.named_parameters()]

    date = sim.config['simulation_metadata']['DATE']
    num_steps = sim.config['simulation_metadata']['num_steps_per_episode']
    num_weeks = sim.config['simulation_metadata']['NUM_WEEKS']
    population = sim.config['simulation_metadata']['POPULATION']
    generating_counterfactual = sim.config['simulation_metadata']['GENERATING_COUNTERFACTUAL']
    initial_rate = sim.config['simulation_metadata']['INITIAL_INFECTION_RATE']
    exposed_to_infected = sim.config['simulation_metadata']['EXPOSED_TO_INFECTED_TIME']
    infected_to_recovered = sim.config['simulation_metadata']['INFECTED_TO_RECOVERED_TIME']
    population = sim.config['simulation_metadata']['POPULATION']
    with_k = sim.config['simulation_metadata']['WITH_K']
    with_vacc = sim.config['simulation_metadata']['WITH_VACC']

    df = pd.read_csv(f"data/processed_data/{population}/daily_data.csv", parse_dates = ["date"])
    case_numbers = df['cases'].values
    case_numbers = torch.tensor(case_numbers, dtype=torch.float, device=DEVICE)

    learn_model = LearnableParams(num_weeks, device=DEVICE)
    learn_model = torch.compile(learn_model)

    def deep_clone_state(state):
        new_state = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                new_state[k] = v.clone()
            elif isinstance(v, dict):
                new_state[k] = deep_clone_state(v)
            else:
                new_state[k] = v
        return new_state
    
    initial_state = deep_clone_state(runner.state)

    if generating_counterfactual:

        base_dir = f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}'
        param_array = np.loadtxt(os.path.join(base_dir, "calibrated_params.txt"))
        param_tensor = torch.tensor(param_array, dtype=torch.float, device=DEVICE)
        param_tensor = param_tensor[:, None] if param_tensor.ndim == 1 else param_tensor

        input_string = learnable_params[1][0]
        tensorfunc = map_and_replace_tensor(input_string)
        tensorfunc(runner, generating_counterfactual, param_tensor[:num_weeks], mode_calibrate=True)

        input_string = learnable_params[3][0]
        tensorfunc = map_and_replace_tensor(input_string)
        tensorfunc(runner, generating_counterfactual, param_tensor[-2], mode_calibrate=True)

        input_string = learnable_params[4][0]
        tensorfunc = map_and_replace_tensor(input_string)
        tensorfunc(runner, generating_counterfactual, param_tensor[-1], mode_calibrate=True)

        _ = execute(sim, runner, case_numbers, epoch=0, epochs=1, n_steps=num_steps)
        return

    opt = optim.Adam(learn_model.parameters(), lr=0.0005)
    x = torch.tensor([1.0], device=DEVICE)
    epochs = 301

    for epoch in range(epochs):
        torch.autograd.set_detect_anomaly(True)

        opt.zero_grad()

        runner.state = deep_clone_state(initial_state)
        runner.state_trajectory = []
        debug_tensor = learn_model(x)
        debug_tensor = debug_tensor[:, None]
        
        input_string = learnable_params[1][0]
        tensorfunc = map_and_replace_tensor(input_string)
        tensorfunc(runner, generating_counterfactual, debug_tensor[:num_weeks], mode_calibrate=True)

        input_string = learnable_params[3][0]
        tensorfunc = map_and_replace_tensor(input_string)
        tensorfunc(runner, generating_counterfactual, debug_tensor[-2], mode_calibrate=True)

        input_string = learnable_params[4][0]
        tensorfunc = map_and_replace_tensor(input_string)
        tensorfunc(runner, generating_counterfactual, debug_tensor[-1], mode_calibrate=True)

        loss = execute(sim, runner, case_numbers, epoch, epochs, num_steps)
        loss.backward()

        opt.step()

        if epoch == epochs - 1:
            base_dir = f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}'
            os.makedirs(base_dir, exist_ok=True)
            np.savetxt(os.path.join(base_dir, "calibrated_params.txt"), debug_tensor.detach().cpu().numpy(), fmt="%.6f")