import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches
#import plotly.graph_objects as go
from datetime import datetime
import os
import pandas as pd
import matplotlib as mpl
from tqdm import tqdm, trange

# ============================================
# Global variables for the model
# ============================================

# General
nation_dimension = 40  # Will be a (nation_dimension,nation_dimension) matrix
vision_agent = 7  # Agent's vision
vision_cop = 7  # Cop's vision
L = 0.82  # Legitimacy, must be element of [0,1]
T = 0.1  # Threshold of agent's activation
Jmax = 15  # Maximal Jail time for type 0
k = 2.3  # Constant for P: estimated arrest probability

# Model bad cops
percentage_bad_cops = 0

# Model two classes
D_const = [1, 1]  # put like 20 here!
p_class_1 = 0.2  # Probability for an agent to be in class 1
prob_arrest_class_1 = 0.6  # Probability, given an arrest is made, that the arrested agent is of type 1
factor_Jmax1 = 1.5  # How many time is Jmax for type 1 bigger than for type 0


# ============================================
# Classes and functions (see end of file for simulation)
# ============================================

class agent():
    def __init__(self, nr, L, Jmax, p_class_1):
        self.nr = nr  # Identifier
        self.vision_agent = vision_agent
        self.position = np.random.randint(0, nation_dimension, (2))  # Assigns initial random position on the matrix
        # locations_agents_dic[nr] = self.position
        self.type = np.random.choice(2, size=1, p=[1 - p_class_1, p_class_1])[0]  # Assigns random type 0 or 1
        self.Jmax = Jmax + self.type * Jmax * (factor_Jmax1 - 1)  # Maximal Jail time for agent (depends on type)
        self.H = np.random.uniform(0, 1)  # Hardship, see paper
        self.status = 0  # 0: quite, 1:active, 2:Jail
        self.G = self.H * (1 - L)  # Grievance, see paper
        self.R = np.random.uniform(0, 1)  # Risk aversion
        self.J = 0  # Remaining jail time, 0 if free
        self.P = 0  # Estimated arrest probability
        self.N = self.R * self.P * self.Jmax  # Agent's net risk
        self.D = 0  # Percieved discrimination
        self.Il = 1 - L
        self.arrested_ratio = p_class_1
    def move(self):
        # Moves the agent if the agent is not in jail
        shift = np.random.randint(-self.vision_agent, self.vision_agent + 1, (2))
        if self.status == 2:  # Check for status
            shift = np.array([0, 0])
        self.position = self.position + shift  # Move
        self.position[0] = max(min(self.position[0], nation_dimension - 1), 0)  # Do not leave the matrix
        self.position[1] = max(min(self.position[1], nation_dimension - 1), 0)  # Do not leave the matrix
        # locations_agents_dic[self.nr] = self.position
        return self

    def update_status(self, arrest=False):
        # Updates the agent's status
        if self.status == 2 and self.J > 0:
            # If in Jail, Jail time reduces by 1
            self.J = self.J - 1

        elif self.status == 2 and self.J == 0:
            # Exits Jail and is now active
            self.status = 1
        elif arrest:
            # Is arrested and assigned a Jail time, see paper
            self.status = 2
            self.J = np.random.randint(1, Jmax)
        elif self.G - (self.N - D_const[self.type] * self.D) > T:
            # Get's active, see paper
            self.status = 1
        else:
            # Keep quite
            self.status = 0

    def active_near_agents(self, agents):
        # Computes number of near active agents
        near_agents = 0
        for agent in agents:
            if agent.status == 1:
                # Only active agents
                pos = agent.position
                if np.linalg.norm(self.position - pos, ord=np.inf) < self.vision_agent:
                    # If within vision, count
                    near_agents += 1
        if not self.status == 1:
            # To avoid double counting but always count the agent self, see paper
            near_agents += 1
        return near_agents

    def compute_arrested_ratio(self, list_arrested_vision):
        # compute the ratio type_1 arrested agents over total arrests
        tot_arrested = 0
        type_1_arrested = 0
        arrested_near = list_arrested_vision[(self.position[0],self.position[1])]
        for arrested_agent in arrested_near:
            if arrested_agent.type == 1:
                type_1_arrested += 1
            tot_arrested += 1
        if tot_arrested == 0:
            ratio = p_class_1
        else:
            ratio = type_1_arrested / tot_arrested
        self.arrested_ratio = ratio
        return ratio, tot_arrested

    def near_cops(self, cops):
        # Counts cops within vision
        near_cops = 0
        for cop in cops:
            pos = cop.position
            if np.linalg.norm(self.position - pos, ord=np.inf) < self.vision_agent:
                # If within vision count
                near_cops += 1
        return near_cops

    def updateP_new(self, list_agent_near_vision_agent, list_cop_near_vision_agent):
        # Updates estimated arrest probability, see paper
        active_agents_near = 1
        # print(len(list_agent_near_vision_agent[self.position[0]][self.position[1]]))
        for ag in list_agent_near_vision_agent[(self.position[0],self.position[1])]:
            if ag.status == 2:
                active_agents_near += 1
        cops_near = len(list_cop_near_vision_agent[(self.position[0],self.position[1])])
        self.P = 1 - np.exp(-k * cops_near / active_agents_near)

    def percieved_agressivity_of_cops(self, cops):
        # Sums over cops agressivity within influence radius
        percieved_agressivity = 0
        for cop in cops:
            pos = cop.position
            if np.linalg.norm(self.position - pos, ord=np.inf) < vision_cop:
                # If within vision count
                percieved_agressivity += cop.agressivity
        return 0  # percieved_agressivity

    def updateIl(self, cops):
        self.Il = self.Il * np.exp(self.percieved_agressivity_of_cops(cops))

    def updateN(self):
        # Update net risk, see paper
        self.N = self.R * self.P * self.Jmax

    def updateG(self):
        # Update net risk, see paper
        self.G = self.Il * self.H

    def updateD(self, list_arrested_vision):
        # update the discrimination factor D
        # radius = 40  # set the radius smaller to let D more local, let it to 40 to keep D global
        ratio, tot_arrested = self.compute_arrested_ratio(list_arrested_vision)
        if tot_arrested < 10:
            self.D = 0.3*abs(p_class_1 - ratio)
        else:
            self.D = abs(p_class_1 - ratio)

    def time_step(self, agent, cops):
        # Comptes one time iteration for the given agent
        self.move()
        self.updateP(agent, cops)
        self.updateN()
        self.updateIl(cops)
        self.updateG()
        self.updateD(agent)
        self.update_status(arrest=False)
        return self

    def time_step_no_move(self, agents, list_agent_near_vision_agent, list_cop_near_vision_agent, list_arrested_vision):
        # Comptes one time iteration for the given agent
        self.updateP_new(list_agent_near_vision_agent, list_cop_near_vision_agent)
        self.updateN()
        # self.updateIl(list_cop_near_vision_agent)
        self.updateG()
        self.updateD(list_arrested_vision)
        self.update_status(arrest=False)
        return self


class cop():
    def __init__(self, nr):
        self.nr = nr  # Identifier
        self.vision_cop = vision_cop
        self.position = np.random.randint(0, nation_dimension, (2))  # Assigns randomly initial position
        self.agressivity = 0  # random.choices([1, -0.1], [percentage_bad_cops, 1 - percentage_bad_cops])[0]

    def move(self):
        # Moves the cop within vision
        shift = np.random.randint(-self.vision_cop, self.vision_cop + 1, (2))
        self.position = self.position + shift
        self.position[0] = max(min(self.position[0], nation_dimension - 1), 0)  # Do not exit matrix
        self.position[1] = max(min(self.position[1], nation_dimension - 1), 0)  # Do not exit matrix
        return self

    def update_agent_status(self, agents):
        # Arrests randomly (with bias based on type) an agent within vision
        near_active_agents_0 = []  # List type 0 agents within vision
        near_active_agents_1 = []  # List type 1 agents within vision
        for agent in agents:
            pos = agent.position
            if np.linalg.norm(self.position - pos, ord=np.inf) < self.vision_cop:
                if agent.status == 1:
                    if agent.type == 0:
                        near_active_agents_0.append(agent)
                    else:
                        near_active_agents_1.append(agent)
        if len(near_active_agents_0) > 0:
            if len(near_active_agents_1) > 0:
                # Both types in vision, compute now which type to arrest
                choice01 = np.random.choice(2, 1, p=[1 - prob_arrest_class_1, prob_arrest_class_1])[0]
                if choice01 == 0:
                    # Arrest randomly in type 0
                    random.choice(near_active_agents_0).update_status(arrest=True)
                else:
                    # Arrest randomly in type 1
                    random.choice(near_active_agents_1).update_status(arrest=True)
            else:
                # No type 1 but type 0 in vision, arrest randomly type 0
                random.choice(near_active_agents_0).update_status(arrest=True)
        elif len(near_active_agents_1) > 0:
            # No type 0 vut type 1 in vision, arrest randomly type 1
            random.choice(near_active_agents_1).update_status(arrest=True)

    # No one in vision, no arrest

    def time_step(self, agents):
        # Compute one time iteration for cops
        self.move()
        self.update_agent_status(agents)  # Do arrest if possible
        return self

    def time_step_no_move(self, agents):
        # Compute one time iteration for cops
        self.update_agent_status(agents)  # Do arrest if possible
        return self


# ============================================
# Simulation data --> Do the tuning here
# ============================================
validation_times = 1
n_agents = 1120  # Number of considerate agents
n_cops = 64  # Number of considerate cops
tfin = 100  # Final time, i.e. number of time steps to consider
agents = [agent(n, L, Jmax, p_class_1) for n in range(n_agents)]  # Generate the agents
cops = [cop(n) for n in range(n_cops)]  # Generate the cops


save = True  # Set to True if want to save the data
interactive = False  # If true computes the html slider stuff
show_plot = False

# ============================================
# Simulation computation
# ============================================

now = datetime.now()  # Gets date and time info
dt_string = now.strftime("%d_%m_%Y_%H_%M")
name_to_save = 'simulation_' + dt_string  # All will be save with this name + counter + extensions
if save:
    if not os.path.isdir(name_to_save):
        # If save and directory does not exists, create one
        os.mkdir(name_to_save)
name_to_save = name_to_save + '/' + name_to_save

positions_data = np.empty([tfin, nation_dimension, nation_dimension])  # Stores positional and type data

color_name_list = ["white", "green", "red", "darkorange", "lime", "fuchsia", "goldenrod", "blue"]

time = range(tfin)

val_D_list = np.zeros((validation_times, tfin))
val_arrested_list = np.zeros((validation_times, tfin))
val_type_1_arrested_list = np.zeros((validation_times, tfin))
val_type_0_arrested_list = np.zeros((validation_times, tfin))
val_active_list = np.zeros((validation_times, tfin))
val_type_1_active_list = np.zeros((validation_times, tfin))
val_type_0_active_list = np.zeros((validation_times, tfin))

# std_D_list = [0] * len(range(tfin))
# std_arrested_list = [0] * len(range(tfin))
# std_type_1_arrested_list = [0] * len(range(tfin))
# std_type_0_arrested_list = [0] * len(range(tfin))
# std_active_list = [0] * len(range(tfin))
# std_type_1_active_list = [0] * len(range(tfin))
# std_type_0_active_list = [0] * len(range(tfin))
for val_round in range(validation_times):
    print('Start validation round nr' + str(val_round))
    D_list = [0] * len(range(tfin))
    arrested_list = [0] * len(range(tfin))
    type_1_arrested_list = [0] * len(range(tfin))
    type_0_arrested_list = [0] * len(range(tfin))
    active_list = [0] * len(range(tfin))
    type_1_active_list = [0] * len(range(tfin))
    type_0_active_list = [0] * len(range(tfin))
    ratio_list = [0]*len(range(tfin))
    type_1_tension_list = [0]*len(range(tfin))
    type_0_tension_list = [0]*len(range(tfin))
    agents_tension_list = [0]*len(range(tfin))
    for t in trange(tfin):
        list_agent_near_vision_agent = {}
        list_agent_near_vision_cop = {}
        list_cop_near_vision_agent = {}
        list_arrested_vision = {}
        for i in range(0, nation_dimension):
            for j in range(0, nation_dimension):
                list_agent_near_vision_agent[(i, j)] = []
                list_agent_near_vision_cop[(i, j)] = []
                list_cop_near_vision_agent[(i, j)] = []
                list_arrested_vision[(i, j)] = []
        arrested = 0
        type_1_arrested = 0
        type_0_arrested = 0
        active = 0
        type_1_active = 0
        type_0_active = 0
        D = 0
        ratio = 0
        type_0_tension = 0
        type_1_tension = 0
        free_agent_0 = 0
        free_agent_1 = 0
        # Does the t-th time iteration
        positions = np.zeros((nation_dimension, nation_dimension)) - 1  # Initialisation of the matrix
        # Values of positions are:
        # * -1: no one here
        # * 0: quite agent type 0 here
        # * 1: active agent type 0 here
        # * 2: agent in jail type 0 here
        # * 3: quite agent type 1 here
        # * 4: active agent type 1 here
        # * 5: agent in jail type 1 here
        # * 6: cop here

        #peak of arrests
        #if t == 20:
        #arrest during a longer periods of time (one also need to adjust the tor_unj_arr condition)
        if t in range(10,21):
           tot_unjustified_arrest = 0
           for agent in agents:
               if agent.type == 1 and agent.status != 2 and tot_unjustified_arrest <= 10:
                   agent.update_status(arrest=True)
                   tot_unjustified_arrest = tot_unjustified_arrest + 1

        for agent in agents:
            if agent.type == 0 and agent.status != 2:
                type_0_tension = type_0_tension + agent.G - (agent.N - D_const[agent.type] * agent.D)
                free_agent_0 = free_agent_0 + 1
            elif agent.type == 1 and agent.status != 2:
                type_1_tension = type_1_tension + agent.G - (agent.N - D_const[agent.type] * agent.D)
                free_agent_1 = free_agent_1 + 1
            pos = agent.position
            positions[
                pos[0], pos[1]] = agent.status + 3 * agent.type  # Updates matrix data with agents position and status
            if agent.status == 2:
                arrested = arrested + 1
                if agent.type == 1:
                    type_1_arrested = type_1_arrested + 1
                else:
                    type_0_arrested = type_0_arrested + 1
            elif agent.status == 1:
                active = active + 1
                if agent.type == 1:
                    type_1_active = type_1_active + 1
                else:
                    type_0_active = type_0_active + 1

        D = agent.D
        ratio = agent.arrested_ratio
        for cop in cops:
            pos = cop.position
            positions[pos[0], pos[1]] = 6  # Updates matrix data with cops position
        positions_data[t, :, :] = positions  # Stores the data of the positons
        if save or show_plot:
            im = plt.imshow(positions, cmap=mpl.colors.ListedColormap(color_name_list))
            values = [-1, 0, 1, 2, 3, 4, 5, 6]
            colors = [im.cmap(im.norm(value)) for value in values]
            label_list = ['', 'quiet agent of type 0','active agent of type 0', 'arrested agent of type 0','quiet agent of type 1','active agent of type 1','arrested agent of type 1','cop']
            patches = [mpatches.Patch(color=colors[i], label=label_list[i]) for i in range(len(values))]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.tight_layout(w_pad = 0.5)
            plt.xticks([])
            plt.yticks([])
            if val_round == 0:
                if save:
                    # Saves the positions matrix
                    plt.savefig(name_to_save + '_time_iter_nr' + str(t) + '.eps',format = 'eps')
            if show_plot:
                # Plots the positions matrix
                plt.show()

        # Compute now one time steps for each cop and each agent
        cops = [cop.move() for cop in cops]
        agents = [ag.move() for ag in agents]
        for agent in agents:
            for i in range(max(0, agent.position[0] - vision_agent),
                           min(nation_dimension, agent.position[0] + vision_agent + 1)):
                for j in range(max(0, agent.position[1] - vision_agent),
                               min(nation_dimension, agent.position[1] + vision_agent + 1)):
                    list_agent_near_vision_agent[(i,j)].append(agent)
        for agent in agents:
            for i in range(max(0, agent.position[0] - vision_cop),
                           min(nation_dimension, agent.position[0] + vision_cop + 1)):
                for j in range(max(0, agent.position[1] - vision_cop),
                               min(nation_dimension, agent.position[1] + vision_cop + 1)):
                    list_agent_near_vision_cop[(i,j)].append(agent)
        for cop in cops:
            for i in range(max(0, cop.position[0] - vision_agent),
                           min(nation_dimension, cop.position[0] + vision_agent + 1)):
                for j in range(max(0, cop.position[1] - vision_agent),
                               min(nation_dimension, cop.position[1] + vision_agent + 1)):
                    list_cop_near_vision_agent[(i,j)].append(agent)
        for agent in agents:
            if agent.status == 2:
                for i in range(max(0, cop.position[0] - 40),
                               min(nation_dimension, cop.position[0] + 40 + 1)):
                    for j in range(max(0, cop.position[1] - 40),
                                   min(nation_dimension, cop.position[1] + 40 + 1)):
                        list_arrested_vision[(i,j)].append(agent)
        cops = [cop.time_step_no_move(agents) for cop in cops]
        agents = [
            ag.time_step_no_move(agents, list_agent_near_vision_agent, list_cop_near_vision_agent, list_arrested_vision)
            for ag in agents]
        D_list[t] = D/D_const[agent.type]
        arrested_list[t] = arrested
        type_1_arrested_list[t] = type_1_arrested
        type_0_arrested_list[t] = type_0_arrested
        active_list[t] = active
        type_1_active_list[t] = type_1_active
        type_0_active_list[t] = type_0_active
        type_0_tension_list[t] = type_0_tension/free_agent_0
        type_1_tension_list[t] = type_1_tension/free_agent_1
        agents_tension_list[t] = (type_0_tension+type_1_tension)/(free_agent_0+free_agent_1)

        val_D_list[val_round, t] = D
        val_arrested_list[val_round, t] = arrested
        val_type_1_arrested_list[val_round, t] = type_1_arrested
        val_type_0_arrested_list[val_round, t] = type_0_arrested
        val_active_list[val_round, t] = active
        val_type_1_active_list[val_round, t] = type_1_active
        val_type_0_active_list[val_round, t] = type_0_active
        ratio_list[t] = ratio

std_D_list, mean_D_list = np.std(val_D_list, axis=0), np.mean(val_D_list, axis=0)
std_arrested_list, mean_arrested_list = np.std(val_arrested_list, axis=0), np.mean(val_arrested_list, axis=0)
std_type_1_arrested_list, mean_type_1_arrested_list = np.std(val_type_1_arrested_list, axis=0), np.mean(
    val_type_1_arrested_list, axis=0)
std_type_0_arrested_list, mean_type_0_arrested_list = np.std(val_type_0_arrested_list, axis=0), np.mean(
    val_type_0_arrested_list, axis=0)
std_active_list, mean_active_list = np.std(val_active_list, axis=0), np.mean(val_active_list, axis=0)
std_type_1_active_list, mean_type_1_active_list = np.std(val_type_1_active_list, axis=0), np.mean(
    val_type_1_active_list, axis=0)
std_type_0_active_list, mean_type_0_active_list = np.std(val_type_0_active_list, axis=0), np.mean(
    val_type_0_active_list, axis=0)

if interactive:
    # Create figure
    fig = go.Figure()
    # Add traces, one for each slider step
    for step in np.arange(0, tfin, 1):
        curent_pd = pd.DataFrame(positions_data[step, :, :])
        fig.add_trace(go.Heatmap(
            z=curent_pd.applymap(str),
            colorscale=color_name_list)
        )
    # Make First trace visible
    fig.data[0].visible = True
    # Create and add slider
    steps = []

    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Slider switched to step: " + str(i)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Time step: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(sliders=sliders)
    fig.show()
    if save:
        fig.write_html(name_to_save + '.html')

if save:
    lines = ['nation_dimension' + ': ' + str(nation_dimension),
             'p_class_1' + ': ' + str(p_class_1),
             'vision_agent' + ': ' + str(vision_agent),
             'vision_cop' + ': ' + str(vision_cop),
             'L' + ': ' + str(L),
             'T' + ': ' + str(T),
             'Jmax' + ': ' + str(Jmax),
             'factor_Jmax1' + ': ' + str(factor_Jmax1),
             'k' + ': ' + str(k),
             'prob_arrest_class_1' + ': ' + str(prob_arrest_class_1),
             'n_agents' + ': ' + str(n_agents),
             'n_cops' + ': ' + str(n_cops),
             'tfin' + ': ' + str(tfin),
             'D_const' + ':' + str(D_const),
             'validation_times' + ':' + str(validation_times)]

    with open(name_to_save + '_par.txt', 'w') as file:
        for line in lines:
            file.write(line + '\n')
        file.close()
    # plot graphics for type0/1 active/arrested and D
    fig, ax = plt.subplots()
    ax.plot(time, D_list)
    ax.set(xlabel='time (epochs)', ylabel="agent's percieved D", title='Discrimination factor')
    ax.grid()
    ax.legend(['D factor'])
    fig.savefig(name_to_save + 'Discrimination.png')

    fig, ax = plt.subplots()
    ax.errorbar(time, mean_D_list, yerr=std_D_list, capsize=5)
    ax.set(xlabel='time (epochs)', ylabel="agent's percieved D", title='Mean Discrimination factor')
    ax.grid()
    ax.legend(['Mean D factor'])
    fig.savefig(name_to_save + 'Mean_Discrimination.png')

    fig, ax = plt.subplots()
    ax.plot(time, arrested_list, label='Total number of arrested agents')
    ax.plot(time, type_1_arrested_list, label='Total number of type 1 arrested agents')
    ax.plot(time, type_0_arrested_list, label='Total number of type 0 arrested agents')
    ax.set(xlabel='time (epochs)', ylabel="number of agents", title='Arrested agents')
    ax.grid()
    ax.legend(['total arrested', 'type 1 arrested', 'type 0 arrested'])
    fig.savefig(name_to_save + 'Arrests.eps',format='eps')

    fig, ax = plt.subplots()
    ax.errorbar(time, mean_arrested_list, yerr=std_arrested_list, label='Total number of arrested agents', capsize=5)
    ax.errorbar(time, mean_type_1_arrested_list, yerr=std_type_1_arrested_list, label='Total number of type 1 arrested agents', capsize=5)
    ax.errorbar(time, mean_type_0_arrested_list,yerr=std_type_0_arrested_list, label='Total number of type 0 arrested agents', capsize=5)
    ax.set(xlabel='time (epochs)', ylabel="number of agents", title='Mean Arrested agents')
    ax.grid()
    ax.legend(['total arrested', 'type 1 arrested', 'type 0 arrested'])
    fig.savefig(name_to_save + 'Mean_Arrests.png')

    fig, ax = plt.subplots(2)
    ax[0].plot(time, active_list, label='Total number of active agents')
    ax[0].plot(time, type_1_active_list, label='Total number of type 1 active agents')
    ax[0].plot(time, type_0_active_list, label='Total number of type 0 active agents')
    ax[0].set(xlabel='time (epochs)', ylabel="number of agents", title='Active agents')
    ax[0].grid()
    ax[0].legend(['total active', 'type 1 active', 'type 0 active'],loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(w_pad = 0.8)

    ax[1].plot(time,ratio_list,label = 'Type1 arrested / Total number of arrested')
    ax[1].plot(time,[p_class_1]*len(time),label = 'Percentage of type 1 agent in nation')
    ax[1].set(xlabel='time (epochs)', title =  'arrested ratio')
    ax[1].grid()
    ax[1].legend(['ratio','type 1 percentage'],loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(w_pad = 0.8)
    fig.subplots_adjust(hspace = 0.5)

    fig.savefig(name_to_save + 'Active.eps', format = 'eps')

    fig, ax = plt.subplots()
    ax.errorbar(time, mean_active_list, yerr=std_active_list, label='Total number of active agents', capsize=5)
    ax.errorbar(time, mean_type_1_active_list, yerr=std_type_1_active_list, label='Total number of type 1 active agents', capsize=5)
    ax.errorbar(time, mean_type_0_active_list, yerr=std_type_0_active_list, label='Total number of type 0 active agents', capsize=5)
    ax.set(xlabel='time (epochs)', ylabel="number of agents", title='Mean Active agents')
    ax.grid()
    ax.legend(['total active', 'type 1 active', 'type 0 active'])
    fig.savefig(name_to_save + 'Mean_Active.png')

    fig, ax = plt.subplots(2)
    ax[0].plot(time, D_list, label = 'Discrimination Factor')
    ax[0].set(xlabel='time (epochs)', title='Discrimination Factor')
    ax[0].grid()
    ax[1].plot(time,ratio_list,label = 'Type1 arrested / Total number of arrested')
    ax[1].set(xlabel='time (epochs)')
    ax[1].grid()
    fig.savefig(name_to_save+'DvsRatio.png')

    fig,ax = plt.subplots(2)
    ax[0].plot(time, active_list, label='Total number of active agents')
    ax[0].plot(time, type_1_active_list, label='Total number of type 1 active agents')
    ax[0].plot(time, type_0_active_list, label='Total number of type 0 active agents')
    ax[0].set(xlabel='time (epochs)', ylabel="number of agents", title='Active agents')
    ax[0].grid()
    ax[0].legend(['total active', 'type 1 active', 'type 0 active'])

    ax[1].plot(time, agents_tension_list, label='Mean of tension value for free agents')
    ax[1].plot(time, type_0_tension_list, label='Mean of tension value for free type 0 agents')
    ax[1].plot(time, type_1_tension_list, label='Mean of tension value for free type 1 agents')
    ax[1].set(xlabel='time (epochs)', ylabel="Tension", title='Tension vs Active agent')
    ax[1].grid()
    ax[1].legend(['Tension', 'Type 0 Tension', 'Type 1 Tension'])
    fig.savefig(name_to_save+'TensionvsActive.png')

    fig, ax = plt.subplots(2)
    ax[0].plot(time,active_list,label = 'Total number of active agents')
    ax[0].set(xlabel='time (epochs)', ylabel="number of agents", title='Active agents')
    ax[0].grid()
    ax[0].legend(['Total active'],loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout(w_pad = 0.5)

    ax[1].plot(time,arrested_list, label = 'Total number of arrested agents')
    ax[1].set(xlabel='time (epochs)', ylabel="number of agents", title='Arrested agents')
    ax[1].grid()
    ax[1].legend(['Total arrested'],loc='center left', bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(hspace = 0.5)
    plt.tight_layout(w_pad = 0.5)

    fig.savefig(name_to_save + 'Active&Arrested.eps',format = 'eps')
    fig, ax = plt.subplots(2)
    ax[0].plot(time,active_list,label = 'Total number of active agents')
    ax[0].set(xlabel='time (epochs)', ylabel="number of agents", title='Active agents')
    ax[0].grid()
    ax[0].legend(['Total active'],loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout(w_pad = 0.5)

    ax[1].plot(time,arrested_list, label = 'Total number of arrested agents')
    ax[1].set(xlabel='time (epochs)', ylabel="number of agents", title='Arrested agents')
    ax[1].grid()
    ax[1].legend(['Total arrested'],loc='center left', bbox_to_anchor=(1, 0.5))
    fig.subplots_adjust(hspace = 0.5)
    plt.tight_layout(w_pad = 0.5)

    fig.savefig(name_to_save + 'Active&Arrested.eps',format = 'eps')
    np.save(name_to_save + 'ratio.npy',ratio_list)
