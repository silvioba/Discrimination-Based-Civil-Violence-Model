import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from datetime import datetime
import os
import pandas as pd
import matplotlib as mpl
from tqdm import tqdm, trange
import matplotlib.animation as animation

# ============================================
# Global variables for the model
# ============================================

nation_dimension = 50       # defines size of matrix (nation_dimension * nation_dimension)
vision = 4                  # vision of each agent / cop
L = 0.3                     # initial legitimacy for all agents
T = 0.1                     # threshold for agent's activation
Jmax = 8                    # maximum jail time
k = 1.5                     # constant for making arrest probability reasonable
percentage_bad_cops = 0.05  # amount of bad cops
bad_cop_influence = 1       # influence of bad cops (should be a big positive value)
good_cop_influence = -0.12  # influence of good cops (should be a small negative value)
C = 0.2                     # constant to make Il reasonable
C2 = 10                     # constant to make the relation between G and N reasonable
C3 = 3                      # constant to scale the influence of active agents

# ============================================
# Simulation data
# ============================================
p_agents = 0.7              # Number of agents
p_cops = 0.08               # Number of cops
tfin = 400                  # Amount of steps to calculate

save = True                 # If true saves the data into a folder in the current directory
interactive = False         # If true opens an webpage with the results after
makeanimation = False       # If true creates an animation



class Agent():
    # Initialization of new agent
    def __init__(self, position, L, Jmax):
        self.status = 0                   # 0 inactive agent, 1 active agent, 2 jail
        self.position = position
        self.H = np.random.uniform(0, 1)  # perceived hardship
        self.G = self.H * (1 - L)         # grievance
        self.R = np.random.uniform(0, 1)  # risk aversion
        self.J = 0                        # jail time
        self.P = 0                        # arrest probablity
        self.N = C2*self.R * self.P       # net risk
        self.Il = 1-L                     # illegitimacy
     

    # updates status -> active / not active / jail
    def update_status(self):
        if self.status == 2 and self.J > 0:
            self.J = self.J - 1
        elif self.status == 2 and self.J == 0:
            self.status = 1
        elif self.G - self.N > T:
            self.status = 1
        else:
            self.status = 0

    # calculate near active agents
    def active_near_agents(self):
        near_activ_agents = 0
        all_near = get_nearby_agents(self.position[0], self.position[1], vision)
        for agnt in all_near:
            if isinstance(agnt, Agent) and agnt.status==1:
               near_activ_agents += 1
        return near_activ_agents

    # find near cops
    def near_cops(self):
        near_cops = 0
        all_near = get_nearby_agents(self.position[0], self.position[1], vision)
        for agent in all_near:
            if isinstance(agent, Cop):
                near_cops += 1
        return near_cops

    # calculate arrest probability
    def updateP(self):
        active_agents_near = self.active_near_agents()
        cops_near = self.near_cops()
        self.P = 1 - np.exp(-k * (1+cops_near) / (1.0+C3*active_agents_near))

    # calculate overall aggressiveness of near cops
    def percieved_aggressivity_of_cops(self):
        percieved_aggressivity = 0
        for cop in [cop for cop in get_nearby_agents(self.position[0], self.position[1], vision) if isinstance(cop, Cop)]:
            percieved_aggressivity += cop.aggressivity
        return percieved_aggressivity

    # move to new empty field around the current field
    def move(self):
        possible_positions = get_empty_field(self.position[0], self.position[1])
        old_position = self.position
        if possible_positions:
            new_position = random.choice(possible_positions)
            positions[new_position[0]][new_position[1]] = positions[old_position[0]][old_position[1]]
            positions[old_position[0]][old_position[1]] = None
            self.position = new_position

    def updateIl(self):
        self.Il=max(self.Il*np.exp(C*self.percieved_aggressivity_of_cops()),1-L)

    def updateN(self):
        self.N = C2*self.R * self.P

    def updateG(self):
        self.G = self.Il*self.H

    # arrests the agent
    def arrest(self):
        self.J = np.random.randint(1, Jmax)
        self.status = 2

    # calculates one time step for the agent
    def time_step(self):
        if self.status != 2:
            self.updateP()
            self.updateN()
            self.updateG()
            self.move()
            self.updateIl()
        self.update_status()
        return self


class Cop():
    # initialize new Cop
    def __init__(self, position,good):
        self.position = position
        if good:
           self.aggressivity= good_cop_influence
        else:
            self.aggressivity=bad_cop_influence
        

    # move to empty field nearby
    def move(self):
        possible_positions = get_empty_field(self.position[0], self.position[1])
        old_position = self.position
        if possible_positions:
            new_position = random.choice(possible_positions)
            positions[new_position[0]][new_position[1]] = positions[old_position[0]][old_position[1]]
            positions[old_position[0]][old_position[1]] = None
            self.position = new_position

    # arrest an active agent
    def update_agent_status(self):
        nearby_agents_and_cops = get_nearby_agents(self.position[0], self.position[1], vision)
        nearby_agents = [agnt for agnt in nearby_agents_and_cops if isinstance(agnt, Agent)]
        near_active_agents = []  # List activ agents within vision

        for agnt in nearby_agents:
            if agnt.status == 1:
                near_active_agents.append(agnt)

        if len(near_active_agents) > 0:
            random.choice(near_active_agents).arrest()

    # calculates one time step for the cop
    def time_step(self):
        self.move()
        self.update_agent_status()
        return self

# ============================================
# Simulation computation
# ============================================

now = datetime.now()    # Gets date and time info for creating the data folder
dt_string = now.strftime("%d_%m_%Y_%H_%M")
name_to_save = 'simulation_' + dt_string
if save:
    if not os.path.isdir(name_to_save):
        # If save and directory does not exists, create one
        os.mkdir(name_to_save)
name_to_save = name_to_save + '/' + name_to_save

# Calculate the amount of good and bad cops to always have the same amount of cops
nsquare=nation_dimension**2
nagent=int(nsquare*p_agents)
ngoodcops=int(nsquare*p_cops*(1-percentage_bad_cops))
nbadcop=int(nsquare*p_cops*percentage_bad_cops)
status_list=[]
for i in range(nsquare):
    if i<=nagent:
        status_list.append(0)
    elif i<=nagent+ngoodcops:
        status_list.append(2)
    elif i<=nagent+ngoodcops+nbadcop:
        status_list.append(3)
    else:
        status_list.append(-1)

# Create the initial array
agents = []
cops = []
positions = []
for i in range(nation_dimension):
    line = []
    for j in range(nation_dimension):
        stat = random.choice(status_list)
        status_list.remove(stat)
        if stat==0:
            agent_instance = Agent([i, j], L, Jmax)
            line.append(agent_instance)
            agents.append(agent_instance)
        elif stat==2:
            cop_instance = Cop([i, j],True)
            line.append(cop_instance)
            cops.append(cop_instance)
        elif stat==3:
            cop_instance = Cop([i, j],False)
            line.append(cop_instance)
            cops.append(cop_instance)
        else:
            line.append(None)
    positions.append(line)

# helper function to find all agents and cops in vision of position (x,y)
def get_nearby_agents(x, y, local_vision):
    nearby_agents = []
    for i in range(max(x - local_vision, 0), min(x + local_vision, nation_dimension)):
        for j in range(max(y - local_vision, 0), min(y + local_vision, nation_dimension)):
            if i is not x or j is not y:
                if positions[i][j] is not None:
                    nearby_agents.append(positions[i][j])
    return nearby_agents

# helper function to find all empty fields around position (x,y). Movement restricted to 1 field
def get_empty_field(x,y):
    empty_fields = []
    for i in range(max(x-1, 0), min(x+2, nation_dimension)):
        for j in range(max(y-1, 0), min(y+2, nation_dimension)):
            if positions[i][j] is None:
                    empty_fields.append([i,j])
    return empty_fields
 


time = range(tfin)
arrested_list = [0]*tfin
active_list = [0]*tfin
positions_data=np.empty([tfin, nation_dimension, nation_dimension])
color_name_list = ["white", "green", "red", "black", "blue","purple"]
values = [-1, 0, 1, 2, 3, 4]
names = ['empty','quiet','active','jail','cop','bad cop']

if makeanimation:
    ims=[]
    fig=plt.figure()

for t in trange(tfin):
    arrested = 0
    active = 0
    # Does the t-th time iteration
    current_status = [] 
    # Values of positions are:
    # * -1: no one here
    # * 0: quite agent type 0 here
    # * 1: active agent type 0 here
    # * 2: agent in jail
    # * 3: cop here
    # * 4: bad cop here
   
    for i in range(nation_dimension):
        line = []
        for j in range(nation_dimension):
            element = positions[i][j]
            if element is None:
               line.append(-1)
            elif isinstance(element, Cop):
               if element.aggressivity < 0:
                   line.append(3)
               else:
                   line.append(4)
            elif isinstance(element, Agent):
               line.append(element.status)
               if element.status == 2:
                   arrested = arrested + 1
               elif element.status == 1:
                   active = active + 1
        current_status.append(line)

    positions_data[t, :, :] = current_status       # Stores the data of the positons
    numb_diff_states = 6
    if percentage_bad_cops==0:
        numb_diff_states=5
    im = plt.imshow(current_status, cmap=mpl.colors.ListedColormap(color_name_list, N=numb_diff_states))

    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i], label=names[i]) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(label="percentage of bad cops: "+str(percentage_bad_cops*100)+"%")
    if save:
        plt.savefig(name_to_save + '_time_iter_nr' + str(t) + '.png')
        if not makeanimation:
            plt.close()
    if makeanimation:
        ims.append([im])
        # Saves the positions matrix
    # Compute now one time steps for each cop and each agent
    for cop in cops:
        cop.time_step()
    for ag in agents:
        ag.time_step()
    
    arrested_list[t] = arrested
    active_list[t] = active

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
        active=60,
        currentvalue={"prefix": "Time step: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(sliders=sliders)
    fig.show()
    if save:
        fig.write_html(name_to_save+'.html')

if save:

    lines = ['nation_dimension' + ': ' + str(nation_dimension),
        'percentage_bad_cops' + ': ' + str(percentage_bad_cops),
        'vision' + ': ' + str(vision),
        'L' + ': ' + str(L),
        'T' + ': ' + str(T),
        'Jmax' + ': ' + str(Jmax),
        'k' + ': ' + str(k),
        'p_agents' + ': ' + str(p_agents),
        'p_cops' + ': ' + str(p_cops),
        'bad_cop_influence'+ ': '+str(bad_cop_influence),
        'good_cop_influence'+ ': '+ str(good_cop_influence),
        'C'+': '+str(C),
        'C3'+': '+str(C3),
        'tfin' + ': ' + str(tfin)]

    with open(name_to_save + '_par.txt', 'w') as file:
        for line in lines:
            file.write(line + '\n')
        file.close()
  
    
    figu, ax = plt.subplots()
    ax.plot(time, arrested_list, label='Total number of arrested agents')
    ax.set(xlabel='time (epochs)', ylabel="number of arrested agents", title='Arrested agents')
    ax.grid()
    ax.legend('total arrested')
    figu.savefig(name_to_save + 'Arrests.png')
    

    figu, ax = plt.subplots()
    ax.plot(time, active_list, label='Total number of active agents')
    ax.set(xlabel='time (epochs)', ylabel="Number of active agents", title='Active agents')
    ax.grid()
    ax.legend('total active')
    figu.savefig(name_to_save + 'Active.png')

    percentage_list=[(100*i)/len(agents) for i in active_list]

    figu, ax = plt.subplots()
    ax.plot(time, percentage_list, label='Percentage of active agents')
    ax.set(xlabel='time (epochs)', ylabel="percentage active", title="Percentage of active agents")
    ax.grid()
    figu.savefig(name_to_save + 'Percentage_active.png')
    if not makeanimation:
        plt.show()

if makeanimation:
    ani = animation.ArtistAnimation(fig, ims, interval=140, blit=True,                            repeat_delay=1000)
    ani.save(name_to_save+'animation.gif')


