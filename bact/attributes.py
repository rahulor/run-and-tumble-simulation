import numpy as np
import pandas as pd
class RunUptoTmax:
    def __init__(self, env):
        from inputs import T_max, memory, bact_speed
        self.R = np.empty((memory, 2))
        self.V = np.empty((memory, 2))
        self.bact_speed = bact_speed
        self.memory = memory
        self.dt = T_max/memory
        self.env = env
    def initialize(self, xy):
        self.R[0] = xy
    def trajectory(self, psi):
        self.V[:,0] = self.bact_speed*np.cos(psi)
        self.V[:,1] = self.bact_speed*np.sin(psi)
        for i in range(0, self.memory-1):
            self.R[i+1] = self.R[i] + self.V[i]*self.dt
    def record_concentration(self):
        conc_and_flag = map(self.env.concentration_at, self.R)
        C      = np.array([item[0] for item in conc_and_flag])
        self.C = np.flip(C, axis=None) # 1st element is the recent one; Volttera kernel applicable
    
def dataset(seed, env, n_sample, D_rot):
    from inputs import memory
    df = random_initial_condition(seed, n_sample)
    psi_0_list = df['psi']
    xy_list = df[['x', 'y']].values
    psi_sample = diffusion_upto_Tmax(seed, psi_0_list, D_rot)
    conc_list = np.empty((n_sample, memory))
    agent = RunUptoTmax(env)
    for i in range(n_sample):
        agent.initialize(xy_list[i])
        agent.trajectory(psi_sample[:,i]) # ith column. row is time
        agent.record_concentration()
        conc_list[i] = agent.C 
    return conc_list

def random_initial_condition(seed, n_sample):
    from inputs import traj_between
    r1, r2 = traj_between[0], traj_between[1]
    rng = np.random.RandomState(seed)
    random_r = rng.uniform(r1**2, r2**2, n_sample)
    random_theta = rng.uniform(-np.pi, np.pi, n_sample)
    psi = rng.uniform(-np.pi, np.pi, n_sample)
    df = pd.DataFrame()
    df['x'] = np.sqrt(random_r) * np.cos(random_theta)
    df['y'] = np.sqrt(random_r) * np.sin(random_theta)
    df['psi'] = psi
    return(df)

def diffusion_upto_Tmax(seed, psi_0_list, D_rot):
    rng = np.random.RandomState(seed)
    n_sample = len(psi_0_list)
    from inputs import T_max, memory
    dt = T_max/memory 
    psi_sample = np.empty((memory, n_sample))
    psi_sample[0] = psi_0_list # set initial conditions
    for i in range(1, memory): # i count the rows
        psi_sample[i] = psi_sample[i-1] + np.sqrt(2*D_rot*dt)*rng.randn(n_sample)
    return(psi_sample)

def normalized_binding(seed, lambda_conc_dt):
    rng = np.random.RandomState(seed)
    bind = rng.poisson(lambda_conc_dt)
    bind_mean = np.mean(bind, axis=1)
    bind_normalized = np.empty(bind.shape)
    for k in range(len(bind)):
        bmean = max(1, bind_mean[k])
        bind_normalized[k] = bind[k]/bmean
    return bind_normalized

def sample_dataset(seed, env, n_sample, D_rot, Lambda):
    from inputs import memory
    df = random_initial_condition(seed, n_sample)
    psi_0_list = df['psi']
    xy_list = df[['x', 'y']].values
    psi_sample = diffusion_upto_Tmax(seed, psi_0_list, D_rot)
    conc_list = np.empty((n_sample, memory))
    traj_list = []
    for i in range(n_sample):
        agent = RunUptoTmax(env)
        agent.initialize(xy_list[i])
        agent.trajectory(psi_sample[:,i]) # ith column. row is time
        agent.record_concentration()
        conc_list[i] = agent.C
        traj_list.append(agent.R)
    return conc_list, df, traj_list

if __name__ == "__main__":
    pass