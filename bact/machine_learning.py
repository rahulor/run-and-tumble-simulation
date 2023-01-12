import numpy as np
import pandas as pd
import conc.profile
class Bacteria:
    def __init__(self, bact_id, seed, directory_name, initial_position, D_rot, Lambda):
        self.id = bact_id
        self.rng = np.random.RandomState(seed)
        self.traj_path = directory_name + '/traj/id_'
        from inputs import T_max, memory, t_sim, bact_speed, horizon, grad_label, food_surface
        self.dim = 2 # 2 for 2D, 3 for 3
        self.Lambda = Lambda
        self.dt = T_max/memory
        self.memory = memory
        sim_time = t_sim + 0.02
        self.n_sim_time = int(sim_time/self.dt) # simulation time -- discrete
        self.bact_speed = bact_speed
        self.horizon = horizon
        self.food_surface = food_surface
        self.k1 = int(grad_label[0]) # grad_label = [k1, k2], where k is the delay. k1<k2
        self.k2 = int(grad_label[1])
        
        self.D_rot = D_rot
        
        self.flag_inside = False
        # Ture -- only if bacteria is inside the food; write traj and stop further simulation.
        self.flag_outside = False
        # Ture -- only if bacteria is outside the horizon; write traj and stop further simulation.
        self.flag_tumble = False
        # Will be set or re-set after each run. Velocity vector changes only if flag is True.
        
        self.flag_force_write = False
        # True for the last run. Used to write the whole traj even if no success (not flag_inside)
        
        self.R = initial_position
        self.trajectory = self.R.reshape(1,-1) 
        # to store the trajectory
        self.label_true = []
        self.label_pred = []
        self.tumble     = []
        self.conc_label_now = False

    def random_psi(self):
        return self.rng.uniform(-np.pi, np.pi)
    def assign_label(self):
        if len(self.conc)<self.memory:
            self.conc_label_prev = self.conc_label_now
            # either self.flag_inside or self.flag_force_write
        else:
            self.conc_label_prev = self.conc_label_now
            self.conc_label_now = (self.conc[self.k1] - self.conc[self.k2]) > 0.0 
        return
    def rotational_diffusion(self, psi, size):
        if size == 1:
            psi_update = psi + np.sqrt(2*self.D_rot*self.dt)*self.rng.randn()
            return psi_update
        elif size > 1:
            psi_list = np.empty(size)
            psi_list[0] = psi
            for i in range(1, size):
                psi_list[i] = psi_list[i-1] + np.sqrt(2*self.D_rot*self.dt)*self.rng.randn()
            return psi_list
        else:
            raise ValueError('check rotational_diffusion(self, psi, size)')
        
    def velocity_for(self, psi_list):
        V = np.empty((psi_list.size, self.dim))
        V[:,0] = self.bact_speed*np.cos(psi_list)
        V[:,1] = self.bact_speed*np.sin(psi_list)
        return V

    def get_new_line(self):
        """
        Returns
        -------
        array of shape memory x dim . Co-ordinates of the new trajectory. 
        """
        self.psi = self.random_psi()
        self.psi_list = self.rotational_diffusion(self.psi, self.memory)
        self.V = self.velocity_for(self.psi_list)  
        line = np.empty((self.memory, self.dim))
        line[0] = self.R
        for i in range(1, self.memory): # i count the rows
            line[i] = line[i-1] + self.V[i-1]*self.dt  # x(t) = x(t-dt) + v(t) * dt
        if len(self.trajectory) + len(line) > self.n_sim_time: # it goes beyond simulation time ? shorten it.
            upto = self.n_sim_time - len(self.trajectory)
            line = line[:upto] 
            
        return line # trajectory as columns (2 or 3)
    
    def update_trajectory(self, line):
        """
        Update the trajectory with current line
        Write to file if flag_inside is True
        Also write if flag_force_write is True -- end of simulation time

        Parameters
        ----------
        

        Returns
        -------
        None.

        """
        self.trajectory = np.vstack((self.trajectory, line))
        # update other labels..
        self.assign_label()
        if (len(line) == 1):
            self.tumble.append(self.flag_tumble)
            self.label_true.append(self.conc_label_prev)
            self.label_pred.append(not self.flag_tumble)
        else:
            nonelist = [None]*(len(line)-1)
            self.tumble.extend([self.flag_tumble] + nonelist)
            self.label_true.extend([self.conc_label_prev] + nonelist)
            self.label_pred.extend([not self.flag_tumble] + nonelist)
        #
        if len(self.trajectory) == self.n_sim_time: # reached simulation time; enough!
            self.flag_force_write = True
            print('TIMEOUT  ---')
        # check this
        if self.flag_inside or self.flag_outside:
            self.flag_force_write = True
        
        if self.flag_force_write: # then write it to file
            if self.dim==2:
                column_name = ['time', 'x', 'y']
            elif self.dim==3:
                column_name = ['time', 'x', 'y', 'z']
            else:
                print('Check self.dim | must be 2 or 3')
            
            n_time = len(self.trajectory)-1
            data = np.empty((n_time, 1+self.dim))
            data[:,0] = np.around(np.arange(n_time)*self.dt,2)
            data[:,1:1+self.dim] = np.around(self.trajectory[1:],4)
            df_traj = pd.DataFrame(data, columns=column_name)
            # other labels..            
            df_traj['tumble'] = self.tumble
            df_traj['label_true'] = self.label_true
            df_traj['label_pred'] = self.label_pred
            #
            output_path= self.traj_path + str(self.id) + '.csv'
            df_traj.to_csv(output_path, index=False)
        pass
    
    def get_concentration(self, food, line):
        """
        To find the concentration along the line (or at a point), due to the food-object
        - If the bacteria is inside the food circle --> set the flag, write the trajectory and exit!
        - otherwise, return the concentration.
        
        Parameters
        ----------
        food : object of type FoodEnviornment(class)
            used to estimate the current distance from the food and concentration along the trajectory.
        line : array (2 or 3 columns) 
            co-ordinates of new line-trajectory; or the co-ordinates of the new point

        Returns
        -------
        array of size len(line). 
        Concentration along the new line-trajectory; or at the new point

        """
        conc_line = np.empty(len(line)) # to store the concentration along trajectory
        for i in range(len(line)):
            conc_line[i], flag_inside, flag_outside = food.concentration_at(line[i]) # calling FoodEnviornment function
            if flag_inside:
                print('REACHED <<---')
                self.flag_inside = flag_inside 
                self.update_trajectory(line[:i+1]) # update trajectory and write it to file. 
                break # exit here
            if flag_outside:
                print('LOST    --->>')
                self.flag_outside = flag_outside 
                self.update_trajectory(line[:i+1]) # update trajectory and write it to file. 
                break # exit here
        return(conc_line)
    def move_and_record(self, food, flag_tumble):
        """
        - bacteria selct a random velocity vector if flag_tumble is True
        - construct a line (trajectory) of fixed length incorporating rotational noise.
        - get the concentration along this new line -- this depends on food enviornment (conc, grad etc)
        Parameters
        ----------
        food : object of type FoodEnviornment(class)
            used to estimate the current distance from the food and concentration along the trajectory.
        flag_tumble : bool
            change the velocity vector if True.
        
        Returns
        -------
        1d array of lenth = memory. Poisson random variable indicating the binding of signaling
        molecules along a fixed line(trajectory). This will be used further to make prediction about
        concentration gradient

        """
        self.flag_tumble = flag_tumble
        #------------------------------------------------------------------------------------------------
        if self.flag_tumble:
            new_line = self.get_new_line()          # pick up a new line-trajectory
            conc_line = self.get_concentration(food, new_line) # concentration along this new line-trajectory
            self.conc = np.flip(conc_line, axis=None)   # 1st element is the recent one; Volttera kernel applicable
            self.R = new_line[-1] # end point; this will be the initial position for the next line.
            self.psi = self.psi_list[-1] # end point; this will be the initial position for the next line.
            if not self.flag_force_write:
                self.update_trajectory(new_line) # update the new line-trajectory
                self.signal = self.rng.poisson(self.Lambda*self.conc*self.dt) # binding events
        else:
            self.psi = self.rotational_diffusion(self.psi, 1) # update psi only once
            self.V = self.velocity_for(self.psi)  # updated velocity
            self.R = self.R + self.V * self.dt  # then find the new point
            conc_R = self.get_concentration(food, self.R) # concentration at the new point
            self.conc = np.roll(self.conc, 1)  # rotating to right
            self.conc[0] = conc_R # new conc as first element
            if not self.flag_force_write:
                self.update_trajectory(self.R.reshape(1,-1)) # update the new point
                signal_R = self.rng.poisson(self.Lambda*conc_R*self.dt) # binding event at the new point
                self.signal = np.roll(self.signal, 1)  # rotating to right
                self.signal[0] = signal_R # new binding event as first element
        #--------------------------------------------------------------------------------------------------
        if self.flag_force_write: # trajectory is already written in file.
            self.signal = None # no need to record further. Assigning None is useful for further check
        return(self.signal) # return value is None only if flag_force_write = True

def normalized_binding(bind):
    bind_mean = np.mean(bind)
    bmean = max(1, bind_mean)
    bind_normalized = bind/bmean
    return bind_normalized.reshape(1,-1)

def trajectory_for(bsim, food, clf):
    flag_tumble = True # Do not change
    flag_go = True
    while flag_go:
        signal = bsim.move_and_record(food, flag_tumble)
        if(signal is None): # bacteria reached the food-circle
            flag_go = False
        else:
            signal_normalized = normalized_binding(signal)
            label_pred = clf.predict(signal_normalized)[0] # list with single element -- [True] or [False]
            flag_tumble = (label_pred == False) #decision is False if conc_gradient is -ve ; then tubmle!
    return

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

def run(args):
    food = conc.profile.DimensionTwo()
    dir_name = args['dir_name']
    clf = args['clf']
    from inputs import n_bacteria
    df_init_condition = random_initial_condition(seed=0, n_sample=n_bacteria)
    initial_position = df_init_condition[['x', 'y']].values
    bsim_list = []
    for i in range(n_bacteria):
        info = f'id: {i}/{n_bacteria}'
        print(info)
        bsim = Bacteria(bact_id=i, seed=i, directory_name = dir_name,
                        initial_position=initial_position[i], D_rot=args['D_rot'], Lambda=args['Lambda'])
        trajectory_for(bsim, food, clf) # for each bact obj, construct trajectory..
        bsim_list.append([bsim.id, bsim.flag_inside, bsim.flag_outside])
    df_bsim_list = pd.DataFrame(bsim_list, columns=['id', 'flag_inside', 'flag_outside'])
    #
    fileno = int(args['fileno'])
    filepath =  dir_name + '/D' + str(fileno) + '_'
    df_bsim_list.to_csv(filepath + 'bsim_flag.csv', index=False)
    return

if __name__ == "__main__":
    pass