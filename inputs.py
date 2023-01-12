# training and testing ...
n_train_sample = 10000      # [int] number of trajectories in a trajectory-set for training
n_test_sample  = 10000      # [int] number of trajectories in a trajectory-set for testing
n_realization_traj   = 10   # [int] number of realizations of the trajectory-set
n_realization_signal = 100  # [int] number of realizations of the biniding events (signals) for each trajectory-set
T_max = 1.0                 # [s] duration of each short trajectory or 'run'
memory = 100                # [int] number of memory units (m) of the bact-agent, to store previous binding events
bact_speed   = 20           # [micrometer/s] speed of the bact-agent
grad_label = [0, 1]         # [k1, k2], where k is the delay. k1<k2. [0, 1] implies the latest tangent direction.
# food (target) / horizon / fixed radial distance ...
food_surface = 10           # [micor m] radius of the target or food. (R_target)
horizon      = 500          # [micro m] boundary of the region of interest. Measured from origin. (R_max)
traj_between = [30, 480]    # [micro m] trajectories initial position lie within this range.
conc_surface  = 1e1         # [microM] concentration at the food-surface. (c at R_target)
conc_horizon  = 1e-4        # [microM] concentration at the horizon. (c at R_max)
grad_fun = 'exp'            # ['lin', 'exp', '1/r'] concentration profile
# sklearn classifier
from sklearn.svm import LinearSVC
clf = LinearSVC(C=10,  dual=False,  max_iter=5000)
# repeat for
D_rot_list  = [0, 0.001, 0.01, 0.1, 0.2, 1.0] # [rad^2/s] rotational diffusion. (D_rot)
Lambda_list = [1e5, 1e6, 1e7] # [(microM s)^-1] normalized rate of molecular binding events. (lambda)
# simulation
n_bacteria = 10000           # number of agents
t_sim = 200                  # [s] maximum simulation time
