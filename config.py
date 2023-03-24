import numpy as np

##############################################################################################################################################################
##############################################################################################################################################################
# CONFIG

#what is this simulation about
short_description ="sim - N8020"  # is included in folder name

################################################################################
# general

random_seed = 42

online_saves = 0  # during simulation, periodically save all network weights
run_analysis = 1  # analyse simulation results and produce plots


################################################################################
# parameters

T = 1*10**4                 # number of simulated timesteps
T_seq = 20                  # length of one sequence - how long one orientation is presented. Corresponds to 200ms, since one iteration (Delta t) corresponds to 10ms .
T_fine = 1*10**4            # record network at single timestep resolution during the last 'T_fine' iterations of the simulation

accuracy        = int(np.ceil(T / 10**4))          # monitor recording variables every 'accuracy' iterations, i.e., 10**x sample points in total
accuracy_states = accuracy * 40                    # monitor full network network every 'accuracy_states' iterations

# number of excitatory neurons (N_pc), inhibitory neurons (N_pv), and input neurons (N_L4)
N_pc = 80
N_pv = 20
N_L4 = 80

# activation functions
gain         = 0.04              # gain 'a' of activation function a(x-b)_+^n
bias         = 0                 # bias 'b' of activation function
n_exp        = 2                 # exponent 'n' of activation function

# Initialze weight norms of E and I synaptic weights same for all E/I neurons
W_EE_norm = 0.0  # becomes non-zero during the simulation
W_EI_norm = 0.3
W_IE_norm = 0.0  # becomes non-zero during the simulation
W_II_norm = 0.35

W_EF_norm = 0.6
W_IF_norm = 0.85

# membrane potential timescales
e_x_pc       = 1/2  # corresponds to tau_E = 1/e_x_pc * Delta_t = 20ms, where Delta_t = 10ms
e_x_pv       = 1/1.7  # corresponds to tau_I = 1/e_x_pv * Delta_t = 17ms, where Delta_t = 10ms

e_y          = 1 * 10**(-4)      # timescale of online exponential weighted average to track mean firing rates and variances

# initial weights drawn from half-normal distribution with following stds
std_W_EL = 0.1     # (L4->PC)
std_W_IL = 0.1     # (L4->PV)
std_W_II = 0.0001  # (PV-|PV)
std_W_EI = 0.0001  # (PV-|PC)
std_W_IE = 0.0001  # (PC->PV)
std_W_EE = 0.0001  # (PC->PC)

# initialize tuned weights
tuned_init = False  # initialize all weights already tuned and distributed across the stimulus space (for testing purposes only)
sigma_ori = 15      #stdv of preinitialized weight kernels.


##############################################################################
# input

# width of input cell's tuning curves
input_sigma = 12

# maximum response of input cells at tuning peak
input_amp = 140


##############################################################################
# plasticity

l_norm = 1  # choose L'n'-norm (L1, L2, ...)
joint_norm = False  # normalize all excitatory and inhibitory inputs together
lateral_norm = False  # normalize all input streams separately: feedforward excitation, lateral excitation, lateral inhibition

# plasticity timescales
e = 0.1 #2 #10

e_w_EE       = e * 0.10  * 10**(-7)   # weight learning rate for excitatory connections onto excitatory neurons
e_w_EI       = e * 0.20  * 10**(-7)   # weight learning rate for inhibitory connections onto excitatory neurons

e_w_IE       = e * 0.15  * 10**(-7)   # weight learning rate for excitatory connections onto inhibitory neurons
e_w_II       = e * 0.25  * 10**(-7)   # weight learning rate for inhibitory connections onto inhibitory neurons

# plasticity on/off switch of specific weight matrices
e_w = 1  # EF
e_k = 1  # IF
e_p = 1  # II
e_m = 1  # EI
e_q = 1  # IE
e_u = 1  # EE

##############################################################################
