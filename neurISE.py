import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import matplotlib.pyplot as plt
import collections

# ========================================================================================
#                                    FUNCTION DEFINITIONS
# ========================================================================================

## Returns the unnormalised probability of a configuration x in an Ising model
# p(x) = sum_{i in V} H_i*x_i + Sum_{(i,j) in E} J_{i,j}*x_i*x_j
# ----------------------------------------------------------------------------------------
def prob_config(x, g, n):
    """
    Arguments: 
    x ∈ R^n configuration, g graph Laplacian, n number of nodes
    """
    H = np.array([np.diagonal(g)[i] * x[i] for i in range(n)])
    
    i1, i2 = np.meshgrid(np.arange(n), np.arange(n))
    i1 = i1.flatten()
    i2 = i2.flatten()
    indices = [(i1[idx], i2[idx]) for idx in range(len(i1)) if i1[idx] != i2[idx]]
    J = np.array([g[i1,i2] * x[i1] * x[i2] for (i1, i2) in indices])

    p = np.exp(sum(H) + 0.5 * sum(J))
    return p

## Returns normalising factor of the probability distribution
# Sum_{configurations in dict} p(config)
# ----------------------------------------------------------------------------------------
def normalize_factor(vec, g, n):
    z = 0
    for row in vec:
        z += prob_config(row, g, n)
    return z

# Function to compute p(x[1]|x[2:N])
# ----------------------------------------------------------------------------------------
def true_cond_prob(x, g, n):
    """
    Compute p(xi | xj) conditional prob of i wrt j
    Arguments:
    - x = [i xi; j xj]: configuration
    - q: graph Laplacian, n number of nodes
    """
    L = np.arange(1, n+1)
    
    # Numerator calculation
    z1 = list(itertools.product([-1, 1], repeat = n-x.shape[0])) # For marginalization
    z1 = np.array(z1)
    Arr1 = np.zeros((len(z1), n))
    ind = np.isin(L, x[:,0]) # check where elements of x are in (0,n)

    A = np.zeros(n)
    for i in L[ind]:
        A[i-1] = x[x[:,0] == i,1][0]

    Arr1[:] = A
    for j in range(len(z1)):
        zero_idx = np.where(Arr1[j,:] == 0)[0]
        Arr1[j, zero_idx] = z1[j]

    print(Arr1)

    pr = [prob_config(Arr1[j, :], g, n) for j in range(len(z1))]
    Num = sum(pr)

    # Denominator calculation
    z2 = list(itertools.product([-1, 1], repeat = n-x.shape[0]+1))
    z2 = np.array(z2)
    Arr2 = np.zeros((len(z2), n))
    ind2 = np.isin(L, x[1:,0])

    A2 = np.zeros(n)
    for i in L[ind2]:
        A2[i-1] = x[x[:,0] == i,1][0]

    Arr2[:] = A2
    for j in range(len(z2)):
        zero_idx = np.where(Arr2[j,:] == 0)[0]
        Arr2[j, zero_idx] = z2[j]

    print(Arr2)

    pr2 = [prob_config(Arr2[j, :], g, n) for j in range(len(z2))]
    Den = sum(pr2)

    return Num / Den

# Compute the loss for a single MLP corresponding to node i
# ----------------------------------------------------------------------------------------
def neurise_loss(i, x, mlp, q, m):
    """
    Arguments:
    - samp: An array (md, n+1) where each row of samp[:,1:] is a unique configuration, 
    samp[:,0] is the number of count
    - mlp
    """
    counts = torch.tensor(x[:, 0]) # Extract sample counts & convert into PyTorch tensor

    # Extract sigma_i and sigma_neighbours
    sigma_i = x[:, i]
    
    sigma_neighbors = np.delete(x[:,1:], i-1, axis=1) # Delete the ith column
    sigma_neighbors = torch.tensor(sigma_neighbors, dtype=torch.float32) # Convert to PyTorch tensor
    
    # Compute MLP output for neighbours
    nn_output = mlp(sigma_neighbors) # size(md,2)
 
    # Compute centered indicators
    phi_in = np.where(sigma_i == -1, 1 - 1/q, -1/q) # Phi(x_i = 1/q)
    phi_in = torch.tensor(phi_in, dtype=torch.float32) 
    
    phi_ip = np.where(sigma_i == 1, 1 - 1/q, -1/q) # Phi(x_i = -1/q)
    phi_ip = torch.tensor(phi_ip, dtype=torch.float32)
   
    # Compute the loss for variable u
    loss = (1 / m) * sum(counts * torch.exp(-phi_in * nn_output[:, 0] - phi_ip * nn_output[:, 1]))
    return loss

# function to learn the conditional probability
# ----------------------------------------------------------------------------------------
def learnt_cond_prob(x, mlp, q):
        """
        Compute p(xi | xj) conditional prob of i wrt j
        Arguments:
        - x = [i,xi]: configuration
        - mlps: List of trained MLPs, one for each node
        - q: Number of discrete states for each variable
        """
        i = x[0,0]
        mlp_i = mlp[i-1]  # Get the MLP for the current variable
        
        # Extract value of sigma_i and sigma_neighbours
        sigma_i = x[0,1]

        sigma_neighbors = x[1:,1]
        sigma_neighbors = torch.tensor(sigma_neighbors, dtype=torch.float32) # Convert to PyTorch tensor
    
        # Compute the MLP output for the neighbors of variable u
        nn_output = mlp_i(sigma_neighbors)
        
        # Compute centered indicators
        phi_in = 1 - 1/q if sigma_i == -1 else -1/q
        phi_ip = 1 - 1/q if sigma_i == 1 else -1/q

        nno = nn_output.detach().numpy()
        cpp = np.exp(phi_in * nno[0] + phi_ip * nno[1])
        cpn = np.exp(-phi_in * nno[0] - phi_ip * nno[1])
        
        return cpp/(cpp+cpn)

# ========================================================================================
#                                      MAIN
# ========================================================================================

# Generate Data Samples

N = 4 # Number of nodes
Ml = 1000 # Number of data samples to be generated

G = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]) # Graph Laplacian

comb = list(itertools.product([-1,1],repeat=N)) # Generate 2^N configurations
comb = np.array(comb)

# Ising probability distribution function
ising_prob = np.zeros(2**N) 
Z = normalize_factor(comb,G,N)
for i in range(2**N):
    ising_prob[i] = prob_config(comb[i,:], G, N)/Z

# Generate samples
smple = np.random.choice(np.arange(0,2**N), size=Ml, p=ising_prob) # Sample from the Ising pdf

counter = collections.Counter(smple) # Create a histogram, outputs a dictionary
comb_no = list(counter.keys()) # configuration number
count = list(counter.values()) # Repetition number
comb_values = np.array([comb[k] for k in comb_no]) # corresponding configuration

samples = np.column_stack((count, comb_values))
print(samples)

# Generate Data Samples

N = 4 # Number of nodes
Ml = 1000 # Number of data samples to be generated

G = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]) # Graph Laplacian

comb = list(itertools.product([-1,1],repeat=N)) # Generate 2^N configurations
comb = np.array(comb)

# Ising probability distribution function
ising_prob = np.zeros(2**N) 
Z = normalize_factor(comb,G,N)
for i in range(2**N):
    ising_prob[i] = prob_config(comb[i,:], G, N)/Z

# Generate samples
smple = np.random.choice(np.arange(0,2**N), size=Ml, p=ising_prob) # Sample from the Ising pdf

counter = collections.Counter(smple) # Create a histogram, outputs a dictionary
comb_no = list(counter.keys()) # configuration number
count = list(counter.values()) # Repetition number
comb_values = np.array([comb[k] for k in comb_no]) # corresponding configuration

samples = np.column_stack((count, comb_values))
print(samples)

# Learn the conditional probabilities
# ----------------------------------------------------------------------------------------
input_dim = N-1 # Defined for each node i and its neighbors
hidden_dim = 20
output_dim = 2

# Construct a NN for each node
mlps = [torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), 
                            torch.nn.ReLU(), 
                            # torch.nn.Linear(hidden_dim, hidden_dim), 
                            # torch.nn.ReLU(), 
                            torch.nn.Linear(hidden_dim, output_dim))
        for _ in range(N)]

# Training parameters
num_epochs = 200  # Number of training epochs
learning_rate = 0.01  # Learning rate

# Loop through each variable (u) and train its MLP independently
for i in range(N):
    
    optimizer = optim.Adam(mlps[i].parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        
        optimizer.zero_grad() # Zero the gradients before the backward pass
        loss_i = neurise_loss(i, samples, mlps[i], output_dim, Ml) # Compute Loss
        loss_i.backward() # Backpropagation to compute gradients   
        optimizer.step() # Update MLP weights

        # Print progress
        # if (epoch + 1) % 10 == 0:
        #     print(f"Epoch [{epoch+1}/{num_epochs}], Loss (Node={i+1}): {loss_i.item():.4f}")
