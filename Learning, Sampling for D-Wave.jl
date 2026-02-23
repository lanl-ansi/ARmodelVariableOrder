Base.run(`Clear`)

using CSV, DataFrames
using JLD2

include("Graph Topology.jl")
include("Probabilities, Moments.jl")
include("ISE Optimization.jl")
include("Generate Samples, Sampling Error.jl")

# ==========================
#      Load Data 
# ==========================

# --- Load Data-Samples ---
data_samples = CSV.read("Data/D-Wave/dwave_samples.csv", DataFrame; delim=',');
data_samples = Matrix(data_samples)

# --- Number of nodes ---
L = size(data_samples, 2) - 1
num_spins = 2
num_samples = sum(data_samples[:, 1])

# ==========================
#      Learn true edges
# ==========================

regularizing_value = 0.2 # L1 regularizer parameter
lambda = regularizing_value * sqrt(log((num_spins^2) / 0.05) / num_samples)

# --- Associated parameters ---
param = build_k_body_interactions(L, nothing, "pairwise", nothing, 2)

# --- Run ISE and save ---
sol = [ISE_hist(k, data_samples, param, lambda) for k in 1:L]

sol_ISE = Dict{Int64,Vector{Float64}}()
[sol_ISE[k] = sol[k] for k in 1:L]
JLD2.@save "Data/D-Wave/sol_ISE.jld2" sol_ISE

# --- Symmeterize and construct the edge weights from ISE output ---
delta = 0.2 # Threshold value
edges = Vector{Tuple{Int,Int}}() # Doesn't include self weights
edge_weights = Float64[]
edge_self_weights = Float64[]

# Diagonal self-edges (i,i)
for i in 1:L
    w = (abs(sol_ISE[i][1]) > delta) ? sol_ISE[i][1] : 0.0
    push!(edge_self_weights, w)
end

# Edges (i,j)
for i = 1:L-1
    for j = i+1:L
        @assert (param[i][j][1] == j) && (param[j][i+1][1] == i) """
        param_seq bookkeeping failed at (i=$i, j=$j):
          param_seq[i][j]   = $(param[i][j])
          param_seq[j][i+1] = $(param[j][i+1])
        """
        println("here")
        println([param[i][j], sol_ISE[i][j]])
        println([param[j][i+1], sol_ISE[j][i+1]])

        w = (sol_ISE[i][j] + sol_ISE[j][i+1]) / 2

        if abs(w) > delta
            push!(edges, (i, j))
            push!(edge_weights, w)
        end
    end
    println(edges, edge_weights)
end
# JLD2.@save "Data/D-Wave/edges.jld2" edge_self_weights edges edge_weights

# --- Load Edges ---
JLD2.@load "Data/D-Wave/edges.jld2" edge_self_weights edges edge_weights

# ==========================
#  AR Learning and Sampling
# ==========================

# --- Define sequences ---
seq = collect(1:L)

seq_cros = [9, 10, 11, 12, 13, 14,          # Block 2
    28, 29, 30, 31,                         # Block 5
    48, 49, 50, 51, 52, 53, 54,             # Block 8
    22, 23, 24, 25, 26, 27,                 # Block 4
    1, 2, 3, 4, 5, 6, 7, 8,                 # Block 1
    40, 41, 42, 43, 44, 45, 46, 47,         # Block 7
    32, 33, 34, 35, 36, 37, 38, 39,         # Block 6
    15, 16, 17, 18, 19, 20, 21,             # Block 3
    55, 56, 57, 58, 59, 60, 61, 62]         # Block 9

# --- Node-Parent Set--with graph information ---
parent_set_seq = build_autoregressive_parents(edges, seq)
parent_set_cros = build_autoregressive_parents(edges, seq_cros)

# --- Edge weight parameters ---
order = 2   # order = true_polynomial_order - 1
param_seq = build_k_body_interactions(L, seq, "general", parent_set_seq, order)
param_cros = build_k_body_interactions(L, seq_cros, "general", parent_set_cros, order)

# --- Learn Conditionals via ISE ---
lambda = 0.001 # L1 regularizer parameter
sol_seq = [ISE_hist(k, data_samples, param_seq, lambda) for k in 1:L]
sol_cros = [ISE_hist(k, data_samples, param_cros, lambda) for k in 1:L]

# Convert to dictionary & save
seq_sol_dict = Dict{Int64,Vector{Float64}}()
[seq_sol_dict[key] = sol_seq[key] for key in 1:L]

sol_cros_dict = Dict{Int64,Vector{Float64}}()
[sol_cros_dict[key] = sol_cros[key] for key in 1:L]

# JLD2.@save "Data/D-Wave/Sol_AR_order_3_new.jld2" seq_sol_dict sol_cros_dict

# --- Empirical Moments ---
emp_mean, emp_cov = empirical_moments(data_samples)

# --- SAMPLING ---
Ms_range = [100_000, 250_000, 500_000, 1_000_000]
J = length(Ms_range)

# --- Moment Vectors ---
avg_seq = zeros(J)
avg_cros = zeros(J)
var_seq = zeros(J)
var_cros = zeros(J)

# --- Sample and generate error over 50 trials ---
for k in eachindex(Ms_range)
    ms = Ms_range[k]
    avg_seq[k], var_seq[k] = error_sample_range(emp_mean, emp_cov, seq, sol_seq_dict, param_seq, ms, L)
    avg_cros[k], var_cros[k] = error_sample_range(emp_mean, emp_cov, seq_cros, sol_cros_dict, param_cros, ms, L)
end
# JLD2.@save "Data/D-Wave/error_moments_spin_hist_order_3.jld2" avg_seq var_seq avg_cros var_cros