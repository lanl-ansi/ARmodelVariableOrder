Base.run(`Clear`)

using JLD2
using Statistics

include("Graph Topology.jl")
include("Probabilities, Moments.jl")
include("ISE Optimization.jl")
include("Generate Samples, Sampling Error.jl")

# --- Number of vertices ---
L = 25
# Number of vertices along each edge of the lattice
N = Int(sqrt(L))

# --- Define graph structure ---
edges = generate_lattice_graph(N, N)
edge_weights = ones(length(edges))
edge_weights = rand([-1.0, 1.0], length(edges))                 # <-————————— Uncomment for random edge weights in {-1,1}
# edge_weights = -1.0 .+ (1.0 .- (-1.0)) .* rand(length(edges)) # <-————————— Uncomment for edge weights in range [-1,1]

# edge_self_weights = zeros(L)                                   # <-————————— Uncomment if self-edges are non-zero   

# --- Define sequences ---
seq = collect(1:L)
seq_skip = lattice_skip_sequence(N)
seq_diag = lattice_diagonal_sequence(N)

# --- Node-Parent Set--with graph information ---
parent_set_seq = build_autoregressive_parents(edges, seq)
parent_set_skip = build_autoregressive_parents(edges, seq_skip)
parent_set_diag = build_autoregressive_parents(edges, seq_diag)

# --- Edge weight parameters ---
order = 5   # order = true_polynomial_order - 1
param_seq = build_k_body_interactions(L, seq, "general", parent_set_seq, order)
param_skip = build_k_body_interactions(L, seq_skip, "general", parent_set_skip, order)
param_diag = build_k_body_interactions(L, seq_diag, "general", parent_set_diag, order)

# --- Generate True Moments for the Graph Topology with Generated Edge Weights ---
Sigma = all_configurations(L)
prob, norm_const, mean_true, cov_true = true_moments(Sigma, edges, edge_weights)
# JLD2.@load "Data/5x5 Experiments/configs_prob_moments.jld2"

# # --- Learn via ISE --- 
lambda = 0.001 # L1 regularizer parameter 

sol_seq_dict = Dict{Int64,Vector{Float64}}()
sol_skip_dict = Dict{Int64,Vector{Float64}}()
sol_diag_dict = Dict{Int64,Vector{Float64}}()

for k in 1:L
    sol_seq_dict[k] = ISE_true(Sigma, k, param_seq, prob, norm_const, lambda)
    sol_skip_dict[k] = ISE_true(Sigma, k, param_skip, prob, norm_const, lambda)
    sol_diag_dict[k] = ISE_true(Sigma, k, param_diag, prob, norm_const, lambda)
end
# JLD2.@save "Data/sol_seq_true_order_6.jld2" sol_seq_dict
# JLD2.@save "Data/sol_skip_true_order_6.jld2" sol_skip_dict
# JLD2.@save "Data/sol_diag_true_order_6.jld2" sol_diag_dict

# --- SAMPLING ---
# JLD2.@load "Data/5x5 Experiments/Learn on True Model/Sol_true_order_2_new.jld2" sol_seq_dict sol_skip_dict sol_diag_dict

Ms_range = [100_000, 250_000, 500_000]
J = length(Ms_range)

# --- Moment Vectors ---
avg_seq = zeros(J)
avg_skip = zeros(J)
avg_diag = zeros(J)
var_seq = zeros(J)
var_skip = zeros(J)
var_diag = zeros(J)

# --- Sample and generate error over 50 trials ---
for k in eachindex(Ms_range)
    ms = Ms_range[k]

    avg_seq[k], var_seq[k] = error_sample_range(mean_true, cov_true, seq, sol_seq_dict, param_seq, ms, L)
    avg_skip[k], var_skip[k] = error_sample_range(mean_true, cov_true, seq_skip, sol_skip_dict, param_skip, ms, L)
    avg_diag[k], var_diag[k] = error_sample_range(mean_true, cov_true, seq_diag, sol_diag_dict, param_diag, ms, L)
end
# JLD2.@save "Data/5x5 Experiments/error_moments_ferro_true_order_6_new.jld2" avg_seq var_seq avg_skip var_skip avg_diag var_diag