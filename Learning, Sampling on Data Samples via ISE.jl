Base.run(`Clear`)

using JLD2
using Statistics

include("Graph Topology.jl")
include("Probabilities, Moments.jl")
include("ISE Optimization.jl")
include("Generate Samples, Sampling Error.jl")

# ==========================
#         FUNCTIONS
# ==========================

function index_to_ij(id::Int, I::Int, J::Int)
    """ row-major: id runs across columns first """

    @boundscheck 1 ≤ id ≤ I * J
    i = fld(id - 1, J) + 1
    j = mod(id - 1, J) + 1
    return i, j
end


function run_one_iteration(
    id::Int,
    Ml_range::Vector{Int64}, Ms_range::Vector{Int64},
    L::Int, N::Int
)
    """ Learn and sample for number of data-samples and generated-samples """

    I = length(Ml_range)
    J = length(Ms_range)
    i, j = index_to_ij(id, I, J)
    ml, ms = Ml_range[i], Ms_range[j]

    # --- Define sequences ---
    seq = collect(1:L)
    seq_skip = lattice_skip_sequence(N)
    seq_diag = lattice_diagonal_sequence(N)

    # --- Number of instances of generating different graph topologies ---
    T = 20

    # --- Error Vectors ---
    err_seq = zeros(T)
    err_skip = zeros(T)
    err_diag = zeros(T)
    err_seq = zeros(T)
    err_skip = zeros(T)
    err_diag = zeros(T)

    # --- Define graph topology ---
    edges = generate_lattice_graph(N, N)

    # --- Node-Parent Set--with graph information ---
    parent_set_seq = build_autoregressive_parents(edges, seq)
    parent_set_skip = build_autoregressive_parents(edges, seq_skip)
    parent_set_diag = build_autoregressive_parents(edges, seq_diag)

    # --- Edge weight parameters ---
    order = 5   # order = true_polynomial_order - 1
    param_seq = build_k_body_interactions(L, seq, "general", parent_set_seq, order)
    param_skip = build_k_body_interactions(L, seq_skip, "general", parent_set_skip, order)
    param_diag = build_k_body_interactions(L, seq_diag, "general", parent_set_diag, order)

    # --- All configurations ---
    Sigma = all_configurations(L)

    for t in 1:T

        # --- Generate edge weights ---
        edge_weights = ones(length(edges))
        # edge_weights = rand([-1.0, 1.0], length(edges))               # <-————————— Uncomment for random edge weights in {-1,1}
        # edge_weights = -1.0 .+ (1.0 .- (-1.0)) .* rand(length(edges)) # <-————————— Uncomment for edge weights in range [-1,1]

        # edge_self_weights = ones(L)                                   # <-————————— Uncomment if self-edges are non-zero

        # --- Generate True Moments for the Graph Topology with Generated Edge Weights ---
        prob, norm_const, mean_true, cov_true = true_moments(Sigma, edges, edge_weights)
        # JLD2.@load "Data/5x5 Experiments/configs_prob_moments.jld2" prob norm_const mean_true cov_true

        # --- Generate a new set of data samples from the Ising probability distribution ---
        data_samples = generate_data_samples(prob, norm_const, ml, L)

        # --- Learn via ISE ---
        lambda = 0.001 # L1 regularizer parameter

        seq_sol = [ISE_hist(k, data_samples, param_seq, lambda) for k in 1:L]
        skip_sol = [ISE_hist(k, data_samples, param_skip, lambda) for k in 1:L]
        diag_sol = [ISE_hist(k, data_samples, param_diag, lambda) for k in 1:L]

        # Convert to dictionary
        seq_sol_dict = Dict{Int64,Vector{Float64}}()
        skip_sol_dict = Dict{Int64,Vector{Float64}}()
        diag_sol_dict = Dict{Int64,Vector{Float64}}()

        for key in 1:L
            seq_sol_dict[key] = seq_sol[key]
            skip_sol_dict[key] = skip_sol[key]
            diag_sol_dict[key] = diag_sol[key]
        end

        # --- Sample and generate error over 50 trials ---
        err_seq[t], _ = error_sample_range(mean_true, cov_true, seq, seq_sol_dict, param_seq, ms, L)
        err_skip[t], _ = error_sample_range(mean_true, cov_true, seq_skip, skip_sol_dict, param_skip, ms, L)
        err_diag[t], _ = error_sample_range(mean_true, cov_true, seq_diag, diag_sol_dict, param_diag, ms, L)

    end
    a_seq = mean(err_seq)
    v_seq = var(err_seq, corrected=false)
    a_skip = mean(err_skip)
    v_skip = var(err_skip, corrected=false)
    a_diag = mean(err_diag)
    v_diag = var(err_diag, corrected=false)

    return a_seq, v_seq, a_skip, v_skip, a_diag, v_diag
end

# ==========================
#           MAIN
# ==========================

# --- Number of vertices ---
L = 25
# Number of vertices along each edge of the lattice
N = Int(sqrt(L))

# --- Range of data and generated samples ---
Ml_range = [250_000, 500_000, 1_000_000, 2_000_000] # Number of data samples
I = length(Ml_range)

Ms_range = [100_000]
J = length(Ms_range)

# --- Error matrices ---
avg_seq = zeros(I, J)
var_seq = zeros(I, J)
avg_skip = zeros(I, J)
var_skip = zeros(I, J)
avg_diag = zeros(I, J)
var_diag = zeros(I, J)

for id in 1:(I*J)

    m, n = index_to_ij(id, I, J)
    abcd = run_one_iteration(id, Ml_range, Ms_range, L, N)

    (avg_seq[m, n], var_seq[m, n], avg_skip[m, n], var_skip[m, n], avg_diag[m, n], var_diag[m, n]) =
        run_one_iteration(id, Ml_range, Ms_range, L, N)

end
# JLD2.@save "Data/5x5 Experiments/error_moments_Ferro.jld2" avg_seq var_seq avg_skip var_skip avg_diag var_diag