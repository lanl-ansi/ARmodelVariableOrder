Base.run(`Clear`)

using JLD2
using Statistics

include("Graph Topology.jl")
include("ISE Optimization.jl")
include("Generate Samples, Sampling Error.jl")
include("Gibbs_Sampling.jl")

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

    # --- Define graph topology ---
    edges = generate_lattice_graph(N, N)

    # --- Node-Parent Set--with graph information ---
    parent_set_seq = build_autoregressive_parents(edges, seq)
    parent_set_skip = build_autoregressive_parents(edges, seq_skip)
    parent_set_diag = build_autoregressive_parents(edges, seq_diag)

    # --- Edge weight parameters ---
    order = 1   # order = true_polynomial_order - 1
    param_seq = build_k_body_interactions(L, seq, "general", parent_set_seq, order)
    param_skip = build_k_body_interactions(L, seq_skip, "general", parent_set_skip, order)
    param_diag = build_k_body_interactions(L, seq_diag, "general", parent_set_diag, order)

    # --- Generate edge weights ---
    edge_weights = ones(length(edges))
    # edge_weights = rand([-1.0, 1.0], length(edges))               # <-————————— Uncomment for random edge weights in {-1,1}
    # edge_weights = -1.0 .+ (1.0 .- (-1.0)) .* rand(length(edges)) # <-————————— Uncomment for edge weights in range [-1,1]

    # edge_self_weights = zeros(L)                                    # <-————————— Uncomment if self-edges are non-zero

    # ========================================================================
    #               Generate Data Samples and Moments using Gibbs Sampling 
    # ========================================================================
    # Initial condition
    s_plus = fill(Int8(1), L)
    s_minus = fill(Int8(-1), L)

    # Sampler parameters
    M = Int(ml / 2) # kept samples per chain
    burn = 500L     # start large for ferromagnet
    thin = 1

    # Sample
    counts1, tr1, mu1, M1 = gibbs_ising_sampler(
        edges, edge_weights, edge_self_weights, s_plus, M, L; burn=burn, thin=thin)

    counts2, tr2, mu2, M2 = gibbs_ising_sampler(
        edges, edge_weights, edge_self_weights, s_minus, M, L; burn=burn, thin=thin)

    # Combine the two sets of samples
    ds1 = counts_to_matrix(counts1)
    ds2 = counts_to_matrix(counts2)
    data_samples_Gibbs = vcat(ds1, ds2)

    # Compute the first two moments
    w1 = 0.5
    w2 = 1 - w1

    mean_true = w1 .* mu1 .+ w2 .* mu2
    d1 = mu1 .- mean_true
    d2 = mu2 .- mean_true
    cov_true = w1 .* M1 .+ w2 .* M2 .+ w1 .* (d1 * d1') .+ w2 .* (d2 * d2')

    # JLD2.@load "Data/10x10 Experiments/Empirical_Moments.jld2" mean_true cov_true

    # -----  Check if mixing is achieved  -----
    Rhat_E = rhat([tr1[:energy], tr2[:energy]])      # target < ~1.01
    Rhat_abs = rhat([tr1[:abs_mag], tr2[:abs_mag]])  # target < ~1.01
    println("Check if Rhat is < 1.01: [$(Rhat_E), $(Rhat_abs)]")

    # ========================================================================
    #                       Learning & Sampling 
    # ========================================================================

    # --- Learn via ISE ---
    lambda = 0.001 # L1 regularizer parameter

    seq_sol = [ISE_hist(k, data_samples_Gibbs, param_seq, lambda) for k in 1:L]
    skip_sol = [ISE_hist(k, data_samples_Gibbs, param_skip, lambda) for k in 1:L]
    diag_sol = [ISE_hist(k, data_samples_Gibbs, param_diag, lambda) for k in 1:L]

    # Convert to dictionary
    seq_sol_dict = Dict{Int64,Vector{Float64}}()
    skip_sol_dict = Dict{Int64,Vector{Float64}}()
    diag_sol_dict = Dict{Int64,Vector{Float64}}()

    for key in 1:L
        seq_sol_dict[key] = seq_sol[key]
        skip_sol_dict[key] = skip_sol[key]
        diag_sol_dict[key] = diag_sol[key]
    end
    # JLD2.@load "Data/10x10 Experiments/Sol_$(ml ÷ 1000)K_order_2.jld2" seq_sol_dict skip_sol_dict diag_sol_dict

    # --- Sample and generate error over 50 trials ---
    err_seq, var_seq = error_sample_range(mean_true, cov_true, seq, param_seq, seq_sol_dict, ms, L)
    err_skip, var_skip = error_sample_range(mean_true, cov_true, seq_skip, param_skip, skip_sol_dict, ms, L)
    err_diag, var_diag = error_sample_range(mean_true, cov_true, seq_diag, param_diag, diag_sol_dict, ms, L)

    return err_seq, var_seq, err_skip, var_skip, err_diag, var_diag
end

# ==========================
#            MAIN
# ==========================

# --- Number of vertices ---
L = 100
# Number of vertices along each edge of the lattice
N = Int(sqrt(L))

# --- Range of data and generated samples ---
Ml_range = [50_000, 250_000, 500_000] # Number of data samples
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

    (avg_seq[m, n], var_seq[m, n], avg_skip[m, n], var_skip[m, n], avg_diag[m, n], var_diag[m, n]) =
        run_one_iteration(id, Ml_range, Ms_range, L, N)

end
# JLD2.@save "Data/10x10 Experiments/error_moments_ferro_order_2.jld2" avg_seq var_seq avg_skip var_skip avg_diag var_diag