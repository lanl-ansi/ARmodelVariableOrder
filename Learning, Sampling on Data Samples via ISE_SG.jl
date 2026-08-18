using GraphicalModelLearning
using JLD2, HDF5, DelimitedFiles
using Statistics
using Random
using Plots

include("Graph Topology.jl")
include("ISE Optimization.jl")
include("Probabilities, Moments.jl")
include("Generate_Test_Samples_Sampling_Error.jl")
include("Generate_Training_Samples_Learning_Error.jl")


# --- Number of vertices ---
N = 25
# Number of vertices along each edge of the lattice
L = Int(sqrt(N))

beta = 0.6
beta_tag = replace(string(beta), "." => "")
# beta_tag = replace(string(beta), ".0" => "")

# --- Range of data and generated samples ---
Ml_range = [500, 1000, 5000, 10_000]
I = length(Ml_range)

Ms = 100_000

# --- Number of iterations ---
T = 10

# --- Define graph topology ---
edges = generate_lattice_graph(L, L)

# --- Define sequences ---
seq = collect(1:N)
seq_skip = lattice_skip_sequence(L)
seq_diag = lattice_diagonal_sequence(L)
seq_greedy = greedy_frontier_sequence(edges, N)

ordering_keys = (:sequential, :checkerboard, :diagonal, :greedy)

# --- Node-Parent Set--with graph information ---
parent_set_seq = build_autoregressive_parents(edges, seq)
parent_set_skip = build_autoregressive_parents(edges, seq_skip)
parent_set_diag = build_autoregressive_parents(edges, seq_diag)
parent_set_greedy = build_autoregressive_parents(edges, seq_greedy)

# --- Edge weight parameters ---
order = 5   # order = true_polynomial_order - 1
param_seq = build_k_body_interactions(N, seq, "general", parent_set_seq, order)
param_skip = build_k_body_interactions(N, seq_skip, "general", parent_set_skip, order)
param_diag = build_k_body_interactions(N, seq_diag, "general", parent_set_diag, order)
param_greedy = build_k_body_interactions(N, seq_greedy, "general", parent_set_greedy, order)

models = Dict(
    :sequential => (
        label="Sequential",
        seq=seq,
        parents=parent_set_seq,
        param=param_seq,
    ),
    :checkerboard => (
        label="Checkerboard",
        seq=seq_skip,
        parents=parent_set_skip,
        param=param_skip,
    ),
    :diagonal => (
        label="Diagonal",
        seq=seq_diag,
        parents=parent_set_diag,
        param=param_diag,
    ),
    :greedy => (
        label="Greedy",
        seq=seq_greedy,
        parents=parent_set_greedy,
        param=param_greedy,
    ),
)

# --- Store solution and errors ---
metrics = (
    :model_tv_error,
    :forward_kl_error,
    :reverse_kl_error,
    :pairmoment_rmse,
    :sample_tv_error,
    :total_kl_error,
    :magnetization_error,
    :overlap_tv_error,
)

# ---- Store solutions and results ---
solutions = Dict(
    ml => Dict(
        key => Vector{Dict{Int,Vector{Float64}}}(undef, T)
        for key in ordering_keys
    )
    for ml in Ml_range
)

mean_errors = Dict(
    key => Dict(
        metric => zeros(I)
        for metric in metrics
    )
    for key in ordering_keys
)

std_errors = Dict(
    key => Dict(
        metric => zeros(I)
        for metric in metrics
    )
    for key in ordering_keys
)

all_trial_errors = Dict(
    key => Dict(
        metric => zeros(I, T)
        for metric in metrics
    )
    for key in ordering_keys
)

results = Dict(
    :trial_errors => all_trial_errors,
    :summary => Dict(
        :mean => mean_errors,
        :std => std_errors,
    ),
)

# --- Load set of training samples ---
fname2 = "Autoregressive Models/Data/5x5/Samples_5x5_M=200K_SpinGlass_beta=$(beta_tag).csv"
data_samples = Int64.(readdlm(fname2, ','))

# ---------- Precompute reference distributions for each t ----------
overlap_prob_true_by_t = Vector{Vector{Float64}}(undef, T)
mag_prob_true_by_t = Vector{Vector{Float64}}(undef, T)

for t in 1:T
    # Overlap reference
    reference_samples = data_samples[(10_001+20_000*(t-1)):(20_000+20_000*(t-1)), :]
    overlap_prob_true_by_t[t] = overlap_distribution(reference_samples)

    # Exact magnetization reference
    fname1 = "Autoregressive Models/Data/5x5/Data_5x5_SpinGlass_beta=$(beta_tag)_t$(t).h5"

    p_unnorm, norm_const = h5open(fname1, "r") do f
        return (
            read(f["p_unnorm"]),
            read(f["norm_const"])
        )
    end

    p_true = Float64.(p_unnorm) ./ Float64(norm_const)

    mag_prob_true = zeros(Float64, N + 1)

    for state_idx in eachindex(p_true)
        sigma = index_to_configuration(state_idx - 1, N)
        nplus = count(==(1), sigma)
        mag_prob_true[nplus+1] += p_true[state_idx]
    end

    mag_prob_true_by_t[t] = mag_prob_true
end


for (j, ml) in enumerate(Ml_range)

    for t in 1:T
        seed = 10_000 * j + t

        # --- Training Samples ---
        train_samples = compress_samples(data_samples[(20_000*(t-1)+1):(20_000*(t-1)+ml), :]) #histogram

        if sum(train_samples[:, 1]) != ml
            throw(ArgumentError("train_samples does not contain ml samples"))
        end

        # --- Load true moments for some error computations ---
        fname1 = "Autoregressive Models/Data/5x5/Data_5x5_SpinGlass_beta=$(beta_tag)_t$(t).h5"
        edge_weights, p_unnorm, norm_const, mean_true, cov_true = h5open(fname1, "r") do f
            return (
                read(f["edge_weights"]),
                read(f["p_unnorm"]),
                read(f["norm_const"]),
                read(f["mean_true"]),
                read(f["cov_true"])
            )
        end
        p_true = deepcopy(p_unnorm)
        p_true ./= Float64(norm_const)
        logp_true = log.(p_true)

        # --- Towards magnetization and overlap error ---
        mag_prob_true = mag_prob_true_by_t[t]
        overlap_prob_true = overlap_prob_true_by_t[t]

        # --- Learn & Sample ---
        for key in ordering_keys
            model = models[key]

            # Learn using GML
            # ----------------
            sol_dict = learn_conditionals(model.seq, model.parents, model.param, order + 1, train_samples)
            solutions[ml][key][t] = deepcopy(sol_dict)

            learning_errors = exact_learning_errors(model.seq, model.param, sol_dict, p_true, logp_true)

            # Sampling
            # ---------
            rng_model = MersenneTwister(seed)

            # --- sampling errors ---
            sampling_errors = finite_sampling_errors(
                model.seq, model.param, sol_dict, Ms,
                mean_true, cov_true, # First moments / pair correlations
                p_true; # TV and KL
                mag_prob_true=nothing, # Magnetization TV
                overlap_prob_true=nothing,
                rng=rng_model,
                compute_moment=false,
                compute_mean_rmse=false,
                compute_paircorr=false,
                compute_tv=false,
                compute_sampling_kl=false,
                compute_total_kl=true,
                compute_magnetization=false,
                compute_overlap=false,
            )

            errors = (
                model_tv_error=learning_errors.tv_error,
                forward_kl_error=learning_errors.forward_kl_error,
                reverse_kl_error=learning_errors.reverse_kl_error,
                mean_rmse=sampling_errors.mean_rmse,
                pairmoment_rmse=sampling_errors.paircorr_error,
                sample_tv_error=sampling_errors.tv_error,
                sampling_kl_error=sampling_errors.sampling_kl_error,
                total_kl_error=sampling_errors.total_kl_error,
                magnetization_error=sampling_errors.magnetization_tv_error,
                overlap_tv_error=sampling_errors.overlap_tv_error,
            )

            for metric in metrics
                results[:trial_errors][key][metric][j, t] = getproperty(errors, metric)
            end

            println("Learning Sampling error done for t=$(t) and ml=$(ml)")
        end

    end

    for key in ordering_keys
        for metric in sampling_metrics
            trial_values =
                @view results[:trial_errors][key][metric][j, :]

            results[:summary][:mean][key][metric][j] =
                mean(trial_values)

            results[:summary][:std][key][metric][j] =
                std(trial_values)
        end
    end
end

# ---- SAVE ---
fname5 = joinpath("Autoregressive Models/Results/5x5", "results_5x5_SpinGlass_beta=$(beta_tag)_updated.jld2")

jldsave(
    fname5;
    beta=beta,
    Ml_range=Ml_range,
    Ms=Ms,
    T=T,
    ordering_keys=ordering_keys,
    metrics=metrics,
    solutions=solutions,
    results=results,
)


# ==========================
#      Post-Processing
# ==========================

using Plots
using LaTeXStrings

function plot_all_errors(
    metrics,
    ml_range,
    mean_errors,
    std_errors;
    ordering_keys=(:sequential, :checkerboard, :diagonal, :greedy),
    use_log_y::Bool=false,
)
    colors = palette(:RdBu_4, length(ordering_keys))
    markers = (:circle, :square, :diamond, :utriangle)

    panels = Plots.Plot[]

    for (metric_index, metric) in enumerate(metrics)
        haskey(METRIC_LABELS, metric) ||
            error("No label has been defined for metric $metric.")

        Pmetric = plot(
            xscale=:log10,
            yscale=use_log_y ? :log10 : :identity,
            xticks=(ml_range, string.(ml_range)),
            xlabel=L"M_l",
            ylabel=METRIC_LABELS[metric],
            title=METRIC_TITLES[metric],
            legend=metric_index == 1 ? :best : false,
            grid=true,
            minorgrid=false,
        )

        for (model_index, key) in enumerate(ordering_keys)
            plot!(
                Pmetric,
                ml_range,
                mean_errors[key][metric][1:length(plot_Ml_range)];
                yerror=std_errors[key][metric][1:length(plot_Ml_range)],
                color=colors[model_index],
                errorcolor=colors[model_index],
                marker=markers[model_index],
                markersize=5,
                linewidth=2,
                capsize=3,
                label=uppercasefirst(String(key)),
            )
        end

        push!(panels, Pmetric)
    end

    return plot(
        panels...;
        layout=(2, 3),
        size=(1500, 800), # 1500,1200 for 3x3, 1500,800 for 2,3
        left_margin=12Plots.mm,
        right_margin=6Plots.mm,
        bottom_margin=8Plots.mm,
        top_margin=6Plots.mm,
    )
end

# -------------------------------------------
const METRIC_LABELS = Dict(
    :model_tv_error => L"\mathrm{TV}(p,q_\theta)",
    :forward_kl_error => L"D_{\mathrm{KL}}(p\Vert q_\theta)",
    :reverse_kl_error => L"D_{\mathrm{KL}}(q_\theta\Vert p)",
    :sample_tv_error => L"\mathrm{TV}(p,\widehat q_{M_s})",
    # :pairmoment_rmse => L"\mathrm{RMSE}_{\mathrm{pair}}",
    :total_kl_error => L"D_{\mathrm{KL}}(\widehat q_{M_s}\Vert p)",
    :magnetization_error => L"\mathrm{TV}\!\left(P_p(m),\widehat{P}_{M_s}(m)\right)",
    # :overlap_tv_error => L"\mathrm{TV}\!\left(P_p(q),P_{q_\theta}(q)\right)",
)

const METRIC_TITLES = Dict(
    :model_tv_error => "Exact model TV",
    :forward_kl_error => "Exact forward KL",
    :reverse_kl_error => "Exact reverse KL",
    :sample_tv_error => "Finite-sample TV",
    # :pairmoment_rmse => "Pair-moment RMSE",
    :total_kl_error => "Total KL",
    :magnetization_error => "Magnetization-distribution TV",
    # :overlap_tv_error => "Replica-overlap TV",
)

plot_Ml_range = [500, 1000, 5000, 10_000]

plot_metrics = (
    :model_tv_error,
    :forward_kl_error,
    :reverse_kl_error,
    :sample_tv_error,
    # :pairmoment_rmse,
    :total_kl_error,
    :magnetization_error,
    # :overlap_tv_error,
)

P = plot_all_errors(
    plot_metrics,
    plot_Ml_range,
    mean_errors,
    std_errors;
    use_log_y=false,
)

display(P)