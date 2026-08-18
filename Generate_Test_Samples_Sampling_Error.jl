using StatsBase
using Random
using LinearAlgebra

function generate_test_samples(
    seq::Vector{Int},                     # Sequence of sites
    param::Dict{Int,Vector{Vector{Int}}},   # Dictionary (nodes, interactions)
    sol_dict::Dict{Int,Vector{Float64}},       # Learned interaction weights
    m::Int,                             # Number of samples to generate
    n::Int;                             # Number of sites
    rng::AbstractRNG=Random.GLOBAL_RNG,
)
    m > 0 || error("m must be positive.")
    n == length(seq) || error("n must equal length(seq).")

    x = zeros(Int8, m, n)
    p = zeros(Float64, m)
    sum_buf = zeros(Float64, m)
    temp = ones(Int8, m)
    rand_vals = zeros(Float64, m)

    for i in seq
        Ei = param[i]
        length(sol_dict[i]) == length(Ei) ||
            error("sol_dict[$i] and param[$i] have inconsistent lengths")

        fill!(sum_buf, 0.0)

        for j in eachindex(Ei)
            fill!(temp, Int8(1))
            interaction = Ei[j]

            for a in interaction
                a == i && continue
                @inbounds @simd for idx in 1:m
                    temp[idx] *= x[idx, a]
                end
            end

            @inbounds @simd for idx in 1:m
                sum_buf[idx] += sol_dict[i][j] * temp[idx]
            end
        end

        @inbounds @simd for idx in 1:m
            p[idx] = 1.0 / (1.0 + exp(-2.0 * sum_buf[idx]))
        end

        rand!(rng, rand_vals)

        @inbounds @simd for k in 1:m
            x[k, i] = rand_vals[k] < p[k] ? Int8(1) : Int8(-1)
        end
    end

    # # Histogram of generated configurations: qhat_Ms.
    # hist = Dict{NTuple{n,Int8},Int}()
    # buf = Vector{Int8}(undef, n)

    # for k in 1:m
    #     @inbounds for j in 1:n
    #         buf[j] = x[k, j]
    #     end
    #     key = Tuple(buf)
    #     hist[key] = get(hist, key, 0) + 1
    # end

    return x
end


@inline function log2cosh(x::Real)
    a = abs(Float64(x))
    return a + log1p(exp(-2.0 * a))
end


function log_q_config(
    sigma::NTuple{N,Int8},
    seq::Vector{Int},
    param::Dict{Int,Vector{Vector{Int}}},
    sol_dict::Dict{Int,Vector{Float64}},
) where {N}
    logq = 0.0

    for node in seq
        interactions = param[node]
        theta = sol_dict[node]

        length(interactions) == length(theta) ||
            error("param[$node] and sol_dict[$node] have different lengths.")

        # Conditional field:
        # h_i(x_Par(i)) = sum_a theta_a prod_{j in interaction_a \ {i}} x_j.
        h = 0.0

        for (interaction, coeff) in zip(interactions, theta)
            monomial = 1.0

            for j in interaction
                j == node && continue
                monomial *= Float64(sigma[j])
            end

            h += coeff * monomial
        end

        s = Float64(sigma[node])

        # log q(x_i | parents) = x_i h_i - log(2 cosh(h_i)).
        logq += s * h - log2cosh(h)
    end

    return logq
end


function overlap_distribution(samples::AbstractMatrix)
    M, N = size(samples)
    npairs = div(M, 2)

    prob = zeros(Float64, N + 1)

    for k in 1:npairs
        sigma1 = @view samples[k, :]
        sigma2 = @view samples[npairs+k, :]

        nagree = 0
        @inbounds for i in 1:N
            nagree += sigma1[i] == sigma2[i]
        end

        prob[nagree+1] += 1.0 / npairs
    end

    return prob
end


function empirical_kl_errors(
    beta::Float64,
    edges::Vector{Tuple{Int,Int}},
    edge_weights::Vector{Float64},
    Z::Real,
    seq::Vector{Int},
    param::Dict{Int,Vector{Vector{Int}}},
    sol_dict::Dict{Int,Vector{Float64}},
    gen_samples::Dict{NTuple{N,Int8},Int},
    m::Int;
    compute_sampling_kl::Bool=false,
    compute_total_kl::Bool=false,
    per_spin::Bool=true,
) where {N}
    """
    Compute KL quantities using the generated histogram qhat_Ms:

    1. `sampling_kl_error = D_KL(qhat_Ms || q_theta)`
       Finite-sampling error relative to the learned model.

    2. `total_kl_error = D_KL(qhat_Ms || p)`
       Total learning-plus-finite-sampling error relative to the true model.
    """

    total = sum(values(gen_samples))
    total == m || error(
        "Inconsistent sample count: m = $m, but sum(counts) = $total"
    )
    m > 0 || error("m must be positive.")

    logZ = log(Float64(Z))

    sampling_kl = 0.0   # D_KL(qhat_Ms || q_theta)
    total_kl = 0.0      # D_KL(qhat_Ms || p)

    for (sigma, cnt) in gen_samples
        qhat = Float64(cnt) / Float64(m)
        logqhat = log(qhat)
        logp = log_prob_config(sigma, beta, edges, edge_weights) - logZ
        logq = log_q_config(sigma, seq, param, sol_dict)

        if compute_sampling_kl
            sampling_kl += qhat * (logqhat - logq)
        end
        if compute_total_kl
            total_kl += qhat * (logqhat - logp)
        end
    end

    scale = per_spin ? N : 1

    return (
        sampling_kl_error=compute_sampling_kl ? sampling_kl / scale : missing,
        total_kl_error=compute_total_kl ? total_kl / scale : missing,
    )
end


function error_computation(
    seq::Vector{Int},
    param::Dict{Int,Vector{Vector{Int}}},
    sol_dict::Dict{Int,Vector{Float64}},
    m::Int,
    mean_true::Vector{Float64},
    cov_true::Matrix{Float64},
    p_true::AbstractVector{<:Real},
    gen_samples::AbstractMatrix{<:Integer};#Dict{NTuple{N,Int8},Int};
    mag_prob_true::Union{Nothing,Vector{Float64}}=nothing,
    overlap_prob_true::Union{Nothing,Vector{Float64}}=nothing,
    compute_moment::Bool=false,
    compute_mean_rmse::Bool=false,
    compute_paircorr::Bool=false,
    compute_tv::Bool=false,
    compute_sampling_kl::Bool=false,
    compute_total_kl::Bool=false,
    compute_magnetization::Bool=false,
    compute_overlap::Bool=false,
    kl_per_spin::Bool=true,
)

    N = size(gen_samples, 2)
    size(gen_samples, 1) == m || error("gen_samples must contain m rows.")
    N == length(seq) || error("Number of columns must equal length(seq).")

    # Histogram of generated configurations: qhat_Ms.
    gen_hist = raw_samples_to_hist(gen_samples, Val(N))

    need_mean = compute_moment || compute_mean_rmse
    need_second = compute_moment || compute_paircorr

    mean_empirical = need_mean ? zeros(Float64, N) : nothing
    second_empirical = need_second ? zeros(Float64, N, N) : nothing

    sum_tv = 0.0
    p_seen = 0.0

    for (sigma, cnt) in gen_hist
        weight = Float64(cnt) / Float64(m)

        if need_mean
            @inbounds for i in 1:N
                mean_empirical[i] += Float64(sigma[i]) * weight
            end
        end

        if need_second
            @inbounds for i in 1:N
                si = Float64(sigma[i])

                for j in 1:N
                    second_empirical[i, j] +=
                        si * Float64(sigma[j]) * weight
                end
            end
        end

        if compute_tv
            idx = configuration_to_index(sigma)
            p_sigma = Float64(p_true[idx+1])

            p_seen += p_sigma
            sum_tv += abs(p_sigma - weight)
        end
    end

    # ---------- Original first-two-moment error ----------
    moment_err = missing

    if compute_moment
        cov_empirical =
            second_empirical - mean_empirical * mean_empirical'

        covariance_error = 0.0

        @inbounds for i in 1:N
            for j in 1:N
                if i != j
                    difference = cov_empirical[i, j] - cov_true[i, j]
                    covariance_error += difference^2
                end
            end
        end

        moment_err = sqrt(
            norm(mean_empirical - mean_true)^2 / N +
            covariance_error / (N * (N - 1))
        )
    end

    # ---------- Mean RMSE ----------

    mean_rmse = missing

    if compute_mean_rmse
        mean_rmse = norm(mean_empirical - mean_true) / sqrt(N)
    end

    # ---------- Pair-second-moment RMSE ----------

    paircorr_err = missing

    if compute_paircorr
        # E[xx'] = Cov(x) + E[x]E[x]'.
        second_true = cov_true + mean_true * mean_true'
        squared_error = 0.0

        @inbounds for i in 1:N
            for j in 1:N
                if i != j
                    difference = second_empirical[i, j] - second_true[i, j]
                    squared_error += difference^2
                end
            end
        end

        paircorr_err = sqrt(
            squared_error / (N * (N - 1))
        )
    end

    # ---------- Empirical TV: TV(p, qhat_Ms) ----------

    tv_err = missing

    if compute_tv
        unseen_probability = max(0.0, 1.0 - p_seen)
        tv_err = 0.5 * (sum_tv + unseen_probability)
    end

    # ---------- KL ----------
    # Compute the generated-histogram KL quantities together so log p and
    # log q_theta are evaluated only once per observed configuration.

    sampling_kl = 0.0   # D_KL(qhat_Ms || q_theta)
    total_kl = 0.0      # D_KL(qhat_Ms || p)

    for (sigma, cnt) in gen_hist
        qhat = Float64(cnt) / Float64(m)
        logqhat = log(qhat)

        if compute_sampling_kl
            logq = log_q_config(sigma, seq, param, sol_dict)
            sampling_kl += qhat * (logqhat - logq)
        end

        if compute_total_kl
            idx = configuration_to_index(sigma)
            logp = log(Float64(p_true[idx+1]))
            total_kl += qhat * (logqhat - logp)
        end
    end

    scale = kl_per_spin ? N : 1

    sampling_kl_err = compute_sampling_kl ? sampling_kl / scale : missing
    total_kl_err = compute_total_kl ? total_kl / scale : missing

    # ---------- Magnetization-distribution TV ----------

    magnetization_tv_err = missing

    if compute_magnetization
        mag_prob_true === nothing &&
            error("mag_prob_true is required when compute_magnetization=true.")

        mag_prob_empirical = zeros(Float64, N + 1)

        for (sigma, cnt) in gen_hist
            weight = Float64(cnt) / Float64(m)
            nplus = count(==(1), sigma)

            mag_prob_empirical[nplus+1] += weight
        end

        magnetization_tv_err =
            0.5 * sum(abs.(mag_prob_empirical .- mag_prob_true))
    end

    # ---------- Replica-overlap distribution TV ----------

    overlap_tv_err = missing

    if compute_overlap
        overlap_prob_true === nothing &&
            error("overlap_prob_true is required when compute_overlap=true.")

        npairs = div(m, 2)
        overlap_prob_empirical = zeros(Float64, N + 1)

        for k in 1:npairs
            sigma1 = @view gen_samples[k, :]
            sigma2 = @view gen_samples[npairs+k, :]

            nagree = 0
            @inbounds for i in 1:N
                nagree += sigma1[i] == sigma2[i]
            end

            overlap_prob_empirical[nagree+1] += 1.0 / npairs
        end

        overlap_tv_err =
            0.5 * sum(abs.(overlap_prob_empirical .- overlap_prob_true))

    end

    # ----------------------------------------------------------------
    return (
        moment_error=moment_err,
        mean_rmse=mean_rmse,
        paircorr_error=paircorr_err,
        tv_error=tv_err,
        sampling_kl_error=sampling_kl_err,
        total_kl_error=total_kl_err,
        magnetization_tv_error=magnetization_tv_err,
        overlap_tv_error=overlap_tv_err,
    )
end


function finite_sampling_errors(
    # ---------- Common: learned model and sampling ----------
    seq::Vector{Int},
    param::Dict{Int,Vector{Vector{Int}}},
    sol_dict::Dict{Int,Vector{Float64}},
    m::Int,

    # ---------- First moments / pair correlations ----------
    mean_true::Vector{Float64},
    cov_true::Matrix{Float64},

    # ---------- TV and KL: true Ising distribution ----------
    p_true::AbstractVector{<:Real};

    # ---------- Magnetization TV -----------
    mag_prob_true::Union{Nothing,Vector{Float64}}=nothing,
    overlap_prob_true::Union{Nothing,Vector{Float64}}=nothing,

    # ---------- Sampling controls ----------
    rng::AbstractRNG=Random.GLOBAL_RNG,
    ntry::Int=20,

    # ---------- Error flags ----------
    compute_moment::Bool=false,
    compute_mean_rmse::Bool=false,
    compute_paircorr::Bool=false,
    compute_tv::Bool=false,
    compute_sampling_kl::Bool=false,
    compute_total_kl::Bool=false,
    compute_magnetization::Bool=false,
    compute_overlap::Bool=false,
    kl_per_spin::Bool=true,
)
    n = length(seq)
    ntry > 0 || error("ntry must be positive.")

    moment_sum = 0.0
    tv_sum = 0.0
    mean_sum = 0.0
    paircorr_sum = 0.0
    sampling_kl_sum = 0.0
    total_kl_sum = 0.0
    magnetization_sum = 0.0
    overlap_sum = 0.0

    for _ in 1:ntry
        gen_samples = generate_test_samples(seq, param, sol_dict, m, n; rng=rng)

        errors = error_computation(
            seq, param, sol_dict, m,
            mean_true, cov_true, p_true,
            gen_samples;
            mag_prob_true=mag_prob_true,
            overlap_prob_true=overlap_prob_true,
            compute_moment=compute_moment,
            compute_tv=compute_tv,
            compute_mean_rmse=compute_mean_rmse,
            compute_paircorr=compute_paircorr,
            compute_sampling_kl=compute_sampling_kl,
            compute_total_kl=compute_total_kl,
            compute_magnetization=compute_magnetization,
            compute_overlap=compute_overlap,
            kl_per_spin=kl_per_spin,
        )

        compute_moment && (moment_sum += errors.moment_error)
        compute_tv && (tv_sum += errors.tv_error)
        compute_mean_rmse && (mean_sum += errors.mean_rmse)
        compute_paircorr && (paircorr_sum += errors.paircorr_error)
        compute_sampling_kl && (sampling_kl_sum += errors.sampling_kl_error)
        compute_total_kl && (total_kl_sum += errors.total_kl_error)
        compute_magnetization && (magnetization_sum += errors.magnetization_tv_error)
        compute_overlap && (overlap_sum += errors.overlap_tv_error)
    end

    return (
        moment_error=compute_moment ? moment_sum / ntry : missing,
        tv_error=compute_tv ? tv_sum / ntry : missing,
        mean_rmse=compute_mean_rmse ? mean_sum / ntry : missing,
        paircorr_error=compute_paircorr ? paircorr_sum / ntry : missing,
        sampling_kl_error=compute_sampling_kl ? sampling_kl_sum / ntry : missing,
        total_kl_error=compute_total_kl ? total_kl_sum / ntry : missing,
        magnetization_tv_error=compute_magnetization ? magnetization_sum / ntry : missing,
        overlap_tv_error=compute_overlap ? overlap_sum / ntry : missing,
    )
end


function raw_samples_to_hist(
    samples::AbstractMatrix{<:Integer},
    ::Val{N},
) where {N}
    hist = Dict{NTuple{N,Int8},Int}()

    for row in axes(samples, 1)
        config = ntuple(j -> Int8(samples[row, j]), N)
        hist[config] = get(hist, config, 0) + 1
    end

    return hist
end


function compress_samples(samples::AbstractMatrix)
    configs = Tuple.(eachrow(samples))
    counts_dict = countmap(configs)

    # Match NumPy's lexicographically sorted np.unique(..., axis=0).
    unique_configs = sort(collect(keys(counts_dict)))

    return reduce(vcat, [
        reshape([counts_dict[config], config...], 1, :)
        for config in unique_configs
    ])
end
