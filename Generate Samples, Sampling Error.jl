using StatsBase, Random
using LinearAlgebra

function generate_data_samples(
    p::AbstractVector{<:Real},   # unnormalized probabilities
    Z::Real,                     # normalizing constant (> 0)
    m::Integer,                  # number of samples to draw
    n::Integer;                  # number of spins
    rng::AbstractRNG=Random.GLOBAL_RNG
)
    """
    Generate data-samples using the generic function StatsBase.sample 
    """

    n_states = 1 << n
    @assert length(p) == n_states "length(p) must equal 2^n"

    # Normalize
    prob = Vector{Float64}(undef, n_states)
    @inbounds for i in 1:n_states
        prob[i] = Float64(p[i]) / Float64(Z)
    end

    # Guard against rounding drift: renormalize to sum == 1
    s = sum(prob)
    @inbounds for i in 1:n_states
        prob[i] /= s
    end

    # Multinomial sampling with replacement
    idxs = StatsBase.sample(rng, 1:n_states, Weights(prob), m; replace=true)

    # Tally counts per state
    counts = zeros(Int, n_states)
    @inbounds for idx in idxs
        counts[idx] += 1
    end

    # Build (n_obs)×(n+1) matrix: first col counts, rest are ±1 spins
    n_obs = count(!=(0), counts)
    samples = Matrix{Int}(undef, n_obs, n + 1)

    row = 1
    @inbounds for idx in 1:n_states
        c = counts[idx]
        if c == 0
            continue
        end
        samples[row, 1] = c
        state = idx - 1
        for j in 1:n
            samples[row, j+1] = ((state >> (j - 1)) & 1 == 1) ? 1 : -1
        end
        row += 1
    end

    return samples
end


function generate_samples(
    O::Vector{Int},                     # Sequence of sites
    E::Dict{Int,Vector{Vector{Int}}},   # Dictionary (nodes, edges)
    C::Dict{Int64,Vector{Float64}},     # Learnt edge weights
    m::Int,                             # Number of samples to be generated
    n::Int                              # Number of sites
)
    x = ones(Int8, m, n)
    p = zeros(Float64, m)
    sum_buf = zeros(Float64, m)
    temp = ones(Int8, m)

    for i in O
        Ei = E[i]
        fill!(sum_buf, 0.0)

        for (j, edge) in enumerate(Ei)
            copyto!(temp, view(x, :, i))  # temp = x[:, i]
            for k in 1:(length(edge)-1)
                a = edge[k]
                @inbounds @simd for idx in 1:m
                    temp[idx] *= x[idx, a]
                end
            end
            @inbounds @simd for idx in 1:m
                sum_buf[idx] += C[i][j] * temp[idx]
            end
        end

        @inbounds @simd for idx in 1:m
            p[idx] = 1.0 / (1.0 + exp(-2 * sum_buf[idx]))
        end

        rand_vals = rand(m)
        @inbounds @simd for k in 1:m
            x[k, i] = rand_vals[k] < (1 - p[k]) ? -1 : 1
        end
    end

    # Convert into a matrix
    hist = Dict{NTuple{n,Int8},Int}()
    buf = Vector{Int8}(undef, n)
    for k in 1:m
        @inbounds for j in 1:n
            buf[j] = x[k, j]
        end
        key = Tuple(buf)  # creates a single tuple per row
        hist[key] = get(hist, key, 0) + 1
    end

    return hist
end


function moments_sampling_error(
    mean_true::Vector{Float64},
    cov_true::Matrix{Float64},
    gen_samples::Dict{NTuple{N,Int8},Int},
    m::Int
) where {N}

    # Check total count consistency
    total = sum(values(gen_samples))
    if total != m
        error("Inconsistent sample count: m = $m, but sum(counts) = $total")
    end
    if m == 0
        error("m must be positive and nonzero.")
    end

    mean_empirical = zeros(Float64, N)
    second = zeros(Float64, N, N)

    for (sigma, cnt) in gen_samples
        w = cnt / Float64(m)           # Float64 weight

        @inbounds for i in 1:N
            si = Float64(sigma[i])
            mean_empirical[i] += si * w # Mean

            for j in 1:N
                second[i, j] += si * Float64(sigma[j]) * w # Covariance
            end
        end
    end

    cov_empirical = second
    @inbounds for i in 1:N, j in 1:N
        cov_empirical[i, j] -= mean_empirical[i] * mean_empirical[j]  # Cov = E[xxᵀ] - μμᵀ
    end

    err = sqrt(norm(mean_empirical .- mean_true) + norm(cov_empirical .- cov_true))
    return err
end


function error_sample_range(
    mean_true::Vector{Float64},
    cov_true::Matrix{Float64},
    O::Vector{Int},                         # Sequence of sites
    E::Dict{Int,Vector{Vector{Int}}},       # Dictionary (nodes, edges)
    C::Dict{Int64,Vector{Float64}},         # Learnt edge weights
    m::Int,                                 # Number of samples to be generated
    n::Int                                  # Number of sites
)
    """
    Returns a range of samples and compute the average sampling error over ntry trials
    """
    ntry = 50 # Average number of trials
    err = zeros(Float64, ntry)

    for t in 1:ntry
        gen_samples = generate_samples(O, E, C, m, n)
        err[t] = moments_sampling_error(mean_true, cov_true, gen_samples, m)
        # sum_err += TV_sampling_error(edges,edge_weights, edge_self_weights, gen_samples, Z, rng[k])
    end
    average = mean(err)
    variance = var(err, corrected=false)

    return average, variance
end


# ====================================================
#                   EXACT SAMPLING
# ====================================================

function sample_index_cdf(cumprobs::Vector{Float64})
    u = rand()
    lo = 1
    hi = length(cumprobs)
    while lo < hi
        mid = (lo + hi) >>> 1  # integer division by 2
        if u <= cumprobs[mid]
            hi = mid
        else
            lo = mid + 1
        end
    end
    return lo
end


function generate_analytical_samples_cdf(
    configs::Matrix{Int8},          # size(configs) == (Ncfg, N), rows are configs
    probs::Vector{Float32},         # length Ncfg, unnormalized probabilities
    Z::Float64,                     # partition function (or sum(probs))
    m::Int,                         # number of samples
    ::Val{N};                       # N = number of spins (columns of configs)
) where {N}

    Ncfg = size(configs, 1)         # number of configs
    @assert size(configs, 2) == N   # N spins per config
    @assert length(probs) == Ncfg

    # cumulative probabilities (normalized by Z)
    cumprobs = cumsum(probs ./ Z)

    counts = Dict{NTuple{N,Int8},Int}()

    @inbounds for s in 1:m
        idx = sample_index_cdf(cumprobs)   # ∈ 1:Ncfg
        cfg = @view configs[idx, :]        # row is the config, length N
        key = ntuple(i -> cfg[i], N)       # NTuple{N,Int8}
        counts[key] = get(counts, key, 0) + 1
    end

    return counts
end


function error_analytical_sample_range(
    configs::Matrix{Int8},
    probs::Vector{Float32},
    Z::Float64,
    mean_true::Vector{Float64},
    cov_true::Matrix{Float64},
    m::Int,
    L::Int
)
    """ Generate a range of samples from the analytical expression and compute the average"""

    ntry = 50 # Average number of trials
    sum_err = 0.0

    for _ in 1:ntry
        gen_samples = generate_analytical_samples_cdf(configs, probs, Z, m, Val(L))
        sum_err += moments_sampling_error(mean_true, cov_true, gen_samples, m)
    end

    return sum_err / ntry
end


# function moments_sampling_error_checked(
#     mean_true::Vector{Float64},
#     cov_true::Matrix{Float64},
#     gen_samples::Dict{NTuple{N,Int8},Int},
#     m::Int;
#     use_frobenius::Bool=true,
#     do_diagnostics::Bool=true,
# ) where {N}

#     # --- shape & finiteness checks on targets ---
#     length(mean_true) == N || error("mean_true must have length $N")
#     size(cov_true) == (N, N) || error("cov_true must be $N×$N")
#     all(isfinite, mean_true) || error("mean_true contains non-finite values")
#     all(isfinite, cov_true) || error("cov_true contains non-finite values")

#     # --- counts consistency ---
#     total = sum(values(gen_samples))
#     total == m || error("Inconsistent sample count: m = $m, but sum(counts) = $total")
#     m > 0 || error("m must be positive")

#     # --- validate dictionary contents ---
#     for (sigma, cnt) in gen_samples
#         cnt ≥ 0 || error("Negative count encountered")
#         @inbounds for i in 1:N
#             s = sigma[i]
#             (s == Int8(1) || s == Int8(-1)) || error("sigma[$i] = $s, expected ±1 (Int8)")
#         end
#     end

#     # --- empirical moments ---
#     mean_empirical = zeros(Float64, N)
#     second = zeros(Float64, N, N)

#     inv_m = 1.0 / m
#     for (sigma, cnt) in gen_samples
#         w = cnt * inv_m                    # ∈ [0,1]
#         @inbounds for i in 1:N
#             si = Float64(sigma[i])         # ±1
#             mean_empirical[i] += si * w
#             for j in 1:N
#                 second[i, j] += si * Float64(sigma[j]) * w
#             end
#         end
#     end

#     # E[xxᵀ] − μμᵀ
#     cov_empirical = copy(second)
#     @inbounds for i in 1:N, j in 1:N
#         cov_empirical[i, j] -= mean_empirical[i] * mean_empirical[j]
#     end

#     # --- quick sanity bounds for ±1 spins ---
#     if do_diagnostics
#         maxabs_mean = maximum(abs.(mean_empirical))
#         maxabs_sec = maximum(abs.(second))
#         maxabs_cov = maximum(abs.(cov_empirical))
#         if maxabs_mean > 1.01 || maxabs_sec > 1.01 || maxabs_cov > 1.01
#             @warn "Empirical moments exceeded expected bounds for ±1 spins" maxabs_mean maxabs_sec maxabs_cov
#         end
#         # also check targets are in a reasonable range
#         if maximum(abs.(cov_true)) > 2.0 || maximum(abs.(mean_true)) > 2.0
#             @warn "Target moments look out of range" maxabs_cov_true = maximum(abs.(cov_true)) maxabs_mean_true = maximum(abs.(mean_true))
#         end
#     end

#     # --- error ---
#     Δμ = mean_empirical .- mean_true
#     ΔΣ = cov_empirical .- cov_true

#     nμ = norm(Δμ)                                   # 2-norm
#     nΣ = use_frobenius ? norm(ΔΣ, fro) : norm(ΔΣ)   # Frobenius or operator 2-norm

#     (isfinite(nμ) && isfinite(nΣ)) || error("Non-finite norm(s): nμ=$nμ, nΣ=$nΣ")

#     return hypot(nμ, nΣ)  # numerically stable sqrt(nμ^2 + nΣ^2)
# end