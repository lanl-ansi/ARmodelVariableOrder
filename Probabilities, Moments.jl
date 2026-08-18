function all_configurations(n::Int)::Matrix{Int8}
    """
    Return Σ ∈ Int8^{(2^n)×n}, with entries in {-1,+1}.
    Row l corresponds to configuration index l-1 (LSB at spin 1).
    """
    n_states = 1 << n
    Sigma = Array{Int8}(undef, n_states, n)
    @inbounds for idx in 0:(n_states-1)
        l = idx + 1
        for j in 1:n
            Sigma[l, j] = ((idx >> (j - 1)) & 1) == 1 ? Int8(1) : Int8(-1)
        end
    end
    return Sigma
end


function index_to_configuration(idx::Integer, n::Int)::Vector{Int8}
    """
    Convert configuration index idx ∈ 0:(2^n - 1) to a spin vector in {-1,+1}^n.

    Uses the same convention as all_configurations:
    LSB corresponds to spin 1.
    """

    sigma = Vector{Int8}(undef, n)

    @inbounds for j in 1:n
        sigma[j] = ((idx >> (j - 1)) & 1) == 1 ? Int8(1) : Int8(-1)
    end

    return sigma
end


function configuration_to_index(sigma)
    idx = 0
    @inbounds for j in eachindex(sigma)
        sigma[j] == 1 && (idx |= 1 << (j - 1))
    end
    return idx
end


@inline function log_prob_config(
    x::Union{AbstractVector{<:Integer},NTuple{N,<:Integer}} where {N},
    beta::Float64,
    edges::Vector{Tuple{Int,Int}},
    edge_weights::Vector{Float64},
)::Float64
    @inbounds begin
        s = 0.0
        for k in eachindex(edge_weights)
            i, j = edges[k]
            s += edge_weights[k] * (Float64(x[i]) * Float64(x[j]))
        end
        return beta * s  # log of unnormalized probability
    end
end


function true_moments(
    Sigma::AbstractMatrix{Int8},
    beta::Float64,
    edges::Vector{Tuple{Int,Int}},
    edge_weights::Vector{Float64},
)
    n_configs, n = size(Sigma)
    log_p = Vector{Float64}(undef, n_configs)
    mean = zeros(Float64, n)
    M2u = zeros(Float64, n, n)

    # Compute log unnormalized probabilities
    @inbounds for l in 1:n_configs
        config = @view Sigma[l, :]
        log_p[l] = log_prob_config(config, beta, edges, edge_weights)
    end

    # Log-sum-exp trick
    log_p_max = maximum(log_p)
    log_Z = log_p_max + log(sum(exp.(log_p .- log_p_max)))
    Z = exp(log_Z)

    # Unnormalized probabilities (for compatibility)
    p_unnorm = exp.(log_p)

    # Normalized probabilities for moment computation
    p_norm = exp.(log_p .- log_Z)

    # Verify normalization (should be ≈ 1.0)
    # println("Sum of p: ", sum(p))

    # Compute moments
    @inbounds for l in 1:n_configs
        config = @view Sigma[l, :]
        pl = p_norm[l]

        @simd for j in 1:n
            mean[j] += Float64(config[j]) * pl
        end

        for i in 1:n
            xi = Float64(config[i])
            @simd for j in i:n
                M2u[i, j] += xi * Float64(config[j]) * pl
            end
        end
    end

    # Symmetrize M2
    M2 = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n
        @simd for j in 1:(i-1)
            M2[i, j] = M2u[j, i]
        end
        @simd for j in i:n
            M2[i, j] = M2u[i, j]
        end
    end

    cov = M2 .- (mean * mean')

    return p_unnorm, Z, mean, cov
end


function empirical_moments(Sigma::AbstractMatrix{Int64})
    # Extract frequencies and spin configurations
    freqs = Sigma[:, 1]
    spins = Sigma[:, 2:end]         # size N × d

    total = sum(freqs)

    # Normalized weights (empirical probability of each configuration)
    w = freqs / total              # size N

    # Empirical mean (magnetization): m_i = ⟨s_i⟩
    m = spins' * w                 # (d×N) * (N) = d
    m = vec(m)                     # ensure it's a Vector

    # Empirical second moment: C = Σ_k w_k * s_k s_kᵀ
    weighted_spins = spins .* w    # broadcast w over columns, N × d
    C = weighted_spins' * spins    # (d×N) * (N×d) = d×d

    # Covariance: Σ_ij = ⟨s_i s_j⟩ − ⟨s_i⟩ ⟨s_j⟩
    Σ = C .- m * m'                # outer product of m

    return m, Σ
end
