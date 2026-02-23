
@inline function prob_config(
    x::AbstractVector{<:Integer},                           # configuration σ ∈ {−1,+1}
    edges::Vector{Tuple{Int,Int}},                          # (i, j)
    edge_weights::Vector{Float64},                          # J_ij
    edge_self_weights::AbstractVector{<:Real}=Float64[]     # H_i (can be all zeros)
)::Float64

    @inbounds begin
        s = 0.0

        # Pairwise term: sum_k J_k * σ_i * σ_j
        @assert length(edges) == length(edge_weights)
        for k in eachindex(edge_weights)
            i, j = edges[k]
            # cast once to Float64 for stable arithmetic
            s += edge_weights[k] * (Float64(x[i]) * Float64(x[j]))
        end

        # Field/self term: only if non-empty
        if !isempty(edge_self_weights)
            @assert length(edge_self_weights) == length(x)
            for i in eachindex(x)
                Hi = edge_self_weights[i]
                if Hi != 0.0
                    s += Hi * Float64(x[i])
                end
            end
        end

        return exp(s)  # unnormalized probability
    end
end


function all_configurations(n::Int)::Matrix{Int8}
    """
    Return Σ ∈ Int8^{(2^n)×n}, with entries in {-1,+1}.
    Row l corresponds to configuration index l-1 (LSB at spin 1).
    """
    n_states = 1 << n
    Sigma = Array{Int8}(undef, n_states, n)
    @inbounds for idx in 0:n_states-1
        l = idx + 1
        for j in 1:n
            Sigma[l, j] = ((idx >> (j - 1)) & 1) == 1 ? Int8(1) : Int8(-1)
        end
    end
    return Sigma
end


function true_moments(
    Sigma::AbstractMatrix{Int8},
    edges::Vector{Tuple{Int,Int}},
    edge_weights::Vector{Float64},
    # edge_self_weights::Vector{Float64},
)
    """
    Sigma: Matrix of all configurations
    Returns Mean and covariance of an Ising model described by edges
    """
    n_states, n = size(Sigma)
    p = Vector{Float32}(undef, n_states)
    mean = zeros(Float64, n)
    M2u = zeros(Float64, n, n) # second moment: E[σ_i σ_j] fill upper triangle and mirror
    Z = 0.0

    @inbounds for l in 1:n_states
        config = @view Sigma[l, :]
        pl = prob_config(config, edges, edge_weights)
        p[l] = pl
        Z += pl

        # mean: mean += σ * pl
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

    # normalize to get expectations
    invZ = 1 / Z
    @. mean = mean * invZ
    @. M2u = M2u * invZ

    # Symmetrize to full M2
    M2 = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n
        @simd for j in 1:i-1
            M2[i, j] = M2u[j, i]
        end
        @simd for j in i:n
            M2[i, j] = M2u[i, j]
        end
    end

    # covariance = E[σ σᵀ] - mean*mean'
    cov = M2 .- (mean * mean')

    return p, Z, mean, cov
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

    # Empirical second moment: C_ij = ⟨s_i s_j⟩
    # C = Σ_k w_k * s_k s_kᵀ
    weighted_spins = spins .* w    # broadcast w over columns, N × d
    C = weighted_spins' * spins    # (d×N) * (N×d) = d×d

    # Covariance: Σ_ij = ⟨s_i s_j⟩ − ⟨s_i⟩ ⟨s_j⟩
    Σ = C .- m * m'                # outer product of m

    return m, Σ
end
