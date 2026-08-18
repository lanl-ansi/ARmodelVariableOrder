using StatsBase
using Random

function generate_training_samples(
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


@inline function log_sigmoid(z::Float64)
    if z >= 0.0
        return -log1p(exp(-z))
    else
        return z - log1p(exp(z))
    end
end


function build_conditional_tables(
    seq::Vector{Int},
    param::Dict{Int,Vector{Vector{Int}}},
    sol_dict::Dict{Int,Vector{Float64}},
)
    tables = []

    for node in seq
        interactions = param[node]
        coefficients = sol_dict[node]

        length(interactions) == length(coefficients) ||
            error("Parameter and coefficient lengths differ for node $node.")

        parents = Int[]

        for clique in interactions
            for variable in clique
                variable != node && push!(parents, variable)
            end
        end

        parents = sort(unique(parents))
        parent_position = Dict(v => k for (k, v) in enumerate(parents))

        number_parent_states = 1 << length(parents)

        logp_minus = zeros(Float64, number_parent_states)
        logp_plus = zeros(Float64, number_parent_states)

        for local_state in 0:(number_parent_states-1)
            field = 0.0

            for (clique, coefficient) in zip(
                interactions,
                coefficients,
            )
                monomial = 1.0

                for variable in clique
                    variable == node && continue

                    position = parent_position[variable]

                    spin = (
                        (local_state >> (position - 1)) & 1
                    ) == 1 ? 1.0 : -1.0

                    monomial *= spin
                end

                field += coefficient * monomial
            end

            logp_plus[local_state+1] =
                log_sigmoid(2.0 * field)

            logp_minus[local_state+1] =
                log_sigmoid(-2.0 * field)
        end

        push!(
            tables,
            (
                node=node,
                parents=parents,
                logp_minus=logp_minus,
                logp_plus=logp_plus,
            ),
        )
    end

    return tables
end


@inline function local_parent_index(
    state::UInt64,
    parents::Vector{Int},
)
    local_state = 0

    @inbounds for (k, parent) in enumerate(parents)
        bit = Int(
            (state >> (parent - 1)) & UInt64(1)
        )

        local_state |= bit << (k - 1)
    end

    return local_state + 1
end


@inline function log_model_probability(
    state::UInt64,
    tables,
)
    logq = 0.0

    @inbounds for table in tables
        local_index = local_parent_index(
            state,
            table.parents,
        )

        node_is_positive =
            ((state >> (table.node - 1)) & UInt64(1)) == 1

        if node_is_positive
            logq += table.logp_plus[local_index]
        else
            logq += table.logp_minus[local_index]
        end
    end

    return logq
end


function exact_learning_errors(
    seq::Vector{Int},
    param::Dict{Int,Vector{Vector{Int}}},
    sol_dict::Dict{Int,Vector{Float64}},
    p_true::AbstractVector{<:Real},
    logp_true::AbstractVector{<:Real},
    per_spin::Bool=true,
)

    length(logp_true) == length(p_true) ||
        error("p_true and logp_true must have the same length.")

    tables = build_conditional_tables(seq, param, sol_dict)

    tv_sum = 0.0
    forward_kl = 0.0
    reverse_kl = 0.0

    for state_idx in eachindex(p_true)
        state = UInt64(state_idx - 1)

        p = Float64(p_true[state_idx])
        logp = Float64(logp_true[state_idx])

        logq = log_model_probability(state, tables)
        q = exp(logq)

        tv_sum += abs(p - q)
        forward_kl += p * (logp - logq)
        reverse_kl += q * (logq - logp)
    end

    if per_spin
        forward_kl /= N
        reverse_kl /= N
    end

    return (
        tv_error=0.5 * tv_sum,
        forward_kl_error=forward_kl,
        reverse_kl_error=reverse_kl,
    )
end
