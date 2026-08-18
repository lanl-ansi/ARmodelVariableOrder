using JuMP, Ipopt
import MathOptInterface as MOI

function regularized_ISE_hist(
    i::Int,
    samples::AbstractMatrix{<:Integer},
    E::Dict{Int,Vector{Vector{Int}}},
    lambda::Real;
    silent::Bool=true
)
    """
      - i: focal node index in 1:n (refers to columns 2:(n+1) of `sample`)
      - sample: md × (n+1) Int/Int8 matrix; sample[:,1] = counts, sample[:,2:end] = configs ∈ {±1}
      - E: Dict{Int, Vector{Vector{Int}}}, where E[i] are the cliques (each a Vector{Int} of node indices)
      - lambda: L1 weight
    Returns: theta::Vector{Float64} (size = length(E[i]))
    """
    Ei = E[i]
    ni = length(Ei)
    md = size(samples, 1)              # n1 = n+1

    # Remove i from each clique once (safe even if i ∉ clique)
    all(i in clq for clq in Ei) || error("Every clique in E[$i] must contain focal node $i")
    Ei_woi = [[a for a in clq if a != i] for clq in Ei]

    # precompute the feature matrix
    Phi = Matrix{Int8}(undef, md, ni)
    @inbounds for l in 1:md, j in 1:ni
        prod = Int8(1)
        for a in Ei_woi[j]
            prod *= samples[l, 1+a] # σ_a(l); note +1 offset
        end
        Phi[l, j] = prod
    end

    counts = view(samples, :, 1)   # md-length vector
    m = sum(counts)                # total sample count

    # Objective: (1/m) * Σ_l counts[l] * exp(-σ_i(l) * Σ_j θ_j * ∏_{a∈Ei[j]\{i}} σ_a(l))
    lse = function (θ...)
        acc = zero(θ[1])
        @inbounds for l in 1:md
            si = samples[l, 1+i]         # σ_i(l); note +1 offset

            Tsum = zero(θ[1])
            @inbounds for j in 1:ni
                Tsum += θ[j] * Phi[l, j]
            end

            acc += counts[l] * exp(-si * Tsum)
        end
        acc / m
    end

    # ---- JuMP model ----
    model = Model(Ipopt.Optimizer)
    silent && set_silent(model)

    @variable(model, theta[1:ni])
    @variable(model, rho[1:ni] >= 0)
    @constraint(model, [j = 1:ni], rho[j] >= theta[j])
    @constraint(model, [j = 1:ni], rho[j] >= -theta[j])

    # Limited-memory Hessian keeps workspace small 
    # set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
    set_optimizer_attribute(model, "tol", 1e-8)
    set_optimizer_attribute(model, "acceptable_tol", 1e-6)

    # Register unique name (avoid collisions if you build many models)
    lname = Symbol("lse_hist_", i)
    JuMP.register(model, lname, ni, lse; autodiff=true)

    # Build lse(theta...) + λ * sum(rho) with interpolation of the actual JuMP vars
    args = [theta[j] for j in 1:ni]
    @eval @NLobjective($model, Min, $(Expr(:call, lname, args...)) + $lambda * $(sum(rho)))

    optimize!(model)

    status = termination_status(model)
    theta_vals = value.(theta)

    max_theta = maximum(abs.(theta_vals))
    max_Tsum = maximum(
        abs(sum(theta_vals[j] * Float64(Phi[l, j]) for j in 1:ni))
        for l in 1:md
    )

    diag = (
        node=i,
        status=string(status),
        objective=try
            objective_value(model)
        catch
            NaN
        end,
        max_theta=max_theta,
        max_Tsum=max_Tsum,
    )

    if !(status in (OPTIMAL, LOCALLY_SOLVED, ALMOST_LOCALLY_SOLVED))
        @warn "ISE solve not fully successful" diag
    end

    return theta_vals, diag
end

function ISE_hist(
    i::Int,
    samples::AbstractMatrix{<:Integer},
    E::Dict{Int,Vector{Vector{Int}}},
    silent::Bool=true
)
    """
      - i: focal node index in 1:n (refers to columns 2:(n+1) of `sample`)
      - sample: md × (n+1) Int/Int8 matrix; sample[:,1] = counts, sample[:,2:end] = configs ∈ {±1}
      - E: Dict{Int, Vector{Vector{Int}}}, where E[i] are the cliques (each a Vector{Int} of node indices)
    Returns: theta::Vector{Float64} (size = length(E[i]))
    """
    Ei = E[i]
    ni = length(Ei)
    md = size(samples, 1)              # n1 = n+1

    # Remove i from each clique once (safe even if i ∉ clique)
    all(i in clq for clq in Ei) || error("Every clique in E[$i] must contain focal node $i")
    Ei_woi = [[a for a in clq if a != i] for clq in Ei]

    # precompute the feature matrix
    Phi = Matrix{Int8}(undef, md, ni)
    @inbounds for l in 1:md, j in 1:ni
        prod = Int8(1)
        for a in Ei_woi[j]
            prod *= samples[l, 1+a] # σ_a(l); note +1 offset
        end
        Phi[l, j] = prod
    end

    counts = view(samples, :, 1)   # md-length vector
    m = sum(counts)                # total sample count

    # Objective: (1/m) * Σ_l counts[l] * exp(-σ_i(l) * Σ_j θ_j * ∏_{a∈Ei[j]\{i}} σ_a(l))
    lse = function (θ...)
        acc = zero(θ[1])
        @inbounds for l in 1:md
            si = samples[l, 1+i]         # σ_i(l); note +1 offset

            Tsum = zero(θ[1])
            @inbounds for j in 1:ni
                Tsum += θ[j] * Phi[l, j]
            end

            acc += counts[l] * exp(-si * Tsum)
        end
        acc / m
    end

    # ---- JuMP model ----
    model = Model(Ipopt.Optimizer)
    # if silent
    #     set_silent(model)
    # end

    @variable(model, theta[1:ni])
    # @variable(model, -10 <= theta[1:ni] <= 10)

    # Limited-memory Hessian keeps workspace small 
    # set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
    set_optimizer_attribute(model, "tol", 1e-8)

    # Register unique name (avoid collisions if you build many models)
    lname = Symbol("lse_hist_", i)
    JuMP.register(model, lname, ni, lse; autodiff=true)

    # Build lse(theta...) + λ * sum(rho) with interpolation of the actual JuMP vars
    args = [theta[j] for j in 1:ni]
    @eval @NLobjective($model, Min, $(Expr(:call, lname, args...)))

    optimize!(model)

    status = termination_status(model)
    theta_vals = value.(theta)

    max_theta = maximum(abs.(theta_vals))
    max_Tsum = maximum(
        abs(sum(theta_vals[j] * Float64(Phi[l, j]) for j in 1:ni))
        for l in 1:md
    )

    diag = (
        node=i,
        status=string(status),
        objective=try
            objective_value(model)
        catch
            NaN
        end,
        max_theta=max_theta,
        max_Tsum=max_Tsum,
    )

    if !(status in (OPTIMAL, LOCALLY_SOLVED, ALMOST_LOCALLY_SOLVED))
        @warn "ISE solve not fully successful" diag
    end

    return theta_vals, diag
end


function ISE_true(
    Sigma::Matrix{Int8},                    # (2^L) × L, entries ∈ {±1}
    i::Int,                                 # focal node
    E::Dict{Int,Vector{Vector{Int}}},       # E[i] = list of cliques (Vector{Int})
    p::AbstractVector{<:Real},              # unnormalized probabilities
    Z::Real,                                # normalizer
    lambda::Real
)
    Ei = E[i]
    ni = length(Ei)
    md = size(Sigma, 1) # md=2^N

    # Remove i from each clique (safe even if i ∉ clique)
    all(i in clq for clq in Ei) || error("Every clique in E[$i] must contain focal node $i")
    Ei_woi = [[a for a in clq if a != i] for clq in Ei]

    # precompute the feature matrix
    Phi = Matrix{Int8}(undef, md, ni)
    @inbounds for l in 1:md, j in 1:ni
        prod = Int8(1)
        for a in Ei_woi[j]
            prod *= Sigma[l, a]
        end
        Phi[l, j] = prod
    end

    # Computes: (1/Z) * Σ_l p[l] * exp( -σ_i(l) * Σ_j θ_j * Π_{a∈Ei[j]\{i}} σ_a(l) )
    lse = function (θ...)
        acc = zero(θ[1])
        @inbounds for l in 1:md
            si = Sigma[l, i] # σ_i(l)

            Tsum = zero(θ[1])
            @inbounds for j in 1:ni
                Tsum += θ[j] * Phi[l, j]
            end

            acc += p[l] * exp(-si * Tsum)
        end
        acc / Z
    end

    # ---- JuMP model ----
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    @variable(model, theta[1:ni])
    @variable(model, rho[1:ni] >= 0)
    @constraint(model, [j = 1:ni], rho[j] >= theta[j])
    @constraint(model, [j = 1:ni], rho[j] >= -theta[j])

    # Set tolerance
    set_optimizer_attribute(model, "tol", 1e-8)

    # Register unique name (avoid collisions if you build many models)
    lname = Symbol("lse_", i)
    JuMP.register(model, lname, ni, lse; autodiff=true)

    # Build lse(theta...) + λ * sum(rho) with interpolation of the actual JuMP vars
    args = [theta[j] for j in 1:ni]
    @eval @NLobjective($model, Min, $(Expr(:call, lname, args...)) + $lambda * $(sum(rho)))

    optimize!(model)
    return value.(theta)
end


function ISE_hist_saturated(
    i::Int,
    samples::AbstractMatrix{<:Integer},
    E::Dict{Int,Vector{Vector{Int}}};
    alpha::Float64=0.5,
)
    Ei = E[i]
    ni = length(Ei)

    # Parent set = all non-i variables appearing in E[i]
    parents = sort(unique(vcat([[a for a in clq if a != i] for clq in Ei]...)))
    d = length(parents)
    nctx = 1 << d

    ni == nctx || error("This function assumes a saturated basis: got ni=$ni, expected 2^$d=$nctx")

    parent_pos = Dict(parents[k] => k for k in 1:d)

    cminus = fill(alpha, nctx)
    cplus = fill(alpha, nctx)

    # Count labels for each parent pattern
    @inbounds for l in axes(samples, 1)
        cnt = Float64(samples[l, 1])
        y = samples[l, 1+i]

        idx = 1
        for k in 1:d
            a = parents[k]
            if samples[l, 1+a] == 1
                idx += 1 << (k - 1)
            end
        end

        if y == 1
            cplus[idx] += cnt
        elseif y == -1
            cminus[idx] += cnt
        else
            error("Spin value must be ±1")
        end
    end

    # Field value for each parent context
    h = Vector{Float64}(undef, nctx)
    @inbounds for idx in 1:nctx
        h[idx] = 0.5 * log(cplus[idx] / cminus[idx])
    end

    # Convert field table h(parent pattern) to monomial coefficients theta
    theta = zeros(Float64, ni)

    @inbounds for j in 1:ni
        clq = Ei[j]
        clq_woi = [a for a in clq if a != i]

        acc = 0.0

        for idx in 1:nctx
            state = idx - 1

            prod = 1.0
            for a in clq_woi
                k = parent_pos[a]
                spin = ((state >> (k - 1)) & 1) == 1 ? 1.0 : -1.0
                prod *= spin
            end

            acc += h[idx] * prod
        end

        theta[j] = acc / nctx
    end

    return theta
end


# Learning using ISE from GML package
# -------------------------------------
function learn_conditionals(
    seq::Vector{Int64},
    parent_set::Dict{Int64,Vector{Int64}},
    param::Dict{Int64,Vector{Vector{Int64}}},
    conditional_order::Integer,
    samples_hist::Matrix{Int64}, # GML sample format: column 1 = counts, variable v is column v+1
    regularizer::Real=0.0
)

    n = length(seq)

    size(samples_hist, 2) == n + 1 ||
        error(
            "Expected count + $n spin columns, " *
            "but received size $(size(samples_hist))."
        )

    conditionals = Dict{Int,Any}()

    for t in eachindex(seq)
        target = seq[t]

        parents = parent_set[target]
        local_vars = vcat(target, parents)

        samples_trunc = samples_hist[:, vcat(1, local_vars .+ 1)]

        learned_local = learn(samples_trunc, multiRISE(regularizer, false, conditional_order))

        conditionals[target] = Dict(
            conditional_key(target, [local_vars[j] for j in k]) => v
            for (k, v) in learned_local.terms
            if !isempty(k) && first(k) == 1
        )
    end

    sol_dict = Dict{Int64,Vector{Float64}}()

    for node in keys(conditionals)
        sol_dict[node] = [
            conditionals[node][conditional_key(node, edge)]
            for edge in param[node]
        ]
    end

    return sol_dict
end


function conditional_key(node::Int, edge)
    others = sort(filter(j -> j != node, collect(edge)))
    return Tuple([node; others])
end
