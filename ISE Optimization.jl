using JuMP, Ipopt

function ISE_hist(
    i::Int,
    samples::AbstractMatrix{<:Integer},
    E::Dict{Int,Vector{Vector{Int}}},
    lambda::Real;
    silent::Bool=true,
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
    md, _ = size(samples)              # n1 = n+1

    # Remove i from each clique once (safe even if i ∉ clique)
    Ei_woi = [[a for a in clq if a != i] for clq in Ei]

    counts = view(samples, :, 1)   # md-length vector
    m = sum(counts)                # total sample count

    # Objective: (1/m) * Σ_l counts[l] * exp(-σ_i(l) * Σ_j θ_j * ∏_{a∈Ei[j]\{i}} σ_a(l))
    lse = function (θ...)
        acc = zero(θ[1])
        @inbounds for l in 1:md
            si = samples[l, 1+i]         # σ_i(l); note +1 offset

            Tsum = zero(θ[1])
            @inbounds for j in 1:ni
                prod = one(θ[1])
                for a in Ei_woi[j]      # empty for the self clique
                    prod *= samples[l, 1+a] # σ_a(l); again +1 offset
                end
                Tsum += θ[j] * prod
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
    set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
    set_optimizer_attribute(model, "tol", 1e-12)

    # Register unique name (avoid collisions if you build many models)
    lname = Symbol("lse_hist_", i)
    JuMP.register(model, lname, ni, lse; autodiff=true)

    # Build lse(theta...) + λ * sum(rho) with interpolation of the actual JuMP vars
    args = [theta[j] for j in 1:ni]
    @eval @NLobjective($model, Min, $(Expr(:call, lname, args...)) + $lambda * $(sum(rho)))

    optimize!(model)
    return value.(theta)
end


function ISE_true(
    Sigma::Matrix{Int8},                    # (2^L) × L, entries ∈ {±1}
    i::Int,                                 # focal node
    E::Dict{Int,Vector{Vector{Int}}},       # E[i] = list of cliques (Vector{Int})
    p::AbstractVector{<:Real},              # weights (Float32/64 or Mmap)
    Z::Real,                                # normalizer
    lambda::Real
)
    Ei = E[i]
    ni = length(Ei)
    md = size(Sigma, 1)

    # Remove i from each clique once (safe even if i ∉ clique)
    Ei_woi = [[a for a in clq if a != i] for clq in Ei]

    # Computes: (1/Z) * Σ_l p[l] * exp( -σ_i(l) * Σ_j θ_j * Π_{a∈Ei[j]\{i}} σ_a(l) )
    lse = function (θ...)
        acc = zero(θ[1])
        @inbounds for l in 1:md
            si = Sigma[l, i] # σ_i(l)

            Tsum = zero(θ[1])
            for j in 1:ni
                prod = one(θ[1])
                @inbounds for a in Ei_woi[j] # empty for the self clique
                    prod *= Sigma[l, a] # σ_a(l)
                end
                Tsum += θ[j] * prod
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
    set_optimizer_attribute(model, "tol", 1e-12)

    # Register unique name (avoid collisions if you build many models)
    lname = Symbol("lse_", i)
    JuMP.register(model, lname, ni, lse; autodiff=true)

    # Build lse(theta...) + λ * sum(rho) with interpolation of the actual JuMP vars
    args = [theta[j] for j in 1:ni]
    @eval @NLobjective($model, Min, $(Expr(:call, lname, args...)) + $lambda * $(sum(rho)))

    optimize!(model)
    return value.(theta)
end


# --------------------------- OPTIMIZATION (Gradient Descent) ------------------------------------------------   

# @inline soft(a::Real, τ::Real) = sign(a) * max(abs(a) - τ, 0.0)

# # S[i,j] = ∏_{a ∈ E[u][j]\{u}} σ_a^(i) * σ_u^(i)
# function build_S(Sigma::AbstractMatrix{<:Integer}, u::Integer, Eu::Vector{Vector{Int}})

#     m, _ = size(Sigma) # number of configurations
#     ni = length(Eu) # number of edges including self edge

#     S = Matrix{Float64}(undef, m, ni)

#     xu = Float64.(Sigma[:, u])                # σ_u across all configurations

#     @inbounds for j in 1:ni                   # for each edge (e.g., [4], [2,4], [3,4], [2,3,4])
#         prod = copy(xu)                       # start at σ_u
#         ej = Eu[j]                            # the j-th edge
#         for k in eachindex(ej)
#             a = ej[k]                         # a is an Int (e.g., 2, 3, or 4)
#             if a != u
#                 prod .*= Float64.(Sigma[:, a])
#             end
#         end
#         S[:, j] = prod
#     end
#     return S
# end


# function iso_obj_grad(
#     theta::AbstractVector{<:Real},
#     S::AbstractMatrix{<:Real},
#     prob::AbstractVector{<:Real},
#     Z::Real
# )
#     """
#     Smooth part: S(θ) = (1/Z) * sum_i prob[i] * exp(-S[i,:]'θ), 
#     gradient = -(1/Z)*S'*(prob .* exp(-Sθ))
#     """
#     y = S * theta
#     w = prob .* exp.(-y)
#     Sm = sum(w) / Z
#     g = -(S' * w) / Z
#     return Sm, g
# end

# function ISE_GradientDescent(
#     Sigma::AbstractMatrix{<:Integer},
#     u::Integer,
#     E::Dict{Int,Vector{Vector{Int}}},
#     prob::AbstractVector{<:Real},
#     Z::Real,
#     λ::Real, η::Real, eps::Real, T::Integer
# )
#     S = build_S(Sigma, u, E[u])                  # m × ni
#     ni = size(S, 2)
#     theta = zeros(Float64, ni)                  # neutral init

#     Sm, g = iso_obj_grad(theta, S, prob, Z)
#     Fprev = Sm + λ * sum(abs, theta) # the previous value of the objective function

#     @inbounds for _ in 1:T
#         # Prox-gradient step
#         theta .= soft.(theta .- η .* g, η * λ)

#         # Refresh objective + gradient
#         Sm, g = iso_obj_grad(theta, S, prob, Z)
#         F = Sm + λ * sum(abs, theta)

#         # Stop on small grad (∞-norm) OR tiny relative objective change
#         if norm(g, Inf) <= eps || abs(F - Fprev) <= eps * max(1.0, abs(Fprev))
#             break
#         end
#         Fprev = F
#     end
#     return theta
# end