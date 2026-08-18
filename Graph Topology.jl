using Combinatorics

# ---------- Generate edges of a given graph ----------
function generate_lattice_graph(rows::Int, cols::Int)
    """
    Generates edges as tuples for a lattice graph
    """
    edges = Vector{Tuple{Int,Int}}()

    for r in 1:rows
        for c in 1:cols
            node = (r - 1) * cols + c

            # Connect to right neighbor
            if c < cols
                right = node + 1
                push!(edges, (node, right))
            end

            # Connect to bottom neighbor
            if r < rows
                below = node + cols
                push!(edges, (node, below))
            end
        end
    end
    return edges
end


# ---------- Generate parent set for each node ----------
function build_autoregressive_parents(edges::Vector{Tuple{Int,Int}}, sequence::Vector{Int})
    """
    Returns a dictionary of nodes and parent nodes, for an autoregressive model.

    This algorithm visits each node sequentially as per the input sequence, 
    node j is denoted as a parent node to i if there exists a path  
    (i,k1),(k1,k2),...,(km,j), with each kl=sequence[i-1], connecting i to j, and
    kl !∊ parent(i). 
    """
    # Build adjacency list
    adj = Dict{Int,Vector{Int}}()
    for (u, v) in edges
        push!(get!(adj, u, Int[]), v)
        push!(get!(adj, v, Int[]), u)
    end

    parent_dict = Dict{Int,Vector{Int}}()

    # Check if there is a path without passing through blocked nodes
    function has_valid_path(parent::Int, child::Int, blocked::Set{Int})
        visited = Set{Int}()
        stack = [parent]

        while !isempty(stack)
            node = pop!(stack)
            if node == child
                return true
            end
            push!(visited, node)
            for neighbor in adj[node]
                if !(neighbor in visited) && !(neighbor in blocked)
                    push!(stack, neighbor)
                end
            end
        end
        return false
    end

    for (i, node) in enumerate(sequence)
        previous_nodes = sequence[1:(i-1)]
        parents = Int[]

        for candidate in previous_nodes
            blocked = Set(previous_nodes)
            delete!(blocked, candidate)  # Only allow the candidate to walk through
            if has_valid_path(candidate, node, blocked)
                push!(parents, candidate)
            end
        end

        parent_dict[node] = parents
    end

    return parent_dict
end


# ---------- Generate edge parameters ----------
function build_k_body_interactions(
    n::Int,
    seq::Union{AbstractVector,Nothing}=nothing,
    mode::String="general",
    par::Union{Dict{Int,Vector{Int}},Nothing}=nothing,
    odr::Union{Int,Nothing}=nothing
)
    """
    For each i and parent nodes k1,...,kn taken one at a time, construct all combinations of 
    parameters [i,k1],[i,k1,k2]..
    """
    param = Dict{Int,Vector{Vector{Int}}}()

    if mode == "pairwise"
        for i in 1:n
            vals = Vector{Vector{Int}}()
            # singleton (i)
            push!(vals, [i])
            # all (j,i) with j ≠ i
            append!(vals, [[j, i] for j in 1:n if j != i])

            param[i] = vals
        end
    end

    # Case 2: General k-body mode
    if mode == "general"
        for i in 1:n
            if odr === nothing
                odr = i - 1
            end
            Comb = [collect(combinations(par[seq[i]], j)) for j in 0:odr]
            param[seq[i]] = [push!(x, seq[i]) for x in vcat(Comb...)]
        end
    end

    return param
end


# ---------- Generate sequences ----------

# Row-wise index for (r,c) on an n×n lattice (1-based)
idx(r, c, n) = (r - 1) * n + c

function lattice_diagonal_sequence(n::Int)
    @assert n ≥ 2 "Use n ≥ 2"

    # Build diagonals keyed by d = c - r (NE-SW). Main diagonal is d = 0.
    diags = Dict{Int,Vector{Int}}()
    for d in (-(n-1)):(n-1)
        v = Int[]
        rmin = max(1, 1 - d)
        rmax = min(n, n - d)
        for r in rmin:rmax
            c = r + d
            push!(v, idx(r, c, n))
        end
        diags[d] = v
    end

    # ---- Main diagonal: center, then skip-one outward (LEFT FIRST) ----
    main = diags[0]
    mid = (n % 2 == 0) ? (n ÷ 2) : ceil(Int, n / 2)  # left-of-center if even
    picked = Int[main[mid]]
    for step in 2:2:n
        l = mid - step
        r = mid + step
        if l >= 1
            push!(picked, main[l])
        end
        if r <= n
            push!(picked, main[r])
        end
    end
    # Append remaining main-diagonal nodes in natural order
    for x in main
        if x ∉ picked
            push!(picked, x)
        end
    end
    seq = copy(picked)

    # ---- Upper triangle (skip one diagonal, pick next) ----
    for d in 2:2:(n-1)
        append!(seq, diags[d])
    end
    for d in 1:2:(n-1)
        append!(seq, diags[d])
    end

    # ---- Lower triangle (skip one diagonal, pick next) ----
    for d in -2:-2:(-(n-1))
        append!(seq, diags[d])
    end
    for d in -1:-2:(-(n-1))
        append!(seq, diags[d])
    end

    return seq
end


function lattice_skip_sequence(n::Int)
    @assert n ≥ 2 "Use n ≥ 2"
    seq = Int[]
    # Pass 1: for each row, take every other column (start aligns with row parity)
    # Odd rows take c = 1,3,5,... ; even rows take c = 2,4,6,...
    for r in 1:n
        startc = isodd(r) ? 1 : 2
        for c in startc:2:n
            push!(seq, (r - 1) * n + c)
        end
    end
    # Pass 2: take the remaining columns in each row
    for r in 1:n
        startc = isodd(r) ? 2 : 1
        for c in startc:2:n
            push!(seq, (r - 1) * n + c)
        end
    end
    return seq
end


function _parent_count_if_next(
    adj::Vector{Vector{Int}},
    selected::BitVector,
    candidate::Int,
)
    n = length(adj)

    visited = falses(n)
    is_parent = falses(n)

    stack = [candidate]
    visited[candidate] = true

    # Explore the unselected connected component containing candidate.
    while !isempty(stack)
        node = pop!(stack)

        for neighbor in adj[node]
            if selected[neighbor]
                # Selected nodes touching this component are parents.
                is_parent[neighbor] = true
            elseif !visited[neighbor]
                visited[neighbor] = true
                push!(stack, neighbor)
            end
        end
    end

    return count(is_parent)
end


function greedy_frontier_sequence(
    edges::Vector{Tuple{Int,Int}},
    n::Int,
)
    # Assumes node labels are 1:n.
    adj = [Int[] for _ in 1:n]

    for (u, v) in edges
        push!(adj[u], v)
        push!(adj[v], u)
    end

    sequence = Int[]
    selected = falses(n)
    realized_sizes = Int[]

    while length(sequence) < n
        best_node = 0
        best_current_size = 0
        best_score = nothing

        for candidate in 1:n
            selected[candidate] && continue

            # Parent count if candidate is selected now.
            current_size = _parent_count_if_next(
                adj, selected, candidate
            )

            # Temporarily select candidate and examine the next step.
            selected[candidate] = true

            prospective_sizes = Int[]

            for other in 1:n
                if !selected[other]
                    push!(
                        prospective_sizes,
                        _parent_count_if_next(
                            adj, selected, other
                        ),
                    )
                end
            end

            selected[candidate] = false

            projected_max = maximum((
                isempty(realized_sizes) ? 0 : maximum(realized_sizes),
                current_size,
                isempty(prospective_sizes) ? 0 :
                maximum(prospective_sizes),
            ))

            projected_count =
                count(==(projected_max), realized_sizes) +
                (current_size == projected_max ? 1 : 0) +
                count(==(projected_max), prospective_sizes)

            future_max = isempty(prospective_sizes) ?
                         0 : maximum(prospective_sizes)

            future_count = count(
                ==(future_max),
                prospective_sizes,
            )

            # Lexicographic priorities:
            # 1. projected maximum parent size
            # 2. number attaining that maximum
            # 3. next-step maximum
            # 4. number attaining next-step maximum
            # 5. total prospective parent sizes
            # 6. current parent size
            # 7. node index as deterministic tie-break
            score = (
                projected_max,
                projected_count,
                future_max,
                future_count,
                sum(prospective_sizes),
                current_size,
                candidate,
            )

            if best_score === nothing || isless(score, best_score)
                best_score = score
                best_node = candidate
                best_current_size = current_size
            end
        end

        push!(sequence, best_node)
        push!(realized_sizes, best_current_size)
        selected[best_node] = true
    end

    return sequence
end