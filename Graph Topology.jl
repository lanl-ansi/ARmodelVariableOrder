# Graph topology related functions
# Table of Contents:
# 1. Generate edges of a given graph
#    function generate_lattice_graph
#    function generate_binary_graph
#
# 2. Generate parent set for each node
#    function build_autoregressive_parents
#
# 3. Generate edge parameters
#    function build_k_body_interactions
# 
# 4. Generate sequences 
#    function lattice_diagonal_sequence

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

function generate_binary_graph(levels::Int)
    """
    Argument: 
    -levels of the binary tree graph
    Returns tuple of edges corresponding to a binary tree
    """
    ed = Vector{Tuple{Int,Int}}()
    current_node = 1  # Root node index now 1

    # Node counter for the next available child
    next_node = 2

    for l in 1:levels
        num_parents = 2^(l - 1)  # Number of parents at this level

        for _ in 1:num_parents
            push!(ed, (current_node, next_node))
            push!(ed, (current_node, next_node + 1))
            current_node += 1
            next_node += 2
        end
    end

    return ed
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
        previous_nodes = sequence[1:i-1]
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


function build_dag_parents(edges::Vector{Tuple{Int,Int}}, root::Int)
    """
    Arguments: 
    - edges: list of edges as (i, j)
    - root: starting node

    Returns DAG from an undirected tree with a user-defined root
    """
    # Build adjacency list for undirected tree
    adj = Dict{Int,Vector{Int}}()
    for (u, v) in edges
        push!(get!(adj, u, Int[]), v)
        push!(get!(adj, v, Int[]), u)
    end

    # DAG will store child => parents (as vectors)
    parent_map = Dict{Int,Vector{Int}}()
    visited = Set{Int}()

    # DFS to build DAG
    function dfs(node::Int, parent::Union{Int,Nothing})
        push!(visited, node)

        # Add parent as a vector (empty if root)
        parent_map[node] = parent === nothing ? Int[] : [parent]

        for neighbor in adj[node]
            if !(neighbor in visited)
                dfs(neighbor, node)
            end
        end
    end

    dfs(root, nothing)
    return parent_map
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
    for d in -(n - 1):(n-1)
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
    for d in -2:-2:-(n - 1)
        append!(seq, diags[d])
    end
    for d in -1:-2:-(n - 1)
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


function dag_sequence(dag::Dict{Int,Vector{Int}}, root::Int)
    """
    Returns a path connecting all nodes of the DAG starting from the root
    """
    # Invert the DAG: child → parents → parent → children
    parent_to_children = Dict{Int,Vector{Int}}()
    for (child, parents) in dag
        for parent in parents
            push!(get!(parent_to_children, parent, Int[]), child)
        end
        # Ensure all nodes are in the map
        if !haskey(parent_to_children, child)
            parent_to_children[child] = Int[]
        end
    end

    visited = Set{Int}()
    order = Vector{Int}()

    function dfs(node)
        push!(visited, node)
        push!(order, node)
        for child in parent_to_children[node]
            if !(child in visited)
                dfs(child)
            end
        end
    end

    dfs(root)
    return order
end