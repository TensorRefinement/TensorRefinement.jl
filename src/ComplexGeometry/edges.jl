using Combinatorics: combinations

########################################
# Abstraction
########################################

abstract type AbstractEdge{D,K} end
const AbstractFace{D} = AbstractEdge{D,1}

dimension(::AbstractEdge{D}) where {D} = D
dimension(::Type{<:AbstractEdge{D}}) where {D} = D
codimension(::AbstractEdge{D,K}) where {D,K} = K
codimension(::Type{<:AbstractEdge{D,K}}) where {D,K} = K

# Interface contract for AbstractEdge:
#   fixed_axes(indexed collection of length K)
#   free_axes(indexed collection of length D-K)
#   Base.getindex
function fixed_axes end
function free_axes end

function coordinates(f::AbstractFace)
    ax = fixed_axes(f)[1]
    lr = f[ax]
    return ax, lr
end

function adjacent_edges(e::AbstractEdge{D,K}, K̂::Int) where {D,K}
    if (K̂ < 1) || (K̂ > D)
        throw(ArgumentError("Codimension of an edge, $(K̂), must lie in [1,$(D)]."))
    end

    if K̂ == K
        return [Edge(e)]
    end

    if K̂ < K
        subaxes_iterator = combinations(fixed_axes(e), K̂)
        axlr_iterator = ([(ax, e[ax]) for ax in subaxes] for subaxes in subaxes_iterator)
        return (Edge{D}(axlr_pairs...) for axlr_pairs in axlr_iterator)
    end

    if K̂ > K
        old_axlr = [(ax, e[ax]) for ax in fixed_axes(e)]
        subaxes_iterator = combinations(free_axes(e), K̂-K)
        bits_iterator = [digits(i, base=2, pad=K̂-K) for i in 0:2^(K̂-K)-1]
        axlr_iterator = ([(subaxes[j], bits[j]+1) for j in 1:K̂-K] for bits in bits_iterator, subaxes in subaxes_iterator)
        return (Edge{D}(old_axlr..., axlr_pairs...) for axlr_pairs in axlr_iterator)
    end
end

function map_and_permute(e::AbstractEdge{D,K}, f₀::AbstractEdge{D,1}, Π::AbstractSignedPermutation, f₁::AbstractEdge{D,1}) where {D,K}
    ax₀ = fixed_axes(f₀)[1]
    ax₁ = fixed_axes(f₁)[1]
    lr₁ = f₁[ax₁]
    return map_and_permute(e, ax₀, Π, ax₁, lr₁)
end

function map_and_permute(e::AbstractEdge{D,K}, ax₀::Int, Π::AbstractSignedPermutation, ax₁::Int, lr₁::Int) where {D,K}
    if (dimension(Π) != D-1)
        throw(ArgumentError("Dimension of the permutation, $(dimension(Π)), must be $(D-1)."))
    end

    if all(ax -> ax!=ax₀, fixed_axes(e))
        throw(ArgumentError("The face $ax₀ must be a fixed axis of the edge $e."))
    end

    new_coords = Dict(ax₁ => lr₁)
    for ax in fixed_axes(e)
        ax == ax₀ && continue
        i = (ax < ax₀) ? ax : ax-1
        j, σ = Π(i)
        new_ax = (j < ax₁) ? j : j+1
        new_coords[new_ax] = σ ? e[ax] : 3-e[ax]
    end
    new_e = Edge{D,K}(new_coords)

    order = Vector{Int}(undef, D-K)
    signs = Vector{Bool}(undef, D-K)
    new_free_axes = free_axes(new_e)
    for (l,ax) in enumerate(free_axes(e))
        i = (ax < ax₀) ? ax : ax-1
        j, σ = Π(i)
        new_ax = (j < ax₁) ? j : j+1
        order[l] = findfirst(isequal(new_ax), new_free_axes)
        signs[l] = σ
    end
    if Base.issorted(order)
        new_Π = IdentityPermutation(signs)
    elseif Base.issorted(order, rev=true)
        new_Π = FlipPermutation(signs)
    else
        new_Π = GeneralPermutation(order, signs)
    end

    return new_e, new_Π
end

########################################
# Concrete implementation
########################################

struct Edge{D,K} <: AbstractEdge{D,K}
    coordinates::Dict{Int, Int}
    sorted_fixed_axes::Vector{Int}
    sorted_free_axes::Vector{Int}

    function Edge{D,K}(coordinates) where {D,K}
        if D < 1
            throw(ArgumentError("Ambient dimension of an edge, $(D), must be at least 1."))
        end

        if (K < 1) || (K > D)
            throw(ArgumentError("Codimension of an edge, $(K), must lie in [1,$(D)]."))
        end

        if length(coordinates) != K
            throw(ArgumentError("Number of coordinates, $(length(coordinates)), of an edge must be $(K)."))
        end
        
        keys_c = keys(coordinates)
        vals_c = values(coordinates)
        
        if any(x -> (x < 1) || (x > D), keys_c)
            throw(ArgumentError("Axes of an edge must lie in [1,$(D)]."))
        end

        if any(x -> (x < 1) || (x > 2), vals_c)
            throw(ArgumentError("Coordinates of an edge must lie in {1,2}."))
        end

        sorted_fixed_axes = Vector{Int}(undef, K)
        sorted_free_axes = Vector{Int}(undef, D-K)

        i_fixed = 1
        i_free = 1
        for i in 1:D
            if i in keys_c
                sorted_fixed_axes[i_fixed] = i
                i_fixed = i_fixed + 1
            else
                sorted_free_axes[i_free] = i
                i_free = i_free + 1
            end
        end

        return new(coordinates, sorted_fixed_axes, sorted_free_axes)       
    end
end

function Edge{D}(coordinates::Dict{Int,Int}) where {D}
    K = length(coordinates)
    return Edge{D,K}(coordinates)
end

function Edge{D}(axlr_pairs...) where {D}
    coordinates = Dict{Int,Int}()
    for axlr in axlr_pairs
        ax, lr = axlr
        if haskey(coordinates, ax)
            throw(ArgumentError("Duplicate axes, $(ax)."))
        end
        coordinates[ax] = lr
    end
    return Edge{D}(coordinates)
end

Edge(e::Edge) = e
function Edge(e::AbstractEdge{D}) where {D}
    coordinates = Dict{Int,Int}()
    for ax in fixed_axes(e)
        coordinates[ax] = e[ax]
    end
    return Edge{D}(coordinates)
end

fixed_axes(e::Edge) = copy(e.sorted_fixed_axes)
free_axes(e::Edge) = copy(e.sorted_free_axes)
Base.getindex(e::Edge, ax) = e.coordinates[ax]

function Base.:(==)(e1::Edge, e2::Edge)
    dimension(e1) != dimension(e2) && return false
    codimension(e1) != codimension(e2) && return false
    e1.sorted_fixed_axes != e2.sorted_fixed_axes && return false
    return all(ax -> e1[ax] == e2[ax], e1.sorted_fixed_axes)
end

function Base.show(io::IO, e::Edge{D,K}) where {D,K}
    print(io, "[ ")
    for i in 1:K-1
        ax = e.sorted_fixed_axes[i]
        print(io, "($ax, $(e[ax])), ")
    end
    ax = e.sorted_fixed_axes[K]
    print(io, "($ax, $(e[ax])) ]")
end

function Base.show(io::IO, ::MIME"text/plain", e::Edge{D,K}) where {D,K}
    print(io, "Edge{$D, $K}:\n ", e)
end
