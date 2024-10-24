########################################
# Abstraction
########################################

abstract type AbstractSubdomain{D,F,T} end

dimension(::AbstractSubdomain{D}) where {D} = D
dimension(::Type{<:AbstractSubdomain{D}}) where {D} = D

# Interface contract for AbstractSubdomain:
#   Base.size
#   Base.length
#   Base.eltype
#   Base.getindex
#   Base.eachindex
#   parametrization
#   faces(iterator)
#   inactive_edges(iterator)

Base.size(::AbstractSubdomain{D}) where {D} = (D, 2)
Base.length(::AbstractSubdomain{D}) where {D} = D*2
Base.eltype(::AbstractSubdomain{D,F}) where {D,F} = F
function Base.getindex(s::AbstractSubdomain{D}, f::AbstractFace{D}) where {D}
    ax, lr = coordinates(f)
    return s[ax,lr]
end

function parametrization end
function faces end
function inactive_edges end

function any_adjacent_dirichlet_faces(faces, edges)
    for e in edges
        for f in adjacent_edges(e, 1)
            ax, lr = coordinates(f)
            is_dirichlet(faces[ax,lr]) && return true
        end
    end
    return false
end

########################################
# Concrete implementation
########################################

struct Subdomain{D,F,T} <: AbstractSubdomain{D,F,T}
    faces::Matrix{F}
    φ::T
    inactive_edges::Set{Edge{D}}

    function Subdomain{D,F,T}(faces,φ,inactive_edges) where {D,F<:AbstractFace{D},T<:Transform{D}}
        for i in 1:D
            for j in 1:2
                ax, lr = coordinates(faces[i,j])
                if (ax != i) || (lr != j)
                    throw(ArgumentError("Face $(faces[i,j]) and its indices [$(i),$(j)] do not match."))
                end
            end
        end

        if any(e -> codimension(e)==1, inactive_edges)
            throw(ArgumentError("Can only specify inactive edges of codimension greater than 1."))
        end

        if any_adjacent_dirichlet_faces(faces, inactive_edges)
            throw(ArgumentError("Dirichlet faces cannot contain inactive edges."))
        end

        return new(faces, φ, inactive_edges)
    end
end

function Subdomain(faces::Vector{F}, φ::T, inactive_edges...) where {F,T}
    D = dimension(F)

    if length(faces) != D*2
        throw(ArgumentError("Wrong number of faces, $(length(faces)), provided, there must be $(D*2)."))
    end

    faces_arranged = Matrix{F}(undef, D, 2)
    for f in faces
        ax, lr = coordinates(f)
        faces_arranged[ax, lr] = f
    end

    if any(i -> !isassigned(faces_arranged, i), eachindex(faces_arranged))
        throw(ArgumentError("Some of the faces have not been specified, check for duplicates."))
    end

    inactive_edges_set = Set{Edge{D}}()
    for e in inactive_edges
        push!(inactive_edges_set, e)
    end

    return Subdomain{D,F,T}(faces_arranged, φ, inactive_edges_set)
end

Base.getindex(s::Subdomain, I::Int...) = s.faces[I...]
Base.eachindex(s::Subdomain) = eachindex(s.faces)
parametrization(s::Subdomain) = s.φ

function faces(s::Subdomain)
    return (f for f in s.faces)
end
function inactive_edges(s::Subdomain)
    return (e for e in s.inactive_edges)
end
