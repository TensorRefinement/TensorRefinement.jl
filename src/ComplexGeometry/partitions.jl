using StaticArrays: MVector

########################################
# Abstraction
########################################

abstract type AbstractPartition{D,S} end

dimension(::AbstractPartition{D}) where {D} = D

# Interface contract for AbstractPartition
#   Base.length
#   Base.eltype
#   Base.IndexStyle
#   Base.getindex
#   Base.eachindex
#   Base.iterate
#   Base.IteratorSize

Base.eltype(::AbstractPartition{D,S}) where {D,S} = S
Base.IndexStyle(::AbstractPartition) = IndexLinear()
Base.IteratorSize(::AbstractPartition) = Base.HasLength()

########################################
# Concrete implementations
########################################

struct DomainPartition{D,S} <: AbstractPartition{D,S}
    subdomains::Vector{S}
    inactive_coupling_edges::Vector{Vector{CouplingEdge{D}}}
    dirichlet_faces::Vector{Vector{Tuple{Int,Int}}}

    function DomainPartition{D,S}(subdomains) where {D,S<:AbstractSubdomain{D}}
        if !correct_coupling_faces(subdomains)
            throw(ArgumentError("Coupling faces are arranged incorrectly."))
        end

        n = length(subdomains)
        inactive_coupling_edges = Vector{Vector{CouplingEdge{D}}}(undef, n)
        dirichlet_faces = Vector{Vector{Tuple{Int,Int}}}(undef, n)
        
        # Add all inactive coupling and Dirichlet faces
        for (ids,s) in enumerate(subdomains)
            inactive_coupling_edges[ids] = Vector{CouplingEdge{D}}(undef, 0)
            dirichlet_faces[ids] = Vector{Tuple{Int,Int}}(undef, 0)
            for f in faces(s)
                if is_coupling(f) && !is_active(f)
                    push!(inactive_coupling_edges[ids], f)
                elseif is_dirichlet(f)
                    push!(dirichlet_faces[ids], coordinates(f))
                end
            end            
        end

        # Check that edges are coupled correctly and add them to the list
        visited = Vector{Bool}(undef, n)
        stack = Vector{CouplingInterface{D}}(undef, 0)
        active_repr = Vector{CouplingInterface{D}}(undef, 0)
        for (ids,s) in enumerate(subdomains)
            for ide in inactive_edges(s)
                visited .= false
                visited[ids] = true
                empty!(active_repr)
                ci = CouplingInterface(ids, ide, IdentityPermutation(ones(Bool,D-codimension(ide))))
                push!(stack, ci)
                collect_active_representatives!(subdomains, stack, visited, active_repr)
                if isempty(active_repr)
                    throw(ArgumentError("Inactive edge $ide in subdomain $ids has no active representatives."))
                end
                all_dirichlet = all(ar->any_adjacent_dirichlet_faces(subdomains[ar.nbr_subdomain], [ar.nbr_edge]), active_repr)
                if (length(active_repr) == 1) || all_dirichlet
                    # activity and coupling can be set arbitrarily here
                    ce = CouplingEdge{Inactive,Interior}(ide, first(active_repr))
                    push!(inactive_coupling_edges[ids], ce)
                else
                    throw(ArgumentError("Inactive edge $ide in subdomain $ids has more than one active representative and not all of them are Dirichlet."))
                end
            end
        end

        return new(subdomains, inactive_coupling_edges, dirichlet_faces)
    end
end

function DomainPartition(subdomains)
    S = eltype(subdomains)
    D = dimension(S)
    array = collect(subdomains)
    ndims(array) != 1 && throw(ArgumentError("Input must be a one-dimensional collection."))
    return DomainPartition{D,S}(array)
end

Base.length(p::DomainPartition) = length(p.subdomains)
Base.eachindex(p::DomainPartition) = eachindex(p.subdomains)
Base.getindex(p::DomainPartition, i::Int) = p.subdomains[i]
Base.getindex(p::DomainPartition, i::Int, J...) = p.subdomains[i][J...]
Base.iterate(p::DomainPartition, i=1) = i > length(p) ? nothing : (p.subdomains[i], i+1)

########################################
# Verification
########################################

function correct_coupling_faces(subdomains::Vector{<:AbstractSubdomain{D}}) where {D}
    x = zeros(MVector{D, Float64})
    y = zeros(MVector{D, Float64})

    for (ids, s) in enumerate(subdomains)
        for f in faces(s)
            !is_coupling(f) && continue

            # Neighbor is a coupling face
            idnbrs, nbrf = neighbor(f)
            ((idnbrs < 1) || (idnbrs > length(subdomains))) && return false
            nbr = subdomains[idnbrs][nbrf]
            !is_coupling(nbr) && return false

            # Neighbor has correct coupling traits
            (is_active(f) == is_active(nbr)) && return false
            (coupling(f) != coupling(nbr)) && return false
    
            # Neighbor of neighbor is self
            idnbrnbrs, nbrnbrf = neighbor(nbr)
            ((ids != idnbrnbrs) || (Edge(f) != nbrnbrf)) && return false

            # Orientations are inverse
            Π = orientation(f)
            Π̂ = orientation(nbr)
            !are_inverse(Π, Π̂) && return false

            (coupling(f) == Periodic) && continue

            # Faces match geometrically
            # (check only vertices here, might want to check random inner points)
            ϕ = parametrization(s)
            ψ = parametrization(subdomains[idnbrs])
            for v in adjacent_edges(f, D)
                w,_ = map_and_permute(v, f, Π, nbr)
                for i in 1:D
                    x[i] = RefIntervalBeg + (v[i]-1) * (RefIntervalEnd-RefIntervalBeg)
                    y[i] = RefIntervalBeg + (w[i]-1) * (RefIntervalEnd-RefIntervalBeg)
                end
                !isapprox(ϕ(x), ψ(y)) && return false
            end
        end
    end
    return true
end

function collect_active_representatives!(subdomains, stack, visited, active_repr)
    while !isempty(stack)
        ci = pop!(stack)
        ids = ci.nbr_subdomain
        ide = ci.nbr_edge
        s = subdomains[ids]

        if ide ∉ inactive_edges(s)
           push!(active_repr, ci)
        end

        for idaf in adjacent_edges(ide, 1)
            !is_coupling(s[idaf]) && continue
            
            # To which neighboring edge and with what orientation does the
            # current edge correspond?
            ids_nbr, idf_nbr = neighbor(s[idaf])
            Π_nbr = orientation(s[idaf])
            ide_nbr, Π_nbre = map_and_permute(ci.nbr_edge, idaf, Π_nbr, idf_nbr)
            
            # Traverse further if not visited yet
            ci_nbr = CouplingInterface(ids_nbr, ide_nbr, Π_nbre ∘ ci.Π)
            if !visited[ids_nbr] 
                visited[ids_nbr] = true
                push!(stack, ci_nbr)
            end
        end
    end
end
