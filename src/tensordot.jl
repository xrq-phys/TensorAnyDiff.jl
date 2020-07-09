# Tensordot.contract implements numpy.tensordot for Julia
# numpy.einsum-like interface is available in einengine.jl.

"""
    contract(T1, T2, axesL, axesR)
Contracts tensor 'T1' with 'T2' in axes specified in axesL and axesR.
"""
contract(TL, TR, axesL::Array{Int}, axesR::Array{Int}) = begin
    # Gets shape and axes information from helper.
    shapeInfo = contractprep(size(TL), size(TR), axesL, axesR)

    # Apply transformation
    contractraw(TL, TR, shapeInfo...)
end

# Raw contraction function.
function contractraw(TL, TR, permL::Array{Int}, permR::Array{Int},
        shapeL::Tuple, shapeR::Tuple, extL::Array, extR::Array)

    # Multiply and restore to original shape.
    reshape((reshape(permutedims(TL, permL), shapeL) *
        reshape(permutedims(TR, permR), shapeR)), (extL..., extR...))
end

"Index preparations for contracting."
function contractprep(shapeL::Tuple, shapeR::Tuple, axesL::Array{Int}, axesR::Array{Int})
    shapeL = [shapeL...]
    shapeR = [shapeR...]
    # TODO: Check axes boundary.

    # Dumb index permuting & size extraction
    dumbperm(shape::Array{Int}, pick::Array{Int}) = begin
        sbarrier = sort(pick) .+ 1
        ebarrier = sort(pick) .- 1
        sbarrier = vcat([1], sbarrier)
        ebarrier = vcat(ebarrier, length(shape))
        regular = Int[]
        for i = 1:length(sbarrier)
            append!(regular, sbarrier[i]:ebarrier[i])
        end # for
        return regular
    end # dumbperm

    # External permutation
    permL = dumbperm(shapeL, axesL)
    permR = dumbperm(shapeR, axesR)
    # External shape
    extL = [shapeL[i] for i in permL]
    extR = [shapeR[i] for i in permR]
    # Contractional permutation
    append!(permL, axesL)
    prepend!(permR, axesR)
    outerL = if (length(extL)==0) 1 else reduce(*, extL) end
    outerR = if (length(extR)==0) 1 else reduce(*, extR) end
    innerL = reduce(*, [shapeL[i] for i in axesL])
    innerR = reduce(*, [shapeR[i] for i in axesR])

    permL, permR, (outerL, innerL), (innerR, outerR), extL, extR
end # contract

# Adjoint of contract should refrain from digging into index processing.
@adjoint contractprep(shapeL::Tuple, shapeR::Tuple, axesL::Array{Int}, axesR::Array{Int}) = begin
    contractprep(shapeL, shapeR, axesL, axesR), _ -> nothing
end

#= Further simplification for multiple gradients.
   Disabled by default.

@adjoint permutedims(A, iperm) = begin
    permutedims(A, iperm), dA -> begin
        rperm = invperm(iperm)
        (permutedims(dA, rperm), nothing)
    end
end

@adjoint invperm(iperm) = begin
    invperm(iperm), _ -> nothing
end
=#

