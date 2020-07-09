# This file loads interface from OMEinsum.
# This file implements alternative driver for EinCode object with Tensordot.contract.
# OMEinsum outperforms Tensordot.contract under Zygote.jl but latter
#  is much more compatible (surviving Zygote.hessian).

# Disable OMEinsum backend by default.
use_einsum = false

"""
    einmm"ij,jk -> ik"(A,B)
String macro interface which understands `numpy.einsum`'s notation.
Translates strings into `EinCode`-structs that can be called to evaluate an `einsum`.

This specific block is copy-paste from under-Peter/OMEinsum.
"""
macro einmm_str(s::AbstractString)
    s = replace(s, " " => "")
    m = match(r"([\(\)a-z,α-ω]*)->([a-zα-ω]*)", s)
    m == nothing && throw(ArgumentError("invalid einsum specification $s"))
    sixs, siy = m.captures
    if '(' in sixs
        error("Nested Einsum not supported at the moment.")
    else
        iy  = Tuple(siy)
        ixs = Tuple(Tuple(ix) for ix in split(sixs,','))
        # TODO: Alias EinCode to switch to Tensordot.contract backend.
        return EinCode(ixs, iy)
    end
end

"""
    execute(s::EinCode{::Any, ::Any}, TL, TR)
Execute operation defined in an EinCode with OMEinsum dispatcher 
or Tensordot.contract compatibility backend.
"""
function execute(s::EinCode, TL, TR)
    if use_einsum
        return s(TL, TR)
    else
        idL, idR, operm = ein_char2idx(s)
        return permutedims(contract(TL, TR, idL, idR), operm)
    end
end

"Converts Einstein-summation indices to dimension index of tensors."
function ein_char2idx(sxL::Tuple, sxR::Tuple, sy::Tuple)
    cidxL = []
    cidxR = []
    # Parse inputs.
    for (ixL, cxL)=enumerate(sxL)
        found = false
        for (ixR, cxR)=enumerate(sxR)
            if (cxL == cxR)
                cidxL = [cidxL..., ixL]
                cidxR = [cidxR..., ixR]
            end
        end
    end
    # Get tensordot out indices.
    lcL, lcR = ((lcT, cidxT) -> begin
        for ixT=reverse(sort(cidxT))
            deleteat!(lcT, ixT)
        end
        lcT
    end).([[sxL...], [sxR...]], [cidxL, cidxR])
    ostr = [lcL..., lcR...]
    # Get transpose operation for final output.
    operm = (cy -> begin
        for (iout, cout)=enumerate(ostr)
            if (cy == cout)
                return iout
            end
        end
    end).(sy)
    cidxL, cidxR, operm
end

"Converts Einstein-summation indices to dimension index of tensors."
function ein_char2idx(::EinCode{ixs, iy}) where {ixs, iy}
    # Currently only able to handle 2-tensor summations.
    sxL, sxR = ixs
    cidxL, cidxR, oidx = ein_char2idx(sxL, sxR, iy)
end

@adjoint ein_char2idx(sxL::Tuple, sxR::Tuple, sy::Tuple) = begin
    ein_char2idx(sxL, sxR, sy), _ -> nothing
end

@adjoint ein_char2idx(s::EinCode) = begin
    ein_char2idx(s), _ -> nothing
end

