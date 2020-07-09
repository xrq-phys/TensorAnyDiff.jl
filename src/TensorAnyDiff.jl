# Simple implementation for tensordot and einsum for Julia,
# giving max compatibility for ForwardDiff and Zygote.

module TensorAnyDiff

using LinearAlgebra
import OMEinsum: EinCode
import Zygote: @adjoint

export @einmm_str, execute, contract

include("tensordot.jl")
include("einengine.jl")

end

