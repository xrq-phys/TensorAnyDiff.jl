TensorAnyDiff
=============

This package is an automatic `numpy.tensordot` and `numpy.einsum`-style wrapper for Julia. It is a continuation of [this Github Gist](https://gist.github.com/xrq-phys/8b6e52f0f371acb0244950f755d7476f).

## Features

- × *Not the fastest*;
- × *No advanced infrastructure*;
- ○ Uses directly matrix multiplication of [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra);
- ○ Plays well with [Zygote.jl](https://github.com/FluxML/Zygote.jl) and [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).

ForwardDiff can theoretically work up to any order and for Zygote.jl derivatives up to 2nd order is supported via `Zygote.hessian` or `Zygote.gradient(Zygote.gradient(...)...)`.

## Usage

### Installation

In Julia REPL console:
``` julia
] add https://github.com/xrq-phys/TensorAnyDiff.jl
using TensorAnyDiff
```

### Calling 2-Tensor Contractions

- `contract(A, B, [2], [2])` is like `numpy.tensordot`, which accepts dimension to be traced out as 3rd and 4th arguments;
- `execute(einmm"il,jl->ij", A, B)` is a einstein summation wrapper made with [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl);

If `TensorAnyDiff.use_einsum` is set to `false` (default), `execute(einmm"...")` would call `contract` as backend so that differentiations like `Zygote.hessian(...)` are supported, otherwise it will call the original `OMEinsum.ein"..."` which is faster but does not support some of the AutoDiff operations.

## Roadmaps

- [ ] Optimize performance or provide 2-order dispatching functionalities for OMEinsum.jl;
- [ ] Add support for some Lapack operations.
