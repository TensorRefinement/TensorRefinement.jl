```@meta
CurrentModule = TensorRefinement
```

# TensorRefinement

Welcome to the documentation for TensorRefinement.jl. Below you will find the index followed by the different sections of the package:


```@index
```

# Auxiliary

This section covers functions related to the Auxiliary module of the package.
```@docs
Indices
FloatRC
Float2 
Float3 
Int2
Int3
indvec 
threshold 
compfloateps 
modemul 
qraddcols
lqaddrows
```

# Chebyshev

This section covers functions related to the Chebyshev module of the package.
```@docs
chebeval 
chebexnodes 
chebtochebex
chebextocheb
chebextoleg
chebextolegn
chebrtnodes
chebtochebrt
chebrttocheb
chebrttoleg
chebrttolegn
chebtoleg
chebtolegn
legtocheb
legntocheb
chebref
chebdiff
chebexdiff
chebdec
chebdeceval!
```

# Exponential

This section covers functions related to the Exponential module of the package.
```@docs
trigevalmask
trigeval
trigrefmask
trigrefmask2
trigdiffmask
trigdiff
trigdec
trigdeceval!
cosfactor
cosdec
```

# FEM

This section covers functions related to the FEM module of the package.
```@docs
extdn
extdd
diffdn 
diffdd 
dint
dintf
bpxdn
bpxdd
extmix
diffbpxdn
diffbpxdd
```

# Legendre

This section covers functions related to the Legendre module of the package.
```@docs
legeval 
legneval 
legtolegn
legntoleg
legdiff
legndiff 
legref
legnref 
legdec
legdeceval!
```


# TensorTrainFactor

This section covers functions related to the TensorTrainFactor module of the package.
```@docs
FactorSize
Factor
VectorFactor 
MatrixFactor
factorsize 
factorranks
factorndims
factornumentries
factorstorage
factor
factormatrix
factorrankselect
block
factorvcat
factorhcat 
factordcat
factorltcat 
factorutcat
factorranktranspose 
factormodetranspose
factormodereshape
factordiagm
factorcontract 
factormp
factorkp
factorhp
factorqr!
factorqradd
factorsvd!
```


# TensorTrainFactorization

This section covers functions related to the TensorTrainFactorization module of the package.
```@docs
DecSize 
DecRank
Dec
VectorDec 
MatrixDec
checkndims
checklength
checksize
checkrank
checkranks
declength
decndims
decsize
decranks
decrank
dec
dec!
vector
decrankselect!
decrankselect
factor!
block! 
decvcat 
dechcat
decdcat
decscale!
decreverse!
decmodetranspose!
decmodereshape
decfill!
decrand!
deczeros 
decones 
decrand
decappend!
decprepend! 
decpush!
decpushfirst! 
decpop!
decpopfirst!
decinsert!
decdeleteat!
decinsertidentity!
decskp!
decskp
decmp 
deckp 
decaxpby!
decadd!
decaxpby
decadd 
dechp
decqr! 
decsvd!
```

# TT

This section covers functions related to the TT module of the package.
```@docs
TT
TensorRefinement.TensorTrain.length
TensorRefinement.TensorTrain.ndims 
TensorRefinement.TensorTrain.size 
TensorRefinement.TensorTrain.rank 
ranks
TensorRefinement.TensorTrain.deepcopy
TensorRefinement.TensorTrain.reverse! 
TensorRefinement.TensorTrain.permutedims!
TensorRefinement.TensorTrain.fill!
TensorRefinement.TensorTrain.rand!
getfirstfactor
getlastfactor 
getfactor
setfirstfactor! 
setlastfactor!
setfactor!
rankselect! 
rankselect
TensorRefinement.TensorTrain.getindex
TensorRefinement.TensorTrain.append!
TensorRefinement.TensorTrain.prepend!
TensorRefinement.TensorTrain.push!
TensorRefinement.TensorTrain.pushfirst!
TensorRefinement.TensorTrain.pop!
TensorRefinement.TensorTrain.popfirst!
TensorRefinement.TensorTrain.insert!
TensorRefinement.TensorTrain.deleteat!
compose!
compose
composecore!
composecore
composeblock!
composeblock
TensorRefinement.TensorTrain.vcat
TensorRefinement.TensorTrain.hcat
dcat
TensorRefinement.TensorTrain.lmul!
TensorRefinement.TensorTrain.rmul!
mul
had 
*
TensorRefinement.TensorTrain.kron
⊗
add
+
TensorRefinement.TensorTrain.qr!
TensorRefinement.TensorTrain.svd!
```

```@autodocs
Modules = [TensorRefinement]
Order = [:function, :type]
Private = true
```
