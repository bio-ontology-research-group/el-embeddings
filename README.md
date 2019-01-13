# EL Embeddings

Here you can find the code to generate spherical embeddings for the description logic EL++ (the logic used in OWL 2 EL).

## How to run

There are two parts that need to be executed sequentially. The first part converts an ontology in OWL 2 EL into a set of one of four normal forms.
The second part of the method generates the embeddings for classes, relations, and individuals.

To build: To generate the normal forms, you need a modified version of the `jcel` reasoner which you find in a submodule here (if you have not cloned this repository with submodules, try
to do `git clone --recurse-submodules https://github.com/leechuck/el-embeddings`.
Next, `cd jcel` and `mvn install` to build the jcel jar files.

For convenience, we include the jar file in the `jar/` subdirectory, so you may simply want to add `jcel.jar` to your CLASSPATH.
Then run `groovy Normalizer.groovy` and follow the instructions (i.e., command line options are in input OWL file and and output file containing the normal forms in OWL functional syntax).

```
usage: groovy Normalizer.groovy -i INPUT -o OUTPUT [-h]
 -h,--help           this information
 -i,--input <arg>    input OWL file
 -o,--output <arg>   output file containing normalized axioms
```

To generate the embeddings, run `julia elembedding.jl --help` and follow instructions. You need CUDA installed to use a GPU, and need to install the packages
`ArgParse`, `MLDataUtils`, `ForwardDiff`, `Distances`, `LinearAlgebra`, `ReverseDiff`, `AutoGrad`, `Random`, `Flux`, `DelimitedFiles` using Julia's package management system (e.g., `Pkg.add()`).
If you want to use a GPU during training, also install the `CuArrays` package.

```
usage: elembedding.jl -i INPUT -o OUTPUT [-e EPOCHS] [-l LR] [-d DIM]
                      [-g] [-h]

optional arguments:
  -i, --input INPUT    input file containing normalized OWL EL axioms;
                       usually the output of Normalize.groovy
  -o, --output OUTPUT  output file containing class, relation, and
                       instance coordinates
  -e, --epochs EPOCHS  number of epochs to train (type: Int64,
                       default: 1000)
  -l, --lr LR          learning rate (type: Float64, default: 0.01)
  -d, --dim DIM        input dimensions (type: Int64, default: 20)
  -g, --gpu            use GPU accelleration
  -h, --help           show this help message and exit
```