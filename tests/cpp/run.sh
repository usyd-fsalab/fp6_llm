#! /bin/bash

# Batch sizes to test
N=(1 2 3 4 5 6 7 8)

# Benchmarking the specific Matrix Shape from llama-1 65b
M=(13824 5120  22016 8192)
K=(5120  13824 8192  22016)
SplitK=(4 10 5 6)      # SplitK for smaller Batch Sizes


# Benchmarking Matrix Shapes from OPT models 
#M=(21504        7168     28672  7168    27648   9216    36864   9216    36864   12288   49152   12288)
#K=(7168         7168     7168   28672   9216    9216    9216    36864   12288   12288   12288   49152)
#SplitK=(2       7       7      7       2       6       3       6       3       4       1       4)        # SplitK for smaller Batch Sizes
#SplitK=(5       7       7      7       1       3       3       3       3       2       1       2)        # SplitK for Batch Sizes 128
#SplitK=(3       2       3      3       1       3       1       3       1       1       1       1)        # SplitK for Batch Sizes 512
#SplitK=(1       2       1      2       1       1       1       1       1       1       1       1)        # SplitK for Batch Sizes 2048

# Benchmarking Matrix Shapes from llama-1 models 
#M=(12288       4096    11008   4096    15360   5120    13824   5120    19968   6656    17920   6656    24576   8192    22016   8192)
#K=(4096        4096    4096    11008   5120    5120    5120    13824   6656    6656    6656    17920   8192    8192    8192    22016)
#SplitK=(9      13      5       13      3       10      4       10      5       8       3       8       2       6       5       6)      # SplitK for smaller Batch Sizes
#SplitK=(2      6       5       6       3       5       2       5       4       4       3       4       1       3       5       3)      # SplitK for Batch Sizes 128
#SplitK=(1      3       3       3       2       4       1       4       1       1       3       1       1       3       2       3)      # SplitK for Batch Sizes 512
#SplitK=(1      2       1       2       1       1       1       1       1       1       1       1       1       1       1       1)      # SplitK for Batch Sizes 2048

#mkdir -p Profiling
for ((i=0;i<${#M[@]};i++)) 
do
    echo "Processing Shape ${i}..."
    for BS in ${N[@]} 
    do
        echo "BS=${BS}"
        #ncu -f -o Profiling/M${M[i]}K${K[i]}N${BS} --set full \
        ./kernel_test ${M[i]} ${K[i]} ${BS} ${SplitK[i]}  
    done
done
