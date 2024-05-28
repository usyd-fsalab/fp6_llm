import argparse
import torch
import fp6_llm

WARMUP = 10
REPEAT = 1000

parser = argparse.ArgumentParser(description='The shape of the MatMul: (M, K)*(K, N)->(M, N).')
parser.add_argument('--OC',        type=int, required=False,     default=4096,   help='number of rows of the weight matrix.')
parser.add_argument('--IC',        type=int, required=False,     default=4096,   help='number of columns of the weight matrix.')
parser.add_argument('--BS',        type=int, required=False,     default=32,     help='inference batch size.')
parser.add_argument('--splitK',    type=int, required=False,     default=1,      help='Split-K parameters allow users to split the GEMM computation along the K dimension so that more CTAs will be created with a better SM utilization.')
parser.add_argument('--EXP',       type=int, required=False,     default=2,      help='number of bits of fpx Exponent, can be set to 2 or 3.')
parser.add_argument('--MAN',       type=int, required=False,     default=2,      help='number of bits of fpx Mantissa, can only be set to 2.')
args = parser.parse_args()

EXPONENT  = args.EXP
MANTISSA  = args.MAN
BIT_WIDTH = 1 + EXPONENT + MANTISSA
assert EXPONENT in [2,3]
assert MANTISSA in [2]

assert(args.OC%256==0)
assert(args.IC%64==0)

print("#"*64)
print(args)

fpx_weight = torch.randint(4294967295, (args.OC,args.IC//32*BIT_WIDTH)).to(torch.int)    # Randomly initialize each bytes. The highest value for randint() is set the the max value of uint32_t.
fp16_scale = torch.rand(args.OC).to(torch.half)+0.5
fp16_activation = torch.rand(args.BS, args.IC).to(torch.half)+0.5

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# fpx-fp16 GEMM (fp6-llm)
####################################################################################################################################
torch.cuda.synchronize()
fpx_weight_packed = fp6_llm.weight_prepacking_eXmY_cpu(EXPONENT, MANTISSA, fpx_weight)
act_cuda = fp16_activation.cuda()
weight_cuda = fpx_weight_packed.cuda()
scale_cuda = fp16_scale.cuda()
for i in range(WARMUP):
    results_fp6_llm = fp6_llm.linear_forward_eXmY_cuda(EXPONENT, MANTISSA, act_cuda, weight_cuda, scale_cuda, args.splitK)
start_event.record()
for i in range(REPEAT):
    results_fp6_llm = fp6_llm.linear_forward_eXmY_cuda(EXPONENT, MANTISSA, act_cuda, weight_cuda, scale_cuda, args.splitK)
end_event.record()
torch.cuda.synchronize()
fp6_llm_time_ms = start_event.elapsed_time(end_event)/REPEAT
fp6_llm_tflops  = args.OC*args.IC*args.BS*2/fp6_llm_time_ms/1e9
####################################################################################################################################

# baseline fp16 GEMM (cuBLAS)
####################################################################################################################################
torch.cuda.synchronize()
fp16_weight = fp6_llm.weight_dequant_eXmY_cpu(EXPONENT, MANTISSA, fpx_weight, fp16_scale)
cuBLAS_MatMul = torch.nn.Linear(args.IC, args.OC, False)
results_cublas = None
with torch.no_grad():
    cuBLAS_MatMul.weight = torch.nn.Parameter(fp16_weight.clone().cuda())
    act_cuda = fp16_activation.cuda()
    for i in range(WARMUP):
        results_cublas = cuBLAS_MatMul(act_cuda)
    start_event.record()
    for i in range(REPEAT):
        results_cublas = cuBLAS_MatMul(act_cuda)
    end_event.record()
torch.cuda.synchronize()
cublas_time_ms = start_event.elapsed_time(end_event)/REPEAT
cublas_tflops  = args.OC*args.IC*args.BS*2/cublas_time_ms/1e9
####################################################################################################################################

# Performance
print( 'cuBLAS    time: {:.3f} ms \t\t cuBLAS    TFLOPs: {:.1f}'.format(cublas_time_ms,  cublas_tflops ) )
print( 'quant-llm time: {:.3f} ms \t\t quant-llm TFLOPs: {:.1f}'.format(fp6_llm_time_ms, fp6_llm_tflops) )
print( 'speedup: {:.2f}'.format(cublas_time_ms/fp6_llm_time_ms) )

# Correctness
error             = results_cublas.cpu() - results_fp6_llm.cpu()
ground_truth      = results_cublas.cpu()
mean_error        = torch.mean(abs(error))
mean_ground_truth = torch.mean(abs(ground_truth))
relative_error    = mean_error.item()/mean_ground_truth.item()
print( "relative error: {:.6f}".format(relative_error) )


# [FP5_e2m2] Setting each element of the input matrices to 1.0f instead of a random value.
#fpx_weight = torch.zeros(args.OC, args.IC//32*BIT_WIDTH).to(torch.int64)  
#for i in range(args.OC):
#    for j in range(args.IC//32):
#        fpx_weight[i][j*BIT_WIDTH+0] = 272762913
#        fpx_weight[i][j*BIT_WIDTH+1] = 1107829124
#        fpx_weight[i][j*BIT_WIDTH+2] = 136414224 
#        fpx_weight[i][j*BIT_WIDTH+3] = 562303042 
#        fpx_weight[i][j*BIT_WIDTH+4] = 2215657992
#fpx_weight = fpx_weight.to(torch.int)
#fp16_scale      = torch.zeros(args.OC).to(torch.half)         +1.0
#fp16_activation = torch.zeros(args.BS, args.IC).to(torch.half)+1.0

# 1.0f values in fp5_e2m2
# 00100 00100 00100 00100 00100 00100 00100 00100       00100 00100 00100 00100 00100 00100 00100 00100     00100 00100 00100 00100 00100 00100 00100 00100     00100 00100 00100 00100 00100 00100 00100 00100
# “00100001 00001000 01000010 00010000”     “10000100 00100001 00001000 01000010”    “00010000 10000100 00100001 00001000”   “01000010 00010000 10000100 00100001”  “00001000 01000010 00010000 10000100”
# Considering Byte order within a INT32
# 00010000 01000010 00001000 00100001       01000010 00001000 00100001 10000100      00001000 00100001 10000100 00010000      00100001 10000100 00010000 01000010   10000100 00010000 01000010 00001000
# 00010000010000100000100000100001          01000010000010000010000110000100         00001000001000011000010000010000         00100001100001000001000001000010      10000100000100000100001000001000
# 272762913                                 1107829124                               136414224                                562303042                             2215657992


# [FP6_e3m2] Setting each element of the input matrices to 1.0f instead of a random value.
#fp6_weight = torch.zeros(args.OC, args.IC//32*BIT_WIDTH).to(torch.int64)
#for i in range(args.OC):
#    for j in range(args.IC//16):
#        fp6_weight[i][j*3+0] = 806142768
#        fp6_weight[i][j*3+1] = 3274706115
#        fp6_weight[i][j*3+2] = 214118412
#        fp6_weight[i][j*3+3] = 806142768
#        fp6_weight[i][j*3+4] = 3274706115
#        fp6_weight[i][j*3+5] = 214118412
#fp6_weight = fp6_weight.to(torch.int32)
#fp16_scale = torch.zeros(args.OC).to(torch.half)+1.0
#fp16_activation = torch.zeros(args.BS, args.IC).to(torch.half)+1.0

# 1.0f values in fp6_e3m2
# 001100 001100 001100 001100 001100 001100 001100 001100 001100 001100 001100 001100 001100 001100 001100 001100 
# "00110000110000110000110000110000"       "11000011000011000011000011000011"     "00001100001100001100001100001100"
# 00110000 11000011 00001100 00110000      11000011 00001100 00110000 11000011    00001100 00110000 11000011 00001100
# Considering Byte order within a INT32
# 00110000 00001100 11000011 00110000      11000011 00110000 00001100 11000011     00001100 11000011 00110000 00001100
# 00110000000011001100001100110000         11000011001100000000110011000011        00001100110000110011000000001100
# 806142768                                3274706115                              214118412