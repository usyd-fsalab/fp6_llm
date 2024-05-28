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
args = parser.parse_args()

assert(args.OC%256==0)
assert(args.IC%64==0)

print("#"*64)
print(args)

fp6_weight = torch.randint(4294967295, (args.OC,args.IC//16*3)).to(torch.int)    # Randomly initialize each bytes. The highest value for randint() is set the the max value of uint32_t.
fp16_scale = torch.rand(args.OC).to(torch.half)+0.5
fp16_activation = torch.rand(args.BS, args.IC).to(torch.half)+0.5
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# fp6-fp16 GEMM (fp6-llm)
####################################################################################################################################
torch.cuda.synchronize()
fp6_weight_packed = fp6_llm.weight_prepacking_cpu(fp6_weight)
act_cuda = fp16_activation.cuda()
weight_cuda = fp6_weight_packed.cuda()
scale_cuda = fp16_scale.cuda()
for i in range(WARMUP):
    results_fp6_llm = fp6_llm.linear_forward_cuda(act_cuda, weight_cuda, scale_cuda, args.splitK);
start_event.record()
for i in range(REPEAT):
    results_fp6_llm = fp6_llm.linear_forward_cuda(act_cuda, weight_cuda, scale_cuda, args.splitK);
end_event.record()
torch.cuda.synchronize()
fp6_llm_time_ms = start_event.elapsed_time(end_event)/REPEAT
fp6_llm_tflops  = args.OC*args.IC*args.BS*2/fp6_llm_time_ms/1e9
####################################################################################################################################

# baseline fp16 GEMM (cuBLAS)
####################################################################################################################################
torch.cuda.synchronize()
fp16_weight = fp6_llm.weight_dequant_cpu(fp6_weight, fp16_scale)
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
print( 'cuBLAS  time: {:.3f} ms \t\t cuBLAS  TFLOPs: {:.1f}'.format(cublas_time_ms,  cublas_tflops) )
print( 'fp6-llm time: {:.3f} ms \t\t fp6-llm TFLOPs: {:.1f}'.format(fp6_llm_time_ms, fp6_llm_tflops) )
print( 'speedup: {:.2f}'.format(cublas_time_ms/fp6_llm_time_ms) )

# Correctness
error             = results_cublas.cpu() - results_fp6_llm.cpu()
ground_truth      = results_cublas.cpu()
mean_error        = torch.mean(abs(error))
mean_ground_truth = torch.mean(abs(ground_truth))
relative_error    = mean_error.item()/mean_ground_truth.item()
print( "relative error: {:.6f}".format(relative_error) )