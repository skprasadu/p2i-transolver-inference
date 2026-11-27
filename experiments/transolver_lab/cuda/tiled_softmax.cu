// Inference-only tiled softmax. Dense arithmetic remains in cuBLAS.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cfloat>
__global__ void softmax4(const float* x,const float* temp,const float* norm,float* y,int heads,int slices,int rows) {
  int row=blockIdx.x, lane=threadIdx.x, h=row%heads;
  if(row>=rows)return;
  __shared__ float reduce[128];
  float t=fminf(5.f,fmaxf(.5f,temp[h]));
  float4 v=reinterpret_cast<const float4*>(x+row*slices)[lane];
  v.x/=t;v.y/=t;v.z/=t;v.w/=t;
  float m=fmaxf(fmaxf(v.x,v.y),fmaxf(v.z,v.w));
  reduce[lane]=m;__syncthreads();
  for(int s=blockDim.x/2;s;s/=2){if(lane<s)reduce[lane]=fmaxf(reduce[lane],reduce[lane+s]);__syncthreads();}
  m=reduce[0];__syncthreads();
  v.x=expf(v.x-m);v.y=expf(v.y-m);v.z=expf(v.z-m);v.w=expf(v.w-m);
  reduce[lane]=(v.x+v.y)+(v.z+v.w);__syncthreads();
  for(int s=blockDim.x/2;s;s/=2){if(lane<s)reduce[lane]+=reduce[lane+s];__syncthreads();}
  float z=reduce[0];v.x/=z;v.y/=z;v.z/=z;v.w/=z;
  if(norm){float4 d=reinterpret_cast<const float4*>(norm+h*slices)[lane];v.x/=d.x;v.y/=d.y;v.z/=d.z;v.w/=d.w;}
  reinterpret_cast<float4*>(y+row*slices)[lane]=v;
}
torch::Tensor softmax(torch::Tensor x,torch::Tensor t,c10::optional<torch::Tensor> norm){
  TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.scalar_type()==torch::kFloat32&&x.dim()==4&&x.size(0)==1,"expected contiguous FP32 CUDA [1,N,H,S]");
  int s=x.size(3),h=x.size(2);TORCH_CHECK(s==64||s==512,"supported slices:64,512");
  TORCH_CHECK(t.is_cuda()&&t.device()==x.device()&&t.is_contiguous()&&t.scalar_type()==torch::kFloat32&&t.numel()==h,"temperature mismatch");
  if(norm)TORCH_CHECK(norm->is_cuda()&&norm->device()==x.device()&&norm->is_contiguous()&&norm->scalar_type()==torch::kFloat32&&norm->numel()==h*s,"norm mismatch");
  c10::cuda::CUDAGuard guard(x.device());auto y=torch::empty_like(x);
  int rows=x.numel()/s;
  if(rows)softmax4<<<rows,s/4,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),t.data_ptr<float>(),norm?norm->data_ptr<float>():nullptr,y.data_ptr<float>(),h,s,rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("softmax",&softmax);}
