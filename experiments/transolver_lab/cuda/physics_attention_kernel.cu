// FP32 inference-only, D=M=64. No global B*H*N*M weights.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
constexpr int D=64, T=32, CHUNK=1024;
__device__ void weights(const float* x,const float* w,const float* bias,float temp,float* s,int n,int base) {
  for(int i=threadIdx.x;i<T*D;i+=blockDim.x){int p=i/D,m=i%D;float v=bias[m];
    if(base+p<n){for(int d=0;d<D;d++)v=fmaf(x[(base+p)*D+d],w[m*D+d],v);s[i]=v/temp;}else s[i]=-INFINITY;
  } __syncthreads();
  if(threadIdx.x<T){int p=threadIdx.x;float mx=-INFINITY,z=0;
    if(base+p<n){for(int m=0;m<D;m++)mx=fmaxf(mx,s[p*D+m]);for(int m=0;m<D;m++){float v=expf(s[p*D+m]-mx);s[p*D+m]=v;z+=v;}for(int m=0;m<D;m++)s[p*D+m]/=z;}
    else for(int m=0;m<D;m++)s[p*D+m]=0;
  } __syncthreads();
}
__global__ void aggregate(const float* x,const float* f,const float* w,const float* bias,const float* temp,float* partial,int n,int heads,int chunks){
  __shared__ float s[T*D],v[T*D];
  int bh=blockIdx.y,k=blockIdx.x; x+= (long long)bh*n*D; f+=(long long)bh*n*D;
  float accum[16]={0};float norm=0;
  for(int base=k*CHUNK;base<min(n,(k+1)*CHUNK);base+=T){
    weights(x,w,bias,temp[bh%heads],s,n,base);
    for(int i=threadIdx.x;i<T*D;i+=blockDim.x)v[i]=(base+i/D<n)?f[(base+i/D)*D+i%D]:0;
    __syncthreads();
    for(int j=0;j<16;j++){int i=threadIdx.x+j*256,m=i/D,d=i%D;float a=accum[j];for(int p=0;p<T;p++)a=fmaf(s[p*D+m],v[p*D+d],a);accum[j]=a;}
    if(threadIdx.x<D)for(int p=0;p<T;p++)norm+=s[p*D+threadIdx.x];
    __syncthreads();
  }
  float* out=partial+((long long)bh*chunks+k)*(D*D+D);
  for(int j=0;j<16;j++)out[threadIdx.x+j*256]=accum[j];
  if(threadIdx.x<D)out[D*D+threadIdx.x]=norm;
}
__global__ void finish(const float* p,float* out,int chunks){
 int i=blockIdx.x*blockDim.x+threadIdx.x,bh=blockIdx.y;if(i>=D*D)return;
 float a=0,z=0;for(int k=0;k<chunks;k++){auto q=p+((long long)bh*chunks+k)*(D*D+D);a+=q[i];z+=q[D*D+i/D];}out[bh*D*D+i]=a/(z+1e-5f);
}
__global__ void deslice(const float* x,const float* tokens,const float* w,const float* bias,const float* temp,float* out,int n,int heads){
 __shared__ float s[T*D];int bh=blockIdx.y,base=blockIdx.x*T;x+=(long long)bh*n*D;
 weights(x,w,bias,temp[bh%heads],s,n,base);
 for(int i=threadIdx.x;i<T*D;i+=blockDim.x){int p=i/D,d=i%D;if(base+p<n){float a=0;for(int m=0;m<D;m++)a=fmaf(s[p*D+m],tokens[bh*D*D+m*D+d],a);out[((long long)bh*n+base+p)*D+d]=a;}}
}
void check(torch::Tensor t,torch::Tensor x){TORCH_CHECK(t.is_cuda()&&t.scalar_type()==torch::kFloat32&&t.is_contiguous(),"CUDA contiguous FP32 required");TORCH_CHECK(t.device()==x.device(),"same device required");}
void validate(torch::Tensor x,torch::Tensor w,torch::Tensor b,torch::Tensor t){
 check(x,x);check(w,x);check(b,x);check(t,x);
 TORCH_CHECK(x.dim()==4&&x.size(3)==64&&x.size(2)>0,"expected B,H,N,64");
 TORCH_CHECK(w.dim()==2&&w.size(0)==64&&w.size(1)==64&&b.numel()==64&&t.numel()==x.size(1),"invalid parameters");
 TORCH_CHECK(x.size(0)*x.size(1)<=65535&&x.size(2)<=10000000,"shape exceeds launch limits");
}
torch::Tensor slice(torch::Tensor x,torch::Tensor f,torch::Tensor w,torch::Tensor b,torch::Tensor t){
 validate(x,w,b,t);check(f,x);TORCH_CHECK(f.sizes()==x.sizes(),"feature shape mismatch");c10::cuda::CUDAGuard guard(x.device());
 int n=x.size(2),h=x.size(1),bh=x.size(0)*h,k=(n+CHUNK-1)/CHUNK;
 auto partial=torch::empty({bh,k,D*D+D},x.options());auto out=torch::empty({x.size(0),h,D,D},x.options());auto stream=at::cuda::getCurrentCUDAStream();
 aggregate<<<dim3(k,bh),256,0,stream>>>(x.data_ptr<float>(),f.data_ptr<float>(),w.data_ptr<float>(),b.data_ptr<float>(),t.data_ptr<float>(),partial.data_ptr<float>(),n,h,k);C10_CUDA_KERNEL_LAUNCH_CHECK();
 finish<<<dim3(16,bh),256,0,stream>>>(partial.data_ptr<float>(),out.data_ptr<float>(),k);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor unslice(torch::Tensor x,torch::Tensor token,torch::Tensor w,torch::Tensor b,torch::Tensor t){
 validate(x,w,b,t);check(token,x);TORCH_CHECK(token.dim()==4&&token.size(0)==x.size(0)&&token.size(1)==x.size(1)&&token.size(2)==64&&token.size(3)==64,"token shape mismatch");c10::cuda::CUDAGuard guard(x.device());auto out=torch::empty_like(x);int n=x.size(2),h=x.size(1),bh=x.size(0)*h;
 deslice<<<dim3((n+T-1)/T,bh),256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),token.data_ptr<float>(),w.data_ptr<float>(),b.data_ptr<float>(),t.data_ptr<float>(),out.data_ptr<float>(),n,h);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
// NVIDIA FP32 path: 512 slices, 32 features. Only tile-local weights exist.
namespace nv {
constexpr int S=512,F=32,P=4,C=128;
__device__ void project(const float*x,const float*w,const float*b,float temp,float*s,int n,int base){
 for(int i=threadIdx.x;i<P*S;i+=blockDim.x){int p=i/S,m=i%S;float a=b[m];
 if(base+p<n){for(int d=0;d<F;d++)a=fmaf(x[(base+p)*F+d],w[m*F+d],a);s[i]=a/fminf(5.f,fmaxf(.5f,temp));}else s[i]=-INFINITY;}
 __syncthreads();
 // One warp per point; 16 slices per lane.
 int p=threadIdx.x/32,l=threadIdx.x%32;
 if(p<P){float mx=-INFINITY,z=0;
 for(int j=0;j<16;j++)mx=fmaxf(mx,s[p*S+l+j*32]);
 for(int k=16;k;k/=2)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,k));
 for(int j=0;j<16;j++){float v=base+p<n?expf(s[p*S+l+j*32]-mx):0;s[p*S+l+j*32]=v;z+=v;}
 for(int k=16;k;k/=2)z+=__shfl_xor_sync(0xffffffff,z,k);
 for(int j=0;j<16;j++)s[p*S+l+j*32]=z?s[p*S+l+j*32]/z:0;}
 __syncthreads();
}
__global__ void aggregate(const float*x,const float*f,const float*w,const float*b,const float*t,float*part,int n,int h,int chunks){
 __shared__ float s[P*S],v[P*F];int bh=blockIdx.y,k=blockIdx.x;
 x+=(long long)bh*n*F;f+=(long long)bh*n*F;
 float acc[64]={0},norm[2]={0};
 for(int base=k*C;base<min(n,(k+1)*C);base+=P){project(x,w,b,t[bh%h],s,n,base);
 for(int i=threadIdx.x;i<P*F;i+=256)v[i]=base+i/F<n?f[(base+i/F)*F+i%F]:0;
 __syncthreads();
 for(int j=0;j<64;j++){int i=threadIdx.x+j*256;for(int p=0;p<P;p++)acc[j]=fmaf(s[p*S+i/F],v[p*F+i%F],acc[j]);}
 for(int j=0;j<2;j++)for(int p=0;p<P;p++)norm[j]+=s[p*S+threadIdx.x+j*256];
 __syncthreads();}
 float*out=part+((long long)bh*chunks+k)*(S*F+S);
 for(int j=0;j<64;j++)out[threadIdx.x+j*256]=acc[j];
 for(int j=0;j<2;j++)out[S*F+threadIdx.x+j*256]=norm[j];
}
__global__ void reduce(const float*p,float*out,int chunks){int i=blockIdx.x*256+threadIdx.x,bh=blockIdx.y;
 double a=0,z=0;for(int k=0;k<chunks;k++){auto q=p+((long long)bh*chunks+k)*(S*F+S);a+=q[i];z+=q[S*F+i/F];}out[bh*S*F+i]=a/(z+.01);}
__global__ void back(const float*x,const float*tok,const float*w,const float*b,const float*t,float*out,int n,int h){
 __shared__ float s[P*S];int bh=blockIdx.y,base=blockIdx.x*P;x+=(long long)bh*n*F;
 project(x,w,b,t[bh%h],s,n,base);
 for(int i=threadIdx.x;i<P*F;i+=256){int p=i/F,d=i%F;if(base+p<n){float a=0;for(int m=0;m<S;m++)a=fmaf(s[p*S+m],tok[bh*S*F+m*F+d],a);out[((long long)bh*n+base+p)*F+d]=a;}}}
void validate(torch::Tensor x,torch::Tensor w,torch::Tensor b,torch::Tensor t){check(x,x);check(w,x);check(b,x);check(t,x);
 TORCH_CHECK(x.dim()==4&&x.size(3)==F&&x.size(2)>0&&x.size(2)<=10000000,"expected B,H,N,32");
 TORCH_CHECK(x.size(0)*x.size(1)<=65535&&w.dim()==2&&w.size(0)==S&&w.size(1)==F&&b.numel()==S&&t.numel()==x.size(1),"NVIDIA parameter mismatch");}
torch::Tensor slice(torch::Tensor x,torch::Tensor f,torch::Tensor w,torch::Tensor b,torch::Tensor t){validate(x,w,b,t);check(f,x);TORCH_CHECK(f.sizes()==x.sizes(),"feature mismatch");c10::cuda::CUDAGuard guard(x.device());
 int n=x.size(2),h=x.size(1),bh=x.size(0)*h,k=(n+C-1)/C;auto part=torch::empty({bh,k,S*F+S},x.options()),out=torch::empty({x.size(0),h,S,F},x.options());auto stream=at::cuda::getCurrentCUDAStream();
 aggregate<<<dim3(k,bh),256,0,stream>>>(x.data_ptr<float>(),f.data_ptr<float>(),w.data_ptr<float>(),b.data_ptr<float>(),t.data_ptr<float>(),part.data_ptr<float>(),n,h,k);C10_CUDA_KERNEL_LAUNCH_CHECK();
 reduce<<<dim3(S*F/256,bh),256,0,stream>>>(part.data_ptr<float>(),out.data_ptr<float>(),k);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}
torch::Tensor unslice(torch::Tensor x,torch::Tensor tok,torch::Tensor w,torch::Tensor b,torch::Tensor t){validate(x,w,b,t);check(tok,x);TORCH_CHECK(tok.dim()==4&&tok.size(0)==x.size(0)&&tok.size(1)==x.size(1)&&tok.size(2)==S&&tok.size(3)==F,"token mismatch");c10::cuda::CUDAGuard guard(x.device());auto out=torch::empty_like(x);int n=x.size(2),h=x.size(1);
 back<<<dim3((n+P-1)/P,x.size(0)*h),256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),tok.data_ptr<float>(),w.data_ptr<float>(),b.data_ptr<float>(),t.data_ptr<float>(),out.data_ptr<float>(),n,h);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("slice",&slice);m.def("unslice",&unslice);m.def("nvidia_slice",&nv::slice);m.def("nvidia_unslice",&nv::unslice);}
