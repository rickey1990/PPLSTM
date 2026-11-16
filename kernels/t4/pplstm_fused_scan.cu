
// NVRTC: no host C math.h header needed; expf/fmaxf are CUDA device built-ins.

// One CTA per (batch, group), one thread per token. Elements beyond T are identity maps.
// Memory: A_t (3x3), b_t (3), f_t, i_t. Fused scan output is normalized and gated.
extern "C" __global__ void pplstm_fused_fwd(
 const float *__restrict__ A, const float *__restrict__ b,
 const float *__restrict__ f, const float *__restrict__ i,
 const float *__restrict__ o, const float *__restrict__ c0,
 float *__restrict__ h, float *__restrict__ cstate, float *__restrict__ nstate,
 int T, int G) {
  const int lane=threadIdx.x;
  const int group=blockIdx.x % G;
  const int batch=blockIdx.x / G;
  const int gidx=(batch*T+lane)*G+group;
  const int bidx=gidx*3, aidx=gidx*9;
  float aa[9], bb[3];
  float ff=1.f, ii=0.f;
  #pragma unroll
  for(int j=0;j<9;++j) aa[j]=(j%4==0)?1.f:0.f;
  bb[0]=0.f;bb[1]=0.f;bb[2]=0.f;
  if(lane<T) {
    #pragma unroll
    for(int j=0;j<9;++j) aa[j]=A[aidx+j];
    #pragma unroll
    for(int j=0;j<3;++j) bb[j]=b[bidx+j];
    ff=f[gidx];ii=i[gidx];
  }
  extern __shared__ float sh[];
  for(int step=1;step<blockDim.x;step<<=1) {
    float *own=sh+lane*14;
    #pragma unroll
    for(int j=0;j<9;++j) own[j]=aa[j];
    #pragma unroll
    for(int j=0;j<3;++j) own[9+j]=bb[j];
    own[12]=ff;own[13]=ii;
    __syncthreads();
    if(lane>=step) {
      const float *prev=sh+(lane-step)*14;
      float na[9],nb[3];
      #pragma unroll
      for(int row=0;row<3;++row) {
        #pragma unroll
        for(int col=0;col<3;++col) {
          na[row*3+col]=aa[row*3]*prev[col]
                         +aa[row*3+1]*prev[3+col]
                         +aa[row*3+2]*prev[6+col];
        }
        nb[row]=bb[row]+aa[row*3]*prev[9]
                           +aa[row*3+1]*prev[10]
                           +aa[row*3+2]*prev[11];
      }
      float next_ii=ii+ff*prev[13];
      ff=ff*prev[12];ii=next_ii;
      #pragma unroll
      for(int j=0;j<9;++j) aa[j]=na[j];
      #pragma unroll
      for(int j=0;j<3;++j) bb[j]=nb[j];
    }
    __syncthreads();
  }
  if(lane<T) {
    const int out=(batch*T+lane)*(G*3)+group*3;
    const int start=(batch*G+group)*3;
    float nn=fmaxf(ff+ii,1.e-12f);
    nstate[gidx]=ff+ii;
    #pragma unroll
    for(int j=0;j<3;++j) {
      float cc=bb[j]+aa[j*3]*c0[start]
                     +aa[j*3+1]*c0[start+1]
                     +aa[j*3+2]*c0[start+2];
      cstate[bidx+j]=cc;
      float sig=1.f/(1.f+expf(-o[out+j]));
      h[out+j]=sig*(cc/nn);
    }
  }
}

// Reverse affine scan for the state and normalizer adjoints.
// q_t = dL/dc_t + A_(t+1)^T q_(t+1),
// u_t = dL/dn_t + f_(t+1) u_(t+1).
extern "C" __global__ void pplstm_fused_bwd(
 const float *__restrict__ A, const float *__restrict__ f,
 const float *__restrict__ o, const float *__restrict__ c0,
 const float *__restrict__ cstate, const float *__restrict__ nstate,
 const float *__restrict__ gh,
 float *__restrict__ ga, float *__restrict__ gb,
 float *__restrict__ gf, float *__restrict__ gi,
 float *__restrict__ go, float *__restrict__ gc0,
 int T, int G) {
  const int lane=threadIdx.x;
  const int t=T-1-lane;
  const int group=blockIdx.x%G, batch=blockIdx.x/G;
  float aa[9],bb[3],ff=1.f,ii=0.f;
  #pragma unroll
  for(int j=0;j<9;++j) aa[j]=(j%4==0)?1.f:0.f;
  bb[0]=0.f;bb[1]=0.f;bb[2]=0.f;
  if(t>=0) {
    int gidx=(batch*T+t)*G+group;
    int bidx=gidx*3;
    int out=(batch*T+t)*(G*3)+group*3;
    float nn=nstate[gidx];
    float denom=fmaxf(nn,1.e-12f);
    float dn=0.f;
    #pragma unroll
    for(int j=0;j<3;++j) {
      float sig=1.f/(1.f+expf(-o[out+j]));
      float grad=gh[out+j];
      float cc=cstate[bidx+j];
      bb[j]=grad*sig/denom;
      dn-=grad*sig*cc/(denom*denom);
      go[out+j]=grad*(cc/denom)*sig*(1.f-sig);
    }
    ii=(nn>1.e-12f)?dn:0.f;
    if(t+1<T) {
      ff=f[(batch*T+t+1)*G+group];
      int nextA=((batch*T+t+1)*G+group)*9;
      #pragma unroll
      for(int row=0;row<3;++row)
        #pragma unroll
        for(int col=0;col<3;++col)
          aa[row*3+col]=A[nextA+col*3+row]; // transpose
    }
  }
  extern __shared__ float sh[];
  for(int step=1;step<blockDim.x;step<<=1) {
    float *own=sh+lane*14;
    #pragma unroll
    for(int j=0;j<9;++j) own[j]=aa[j];
    #pragma unroll
    for(int j=0;j<3;++j) own[9+j]=bb[j];
    own[12]=ff;own[13]=ii;
    __syncthreads();
    if(lane>=step) {
      const float *prev=sh+(lane-step)*14;
      float na[9],nb[3];
      #pragma unroll
      for(int row=0;row<3;++row) {
        #pragma unroll
        for(int col=0;col<3;++col)
          na[row*3+col]=aa[row*3]*prev[col]
                         +aa[row*3+1]*prev[3+col]
                         +aa[row*3+2]*prev[6+col];
        nb[row]=bb[row]+aa[row*3]*prev[9]
                           +aa[row*3+1]*prev[10]
                           +aa[row*3+2]*prev[11];
      }
      float new_ii=ii+ff*prev[13];
      ff=ff*prev[12];ii=new_ii;
      #pragma unroll
      for(int j=0;j<9;++j) aa[j]=na[j];
      #pragma unroll
      for(int j=0;j<3;++j) bb[j]=nb[j];
    }
    __syncthreads();
  }
  if(t>=0) {
    int gidx=(batch*T+t)*G+group;
    int bidx=gidx*3;
    int aidx=gidx*9;
    int start=(batch*G+group)*3;
    float prevc[3], prevn;
    if(t==0) {
      prevc[0]=c0[start];prevc[1]=c0[start+1];prevc[2]=c0[start+2];
      prevn=1.f;
    } else {
      int previous=(batch*T+t-1)*G+group;
      prevc[0]=cstate[previous*3];
      prevc[1]=cstate[previous*3+1];
      prevc[2]=cstate[previous*3+2];
      prevn=nstate[previous];
    }
    #pragma unroll
    for(int row=0;row<3;++row) {
      gb[bidx+row]=bb[row];
      #pragma unroll
      for(int col=0;col<3;++col)
        ga[aidx+row*3+col]=bb[row]*prevc[col];
    }
    gf[gidx]=ii*prevn;
    gi[gidx]=ii;
    if(t==0) {
      #pragma unroll
      for(int col=0;col<3;++col) {
        gc0[start+col]=A[aidx+col]*bb[0]
                       +A[aidx+3+col]*bb[1]
                       +A[aidx+6+col]*bb[2];
      }
    }
  }
}