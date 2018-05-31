/*
<DUALSPHYSICS>  Copyright (C) 2013 by Jose M. Dominguez, Dr Alejandro Crespo, Prof. M. Gomez Gesteira, Anxo Barreiro, Ricardo Canelas
                                      Dr Benedict Rogers, Dr Stephen Longshaw, Dr Renato Vacondio

EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

This file is part of DualSPHysics. 

DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your option) any later version. 

DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. 

You should have received a copy of the GNU General Public License, along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>. 
*/

/// \file JSphGpu_ker.cu \brief Implements functions and CUDA kernels for the particle interaction and system update..

#include "JSphGpu_ker.h"
#include <float.h>
#include "JLog2.h"
#include <math_constants.h>
//#include "JDgKerPrint.h"
//#include "JDgKerPrint_ker.h"

__constant__ StCteInteraction CTE;
__constant__ StPhaseCteg PHASECTE[2];
__constant__ StPhaseArrayg PHASEARRAY[2];

namespace cusph{

//==============================================================================
/// Checks error and ends execution.
//==============================================================================
#define CheckErrorCuda(text)  __CheckErrorCuda(text,__FILE__,__LINE__)
void __CheckErrorCuda(const char *text,const char *file,const int line){
  cudaError_t err=cudaGetLastError();
  if(cudaSuccess!=err){
    char cad[2048]; 
    sprintf(cad,"%s (CUDA error: %s -> %s:%i).\n",text,cudaGetErrorString(err),file,line); 
    throw std::string(cad);
  }
}

//==============================================================================
/// Returns dimensions of gridsize according to parameters.
//==============================================================================
dim3 GetGridSize(unsigned n,unsigned blocksize){
  dim3 sgrid;//=dim3(1,2,3);
  unsigned nb=unsigned(n+blocksize-1)/blocksize;//-Total number of blocks to be launched.
  sgrid.x=(nb<=65535? nb: unsigned(sqrt(float(nb))));
  sgrid.y=(nb<=65535? 1: unsigned((nb+sgrid.x-1)/sgrid.x));
  sgrid.z=1;
  return(sgrid);
}

//==============================================================================
/// Reduction using maximum of float values in shared memory for a warp.
//==============================================================================
template <unsigned blockSize> __device__ __forceinline__ void KerReduMaxFloatWarp(volatile float* sdat,unsigned tid) {
  if(blockSize>=64)sdat[tid]=max(sdat[tid],sdat[tid+32]);
  if(blockSize>=32)sdat[tid]=max(sdat[tid],sdat[tid+16]);
  if(blockSize>=16)sdat[tid]=max(sdat[tid],sdat[tid+8]);
  if(blockSize>=8)sdat[tid]=max(sdat[tid],sdat[tid+4]);
  if(blockSize>=4)sdat[tid]=max(sdat[tid],sdat[tid+2]);
  if(blockSize>=2)sdat[tid]=max(sdat[tid],sdat[tid+1]);
}

//==============================================================================
/// Accumulates the summation of n values of array dat[], storing the result in the beginning of res[].
/// As many positions of res[] as blocks are used, storing the final result in res[0]).
//==============================================================================
template <unsigned blockSize> __global__ void KerReduMaxFloat(unsigned n,unsigned ini,const float *dat,float *res){
  extern __shared__ float sdat[];
  unsigned tid=threadIdx.x;
  unsigned c=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
  //unsigned c=blockIdx.x*blockDim.x + threadIdx.x;
  sdat[tid]=(c<n? dat[c+ini]: -FLT_MAX);
  __syncthreads();
  if(blockSize>=512){ if(tid<256)sdat[tid]=max(sdat[tid],sdat[tid+256]);  __syncthreads(); }
  if(blockSize>=256){ if(tid<128)sdat[tid]=max(sdat[tid],sdat[tid+128]);  __syncthreads(); }
  if(blockSize>=128){ if(tid<64) sdat[tid]=max(sdat[tid],sdat[tid+64]);   __syncthreads(); }
  if(tid<32)KerReduMaxFloatWarp<blockSize>(sdat,tid);
  if(tid==0)res[blockIdx.y*gridDim.x + blockIdx.x]=sdat[0];
}


////==============================================================================
//// CUDA Kernel that computes Fa2=ace[].x*ace[].x+ace[].y*ace[].y+ace[].z*ace[].z
////------------------------------------------------------------------------------
//__global__ void KerCalcMiou(unsigned n,float *viscop,float *rhop,float *resu)
//{
//  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; 
//  if(p<n){
//    float rvisco=viscop[p];
//	float rrhop=rhop[p];
//	resu[p]=0.5f*rrhop/rvisco;
//  }
//}
//
////------------------------------------------------------------------------------
//void CalcMiou(unsigned n,float *viscop,float *rhop,float *resu){
//  dim3 sgrid=GetGridSize(n,SPHBSIZE);
//  cusph::KerCalcMiou<<<sgrid,SPHBSIZE>>>(n,viscop,rhop,resu);
//}
////==============================================================================

//==============================================================================
// CUDA Kernel that computes Fa2=ace[].x*ace[].x+ace[].y*ace[].y+ace[].z*ace[].z
//------------------------------------------------------------------------------
__global__ void KerCalcFa2(unsigned n,float3 *ace,float *fa2)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; 
  if(p<n){
    float3 race=ace[p];
    fa2[p]=sqrt(race.x*race.x+race.y*race.y+race.z*race.z);
  }
}

//------------------------------------------------------------------------------
void CalcFa2(unsigned n,float3 *ace,float *fa2){
  dim3 sgrid=GetGridSize(n,SPHBSIZE);
  cusph::KerCalcFa2<<<sgrid,SPHBSIZE>>>(n,ace,fa2);
}
//==============================================================================

//==============================================================================
/// Returns the maximum of an array, using resu[] as auxiliar array.
/// Size of resu[] msut be >= N/SPHBSIZE+1)+(N/(SPHBSIZE*SPHBSIZE)+SPHBSIZE)
//==============================================================================
float ReduMaxFloat(unsigned ndata,unsigned inidata,float* data,float* resu){
  //printf("[ReduMaxF ndata:%d  SPHBSIZE:%d]\n",ndata,SPHBSIZE);
  unsigned n=ndata,ini=inidata;
  unsigned smemSize=SPHBSIZE*sizeof(float);
  dim3 sgrid=GetGridSize(n,SPHBSIZE);
  unsigned n_blocks=sgrid.x*sgrid.y;
  //printf("n:%d  n_blocks:%d]\n",n,n_blocks);
  float *dat=data;
  float *resu1=resu,*resu2=resu+n_blocks;
  float *res=resu1;
  while(n>1){
    //printf("##>ReduMaxF n:%d  n_blocks:%d  ini:%d\n",n,n_blocks,ini);
    //printf("##>ReduMaxF>sgrid=(%d,%d,%d)\n",sgrid.x,sgrid.y,sgrid.z);
    KerReduMaxFloat<SPHBSIZE><<<sgrid,SPHBSIZE,smemSize>>>(n,ini,dat,res);
    //KerReduMaxF<SPHBSIZE><<<n_blocks,SPHBSIZE,smemSize>>>(n,dat,res);
    //CheckErrorCuda("#>ReduMaxF KerReduMaxF  failed.");
    n=n_blocks; ini=0;
    sgrid=GetGridSize(n,SPHBSIZE);  
    n_blocks=sgrid.x*sgrid.y;
    if(n>1){
      //n_blocks=(n+SPHBSIZE-1)/SPHBSIZE;
      dat=res; res=(dat==resu1? resu2: resu1); 
    }
  }
  float resf;
  if(ndata>1)cudaMemcpy(&resf,res,sizeof(float),cudaMemcpyDeviceToHost);
  else cudaMemcpy(&resf,data,sizeof(float),cudaMemcpyDeviceToHost);
  //CheckErrorCuda("#>ReduMaxF cudaMemcpy  failed.");
#ifdef DG_ReduMaxFloat
  if(1){ //-Checks the reduction <DEBUG>
    float *vdat=new float[ndata];
    cudaMemcpy(vdat,data+inidata,sizeof(float)*ndata,cudaMemcpyDeviceToHost);
    float maxi=vdat[0];
    //for(unsigned c=0;c<ndata;c++){ printf("ReduMaxF>vdat[%u]=%f\n",c,vdat[c]); }      
    for(unsigned c=1;c<ndata;c++)if(maxi<vdat[c])maxi=vdat[c];
    if(resf!=maxi){
      printf("ReduMaxF>ERRORRRRR... Maximo:; %f; %f\n",resf,maxi);
      printf("ReduMaxF>sgrid=(%d,%d,%d)\n",sgrid.x,sgrid.y,sgrid.z);
      exit(0);
    }
    delete[] vdat;
  }
#endif
  return(resf);
}

//==============================================================================
/// Stores constants for interaction in GPU.
//==============================================================================
void CteInteractionUp(const StCteInteraction *cte, const StPhaseCteg *phasecte, const StPhaseArrayg *phasearray,unsigned phases){
  cudaMemcpyToSymbol(CTE,cte,sizeof(StCteInteraction));

  //multiphase
  cudaMemcpyToSymbol(PHASECTE,phasecte,sizeof(StPhaseCteg)*phases);
  printf("PhaseCte copied to memory\n");
  cudaMemcpyToSymbol(PHASEARRAY,phasearray,sizeof(StPhaseArrayg)*phases);
  printf("PhaseArray copied to memory\n");

}

//------------------------------------------------------------------------------
/// Sets v[].y to zero.
//------------------------------------------------------------------------------
__global__ void KerResety(unsigned n,unsigned pini,float3 *v)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n)v[p+pini].y=0;
}

//------------------------------------------------------------------------------
/// Sets v[].y to zero.
//------------------------------------------------------------------------------
__global__ void KerResety(unsigned n,const unsigned *list,float3 *v)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n)v[list[p]].y=0;
}

//------------------------------------------------------------------------------
/// Prepares derivated variables for force computation.
//------------------------------------------------------------------------------
__global__ void KerPreInteraction_Forces(unsigned n,unsigned npb,const float3 *pos,const float3 *vel,const float *rhop,float4 *pospres,float4 *velrhop,float3 *ace,float3 gravity,float *press,const unsigned *idpm)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
	unsigned pp=idpm[p];
	float rrhop0=PHASEARRAY[pp].rho_ph;
	float gamma0=PHASEARRAY[pp].Gamma_ph;
	float b_ph=PHASEARRAY[pp].b_ph;
	float rrhop=rhop[p];
    float rpress=b_ph*(powf(rrhop/rrhop0,gamma0)-1.0f);
	
	press[p]=rpress;
    float3 r=pos[p];
    pospres[p]=make_float4(r.x,r.y,r.z,rpress);
    r=vel[p];
    velrhop[p]=make_float4(r.x,r.y,r.z,rrhop);
    ace[p]=(p<npb? make_float3(0,0,0): gravity);
  }
 }

//==============================================================================
/// Prepares variables for interaction between cells.
//==============================================================================
void PreInteraction_Forces(unsigned np,unsigned npb,const float3 *pos,const float3 *vel,const float *rhop,float4 *pospres,float4 *velrhop,float3 *ace,tfloat3 gravity,float *press, const unsigned *idpm){
  if(np){
    dim3 sgrid=GetGridSize(np,SPHBSIZE);
    KerPreInteraction_Forces <<<sgrid,SPHBSIZE>>> (np,npb,pos,vel,rhop,pospres,velrhop,ace,Float3(gravity),press,idpm);
  }
}

//------------------------------------------------------------------------------
/// Computes sub-particle stress tensor (Tau) for SPS turbulence model.
//------------------------------------------------------------------------------
template <bool floating> __global__ void KerSPSCalcTau(unsigned n,unsigned pini,float smag,float blin,const float *rhop,const word *code,const tsymatrix3f *csph,tsymatrix3f *tau)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; 
  if(p<n){
    const unsigned p1=p+pini;
    tsymatrix3f rcsph=csph[p1];
    const float pow1=rcsph.xx*rcsph.xx + rcsph.yy*rcsph.yy + rcsph.zz*rcsph.zz;
    const float prr=pow1+pow1 + rcsph.xy*rcsph.xy + rcsph.xz*rcsph.xz + rcsph.yz*rcsph.yz;
    const float visc_SPS=smag*sqrt(prr);
    const float div_u=rcsph.xx+rcsph.yy+rcsph.zz;
    const float sps_k=(2.0f/3.0f)*visc_SPS*div_u;
    const float sps_Blin=blin*prr;
    const float sumsps=-(sps_k+sps_Blin);
    const float twovisc_SPS=(visc_SPS+visc_SPS);
    float one_rho2=1.0f/rhop[p1];
    tsymatrix3f rtau;
    if(floating && CODE_GetType(code[p1])==CODE_TYPE_FLOATING)one_rho2=0;
    rtau.xx=one_rho2*(twovisc_SPS*rcsph.xx +sumsps);
    rtau.xy=one_rho2*(visc_SPS*rcsph.xy);
    rtau.xz=one_rho2*(visc_SPS*rcsph.xz);
    rtau.yy=one_rho2*(twovisc_SPS*rcsph.yy +sumsps);
    rtau.yz=one_rho2*(visc_SPS*rcsph.yz);
    rtau.zz=one_rho2*(twovisc_SPS*rcsph.zz +sumsps);
    tau[p1]=rtau;
  }
}

//==============================================================================
/// Computes sub-particle stress tensor (Tau) for SPS turbulence model.
//==============================================================================
void SPSCalcTau(bool floating,unsigned np,unsigned npb,float smag,float blin,const float *rhop,const word *code,const tsymatrix3f *csph,tsymatrix3f *tau){
  const unsigned npf=np-npb;
  if(npf){
    dim3 sgridf=GetGridSize(npf,SPHBSIZE);
    if(floating)KerSPSCalcTau<true>  <<<sgridf,SPHBSIZE>>> (npf,npb,smag,blin,rhop,code,csph,tau);
    else        KerSPSCalcTau<false> <<<sgridf,SPHBSIZE>>> (npf,npb,smag,blin,rhop,NULL,csph,tau);
  }
}


//##############################################################################
//# Kernels to compute forces without floating bodies.
//##############################################################################
//------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles (Bound-Fluid).
//------------------------------------------------------------------------------
template<TpKernel tkernel> __device__ __forceinline__ void KerInteractionForcesBoundBox
  (unsigned p1,const unsigned &pini,const unsigned &pfin,const float4 *pospres,const float4 *velrhop,const unsigned* idp
  ,float massf,float3 posp1,float3 velp1,float &arp1,float &visc)
{
  for(int p2=pini;p2<pfin;p2++){
    float4 pospres2=pospres[p2];
    float drx=posp1.x-pospres2.x;
    float dry=posp1.y-pospres2.y;
    float drz=posp1.z-pospres2.z;
    float rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=1e-18f){
      const float4 velrhop2=velrhop[p2];
      float frx,fry,frz;
      {//===== Kernel =====
        const float rad=sqrt(rr2);
        const float qq=rad/CTE.h;
        float fac;
        if(tkernel==KERNEL_Cubic){     //-Cubic kernel.
          const bool radgt=qq>1;
          float wqq2=(radgt? 2.0f-qq: qq); wqq2*=wqq2;
          fac=(radgt? CTE.cubic_c2*wqq2: (CTE.cubic_c1*qq+CTE.cubic_d1*wqq2))/rad;
        }
        if(tkernel==KERNEL_Wendland){  //-Wendland kernel.
          const float wqq1=1.f-0.5f*qq;
          fac=CTE.wendland_bwen*qq*wqq1*wqq1*wqq1/rad;
        } 
        frx=fac*drx; fry=fac*dry; frz=fac*drz;
      }

      float dvx=velp1.x-velrhop2.x, dvy=velp1.y-velrhop2.y, dvz=velp1.z-velrhop2.z;
      {//===== Viscosity ===== 
        const float dot=drx*dvx + dry*dvy + drz*dvz;
        const float dot_rr2=dot/(rr2+CTE.eta2);
        visc=max(dot_rr2,visc);   // <----- Reduction to only one value. 
      }
      //===== Density derivative =====
      arp1+=massf*(dvx*frx+dvy*fry+dvz*frz);
    }
  }
}
//------------------------------------------------------------------------------
/// Particle interaction Bound-Fluid.
//------------------------------------------------------------------------------
template<TpKernel tkernel,unsigned hdiv> __global__ void KerInteractionForcesBound
  (unsigned n,uint4 nc,const unsigned *cellpart,const int2 *begincell
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp,float *viscdt,float *ar)
{
  unsigned p1=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p1<n){
    float arp1=0,visc=0;
    float4 r=velrhop[p1];
    float3 velp1=make_float3(r.x,r.y,r.z);
    r=pospres[p1];
    float3 posp1=make_float3(r.x,r.y,r.z);
    const float massf=CTE.massf;
    const int cel=cellpart[p1];
    //-Gets limits of interaction.
    int cx=cel%nc.x;
    int cz=int(cel/(nc.w));
    int cy=int((cel%(nc.w))/nc.x);

    //-Code for hdiv 1 or 2.
    int cxini=cx-min(cx,hdiv);
    int cxfin=cx+min(nc.x-cx-1,hdiv)+1;
    int yini=cy-min(cy,hdiv);
    int yfin=cy+min(nc.y-cy-1,hdiv)+1;
    int zini=cz-min(cz,hdiv);
    int zfin=cz+min(nc.z-cz-1,hdiv)+1;

    //-Interaction of Boundary with Fluid.
    for(int z=zini;z<zfin;z++){
      int zmod=(nc.w)*z+(nc.w*nc.z+1);//-Adds Nct+1 that is the first cell of fluid.
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        unsigned pini,pfin=0;
        for(int x=cxini;x<cxfin;x++){
          int2 cbeg=begincell[x+ymod];
          if(cbeg.y){
            if(!pfin)pini=cbeg.x;
            pfin=cbeg.y;
          }
        }
        if(pfin)KerInteractionForcesBoundBox<tkernel>(p1,pini,pfin,pospres,velrhop,idp,massf,posp1,velp1,arp1,visc);
      }
    }
    //-Stores results.
    if(arp1 || visc){
      ar[p1]+=arp1;
      if(visc>viscdt[p1])viscdt[p1]=visc;
    }
  }
}

//------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles (Fluid-Fluid/Bound).
//------------------------------------------------------------------------------
template<TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph> __device__ __forceinline__ void KerInteractionForcesFluidBox
  (bool bound,unsigned p1,const unsigned &pini,const unsigned &pfin
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp,const tsymatrix3f *tau
  ,float massp2,float3 posp1,float3 velp1,float3 devsp1,float rhopp1,const tsymatrix3f &taup1
  ,float3 &acep1,float3 &vcor,float &arp1,float &visc,tsymatrix3f &csphp1,float &deltap1)
{
  for(int p2=pini;p2<pfin;p2++){
    float4 pospres2=pospres[p2];
    float drx=posp1.x-pospres2.x;
    float dry=posp1.y-pospres2.y;
    float drz=posp1.z-pospres2.z;
    float rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=1e-18f){
      const float4 velrhop2=velrhop[p2];
      const float prrhop2=pospres2.w/(velrhop2.w*velrhop2.w);
      float prs=devsp1.x+prrhop2;
      float wab,frx,fry,frz;
      {//===== Kernel =====
        const float rad=sqrt(rr2);
        const float qq=rad/CTE.h;
        float fac;
        if(tkernel==KERNEL_Cubic){     //-Cubic kernel.
          const bool radgt=qq>1;
          const float wqq1=(radgt? 2.0f-qq: qq);
          const float wqq2=wqq1*wqq1;
          const float wqq3=wqq2*wqq1;
          wab=(radgt? CTE.cubic_a24*wqq3: CTE.cubic_a2*(1.0f-1.5f*wqq2+0.75f*wqq3));
          fac=(radgt? CTE.cubic_c2*wqq2: (CTE.cubic_c1*qq+CTE.cubic_d1*wqq2))/rad;
          //-Tensile correction.
          float fab=wab*CTE.cubic_odwdeltap;
          fab*=fab; fab*=fab; //fab=fab^4
          prs+=fab*(devsp1.y+ prrhop2*(pospres2.w>0? 0.01f: -0.2f) );
        }
        if(tkernel==KERNEL_Wendland){  //-Wendland kernel.
          const float wqq=2.f*qq+1.f;
          const float wqq1=1.f-0.5f*qq;
          const float wqq2=wqq1*wqq1;
          wab=CTE.wendland_awen*wqq*wqq2*wqq2;
          fac=CTE.wendland_bwen*qq*wqq2*wqq1/rad;
        } 
        frx=fac*drx; fry=fac*dry; frz=fac*drz;
      }

      {//===== Aceleration ===== 
        const float p_vpm=-prs*massp2;
        acep1.x+=p_vpm*frx; acep1.y+=p_vpm*fry; acep1.z+=p_vpm*frz;
      }

      //===== Density derivative =====
      const float dvx=velp1.x-velrhop2.x, dvy=velp1.y-velrhop2.y, dvz=velp1.z-velrhop2.z;
      arp1+=massp2*(dvx*frx+dvy*fry+dvz*frz);

      const float csoun=velrhop2.w*CTE.overrhop0;  //const float csound=CTE.cs0*powf(rrhop*OVERRHOPCERO,3); 
      const float cbar=(devsp1.z+ CTE.cs0*(csoun*csoun*csoun) )*0.5f;
      //===== DeltaSPH =====
      if(tdelta==DELTA_DBC || tdelta==DELTA_DBCExt){
        const float rhop1over2=rhopp1/velrhop2.w;
        const float visc_densi=CTE.delta2h*cbar*(rhop1over2-1)/(rr2+CTE.eta2);
        const float dot3=(drx*frx+dry*fry+drz*frz);
        const float delta=visc_densi*dot3*massp2;
        deltap1=(bound? FLT_MAX: deltap1+delta);
      }

      float robar=(rhopp1+velrhop2.w)*0.5f;
      {//===== Viscosity ===== 
        const float dot=drx*dvx + dry*dvy + drz*dvz;
        const float dot_rr2=dot/(rr2+CTE.eta2);
        //-Artificial viscosity.
        if(tvisco==VISCO_Artificial && dot<0){
          const float amubar=CTE.h*dot_rr2;   //amubar=CTE.h*dot/(rr2+CTE.eta2);
          const float pi_visc=(-CTE.visco*cbar*amubar/robar)*massp2;
          acep1.x-=pi_visc*frx; acep1.y-=pi_visc*fry; acep1.z-=pi_visc*frz;
        }
        //-Laminar+SPS viscosity.
        if(tvisco==VISCO_LaminarSPS){ 
          const float temp=2.0f*CTE.visco/((rr2+CTE.eta2)*robar);
          const float vtemp=massp2*temp*(drx*frx+dry*fry+drz*frz);  
          acep1.x+=vtemp*dvx; acep1.y+=vtemp*dvy; acep1.z+=vtemp*dvz;
          // SPS turbulence model.
          tsymatrix3f tausum=taup1;
          if(tau){ //-Only with fluid.
            tausum=tau[p2];
            tausum.xx+=taup1.xx;
            tausum.xy+=taup1.xy;
            tausum.xz+=taup1.xz;
            tausum.yy+=taup1.yy;
            tausum.yz+=taup1.yz;
            tausum.zz+=taup1.zz;
          }
          acep1.x+=massp2*(tausum.xx*frx+tausum.xy*fry+tausum.xz*frz);
          acep1.y+=massp2*(tausum.xy*frx+tausum.yy*fry+tausum.yz*frz);
          acep1.z+=massp2*(tausum.xz*frx+tausum.yz*fry+tausum.zz*frz);
          // CSPH terms.
          const float volp2=-massp2/velrhop2.w;
          float dv=dvx*volp2; csphp1.xx+=dv*frx; csphp1.xy+=dv*fry; csphp1.xz+=dv*frz;
                dv=dvy*volp2; csphp1.xy+=dv*frx; csphp1.yy+=dv*fry; csphp1.yz+=dv*frz;
                dv=dvz*volp2; csphp1.xz+=dv*frx; csphp1.yz+=dv*fry; csphp1.zz+=dv*frz;
        }
        visc=max(dot_rr2,visc);  //ViscDt=max(dot/(rr2+Eta2),ViscDt); // <----- Reduction to only one value. 
      }

      //===== XSPH correction =====
      if(xsph){
        const float wab_rhobar=massp2*(wab/robar);
        vcor.x-=wab_rhobar * dvx;
        vcor.y-=wab_rhobar * dvy;
        vcor.z-=wab_rhobar * dvz;
      }
    }
  }
}
//------------------------------------------------------------------------------
/// Particle interaction Fluid-Fluid & Fluid-Bound.
//------------------------------------------------------------------------------
template<TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph,unsigned hdiv> __global__ void KerInteractionForcesFluid
  (unsigned n,unsigned pinit,uint4 nc,unsigned cellfluid,const unsigned *cellpart,const int2 *begincell
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp,const tsymatrix3f *tau
  ,float *viscdt,float *ar,float3 *ace,float3 *velxcor,tsymatrix3f *csph,float *delta)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    unsigned p1=p+pinit;     //-Number of particle.
    float arp1=0,visc=0,deltap1=0;
    float3 acep1=make_float3(0,0,0);
    float3 vcor;
    if(xsph)vcor=acep1;
    float4 r=velrhop[p1];
    float3 velp1=make_float3(r.x,r.y,r.z);
    float rhopp1=r.w;
    r=pospres[p1];
    float3 posp1=make_float3(r.x,r.y,r.z);
    const float csoun=rhopp1*CTE.overrhop0;
    float3 devsp1; devsp1.x=r.w/(rhopp1*rhopp1); devsp1.y=devsp1.x*(r.w>0? 0.01f: -0.2f); devsp1.z=CTE.cs0*(csoun*csoun*csoun);
    tsymatrix3f taup1=tau[p1];
    tsymatrix3f csphp1={0,0,0,0,0,0};
    const int cel=cellpart[p1]-cellfluid; //-Substracts where the cells of fluid start.
//  if(ik&&p==0){ KPrintt(ik,"id:",idp[p1]," cel:",true); KPrint(ik,cel); }
    //-Gets limits of interaction.
    int cx=cel%nc.x;
    int cz=int(cel/(nc.w));
    int cy=int((cel%(nc.w))/nc.x);

    //-Code for hdiv 1 or 2.
    int cxini=cx-min(cx,hdiv);
    int cxfin=cx+min(nc.x-cx-1,hdiv)+1;
    int yini=cy-min(cy,hdiv);
    int yfin=cy+min(nc.y-cy-1,hdiv)+1;
    int zini=cz-min(cz,hdiv);
    int zfin=cz+min(nc.z-cz-1,hdiv)+1;

    //-Interaction with Fluids.
    for(int z=zini;z<zfin;z++){
      int zmod=(nc.w)*z+cellfluid; //-Adds when the cells of fluid start.
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        unsigned pini,pfin=0;
        for(int x=cxini;x<cxfin;x++){
          int2 cbeg=begincell[x+ymod];
          if(cbeg.y){
            if(!pfin)pini=cbeg.x;
            pfin=cbeg.y;
          }
        }
        if(pfin)KerInteractionForcesFluidBox<tdelta,tkernel,tvisco,xsph> (false,p1,pini,pfin,pospres,velrhop,idp,tau,CTE.massf,posp1,velp1,devsp1,rhopp1,taup1,acep1,vcor,arp1,visc,csphp1,deltap1);
      }
    }
    //-Interaction with Boundary.
    for(int z=zini;z<zfin;z++){
      int zmod=(nc.w)*z;
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        unsigned pini,pfin=0;
        for(int x=cxini;x<cxfin;x++){
          int2 cbeg=begincell[x+ymod];
          if(cbeg.y){
            if(!pfin)pini=cbeg.x;
            pfin=cbeg.y;
          }
        }
        if(pfin)KerInteractionForcesFluidBox<tdelta,tkernel,tvisco,xsph> (true,p1,pini,pfin,pospres,velrhop,idp,NULL,CTE.massb,posp1,velp1,devsp1,rhopp1,taup1,acep1,vcor,arp1,visc,csphp1,deltap1);
      }
    }
    //-Stores resutls.
    if(arp1 || acep1.x || acep1.y || acep1.z || visc){
      if(tdelta==DELTA_DBC && deltap1!=FLT_MAX)arp1+=deltap1;
      if(tdelta==DELTA_DBCExt){
        float rdelta=delta[p1];
        delta[p1]=(rdelta==FLT_MAX || deltap1==FLT_MAX? FLT_MAX: rdelta+deltap1);
      }
      ar[p1]+=arp1;
      float3 k=ace[p1]; k.x+=acep1.x; k.y+=acep1.y; k.z+=acep1.z; ace[p1]=k;
      if(xsph){
        k=velxcor[p1]; k.x+=vcor.x; k.y+=vcor.y; k.z+=vcor.z; velxcor[p1]=k;
      }
      if(visc>viscdt[p1])viscdt[p1]=visc;
      if(tvisco==VISCO_LaminarSPS)csph[p1]=csphp1;
    }
  }
}

//==============================================================================
/// Interaction to compute forces.
//==============================================================================
template<TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph> void InteractionForces(TpCellMode cellmode,unsigned bsbound,unsigned bsfluid
  ,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells,const unsigned *cellpart,const int2 *begincell
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp,const tsymatrix3f* tau
  ,float *viscdt,float* ar,float3 *ace,float3 *velxcor,tsymatrix3f* csph,float *delta,bool simulate2d)
{
  const unsigned npf=np-npb;
  uint4 nc=make_uint4(ncells.x,ncells.y,ncells.z,ncells.x*ncells.y);
  //-Interaction Fluid-Fluid & Fluid-Bound.
  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    if(cellmode==CELLMODE_H)KerInteractionForcesFluid<tdelta,tkernel,tvisco,xsph,2> <<<sgridf,bsfluid>>> (npf,npb,nc,nc.w*nc.z+1,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta);
    else                    KerInteractionForcesFluid<tdelta,tkernel,tvisco,xsph,1> <<<sgridf,bsfluid>>> (npf,npb,nc,nc.w*nc.z+1,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta);
  }
  //-Interaction Boundary-Fluid.
  if(npbok){
    dim3 sgridb=GetGridSize(npbok,bsbound);
    if(cellmode==CELLMODE_H)KerInteractionForcesBound<tkernel,2> <<<sgridb,bsbound>>> (npbok,nc,cellpart,begincell,pospres,velrhop,idp,viscdt,ar);
    else                    KerInteractionForcesBound<tkernel,1> <<<sgridb,bsbound>>> (npbok,nc,cellpart,begincell,pospres,velrhop,idp,viscdt,ar);
  }
  //-For 2D simulations, second component (.y) is removed.
  if(simulate2d && npf){
    dim3 sgrid=GetGridSize(npf,SPHBSIZE);
    KerResety <<<sgrid,SPHBSIZE>>> (npf,npb,ace);
  }
}
//==============================================================================
void Interaction_Forces(TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph,TpCellMode cellmode,unsigned bsbound,unsigned bsfluid
  ,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells,const unsigned *cellpart,const int2 *begincell
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp,const tsymatrix3f* tau
  ,float *viscdt,float* ar,float3 *ace,float3 *velxcor,tsymatrix3f* csph,float *delta,bool simulate2d)
{
  if(tdelta==DELTA_None){ const TpDeltaSph tdel=DELTA_None;
    if(tkernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)InteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)InteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
    else if(tkernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)InteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)InteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
  }
  else if(tdelta==DELTA_DBC){ const TpDeltaSph tdel=DELTA_DBC;
    if(tkernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)InteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)InteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
    else if(tkernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)InteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)InteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
  }
  else if(tdelta==DELTA_DBCExt){ const TpDeltaSph tdel=DELTA_DBCExt;
    if(tkernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)InteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)InteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
    else if(tkernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)InteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)InteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
  }
}

//---------------------------**********************************MULTIPHASE BEGIN**********************************---------------------------
//##############################################################################
//# Kernels to compute forces with multi-phase model.
//##############################################################################



//------------------------------------------------------------------------------
/// Computes yielding and zeros ace and some other stuff
//------------------------------------------------------------------------------
template <bool gswitch> __global__ void KerYieldResetAce(unsigned n,unsigned pini,tsymatrix3f *vtau,float *viscop,float3 *ace,unsigned *idpm, float3 *vel,float *ar)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; 
  if(p<n){
    const unsigned p1=p;//+pini;
	float tauy=viscop[p1];
	tsymatrix3f rvtau=vtau[p1];
	//Invariant
	const float pow1=rvtau.xx*rvtau.xx + rvtau.yy*rvtau.yy + rvtau.zz*rvtau.zz;
    const float prr=pow1+pow1 + rvtau.xy*rvtau.xy + rvtau.xz*rvtau.xz + rvtau.yz*rvtau.yz;
	const float vtau2inv=sqrt(prr);

	if (tauy>vtau2inv){
	  //Ace debug {0,0,-9.81}
	  ace[p].x=0.f;
	  ace[p].y=0.f;
	  ace[p].z=0.f;

	  //vel[p].x=0.f; //debug
	  //vel[p].y=0.f;
	  //vel[p].z=0.f;
	  
	  //ar[p]=0.0f; //debug
	}

	viscop[p1]=vtau2inv; //debug
  }
}
//==============================================================================
///  Computes yielding and zeros ace and some other stuff
//==============================================================================
void YieldResetAce(unsigned np,unsigned pini,tsymatrix3f *vtau,float *viscop,float3 *ace,unsigned *idpm, float3 *vel,float *ar){
  if(np){
	//unsigned npf=np-pini; //debug do them all as with preyield
    dim3 sgrid=GetGridSize(np,SPHBSIZE);
    KerYieldResetAce<false> <<<sgrid,SPHBSIZE>>> (np,pini,vtau,viscop,ace,idpm,vel,ar);
  }
}

//------------------------------------------------------------------------------
/// Prepares derivated variables for force computation.
//------------------------------------------------------------------------------
__global__ void KerPreYieldStress(unsigned n,unsigned pini,const float *rhop,float4 *pospres,const unsigned *idpm,float *viscop)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
	const unsigned p1=p;//+pini;
	const unsigned pp1=idpm[p1];
	const float pressat_temp=pospres[p1].w;
	float pressat=(pressat_temp>0.f?pressat_temp:0.f);

	const float tauyield=PHASECTE[pp1].DP_alpha*pressat+PHASECTE[pp1].DP_kappa*PHASECTE[pp1].coh;
	viscop[p1]=tauyield;
  }
 }

//==============================================================================
/// Prepares variables for interaction between cells.
//==============================================================================
void PreYieldStress(unsigned np,unsigned npb,const float *rhop,float4 *pospres,const unsigned *idpm,float *viscop){
  if(np){
	//unsigned npf=np-npb;//debug do them all np to make sure you have no mistakes
    dim3 sgrid=GetGridSize(np,SPHBSIZE);
    KerPreYieldStress <<<sgrid,SPHBSIZE>>> (np,npb,rhop,pospres,idpm,viscop);
  }
}

//------------------Stress Begin------------------
//------------------------------------------------------------------------------
/// Computes sub-particle stress tensor (Tau) for SPS turbulence model for the shear Stress model only
//------------------------------------------------------------------------------
template <bool gswitch> __global__ void KerSPSCalcStressTau(unsigned n,unsigned pini,const float4 *velrhop/*dont need*/,tsymatrix3f *csph, tsymatrix3f *tau)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; 
  if(p<n){
    const unsigned p1=p+pini;
    tsymatrix3f rcsph=csph[p1];
    const float pow1=rcsph.xx*rcsph.xx + rcsph.yy*rcsph.yy + rcsph.zz*rcsph.zz;
    const float prr=pow1+pow1 + rcsph.xy*rcsph.xy + rcsph.xz*rcsph.xz + rcsph.yz*rcsph.yz;
	const float visc_SPS=CTE.spssmag*sqrt(prr);//cte
    const float div_u=rcsph.xx+rcsph.yy+rcsph.zz;
    const float sps_k=(2.0f/3.0f)*visc_SPS*div_u;
    const float sps_Blin=CTE.spsblin*prr; //cte
    const float sumsps=-(sps_k+sps_Blin);
    const float twovisc_SPS=(visc_SPS+visc_SPS);
    //float one_rho2=1.0f/rhop[p1];
	//float rhop1=velrhop[p1].w;
    tsymatrix3f rtau;
    //if(floating && CODE_GetType(code[p1])==CODE_TYPE_FLOATING)rhop1=0; // No floating -> sorry
    rtau.xx=(twovisc_SPS*rcsph.xx +sumsps);
    rtau.xy=(visc_SPS*rcsph.xy);
    rtau.xz=(visc_SPS*rcsph.xz);
    rtau.yy=(twovisc_SPS*rcsph.yy +sumsps);
    rtau.yz=(visc_SPS*rcsph.yz);
    rtau.zz=(twovisc_SPS*rcsph.zz +sumsps);
    tau[p1]=rtau;
  }
}

//------------------------------------------------------------------------------
/// Computes shear stress tensor (vTau) for the Shear stress model only
//------------------------------------------------------------------------------
template <bool gswitch> __global__ void KerShearCalcvTau(unsigned n,unsigned pini,const float4 *velrhop/*dont need*/,tsymatrix3f *csph, tsymatrix3f *vtau,const unsigned *idpm,float *viscop,float *cv)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; 
  if(p<n){
    const unsigned p1=p;//+pini;
	unsigned pp=idpm[p1];
	tsymatrix3f rcsph=csph[p1];
    tsymatrix3f rvtau;
	const float div_u=rcsph.xx+rcsph.yy+rcsph.zz;

	//**variable tu
	const float miou=PHASECTE[pp].visco; //viscosity or consistency of the phase
	const float cvp1=cv[p1];			 //volumetric concentration
	const float n_param=PHASECTE[pp].hb_n;
	const float m_param=PHASECTE[pp].hbp_m;
    const float twothirds=2.f/3.f;
    const float tauy=(PHASECTE[pp].ys!=0.0f? PHASECTE[pp].ys: viscop[p1]);

    ////***Newtonian constitutive model***				//if you need to run newtonian/newtonian or debug
    //rvtau.xx=miou*(2.f*rcsph.xx-twothirds*div_u); 
    //rvtau.xy=miou*(rcsph.xy);
    //rvtau.xz=miou*(rcsph.xz);
    //rvtau.yy=miou*(2.f*rcsph.yy-twothirds*div_u);
    //rvtau.yz=miou*(rcsph.yz);								
    //rvtau.zz=miou*(2.f*rcsph.zz-twothirds*div_u);
    //vtau[p1]=rvtau;

    //***Non Newtonian***** 
	//II invariant of deformation
	const float pow1=rcsph.xx*rcsph.xx + rcsph.yy*rcsph.yy + rcsph.zz*rcsph.zz;
    const float prr=pow1+pow1 + rcsph.xy*rcsph.xy + rcsph.xz*rcsph.xz + rcsph.yz*rcsph.yz;
	float IIdef_temp=sqrt(prr);
	float IIdef=(IIdef_temp==0.f? FLT_MIN : IIdef_temp); // just in case CUDA through a NaN (remove if you are brave)
	
    //***HB-Papanastasiou constitutive model (HBP)***
	float IIdef_n_exp=pow(IIdef,n_param-1.f);
	float Pap_m_exp=1-exp(-m_param*IIdef);
	float htaHBP=(tauy/IIdef*Pap_m_exp + miou*IIdef_n_exp); 
	htaHBP=(htaHBP!=htaHBP?miou+tauy:htaHBP); // just in case CUDA through a NaN (remove if you are brave)

	if (cvp1<0.3f && PHASECTE[pp].sed==1){
	  htaHBP=CTE.minvisc*exp(2.5f*cvp1/(1-0.609f*cvp1));
	}
	
    rvtau.xx=htaHBP*(2.f*rcsph.xx-twothirds*div_u);
    rvtau.xy=htaHBP*(rcsph.xy);								
    rvtau.xz=htaHBP*(rcsph.xz);								
    rvtau.yy=htaHBP*(2.f*rcsph.yy-twothirds*div_u);
    rvtau.yz=htaHBP*(rcsph.yz);								
    rvtau.zz=htaHBP*(2.f*rcsph.zz-twothirds*div_u);
    vtau[p1]=rvtau;
	
	//viscop[p1]=cvp1; //debug for output only
	viscop[p1]=htaHBP; //debug for output only (this should go to the ViscoDt calc in the JSphGpuSingle::Interaction_Forces )

  }
}

//------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles (Fluid-Fluid/Bound), Stress computation
//------------------------------------------------------------------------------
template<TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph> __device__ __forceinline__ void KerInteractionMultiStressFluidBox
  (bool bound,unsigned p1,const unsigned &pini,const unsigned &pfin
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp,const unsigned *idpm,const tsymatrix3f *tau,tsymatrix3f *vtau
  ,float massp1,float3 posp1,float3 velp1,/*float3 devsp1,*/float rhopp1, tsymatrix3f taup1,tsymatrix3f vtaup1
  ,float3 &acep1)
{
  for(int p2=pini;p2<pfin;p2++){
    float4 pospres2=pospres[p2];
    float drx=posp1.x-pospres2.x;
    float dry=posp1.y-pospres2.y;
    float drz=posp1.z-pospres2.z;
    float rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=1e-18f){
	  //multi
	  unsigned pp2=idpm[p2];
	  float massp2=PHASEARRAY[pp2].mass_ph;
      const float4 velrhop2=velrhop[p2];
      
      float /*wab,*/frx,fry,frz;
      {//===== Kernel =====
        const float rad=sqrt(rr2);
        const float qq=rad/CTE.h;
        float fac;
        if(tkernel==KERNEL_Cubic){     //-Cubic kernel.
          const bool radgt=qq>1;
          const float wqq1=(radgt? 2.0f-qq: qq);
          const float wqq2=wqq1*wqq1;
          //const float wqq3=wqq2*wqq1;
          //wab=(radgt? CTE.cubic_a24*wqq3: CTE.cubic_a2*(1.0f-1.5f*wqq2+0.75f*wqq3));
          fac=(radgt? CTE.cubic_c2*wqq2: (CTE.cubic_c1*qq+CTE.cubic_d1*wqq2))/rad;
         
        }
        if(tkernel==KERNEL_Wendland){  //-Wendland kernel.
          //const float wqq=2.f*qq+1.f;
          const float wqq1=1.f-0.5f*qq;
          const float wqq2=wqq1*wqq1;
          //wab=CTE.wendland_awen*wqq*wqq2*wqq2;
          fac=CTE.wendland_bwen*qq*wqq2*wqq1/rad;
        } 
        frx=fac*drx; fry=fac*dry; frz=fac*drz;
      }

      {//===== Viscosity ===== 
        const float mass_over_rhopp12=massp2/(rhopp1*velrhop2.w);		
		//Shear stresses
		tsymatrix3f vtausum=vtaup1;
		//if (!bound){ //debug
		  vtausum=vtau[p2];
          vtausum.xx+=vtaup1.xx; 
		  vtausum.xy+=vtaup1.xy; 
		  vtausum.xz+=vtaup1.xz;	   
		  vtausum.yy+=vtaup1.yy; 
		  vtausum.yz+=vtaup1.yz;
		  vtausum.zz+=vtaup1.zz;
		//}

		acep1.x+=mass_over_rhopp12*(vtausum.xx*frx+vtausum.xy*fry+vtausum.xz*frz); //expensive and same as with SPS -> combine shear & SPS
        acep1.y+=mass_over_rhopp12*(vtausum.xy*frx+vtausum.yy*fry+vtausum.yz*frz);
        acep1.z+=mass_over_rhopp12*(vtausum.xz*frx+vtausum.yz*fry+vtausum.zz*frz);

		//Coment out for debug
        //// SPS turbulence model.
        //tsymatrix3f tausum=taup1;
        //if(tau){ //-Only with fluid.
        //  tausum=tau[p2];
        //  tausum.xx+=taup1.xx;
        //  tausum.xy+=taup1.xy;
        //  tausum.xz+=taup1.xz;
        //  tausum.yy+=taup1.yy;
        //  tausum.yz+=taup1.yz;
        //  tausum.zz+=taup1.zz;
        //}
		
        //acep1.x+=mass_over_rhopp12*(tausum.xx*frx+tausum.xy*fry+tausum.xz*frz);
        //acep1.y+=mass_over_rhopp12*(tausum.xy*frx+tausum.yy*fry+tausum.yz*frz);
        //acep1.z+=mass_over_rhopp12*(tausum.xz*frx+tausum.yz*fry+tausum.zz*frz);
      }
    }
  }
}

//------------------------------------------------------------------------------
/// Particle interaction Fluid-Fluid & Fluid-Bound, Stress computation 
//------------------------------------------------------------------------------
template<TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph,unsigned hdiv> __global__ void KerInteractionMultiStressFluid
  (unsigned n,unsigned pinit,uint4 nc,unsigned cellfluid,const unsigned *cellpart,const int2 *begincell
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp,const unsigned *idpm,float3 *ace,const tsymatrix3f *tau,tsymatrix3f *vtau)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    unsigned p1=p+pinit;     //-Number of particle.
    //float visc=0;
    float3 acep1=make_float3(0,0,0);
	
	//multi
	unsigned pp1=idpm[p1];
	float massp1=PHASEARRAY[pp1].mass_ph;

    float4 r=velrhop[p1];
    float3 velp1=make_float3(r.x,r.y,r.z);
    float rhopp1=r.w;
    r=pospres[p1];
    float3 posp1=make_float3(r.x,r.y,r.z);
    //const float csoun=rhopp1*CTE.overrhop0;
    //float3 devsp1; devsp1.x=r.w/(rhopp1*rhopp1); devsp1.y=devsp1.x*(r.w>0? 0.01f: -0.2f); devsp1.z=CTE.cs0*(csoun*csoun*csoun);
    
	//Stresses
	tsymatrix3f taup1=tau[p1];
	tsymatrix3f vtaup1=vtau[p1]; //or ={0,0,0,0,0,0};
   
    const int cel=cellpart[p1]-cellfluid; //-Substracts where the cells of fluid start.
//  if(ik&&p==0){ KPrintt(ik,"id:",idp[p1]," cel:",true); KPrint(ik,cel); }
    //-Gets limits of interaction.
    int cx=cel%nc.x;
    int cz=int(cel/(nc.w));
    int cy=int((cel%(nc.w))/nc.x);

    //-Code for hdiv 1 or 2.
    int cxini=cx-min(cx,hdiv);
    int cxfin=cx+min(nc.x-cx-1,hdiv)+1;
    int yini=cy-min(cy,hdiv);
    int yfin=cy+min(nc.y-cy-1,hdiv)+1;
    int zini=cz-min(cz,hdiv);
    int zfin=cz+min(nc.z-cz-1,hdiv)+1;

    //-Interaction with Fluids.
    for(int z=zini;z<zfin;z++){
      int zmod=(nc.w)*z+cellfluid; //-Adds when the cells of fluid start.
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        unsigned pini,pfin=0;
        for(int x=cxini;x<cxfin;x++){
          int2 cbeg=begincell[x+ymod];
          if(cbeg.y){
            if(!pfin)pini=cbeg.x;
            pfin=cbeg.y;
          }
        }
		if(pfin)KerInteractionMultiStressFluidBox<tdelta,tkernel,tvisco,xsph> (false,p1,pini,pfin,pospres,velrhop,idp,idpm,tau,vtau,massp1,posp1,velp1,/*devsp1,*/rhopp1,taup1,vtaup1,acep1);
      }
    }
    //-Interaction with Boundary.
    for(int z=zini;z<zfin;z++){
      int zmod=(nc.w)*z;
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        unsigned pini,pfin=0;
        for(int x=cxini;x<cxfin;x++){
          int2 cbeg=begincell[x+ymod];
          if(cbeg.y){
            if(!pfin)pini=cbeg.x;
            pfin=cbeg.y;
          }
        }
        if(pfin)KerInteractionMultiStressFluidBox<tdelta,tkernel,tvisco,xsph> (true,p1,pini,pfin,pospres,velrhop,idp,idpm,NULL,vtau,massp1,posp1,velp1,/*,devsp1,*/rhopp1,taup1,vtaup1,acep1);
      }
    }
    //-Stores resutls.
    if( acep1.x || acep1.y || acep1.z ){
      float3 k=ace[p1]; k.x+=acep1.x; k.y+=acep1.y; k.z+=acep1.z; ace[p1]=k;
    }
  }
}
//------------------Stress End------------------

//------------------Press Begin-----------------
//------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles (Bound-Fluid).
//------------------------------------------------------------------------------
template<TpKernel tkernel> __device__ __forceinline__ void KerInteractionMultiForcesBoundBox
  (unsigned p1,const unsigned &pini,const unsigned &pfin,const float4 *pospres,const float4 *velrhop,const unsigned* idp, const unsigned *idpm,float rhopp1
  ,float3 posp1,float3 velp1,float &arp1,tsymatrix3f &csphp1)
{
  for(int p2=pini;p2<pfin;p2++){
    float4 pospres2=pospres[p2];
    float drx=posp1.x-pospres2.x;
    float dry=posp1.y-pospres2.y;
    float drz=posp1.z-pospres2.z;
    float rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=1e-18f){
	  //multi
	  const unsigned pp2=idpm[p2];
	  const float massp2=PHASEARRAY[pp2].mass_ph;

      const float4 velrhop2=velrhop[p2];
      float frx,fry,frz;
      {//===== Kernel =====
        const float rad=sqrt(rr2);
        const float qq=rad/CTE.h;
        float fac;
        if(tkernel==KERNEL_Cubic){     //-Cubic kernel.
          const bool radgt=qq>1;
          float wqq2=(radgt? 2.0f-qq: qq); wqq2*=wqq2;
          fac=(radgt? CTE.cubic_c2*wqq2: (CTE.cubic_c1*qq+CTE.cubic_d1*wqq2))/rad;
        }
        if(tkernel==KERNEL_Wendland){  //-Wendland kernel.
          const float wqq1=1.f-0.5f*qq;
          fac=CTE.wendland_bwen*qq*wqq1*wqq1*wqq1/rad;
        } 
        frx=fac*drx; fry=fac*dry; frz=fac*drz;
      }

      float dvx=velp1.x-velrhop2.x, dvy=velp1.y-velrhop2.y, dvz=velp1.z-velrhop2.z;
     
      //===== Density derivative =====
      arp1+=massp2*(dvx*frx+dvy*fry+dvz*frz)*rhopp1/velrhop2.w;

	  {// CSPH terms.
        const float volp2=-massp2/velrhop2.w;
        float dv=dvx*volp2; csphp1.xx+=dv*frx; csphp1.xy+=dv*fry; csphp1.xz+=dv*frz;
              dv=dvy*volp2; csphp1.xy+=dv*frx; csphp1.yy+=dv*fry; csphp1.yz+=dv*frz;
              dv=dvz*volp2; csphp1.xz+=dv*frx; csphp1.yz+=dv*fry; csphp1.zz+=dv*frz;
	  }
    }
  }
}

//------------------------------------------------------------------------------
/// Particle interaction Bound-Fluid.
//------------------------------------------------------------------------------
template<TpKernel tkernel,unsigned hdiv> __global__ void KerInteractionMultiForcesBound
  (unsigned n,uint4 nc,const unsigned *cellpart,const int2 *begincell
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp,float *viscdt,float *ar,const unsigned *idpm,tsymatrix3f *csph)
{
  unsigned p1=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p1<n){
    float arp1=0,visc=0;
    float4 r=velrhop[p1];
    float3 velp1=make_float3(r.x,r.y,r.z);
	float rhopp1=r.w;
    r=pospres[p1];
    float3 posp1=make_float3(r.x,r.y,r.z);
	tsymatrix3f csphp1={0,0,0,0,0,0};
	
    const int cel=cellpart[p1];
    //-Gets limits of interaction.
    int cx=cel%nc.x;
    int cz=int(cel/(nc.w));
    int cy=int((cel%(nc.w))/nc.x);

    //-Code for hdiv 1 or 2.
    int cxini=cx-min(cx,hdiv);
    int cxfin=cx+min(nc.x-cx-1,hdiv)+1;
    int yini=cy-min(cy,hdiv);
    int yfin=cy+min(nc.y-cy-1,hdiv)+1;
    int zini=cz-min(cz,hdiv);
    int zfin=cz+min(nc.z-cz-1,hdiv)+1;

    //-Interaction of Boundary with Fluid.
    for(int z=zini;z<zfin;z++){
      int zmod=(nc.w)*z+(nc.w*nc.z+1);//-Adds Nct+1 that is the first cell of fluid.
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        unsigned pini,pfin=0;
        for(int x=cxini;x<cxfin;x++){
          int2 cbeg=begincell[x+ymod];
          if(cbeg.y){
            if(!pfin)pini=cbeg.x;
            pfin=cbeg.y;
          }
        }
        if(pfin)KerInteractionMultiForcesBoundBox<tkernel>(p1,pini,pfin,pospres,velrhop,idp,idpm,rhopp1,posp1,velp1,arp1,csphp1);
      }
    }
    //-Stores results.
    if(arp1 || visc){
      ar[p1]+=arp1;
      if(visc>viscdt[p1])viscdt[p1]=visc;
	  csph[p1]=csphp1;
    }
  }
}

//------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles (Fluid-Fluid/Bound).
//------------------------------------------------------------------------------
template<TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph> __device__ __forceinline__ void KerInteractionMultiForcesFluidBox
  (bool bound,unsigned p1,const unsigned &pini,const unsigned &pfin
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp,const unsigned *idpm,const unsigned pp1,float massp1
  ,float3 posp1,float3 velp1,float3 devsp1,float rhopp1,float3 &acep1,float3 &vcor,float &arp1,tsymatrix3f &csphp1,float &deltap1,float &visc,float &cvs,float &cvt)
{
  for(int p2=pini;p2<pfin;p2++){
    float4 pospres2=pospres[p2];
    float drx=posp1.x-pospres2.x;
    float dry=posp1.y-pospres2.y;
    float drz=posp1.z-pospres2.z;
    float rr2=drx*drx+dry*dry+drz*drz;
    
	if(rr2<=CTE.fourh2 && rr2>=1e-18f){
      //multi
	  const unsigned pp2=idpm[p2];
	  const float massp2=PHASEARRAY[pp2].mass_ph;
	  //press=p1/r1r2+p2/r1r2
	  const float4 velrhop2=velrhop[p2];
      const float prrhop2=pospres2.w/(rhopp1*velrhop2.w);
      float prs=devsp1.x/(rhopp1*velrhop2.w)+prrhop2;

      float wab,frx,fry,frz;
      {//===== Kernel =====
        const float rad=sqrt(rr2);
        const float qq=rad/CTE.h;
        float fac;
        if(tkernel==KERNEL_Cubic){     //-Cubic kernel.
          const bool radgt=qq>1;
          const float wqq1=(radgt? 2.0f-qq: qq);
          const float wqq2=wqq1*wqq1;
          const float wqq3=wqq2*wqq1;
          wab=(radgt? CTE.cubic_a24*wqq3: CTE.cubic_a2*(1.0f-1.5f*wqq2+0.75f*wqq3));
          fac=(radgt? CTE.cubic_c2*wqq2: (CTE.cubic_c1*qq+CTE.cubic_d1*wqq2))/rad;
          //-Tensile correction.
          float fab=wab*CTE.cubic_odwdeltap;
          fab*=fab; fab*=fab; //fab=fab^4
          prs+=fab*(devsp1.y/(rhopp1*velrhop2.w)+ prrhop2*(pospres2.w>0? 0.01f: -0.2f) );
        }
        if(tkernel==KERNEL_Wendland){  //-Wendland kernel.
          const float wqq=2.f*qq+1.f;
          const float wqq1=1.f-0.5f*qq;
          const float wqq2=wqq1*wqq1;
          wab=CTE.wendland_awen*wqq*wqq2*wqq2;
          fac=CTE.wendland_bwen*qq*wqq2*wqq1/rad;
        } 
        frx=fac*drx; fry=fac*dry; frz=fac*drz;
      }

      {//===== Aceleration ===== 
        const float p_vpm=-prs*massp2;
        acep1.x+=p_vpm*frx; acep1.y+=p_vpm*fry; acep1.z+=p_vpm*frz;
      }

      //===== Density derivative =====
      const float dvx=velp1.x-velrhop2.x, dvy=velp1.y-velrhop2.y, dvz=velp1.z-velrhop2.z;
	  arp1+=massp2*(dvx*frx+dvy*fry+dvz*frz)*rhopp1/velrhop2.w;

      //const float csoun=velrhop2.w*CTE.overrhop0;  //const float csound=CTE.cs0*powf(rrhop*OVERRHOPCERO,3); 
	  //const float cs0p2=PHASEARRAY[pp2].Cs0_ph;
      //const float cbar=(devsp1.z+ cs0p2 )*0.5f; // *no need never different phase in DSPH*
      //===== DeltaSPH =====
      if(tdelta==DELTA_DBC || tdelta==DELTA_DBCExt){
		if (pp1==pp2){
        const float rhop1over2=rhopp1/velrhop2.w;
        const float visc_densi=CTE.delta2h*devsp1.z*(rhop1over2-1)/(rr2+CTE.eta2);
        const float dot3=(drx*frx+dry*fry+drz*frz);
        const float delta=visc_densi*dot3*massp2;
        deltap1=(bound? FLT_MAX: deltap1+delta);
		}
      }

      {// CSPH terms.
      const float volp2=-massp2/velrhop2.w;
      float dv=dvx*volp2; csphp1.xx+=dv*frx; csphp1.xy+=dv*fry; csphp1.xz+=dv*frz;
            dv=dvy*volp2; csphp1.xy+=dv*frx; csphp1.yy+=dv*fry; csphp1.yz+=dv*frz;
			dv=dvz*volp2; csphp1.xz+=dv*frx; csphp1.yz+=dv*fry; csphp1.zz+=dv*frz;
	  }

	  {//-temp viscdt
		const float dot=drx*dvx + dry*dvy + drz*dvz;
        const float dot_rr2=dot/(rr2+CTE.eta2);
        visc=max(dot_rr2,visc);  //ViscDt=max(dot/(rr2+Eta2),ViscDt); // <----- Reduction to only one value. 
	  }
	  
	  {//Volumetric concentration  //this can be cheaper //debug
	    const unsigned sed_phase=PHASECTE[pp2].sed;
	    if(!bound){
		  const float vol=massp2/velrhop2.w;
		  if (sed_phase==1)cvs+=vol;
		  cvt+=vol;
		}
	  }

      //===== XSPH correction =====
      if(xsph){
		float robar=(rhopp1+velrhop2.w)*0.5f;
        const float wab_rhobar=massp2*(wab/robar);
        vcor.x-=wab_rhobar * dvx;
        vcor.y-=wab_rhobar * dvy;
        vcor.z-=wab_rhobar * dvz;
      }
    }
  }
}

//------------------------------------------------------------------------------
/// Particle interaction Fluid-Fluid & Fluid-Bound.
//------------------------------------------------------------------------------
template<TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph,unsigned hdiv> __global__ void KerInteractionMultiForcesFluid
  (unsigned n,unsigned pinit,uint4 nc,unsigned cellfluid,const unsigned *cellpart,const int2 *begincell
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp,float *ar,float3 *ace,float3 *velxcor,tsymatrix3f *csph,float *delta,const unsigned *idpm,float *viscdt,float *cv)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    unsigned p1=p+pinit;     //-Number of particle.
    float arp1=0,deltap1=0;
	float visc=0;
    float3 acep1=make_float3(0,0,0);
    float3 vcor;
    if(xsph)vcor=acep1;
    float4 r=velrhop[p1];
    float3 velp1=make_float3(r.x,r.y,r.z);
    float rhopp1=r.w;
    r=pospres[p1];
    float3 posp1=make_float3(r.x,r.y,r.z);
	//multi
	unsigned pp1=idpm[p1];
	float cs0p1=PHASEARRAY[pp1].Cs0_ph;
	float massp1=PHASEARRAY[pp1].mass_ph;
	float cvs=0,cvt=0;
	
    float3 devsp1; devsp1.x=r.w; devsp1.y=devsp1.x*(r.w>0? 0.01f: -0.2f); devsp1.z=cs0p1;
	tsymatrix3f csphp1={0,0,0,0,0,0};

	//cells
    const int cel=cellpart[p1]-cellfluid; //-Substracts where the cells of fluid start.
//  if(ik&&p==0){ KPrintt(ik,"id:",idp[p1]," cel:",true); KPrint(ik,cel); }
    //-Gets limits of interaction.
    int cx=cel%nc.x;
    int cz=int(cel/(nc.w));
    int cy=int((cel%(nc.w))/nc.x);

    //-Code for hdiv 1 or 2.
    int cxini=cx-min(cx,hdiv);
    int cxfin=cx+min(nc.x-cx-1,hdiv)+1;
    int yini=cy-min(cy,hdiv);
    int yfin=cy+min(nc.y-cy-1,hdiv)+1;
    int zini=cz-min(cz,hdiv);
    int zfin=cz+min(nc.z-cz-1,hdiv)+1;

    //-Interaction with Fluids.
    for(int z=zini;z<zfin;z++){
      int zmod=(nc.w)*z+cellfluid; //-Adds when the cells of fluid start.
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        unsigned pini,pfin=0;
        for(int x=cxini;x<cxfin;x++){
          int2 cbeg=begincell[x+ymod];
          if(cbeg.y){
            if(!pfin)pini=cbeg.x;
            pfin=cbeg.y;
          }
        }
        if(pfin)KerInteractionMultiForcesFluidBox<tdelta,tkernel,tvisco,xsph> (false,p1,pini,pfin,pospres,velrhop,idp,idpm,pp1,massp1,posp1,velp1,devsp1,rhopp1,acep1,vcor,arp1,csphp1,deltap1,visc,cvs,cvt);
      }
    }
    //-Interaction with Boundary.
    for(int z=zini;z<zfin;z++){
      int zmod=(nc.w)*z;
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        unsigned pini,pfin=0;
        for(int x=cxini;x<cxfin;x++){
          int2 cbeg=begincell[x+ymod];
          if(cbeg.y){
            if(!pfin)pini=cbeg.x;
            pfin=cbeg.y;
          }
        }
        if(pfin)KerInteractionMultiForcesFluidBox<tdelta,tkernel,tvisco,xsph> (true,p1,pini,pfin,pospres,velrhop,idp,idpm,pp1,massp1,posp1,velp1,devsp1,rhopp1,acep1,vcor,arp1,csphp1,deltap1,visc,cvs,cvt);
      }
    }
    //-Stores resutls.
    if(arp1 || acep1.x || acep1.y || acep1.z ){
      if(tdelta==DELTA_DBC && deltap1!=FLT_MAX)arp1+=deltap1;
      if(tdelta==DELTA_DBCExt){
        float rdelta=delta[p1];
        delta[p1]=(rdelta==FLT_MAX || deltap1==FLT_MAX? FLT_MAX: rdelta+deltap1);
      }
      ar[p1]+=arp1;
      float3 k=ace[p1]; k.x+=acep1.x; k.y+=acep1.y; k.z+=acep1.z; ace[p1]=k;
      if(xsph){
        k=velxcor[p1]; k.x+=vcor.x; k.y+=vcor.y; k.z+=vcor.z; velxcor[p1]=k;
      }
	  if(visc>viscdt[p1])viscdt[p1]=visc;
	  csph[p1]=csphp1;
    }
	if (PHASECTE[pp1].sed==1 &&cvt&&cvs )cv[p1]=cvs/cvt; // I am not using +cvp1 here but small change since this is NOT a Sum{W} just a vol concentration 

  }
}

//==============================================================================
/// Interaction to compute forces Multiphase only
//==============================================================================
template<TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph> void MultiInteractionForces(TpCellMode cellmode,unsigned bsbound,unsigned bsfluid
  ,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells,const unsigned *cellpart,const int2 *begincell
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp, tsymatrix3f* tau,tsymatrix3f* vtau
  ,float *viscdt,float* ar,float3 *ace,float3 *velxcor,tsymatrix3f* csph,float *delta,const unsigned *idpm,float *viscop,float *cv,bool simulate2d)
{
  const unsigned npf=np-npb;
  uint4 nc=make_uint4(ncells.x,ncells.y,ncells.z,ncells.x*ncells.y);
  //-Interaction Fluid-Fluid & Fluid-Bound, Pressure terms and CSPH
  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    if(cellmode==CELLMODE_H)KerInteractionMultiForcesFluid<tdelta,tkernel,tvisco,xsph,2> <<<sgridf,bsfluid>>> (npf,npb,nc,nc.w*nc.z+1,cellpart,begincell,pospres,velrhop,idp,ar,ace,velxcor,csph,delta,idpm,viscdt,cv);
    else                    KerInteractionMultiForcesFluid<tdelta,tkernel,tvisco,xsph,1> <<<sgridf,bsfluid>>> (npf,npb,nc,nc.w*nc.z+1,cellpart,begincell,pospres,velrhop,idp,ar,ace,velxcor,csph,delta,idpm,viscdt,cv);
  }
  //-Interaction Boundary-Fluid, Pressure terms and CSPH
  if(npbok){
    dim3 sgridb=GetGridSize(npbok,bsbound);
	if(cellmode==CELLMODE_H)KerInteractionMultiForcesBound<tkernel,2> <<<sgridb,bsbound>>> (npbok,nc,cellpart,begincell,pospres,velrhop,idp,viscdt,ar,idpm,csph);
    else                    KerInteractionMultiForcesBound<tkernel,1> <<<sgridb,bsbound>>> (npbok,nc,cellpart,begincell,pospres,velrhop,idp,viscdt,ar,idpm,csph);
  }
    
  //tsymatrix3f *auxc=new tsymatrix3f[np];
  //cudaMemcpy(auxc,csph,sizeof(tsymatrix3f)*np,cudaMemcpyDeviceToHost);
  ////for(unsigned p=0;p<np;p++)printf("Array[%d]=%f\n",p,auxc[p]);
  //for(unsigned p=0;p<np;p++)printf("Array[%d].xx=%f Array[%d].xy=%f Array[%d].xz=%f Array[%d].yy=%f Array[%d].yz=%f Array[%d].zz=%f \n",p,auxc[p].xx,p,auxc[p].xy,p,auxc[p].xz,p,auxc[p].yy,p,auxc[p].yz,p,auxc[p].zz);
  //delete[] auxc;
  //getchar();
    
  //Now that we have all csph fluid+bound we can calculate the shear stresses for each particle 
  if(np){
   /* dim3 sgridf=GetGridSize(npf,bsfluid);
	KerSPSCalcStressTau<false> <<<sgridf,bsfluid>>> (npf,npb,velrhop,csph,tau); */ //remove it for debug
	dim3 sgrid=GetGridSize(np,SPHBSIZE);
	KerShearCalcvTau<false> <<<sgrid,SPHBSIZE>>> (np,npb,velrhop,csph,vtau,idpm,viscop,cv); 
  }

  //-Interaction Fluid-Fluid & Fluid-Bound, Viscous/shear stresses only
  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    if(cellmode==CELLMODE_H)KerInteractionMultiStressFluid<tdelta,tkernel,tvisco,xsph,2> <<<sgridf,bsfluid>>> (npf,npb,nc,nc.w*nc.z+1,cellpart,begincell,pospres,velrhop,idp,idpm,ace,tau,vtau);
    else                    KerInteractionMultiStressFluid<tdelta,tkernel,tvisco,xsph,1> <<<sgridf,bsfluid>>> (npf,npb,nc,nc.w*nc.z+1,cellpart,begincell,pospres,velrhop,idp,idpm,ace,tau,vtau);
  }
  
  //-For 2D simulations, second component (.y) is removed.
  if(simulate2d && npf){
    dim3 sgrid=GetGridSize(npf,SPHBSIZE);
    KerResety <<<sgrid,SPHBSIZE>>> (npf,npb,ace);
  }
}

//==============================================================================
void MultiInteraction_Forces(TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph,TpCellMode cellmode,unsigned bsbound,unsigned bsfluid
  ,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells,const unsigned *cellpart,const int2 *begincell
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp, tsymatrix3f* tau,tsymatrix3f* vtau
  ,float *viscdt,float* ar,float3 *ace,float3 *velxcor,tsymatrix3f* csph,float *delta,const unsigned *idpm,float *viscop,float *cv,bool simulate2d)
{
  if(tdelta==DELTA_None){ const TpDeltaSph tdel=DELTA_None;
    if(tkernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
	  const TpVisco tvis=VISCO_SumSPS;
	  if(xsph)MultiInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,vtau,viscdt,ar,ace,velxcor,csph,delta,idpm,viscop,cv,simulate2d);
	  else    MultiInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,vtau,viscdt,ar,ace,velxcor,csph,delta,idpm,viscop,cv,simulate2d);
    }
    else if(tkernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland; 
	  const TpVisco tvis=VISCO_SumSPS;
	  if(xsph)MultiInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,vtau,viscdt,ar,ace,velxcor,csph,delta,idpm,viscop,cv,simulate2d);
	  else    MultiInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,vtau,viscdt,ar,ace,velxcor,csph,delta,idpm,viscop,cv,simulate2d);
	}
  }

  else if(tdelta==DELTA_DBC){ const TpDeltaSph tdel=DELTA_DBC;
    if(tkernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;     
	  const TpVisco tvis=VISCO_SumSPS;
      if(xsph)MultiInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,vtau,viscdt,ar,ace,velxcor,csph,delta,idpm,viscop,cv,simulate2d);
      else    MultiInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,vtau,viscdt,ar,ace,velxcor,csph,delta,idpm,viscop,cv,simulate2d);
	}
    else if(tkernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
      const TpVisco tvis=VISCO_SumSPS;
      if(xsph)MultiInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,vtau,viscdt,ar,ace,velxcor,csph,delta,idpm,viscop,cv,simulate2d);
      else    MultiInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,vtau,viscdt,ar,ace,velxcor,csph,delta,idpm,viscop,cv,simulate2d);
    }
  }

  else if(tdelta==DELTA_DBCExt){ const TpDeltaSph tdel=DELTA_DBCExt;
    if(tkernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
      const TpVisco tvis=VISCO_SumSPS;
      if(xsph)MultiInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,vtau,viscdt,ar,ace,velxcor,csph,delta,idpm,viscop,cv,simulate2d);
      else    MultiInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,vtau,viscdt,ar,ace,velxcor,csph,delta,idpm,viscop,cv,simulate2d);
    }
    else if(tkernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
      const TpVisco tvis=VISCO_LaminarSPS;
      if(xsph)MultiInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,vtau,viscdt,ar,ace,velxcor,csph,delta,idpm,viscop,cv,simulate2d);
      else    MultiInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,vtau,viscdt,ar,ace,velxcor,csph,delta,idpm,viscop,cv,simulate2d);
    }
  }
}

//---------------------------**********************************MULTIPHASE END**********************************---------------------------



//##############################################################################
//# Kernels to compute forces with periodic zone.
//##############################################################################
//------------------------------------------------------------------------------
/// Interaction of particles with periodic zone (Bound-Fluid).
//------------------------------------------------------------------------------
template<TpKernel tkernel,unsigned hdiv> __global__ void KerInteractionPeriForcesBound
  (unsigned n,const unsigned *list,unsigned listpini,const unsigned *cellpart
  ,unsigned ncx,unsigned cellfluid,const int2 *zobegincell
  ,const float4 *zopospres,const float4 *zovelrhop,const unsigned *zoidp
  ,const float4 *pospres,const float4 *velrhop,float *viscdt,float *ar)
{
  unsigned pp=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p1=(list? list[pp]: pp+listpini);  //-Number of particle.
    float arp1=0,visc=0;
    float4 r=velrhop[p1];
    float3 velp1=make_float3(r.x,r.y,r.z);
    r=pospres[p1];
    float3 posp1=make_float3(r.x,r.y,r.z);
    const float massf=CTE.massf;
    const unsigned cel=cellpart[pp];
    //-Gets limits of interaction.
    unsigned cx=(cel>>16)+cellfluid;
    unsigned cy=(cel&0xffff);
    unsigned cxini=cx-hdiv;
    unsigned cxfin=cx+hdiv+1;
    unsigned yini=cy-hdiv;
    unsigned yfin=cy+hdiv+1;
    //-Interaction of Boundary with Fluids.
    for(int y=yini;y<yfin;y++){
      int ymod=ncx*y;
      unsigned pini,pfin=0;
      for(int x=cxini;x<cxfin;x++){
        int2 cbeg=zobegincell[x+ymod];
        if(cbeg.y){
          if(!pfin)pini=cbeg.x;
          pfin=cbeg.y;
        }
      }
      if(pfin)KerInteractionForcesBoundBox<tkernel>(p1,pini,pfin,zopospres,zovelrhop,zoidp,massf,posp1,velp1,arp1,visc);
    }
    //-Stores results.
    if(arp1 || visc){
      ar[p1]+=arp1;
      if(visc>viscdt[p1])viscdt[p1]=visc;
    }
  }
}



//------------------------------------------------------------------------------
/// Interaction of particles with periodic zone (Fluid-Fluid/Bound).
//------------------------------------------------------------------------------
template<TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph,unsigned hdiv> __global__ void KerInteractionPeriForcesFluid
  (unsigned n,const unsigned *list,unsigned listpini,const unsigned *cellpart,unsigned ncx,unsigned cellfluid,const int2 *zobegincell
  ,const float4 *zopospres,const float4 *zovelrhop,const unsigned *zoidp,const tsymatrix3f *zotau
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp,const tsymatrix3f *tau
  ,float *viscdt,float *ar,float3 *ace,float3 *velxcor,tsymatrix3f *csph,float *delta)
{
  unsigned pp=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p1=(list? list[pp]: pp+listpini);       //-Number of particle.
    float arp1=0,visc=0,deltap1=0;
    float3 acep1=make_float3(0,0,0);
    float3 vcor;
    if(xsph)vcor=acep1;
    float4 r=velrhop[p1];
    float3 velp1=make_float3(r.x,r.y,r.z);
    float rhopp1=r.w;
    r=pospres[p1];
    float3 posp1=make_float3(r.x,r.y,r.z);
    const float csoun=rhopp1*CTE.overrhop0;
    float3 devsp1; devsp1.x=r.w/(rhopp1*rhopp1); devsp1.y=devsp1.x*(r.w>0? 0.01f: -0.2f); devsp1.z=CTE.cs0*(csoun*csoun*csoun);
    tsymatrix3f taup1=tau[p1];
    tsymatrix3f csphp1={0,0,0,0,0,0};
    const unsigned cel=cellpart[pp];
    //-Gets limits of interaction.
    unsigned cx=(cel>>16)+cellfluid;
    unsigned cy=(cel&0xffff);
    unsigned cxini=cx-hdiv;
    unsigned cxfin=cx+hdiv+1;
    unsigned yini=cy-hdiv;
    unsigned yfin=cy+hdiv+1;
    //-Interaction with Fluids.
    for(int y=yini;y<yfin;y++){
      int ymod=ncx*y;
      unsigned pini,pfin=0;
      for(int x=cxini;x<cxfin;x++){
        int2 cbeg=zobegincell[x+ymod];
        if(cbeg.y){
          if(!pfin)pini=cbeg.x;
          pfin=cbeg.y;
        }
      }
      if(pfin)KerInteractionForcesFluidBox<tdelta,tkernel,tvisco,xsph> (false,p1,pini,pfin,zopospres,zovelrhop,zoidp,zotau,CTE.massf,posp1,velp1,devsp1,rhopp1,taup1,acep1,vcor,arp1,visc,csphp1,deltap1);
    }
    //-Interaction with Boundary.
    cxini-=cellfluid; cxfin-=cellfluid;
    for(int y=yini;y<yfin;y++){
      int ymod=ncx*y;
      unsigned pini,pfin=0;
      for(int x=cxini;x<cxfin;x++){
        int2 cbeg=zobegincell[x+ymod];
        if(cbeg.y){
          if(!pfin)pini=cbeg.x;
          pfin=cbeg.y;
        }
      }
      if(pfin)KerInteractionForcesFluidBox<tdelta,tkernel,tvisco,xsph> (true,p1,pini,pfin,zopospres,zovelrhop,zoidp,NULL,CTE.massb,posp1,velp1,devsp1,rhopp1,taup1,acep1,vcor,arp1,visc,csphp1,deltap1);
    }
    //-Stores resutls.
    if(arp1 || acep1.x || acep1.y || acep1.z || visc){
      if(tdelta==DELTA_DBC && deltap1!=FLT_MAX)arp1+=deltap1;
      if(tdelta==DELTA_DBCExt){
        float rdelta=delta[p1];
        delta[p1]=(rdelta==FLT_MAX || deltap1==FLT_MAX? FLT_MAX: rdelta+deltap1);
      }
      ar[p1]+=arp1;
      float3 r=ace[p1]; r.x+=acep1.x; r.y+=acep1.y; r.z+=acep1.z; ace[p1]=r;
      if(xsph){
        r=velxcor[p1]; r.x+=vcor.x; r.y+=vcor.y; r.z+=vcor.z; velxcor[p1]=r;
      }
      if(visc>viscdt[p1])viscdt[p1]=visc;
      if(tvisco==VISCO_LaminarSPS)csph[p1]=csphp1;
    }
  }
}

//==============================================================================
/// Interaction to compute forces with periodic zone.
//==============================================================================
template<TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph> void InteractionPeriForces
  (TpCellMode cellmode,unsigned bsbound,unsigned bsfluid,unsigned np,unsigned npb
  ,const unsigned *list,unsigned listbini,unsigned listfini,const unsigned *cellpart
  ,unsigned ncx,unsigned cellfluid,const int2 *zobegincell
  ,const float4 *zopospres,const float4 *zovelrhop,const unsigned *zoidp,const tsymatrix3f *zotau
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp,const tsymatrix3f *tau
  ,float *viscdt,float *ar,float3 *ace,float3 *velxcor,tsymatrix3f *csph,float *delta,bool simulate2d)
{
  const unsigned npf=np-npb;
  //-Interaction Fluid-Fluid & Fluid-Bound.
  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    //JDgKerPrint info;
    //byte* ik=NULL;//info.GetInfoPointer(sgridf,bsfluid);
    if(cellmode==CELLMODE_H)KerInteractionPeriForcesFluid<tdelta,tkernel,tvisco,xsph,2> <<<sgridf,bsfluid>>> (npf,(list? list+npb: NULL),listfini,cellpart+npb,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta);
    else                    KerInteractionPeriForcesFluid<tdelta,tkernel,tvisco,xsph,1> <<<sgridf,bsfluid>>> (npf,(list? list+npb: NULL),listfini,cellpart+npb,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta);
    //info.PrintValuesFull(true); //info.PrintValuesInfo();
  }
  //-Interaction Boundary-Fluid.
  if(npb){
    dim3 sgridb=GetGridSize(npb,bsbound);
    if(cellmode==CELLMODE_H)KerInteractionPeriForcesBound<tkernel,2> <<<sgridb,bsbound>>> (npb,list,listbini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,pospres,velrhop,viscdt,ar);
    else                    KerInteractionPeriForcesBound<tkernel,1> <<<sgridb,bsbound>>> (npb,list,listbini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,pospres,velrhop,viscdt,ar);
  }
  //-For 2D simulations, second component (.y) is removed.
  if(simulate2d && npf){
    dim3 sgrid=GetGridSize(npf,SPHBSIZE);
    if(list)KerResety <<<sgrid,SPHBSIZE>>> (npf,list+npb,ace);
    else    KerResety <<<sgrid,SPHBSIZE>>> (npf,npb,ace);
  }
}
//==============================================================================
void InteractionPeri_Forces(TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph,TpCellMode cellmode,unsigned bsbound,unsigned bsfluid
  ,unsigned np,unsigned npb,const unsigned *list,unsigned listbini,unsigned listfini,const unsigned *cellpart,unsigned ncx,unsigned cellfluid,const int2 *zobegincell
  ,const float4 *zopospres,const float4 *zovelrhop,const unsigned *zoidp,const tsymatrix3f *zotau
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp,const tsymatrix3f *tau
  ,float *viscdt,float *ar,float3 *ace,float3 *velxcor,tsymatrix3f *csph,float *delta,bool simulate2d)
{
  if(tdelta==DELTA_None){ const TpDeltaSph tdel=DELTA_None;
    if(tkernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)InteractionPeriForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionPeriForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)InteractionPeriForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionPeriForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
    else if(tkernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)InteractionPeriForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionPeriForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)InteractionPeriForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionPeriForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
  }
  else if(tdelta==DELTA_DBC){ const TpDeltaSph tdel=DELTA_DBC;
    if(tkernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)InteractionPeriForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionPeriForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)InteractionPeriForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionPeriForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
    else if(tkernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)InteractionPeriForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionPeriForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)InteractionPeriForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionPeriForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
  }
  else if(tdelta==DELTA_DBCExt){ const TpDeltaSph tdel=DELTA_DBCExt;
    if(tkernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)InteractionPeriForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionPeriForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)InteractionPeriForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionPeriForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
    else if(tkernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)InteractionPeriForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionPeriForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)InteractionPeriForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    InteractionPeriForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,list,listbini,listfini,cellpart,ncx,cellfluid,zobegincell,zopospres,zovelrhop,zoidp,zotau,pospres,velrhop,idp,tau,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
  }
}

//##############################################################################
//# Kernels to add forces-
//##############################################################################
//------------------------------------------------------------------------------
/// Adds value of force to particles.
//------------------------------------------------------------------------------
__global__ void KerAddForceFluid(unsigned n,unsigned pini,word codesel,tfloat3 force,const word *code,float3 *ace)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned p1=p+pini;
    word rcode=code[p];
    if(CODE_GetValue(rcode)==codesel){
      float3 r=ace[p1];
      r.x+=force.x; r.y+=force.y; r.z+=force.z;
      ace[p1]=r;
    }
  }
}

//==============================================================================
/// Adds value of force to particles.
//==============================================================================
void AddForceFluid(unsigned n,unsigned pini,word tpvalue,tfloat3 force,const word *code,float3 *ace){
  if(n){
    word codesel=word(CODE_TYPE_FLUID|tpvalue);
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    KerAddForceFluid <<<sgrid,SPHBSIZE>>> (n,pini,codesel,force,code,ace);
  }
}



//##############################################################################
//# Kernels for Delta-SPH.
//##############################################################################

//------------------------------------------------------------------------------
/// Adds value of delta[] to ar[] while different from FLT_MAX.
//------------------------------------------------------------------------------
__global__ void KerAddDelta(unsigned n,const float *delta,float *ar)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    float rdelta=delta[p];
    if(rdelta!=FLT_MAX)ar[p]+=rdelta;
  }
}

//==============================================================================
/// Adds value of delta[] to ar[] while different from FLT_MAX.
//==============================================================================
void AddDelta(unsigned n,const float *delta,float *ar){
  if(n){
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    KerAddDelta <<<sgrid,SPHBSIZE>>> (n,delta,ar);
  }
}


//##############################################################################
//# Kernels for Shepard density filter.
//##############################################################################
//------------------------------------------------------------------------------
/// Prepares variables for Shepard interaction.
//------------------------------------------------------------------------------
template<bool floating> __global__ void KerPreInteraction_Shepard(unsigned npf,unsigned ini,const float3 *pos,const float *rhop,const unsigned *idp,unsigned nbound,float ftposout,float massf,float4 *posvol)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<npf){
    p+=ini;
    float3 rpos=pos[p]; 
    if(floating && idp[p]<nbound)rpos=make_float3(ftposout,ftposout,ftposout);
    posvol[p]=make_float4(rpos.x,rpos.y,rpos.z,massf/rhop[p]);
  }
}

//==============================================================================
/// Prepares variables for Shepard interaction.
//==============================================================================
void PreInteraction_Shepard(bool floating,unsigned pini,unsigned pfin,const float3 *pos,const float *rhop,const unsigned *idp,unsigned nbound,float ftposout,float massf,float4 *posvol){
  const unsigned npf=pfin-pini;
  if(npf){
    dim3 sgrid=GetGridSize(npf,SPHBSIZE);
    if(floating)KerPreInteraction_Shepard<true>  <<<sgrid,SPHBSIZE>>> (npf,pini,pos,rhop,idp,nbound,ftposout,massf,posvol);
    else        KerPreInteraction_Shepard<false> <<<sgrid,SPHBSIZE>>> (npf,pini,pos,rhop,idp,nbound,ftposout,massf,posvol);
  }
}

//------------------------------------------------------------------------------
/// Shepard interaction of a particle with a set of particles.
//------------------------------------------------------------------------------
template<TpKernel tkernel> __device__ __forceinline__ void KerInteractionShepardBox
  (unsigned p1,const unsigned &pini,const unsigned &pfin,const float4 *posvol,const float4 &posvol1,float &fdwabp1,float &fdrhopp1)
{
  for(int p2=pini;p2<pfin;p2++){
    float4 posvol2=posvol[p2];
    float drx=posvol1.x-posvol2.x;
    float dry=posvol1.y-posvol2.y;
    float drz=posvol1.z-posvol2.z;
    float rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=1e-18f){
      float wab;
      {//===== Kernel =====
        const float rad=sqrt(rr2);
        const float qq=rad/CTE.h;
        if(tkernel==KERNEL_Cubic){     //-Cubic kernel.
          const bool radgt=qq>1;
          const float wqq1=(radgt? 2.0f-qq: qq);
          const float wqq2=wqq1*wqq1;
          const float wqq3=wqq2*wqq1;
          wab=(radgt? CTE.cubic_a24*wqq3: CTE.cubic_a2*(1.0f-1.5f*wqq2+0.75f*wqq3));
        }
        if(tkernel==KERNEL_Wendland){  //-Wendland kernel.
          float wqq2=1.f-0.5f*qq; wqq2*=wqq2;
          wab=CTE.wendland_awen*(2*qq+1)*wqq2*wqq2;
        }
      }
      fdwabp1+=wab*posvol2.w;
      fdrhopp1+=wab;
    }
  }
}

//------------------------------------------------------------------------------
/// Shepard interaction between fluid particles.
//------------------------------------------------------------------------------
template<TpKernel tkernel,unsigned hdiv> __global__ void KerInteractionShepard
  (unsigned n,unsigned pini,uint4 nc,const unsigned *cellpart,const int2 *begincell,const float4 *posvol,float *fdrhop,float *fdwab)
{
  unsigned p1=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
  if(p1<n){
    p1+=pini;      //-Number of particle.
    float4 posvol1=posvol[p1];
    float fdrhopp1=0;
    float fdwabp1=0;
    const unsigned cellfluid=nc.w*nc.z+1; 
    const int cel=cellpart[p1]-cellfluid;
    //-Gets limits of interaction.
    int cx=cel%nc.x;
    int cz=int(cel/(nc.w));
    int cy=int((cel%(nc.w))/nc.x);
    int cxini=cx-min(cx,hdiv);
    int cxfin=cx+min(nc.x-cx-1,hdiv)+1;
    int yini=cy-min(cy,hdiv);
    int yfin=cy+min(nc.y-cy-1,hdiv)+1;
    int zini=cz-min(cz,hdiv);
    int zfin=cz+min(nc.z-cz-1,hdiv)+1;
    //-Interaction with Fluids.
    for(int z=zini;z<zfin;z++){
      int zmod=(nc.w)*z+cellfluid;
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        unsigned pini,pfin=0;
        for(int x=cxini;x<cxfin;x++){
          int2 cbeg=begincell[x+ymod];
          if(cbeg.y){
            if(!pfin)pini=cbeg.x;
            pfin=cbeg.y;
          }
        }
        if(pfin)KerInteractionShepardBox <tkernel> (p1,pini,pfin,posvol,posvol1,fdwabp1,fdrhopp1);
      }
    }
    //-Interaction by parts: Recovers data of other interaction and stores partial results.
    if(fdwab!=NULL){
      fdwab[p1]=fdwabp1;
      fdrhop[p1]=fdrhopp1;
    }
    //-Complete interation: FdWab[] is not used and stores final density in FdRhop[].
    else fdrhop[p1]=((fdrhopp1+CTE.cteshepard)*CTE.massf)/(fdwabp1+CTE.cteshepard*posvol1.w);
  }
}

//==============================================================================
/// Shepard interaction.
//==============================================================================
template<TpKernel tkernel> void InteractionShepard(TpCellMode cellmode,unsigned bsshepard,unsigned pini,unsigned pfin,tuint3 ncells,const unsigned *cellpart,const int2 *begincell,const float4 *posvol,float *fdrhop,float *fdwab){
  const unsigned npf=pfin-pini;
  if(npf){
    uint4 nc=make_uint4(ncells.x,ncells.y,ncells.z,ncells.x*ncells.y);
    dim3 sgridf=GetGridSize(npf,bsshepard);
    if(cellmode==CELLMODE_H)KerInteractionShepard<tkernel,2> <<<sgridf,bsshepard>>> (npf,pini,nc,cellpart,begincell,posvol,fdrhop,fdwab);
    else                    KerInteractionShepard<tkernel,1> <<<sgridf,bsshepard>>> (npf,pini,nc,cellpart,begincell,posvol,fdrhop,fdwab);
  }
}
//==============================================================================
void Interaction_Shepard(TpKernel tkernel,TpCellMode cellmode,unsigned bsshepard,unsigned pini,unsigned pfin,tuint3 ncells,const unsigned *cellpart,const int2 *begincell,const float4 *posvol,float *fdrhop,float *fdwab){
  if(tkernel==KERNEL_Cubic)        InteractionShepard<KERNEL_Cubic>   (cellmode,bsshepard,pini,pfin,ncells,cellpart,begincell,posvol,fdrhop,fdwab);
  else if(tkernel==KERNEL_Wendland)InteractionShepard<KERNEL_Wendland>(cellmode,bsshepard,pini,pfin,ncells,cellpart,begincell,posvol,fdrhop,fdwab);
}

//------------------------------------------------------------------------------
/// Computes new value of density using Shepard filter.
//------------------------------------------------------------------------------
__global__ void KerCompute_Shepard
  (unsigned n,unsigned pini,float massf,const float *fdrhop,const float *fdwab,float *rhop)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
  if(p<n){
    p+=pini;       //-Number of particle.
    float rvol=massf/rhop[p];
    rhop[p]=((fdrhop[p]+CTE.cteshepard)*massf)/(fdwab[p]+CTE.cteshepard*rvol);
  }
}

//==============================================================================
/// Computes new value of density according to FdRhop[] and FdWab[].
//==============================================================================
void Compute_Shepard(unsigned pini,unsigned pfin,float massf,const float *fdrhop,const float *fdwab,float *rhop){
  const unsigned npf=pfin-pini;
  if(npf){
    dim3 sgridf=GetGridSize(npf,SPHBSIZE);
    KerCompute_Shepard<<<sgridf,SPHBSIZE>>>(npf,pini,massf,fdrhop,fdwab,rhop);
  }
}

//##############################################################################
//# Kernels for Shepard density filter with periodic zone.
//##############################################################################
//------------------------------------------------------------------------------
/// Shepard interaction of fluid particles with periodic zone.
//------------------------------------------------------------------------------
template<TpKernel tkernel,unsigned hdiv> __global__ void KerInteractionPeriShepard
  (unsigned n,const unsigned *list,unsigned listpini,const unsigned *cellpart
  ,unsigned ncx,unsigned cellfluid,const int2 *zobegincell
  ,const float4 *zoposvol
  ,const float4 *posvol,float *fdrhop,float *fdwab)
{
  unsigned pp=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
  if(pp<n){
    const unsigned p1=(list? list[pp]: pp+listpini);  //-Number of particle.
    float4 posvol1=posvol[p1];
    float fdrhopp1=0;
    float fdwabp1=0;
    const unsigned cel=cellpart[pp];
    //-Gets limits of interaction.
    unsigned cx=(cel>>16)+cellfluid;
    unsigned cy=(cel&0xffff);
    unsigned cxini=cx-hdiv;
    unsigned cxfin=cx+hdiv+1;
    unsigned yini=cy-hdiv;
    unsigned yfin=cy+hdiv+1;

    //-Interaction with Fluids.
    for(int y=yini;y<yfin;y++){
      int ymod=ncx*y;
      unsigned pini,pfin=0;
      for(int x=cxini;x<cxfin;x++){
        int2 cbeg=zobegincell[x+ymod];
        if(cbeg.y){
          if(!pfin)pini=cbeg.x;
          pfin=cbeg.y;
        }
      }
      if(pfin)KerInteractionShepardBox <tkernel> (p1,pini,pfin,zoposvol,posvol1,fdwabp1,fdrhopp1);
    }
    //-Adds values computed in the periodic zone.
    fdwab[p1]+=fdwabp1;
    fdrhop[p1]+=fdrhopp1;
  }
}

//==============================================================================
/// Shepard interaction.
//==============================================================================
template<TpKernel tkernel> void InteractionPeriShepard(TpCellMode cellmode,unsigned bsshepard,unsigned npf,const unsigned *list,unsigned listpini,const unsigned *cellpart,unsigned ncx,unsigned cellfluid,const int2 *zobegincell,const float4 *zoposvol,const float4 *posvol,float *fdrhop,float *fdwab){
  if(npf){
    dim3 sgridf=GetGridSize(npf,bsshepard);
    if(cellmode==CELLMODE_H)KerInteractionPeriShepard<tkernel,2> <<<sgridf,bsshepard>>> (npf,list,listpini,cellpart,ncx,cellfluid,zobegincell,zoposvol,posvol,fdrhop,fdwab);
    else                    KerInteractionPeriShepard<tkernel,1> <<<sgridf,bsshepard>>> (npf,list,listpini,cellpart,ncx,cellfluid,zobegincell,zoposvol,posvol,fdrhop,fdwab);
  }
}
//==============================================================================
void InteractionPeri_Shepard(TpKernel tkernel,TpCellMode cellmode,unsigned bsshepard,unsigned npf,const unsigned *list,unsigned listpini,const unsigned *cellpart,unsigned ncx,unsigned cellfluid,const int2 *zobegincell,const float4 *zoposvol,const float4 *posvol,float *fdrhop,float *fdwab){
  if(tkernel==KERNEL_Cubic)        InteractionPeriShepard<KERNEL_Cubic>   (cellmode,bsshepard,npf,list,listpini,cellpart,ncx,cellfluid,zobegincell,zoposvol,posvol,fdrhop,fdwab);
  else if(tkernel==KERNEL_Wendland)InteractionPeriShepard<KERNEL_Wendland>(cellmode,bsshepard,npf,list,listpini,cellpart,ncx,cellfluid,zobegincell,zoposvol,posvol,fdrhop,fdwab);
}


//##############################################################################
//# Kernels for ComputeStep.
//##############################################################################
//------------------------------------------------------------------------------
/// Computes new values of Pos, Vel and Rhop using VERLET.
//------------------------------------------------------------------------------
template<bool rhopbound,bool floating> __global__ void KerComputeStepVerlet
  (unsigned n,unsigned npb,const float3 *vel1,const float3 *vel2,const float *rhop,const unsigned *idp,const float *ar,const float3 *ace,const float3 *velxcor
  ,float dt,float dt205,float dt2,float eps,float movlimit,float3 *pos,word *code,float3 *velnew,float *rhopnew,float rhop0)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    if(p<npb){     //-Particles: Fixed & Moving.
      if(rhopbound){
        float rrhop=rhop[p]+ar[p]*dt2;
        rhopnew[p]=(rrhop<rhop0? rhop0: rrhop);  //-To prevent absorption of fluid particles by boundaries.
      }
      else rhopnew[p]=rhop[p];
    }
    else{          //-Particles: Floating & Fluid.
      float rrhop=rhop[p]+ar[p]*dt2;
      float3 rvel=vel1[p];
      word rcode=code[p];
      if(CODE_GetType(rcode)==CODE_TYPE_FLUID){  //-Particles: Fluid.
        float3 rpos=pos[p],rvelxcor=velxcor[p],race=ace[p];
        float dx=(rvel.x+rvelxcor.x*eps)*dt + race.x*dt205;
        float dy=(rvel.y+rvelxcor.y*eps)*dt + race.y*dt205;
        float dz=(rvel.z+rvelxcor.z*eps)*dt + race.z*dt205;
        if(fmaxf(fabsf(dx),fmaxf(fabsf(dy),fabsf(dz)))>movlimit)code[p]=CODE_SetOutMove(rcode);
        rpos.x+=dx; rpos.y+=dy; rpos.z+=dz;
        pos[p]=rpos;
        rvel=vel2[p];
        rvel.x+=race.x*dt2;  rvel.y+=race.y*dt2;  rvel.z+=race.z*dt2;
        velnew[p]=rvel;
      }
      else{                                      //-Particles: Floating.
        rrhop=(rrhop<rhop0? rhop0: rrhop);  //-To prevent absorption of fluid particles by floating.
      }
      rhopnew[p]=rrhop;
      velnew[p]=rvel;
    }
  }
}

//==============================================================================
/// Updates particles according to forces and dt using VERLET. 
//==============================================================================
void ComputeStepVerlet(bool rhopbound,bool floating,unsigned np,unsigned npb,const float3 *vel1,const float3 *vel2,const float *rhop,const unsigned *idp
  ,const float *ar,const float3 *ace,const float3 *velxcor,float dt,float dt2,float eps
  ,float movlimit,float3 *pos,word *code,float3 *velnew,float *rhopnew,float rhop0)
{
  if(np){
    dim3 sgrid=GetGridSize(np,SPHBSIZE);
    if(rhopbound){
      if(floating)KerComputeStepVerlet<true,true>   <<<sgrid,SPHBSIZE>>> (np,npb,vel1,vel2,rhop,idp,ar,ace,velxcor,dt,(0.5f*dt*dt),dt2,eps,movlimit,pos,code,velnew,rhopnew,rhop0);
      else        KerComputeStepVerlet<true,false>  <<<sgrid,SPHBSIZE>>> (np,npb,vel1,vel2,rhop,idp,ar,ace,velxcor,dt,(0.5f*dt*dt),dt2,eps,movlimit,pos,code,velnew,rhopnew,rhop0);
    }
    else{
      if(floating)KerComputeStepVerlet<false,true>  <<<sgrid,SPHBSIZE>>> (np,npb,vel1,vel2,rhop,idp,ar,ace,velxcor,dt,(0.5f*dt*dt),dt2,eps,movlimit,pos,code,velnew,rhopnew,rhop0);
      else        KerComputeStepVerlet<false,false> <<<sgrid,SPHBSIZE>>> (np,npb,vel1,vel2,rhop,idp,ar,ace,velxcor,dt,(0.5f*dt*dt),dt2,eps,movlimit,pos,code,velnew,rhopnew,rhop0);
    }
  }
}

//---------------------------------------------------------------------------------
/// Computes new values of Pos, Vel and Rhop using SYMPLECTIC-Predictor.
//---------------------------------------------------------------------------------
template<bool rhopbound,bool floating> __global__ void KerComputeStepSymplecticPre
  (unsigned n,unsigned npb,const unsigned *idp,const float3 *pospre,const float3 *velpre,const float *rhoppre,const float *ar,const float3 *velxcor
  ,const float3 *ace,float dtm,float eps,float movlimit,float3 *pos,word *code,float3 *vel,float *rhop,float rhop0)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    if(p<npb){     //-Particles: Fixed & Moving.
      if(rhopbound){
        float rrhop=rhoppre[p]+ar[p]*dtm;
        //rhop[p]=rrhop;//(rrhop<rhop0? rhop0: rrhop);     //-To prevent absorption of fluid particles by boundaries.
		rhop[p]=(rrhop<rhop0? rhop0: rrhop);     //-To prevent absorption of fluid particles by boundaries.
      }
      else rhop[p]=rhoppre[p];
    }
    else{          //-Particles: Floating & Fluid.
      float rrhop=rhoppre[p]+ar[p]*dtm;
      float3 rpos=pospre[p],rvel=velpre[p];
      word rcode=code[p];
      if(CODE_GetType(rcode)==CODE_TYPE_FLUID){  //-Particles: Fluid.
        float3 rvelxcor=velxcor[p],race=ace[p];
        float dx=(rvel.x+rvelxcor.x*eps)*dtm;
        float dy=(rvel.y+rvelxcor.y*eps)*dtm;
        float dz=(rvel.z+rvelxcor.z*eps)*dtm;
        if(fmaxf(fabsf(dx),fmaxf(fabsf(dy),fabsf(dz)))>movlimit)code[p]=CODE_SetOutMove(rcode);
        rpos.x+=dx; rpos.y+=dy; rpos.z+=dz;
        rvel.x+=race.x*dtm;
        rvel.y+=race.y*dtm;
        rvel.z+=race.z*dtm;
      }
      else{                                      //-Particles: Floating.
        rrhop=(rrhop<rhop0? rhop0: rrhop); //-To prevent absorption of fluid particles by floating.
      }
      rhop[p]=rrhop;
      pos[p]=rpos; 
      vel[p]=rvel;
    }
  }
}

//==============================================================================
/// Updates particles according to forces and dt using SYMPLECTIC-Predictor. 
//==============================================================================   
void ComputeStepSymplecticPre(bool rhopbound,bool floating,unsigned np,unsigned npb,const unsigned *idp
  ,const float3 *pospre,const float3 *velpre,const float *rhoppre,const float *ar,const float3 *velxcor,const float3 *ace
  ,float dtm,float eps,float movlimit,float3 *pos,word *code,float3 *vel,float *rhop,float rhop0)
{
  if(np){
    dim3 sgrid=GetGridSize(np,SPHBSIZE);
    if(rhopbound){
      if(floating)KerComputeStepSymplecticPre<true,true>   <<<sgrid,SPHBSIZE>>> (np,npb,idp,pospre,velpre,rhoppre,ar,velxcor,ace,dtm,eps,movlimit,pos,code,vel,rhop,rhop0);
      else        KerComputeStepSymplecticPre<true,false>  <<<sgrid,SPHBSIZE>>> (np,npb,idp,pospre,velpre,rhoppre,ar,velxcor,ace,dtm,eps,movlimit,pos,code,vel,rhop,rhop0);
    }
    else{
      if(floating)KerComputeStepSymplecticPre<false,true>  <<<sgrid,SPHBSIZE>>> (np,npb,idp,pospre,velpre,rhoppre,ar,velxcor,ace,dtm,eps,movlimit,pos,code,vel,rhop,rhop0);
      else        KerComputeStepSymplecticPre<false,false> <<<sgrid,SPHBSIZE>>> (np,npb,idp,pospre,velpre,rhoppre,ar,velxcor,ace,dtm,eps,movlimit,pos,code,vel,rhop,rhop0);
    }
  }
}

//---------------------------------------------------------------------------------
/// Computes new values of Pos, Vel and Rhop using SYMPLECTIC-Corrector.
//---------------------------------------------------------------------------------
template<bool rhopbound,bool floating> __global__ void KerComputeStepSymplecticCor
  (unsigned n,unsigned npb,const unsigned *idp,const float3 *pospre,const float3 *velpre,const float *rhoppre,const float *ar,const float3 *ace
  ,float dtm,float dt,float movlimit,float3 *pos,word *code,float3 *vel,float *rhop,float rhop0)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    if(p<npb){     //-Particles: Fixed & Moving.
      if(rhopbound){
        float epsilon_rdot=(-ar[p]/rhop[p])*dt;
        float rrhop=rhoppre[p] * (2.f-epsilon_rdot)/(2.f+epsilon_rdot);
        //rhop[p]=rrhop; //(rrhop<rhop0? rhop0: rrhop);     //-To prevent absorption of fluid particles by boundaries.
		rhop[p]=(rrhop<rhop0? rhop0: rrhop);     //-To prevent absorption of fluid particles by boundaries.
      }
      else rhop[p]=rhoppre[p];
    }
    else{          //-Particles: Floating & Fluid.
      float epsilon_rdot=(-ar[p]/rhop[p])*dt;
      float rrhop=rhoppre[p] * (2.f-epsilon_rdot)/(2.f+epsilon_rdot);
      float3 rpos=pospre[p],rvel=velpre[p];
      word rcode=code[p];
      if(CODE_GetType(rcode)==CODE_TYPE_FLUID){  //-Particles: Fluid.
        float3 rvelp=rvel,race=ace[p];
        rvel.x+=race.x*dt;
        rvel.y+=race.y*dt;
        rvel.z+=race.z*dt;
        float dx=(rvelp.x+rvel.x)*dtm;
        float dy=(rvelp.y+rvel.y)*dtm;
        float dz=(rvelp.z+rvel.z)*dtm;
        if(fmaxf(fabsf(dx),fmaxf(fabsf(dy),fabsf(dz)))>movlimit)code[p]=CODE_SetOutMove(rcode);
        rpos.x+=dx; rpos.y+=dy; rpos.z+=dz;
      }
      else{                                      //-Particles: Floating.
        rrhop=(rrhop<rhop0? rhop0: rrhop); //-To prevent absorption of fluid particles by floating.
      }
      rhop[p]=rrhop;
      pos[p]=rpos; 
      vel[p]=rvel;
    }
  }
}

//==============================================================================
/// Updates particles according to forces and dt using SYMPLECTIC-Corrector.
//==============================================================================   
void ComputeStepSymplecticCor(bool rhopbound,bool floating,unsigned np,unsigned npb,const unsigned *idp
  ,const float3 *pospre,const float3 *velpre,const float *rhoppre,const float *ar,const float3 *ace
  ,float dtm,float dt,float movlimit,float3 *pos,word *code,float3 *vel,float *rhop,float rhop0)
{
  if(np){
    dim3 sgrid=GetGridSize(np,SPHBSIZE);
    if(rhopbound){
      if(floating)KerComputeStepSymplecticCor<true,true>   <<<sgrid,SPHBSIZE>>> (np,npb,idp,pospre,velpre,rhoppre,ar,ace,dtm,dt,movlimit,pos,code,vel,rhop,rhop0);
      else        KerComputeStepSymplecticCor<true,false>  <<<sgrid,SPHBSIZE>>> (np,npb,idp,pospre,velpre,rhoppre,ar,ace,dtm,dt,movlimit,pos,code,vel,rhop,rhop0);
    }
    else{
      if(floating)KerComputeStepSymplecticCor<false,true>  <<<sgrid,SPHBSIZE>>> (np,npb,idp,pospre,velpre,rhoppre,ar,ace,dtm,dt,movlimit,pos,code,vel,rhop,rhop0);
      else        KerComputeStepSymplecticCor<false,false> <<<sgrid,SPHBSIZE>>> (np,npb,idp,pospre,velpre,rhoppre,ar,ace,dtm,dt,movlimit,pos,code,vel,rhop,rhop0);
    }
  }
}


//------------------------------------------------------------------------------
/// Computes for a range of particles, their position according to idp[].
//------------------------------------------------------------------------------
__global__ void KerCalcRidp(unsigned n,unsigned ini,unsigned idini,unsigned idfin,const unsigned *idp,unsigned *ridp)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    p+=ini; 
    unsigned id=idp[p];
    if(idini<=id && id<idfin)ridp[id-idini]=p;
  }
}

//==============================================================================
/// Computes for a range of particles, their position according to idp[].
//==============================================================================
void CalcRidp(unsigned np,unsigned pini,unsigned idini,unsigned idfin,const unsigned *idp,unsigned *ridp){
  dim3 sgrid=GetGridSize(np,SPHBSIZE);
  KerCalcRidp <<<sgrid,SPHBSIZE>>> (np,pini,idini,idfin,idp,ridp);
}


//------------------------------------------------------------------------------
/// Applies a linear movement to a set of particles.
//------------------------------------------------------------------------------
__global__ void KerMoveLinBound(unsigned n,unsigned ini,float3 mvpos,float3 mvvel,const unsigned *ridpmv,float3 *pos,float3 *vel,word *code)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    int pid=ridpmv[p+ini];
    if(pid>=0){
      float3 rpos=pos[pid];
      rpos.x+=mvpos.x; rpos.y+=mvpos.y; rpos.z+=mvpos.z;
      pos[pid]=rpos;
      vel[pid]=mvvel;
      if(code)code[pid]=CODE_SetOutMove(code[pid]);
    }
  }
}

//==============================================================================
/// Applies a matrix movement to a set of particles.
//==============================================================================
void MoveLinBound(bool simulate2d,unsigned np,unsigned ini,tfloat3 mvpos,tfloat3 mvvel,const unsigned *ridp,float3 *pos,float3 *vel,word *code,float movlimit){
  if(simulate2d){ mvpos.y=0; mvvel.y=0; }
  if(fabs(mvpos.x)<=movlimit && fabs(mvpos.y)<=movlimit && fabs(mvpos.z)<=movlimit)code=NULL; 
  dim3 sgrid=GetGridSize(np,SPHBSIZE);
  KerMoveLinBound <<<sgrid,SPHBSIZE>>> (np,ini,Float3(mvpos),Float3(mvvel),ridp,pos,vel,code);
}


//------------------------------------------------------------------------------
/// Applies a matrix movement to a set of particles.
//------------------------------------------------------------------------------
template<bool simulate2d> __global__ void KerMoveMatBound(unsigned n,unsigned ini,tmatrix4f m,float dt,const unsigned *ridpmv,float3 *pos,float3 *vel,word *code,float movlimit)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    int pid=ridpmv[p+ini];
    if(pid>=0){
      float3 rpos=pos[pid];
      float3 rpos2;
      rpos2.x= m.a11*rpos.x + m.a12*rpos.y + m.a13*rpos.z + m.a14;
      rpos2.y= m.a21*rpos.x + m.a22*rpos.y + m.a23*rpos.z + m.a24;
      rpos2.z= m.a31*rpos.x + m.a32*rpos.y + m.a33*rpos.z + m.a34;
      if(simulate2d)rpos2.y=rpos.y;
      float dx=rpos2.x-rpos.x;
      float dy=rpos2.y-rpos.y;
      float dz=rpos2.z-rpos.z;
      if(fmaxf(fabsf(dx),fmaxf(fabsf(dy),fabsf(dz)))>movlimit)code[pid]=CODE_SetOutMove(code[pid]);
      pos[pid]=rpos2;
      vel[pid]=make_float3(dx/dt,dy/dt,dz/dt);
    }
  }
}

//==============================================================================
/// Applies a matrix movement to a set of particles.
//==============================================================================
void MoveMatBound(bool simulate2d,unsigned np,unsigned ini,tmatrix4f m,float dt,const unsigned *ridpmv,float3 *pos,float3 *vel,word *code,float movlimit){
  dim3 sgrid=GetGridSize(np,SPHBSIZE);
  if(simulate2d)KerMoveMatBound<true>  <<<sgrid,SPHBSIZE>>> (np,ini,m,dt,ridpmv,pos,vel,code,movlimit);
  else          KerMoveMatBound<false> <<<sgrid,SPHBSIZE>>> (np,ini,m,dt,ridpmv,pos,vel,code,movlimit);
}


//##############################################################################
//# Kernels for Floating bodies.
//##############################################################################
//------------------------------------------------------------------------------
/// Computes FtDist[] for the particles of a floating body.
//------------------------------------------------------------------------------
__global__ void KerFtCalcDist(unsigned n,unsigned pini,float3 center,const unsigned *ftridp,const float3 *pos,float3 *ftdist)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; 
  if(p<n){
    p+=pini;
    float3 rpos=pos[ftridp[p]];
    ftdist[p]=make_float3(rpos.x-center.x,rpos.y-center.y,rpos.z-center.z); 
  }
}

//==============================================================================
/// Computes FtDist[] for the particles of a floating body.
//==============================================================================
void FtCalcDist(unsigned n,unsigned pini,tfloat3 center,const unsigned *ftridp,const float3 *pos,float3 *ftdist){
  dim3 sgrid=GetGridSize(n,SPHBSIZE);
  KerFtCalcDist <<<sgrid,SPHBSIZE>>> (n,pini,Float3(center),ftridp,pos,ftdist);
}


//------------------------------------------------------------------------------
/// Computes values for a floating body.
//------------------------------------------------------------------------------
__global__ void KerFtCalcOmega(unsigned n,unsigned pini,float3 gravity,float ftmass,const unsigned *ftridp,const float3 *ftdist,const float3 *ace,float3 *result)
{
  unsigned tid=threadIdx.x;
  if(!tid){
    float rfacex=0,rfacey=0,rfacez=0;
    float rfomegavelx=0,rfomegavely=0,rfomegavelz=0;
    const unsigned pfin=pini+n;
    for(unsigned p=pini;p<pfin;p++){
      float3 race=ace[ftridp[p]];
      race.x-=gravity.x; race.y-=gravity.y; race.z-=gravity.z;
      rfacex+=race.x; rfacey+=race.y; rfacez+=race.z;
      float3 rdist=ftdist[p];
      rfomegavelx+=(race.z*rdist.y - race.y*rdist.z);
      rfomegavely+=(race.x*rdist.z - race.z*rdist.x);
      rfomegavelz+=(race.y*rdist.x - race.x*rdist.y);
    }
    rfacex=(rfacex+ftmass*gravity.x)/ftmass;
    rfacey=(rfacey+ftmass*gravity.y)/ftmass;
    rfacez=(rfacez+ftmass*gravity.z)/ftmass;
    result[0]=make_float3(rfacex,rfacey,rfacez);
    result[1]=make_float3(rfomegavelx,rfomegavely,rfomegavelz);
  }
}


//==============================================================================
/// Computes values for a floating body.
//==============================================================================
void FtCalcOmega(unsigned n,unsigned pini,tfloat3 gravity,float ftmass,const unsigned *ftridp,const float3 *ftdist,const float3 *ace,float3 *result){
  if(n)KerFtCalcOmega <<<1,32>>> (n,pini,Float3(gravity),ftmass,ftridp,ftdist,ace,result);
}

//------------------------------------------------------------------------------
/// Updates particles of a floating body.
//------------------------------------------------------------------------------
template<bool predictor> __global__ void KerFtUpdate(unsigned n,unsigned pini,float dt,float3 center,float3 fvel,float3 fomega,const unsigned *ftridp,float3 *ftdist,float3 *pos,float3 *vel)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; 
  if(p<n){
    p+=pini;
    unsigned cp=ftridp[p];
    float3 rpos=pos[cp];
    float3 rvel=vel[cp];
    rpos.x+=dt*rvel.x;  rpos.y+=dt*rvel.y;  rpos.z+=dt*rvel.z;
    pos[cp]=rpos;
    float3 rdist=make_float3(rpos.x-center.x,rpos.y-center.y,rpos.z-center.z);  
    if(!predictor)ftdist[p]=rdist;
    rvel.x=fvel.x+(fomega.y*rdist.z-fomega.z*rdist.y);
    rvel.y=fvel.y+(fomega.z*rdist.x-fomega.x*rdist.z);
    rvel.z=fvel.z+(fomega.x*rdist.y-fomega.y*rdist.x);
    vel[cp]=rvel;
  }
}

//==============================================================================
/// Updates particles of a floating body.
//==============================================================================
void FtUpdate(bool predictor,unsigned n,unsigned pini,float dt,tfloat3 center,tfloat3 fvel,tfloat3 fomega,const unsigned *ftridp,float3 *ftdist,float3 *pos,float3 *vel){
  if(n){
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    if(predictor)KerFtUpdate<true>  <<<sgrid,SPHBSIZE>>> (n,pini,dt,Float3(center),Float3(fvel),Float3(fomega),ftridp,ftdist,pos,vel);
    else         KerFtUpdate<false> <<<sgrid,SPHBSIZE>>> (n,pini,dt,Float3(center),Float3(fvel),Float3(fomega),ftridp,ftdist,pos,vel);
  }
}

//##############################################################################
//# Kernels to compute forces with floating bodies.
//##############################################################################
//------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles (Bound-Fluid).
//------------------------------------------------------------------------------
template<TpKernel tkernel> __device__ __forceinline__ void KerFtInteractionForcesBoundBox
  (unsigned p1,const unsigned &pini,const unsigned &pfin,const float4 *pospres,const float4 *velrhop
  ,const unsigned* idp,const word *code,const float *ftomassp
  ,float3 posp1,float3 velp1,float &arp1,float &visc)
{
  for(int p2=pini;p2<pfin;p2++){
    float4 pospres2=pospres[p2];
    float drx=posp1.x-pospres2.x;
    float dry=posp1.y-pospres2.y;
    float drz=posp1.z-pospres2.z;
    float rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=1e-18f){
      const float4 velrhop2=velrhop[p2];
      float frx,fry,frz;
      {//===== Kernel =====
        const float rad=sqrt(rr2);
        const float qq=rad/CTE.h;
        float fac;
        if(tkernel==KERNEL_Cubic){     //-Cubic kernel.
          const bool radgt=qq>1;
          float wqq2=(radgt? 2.0f-qq: qq); wqq2*=wqq2;
          fac=(radgt? CTE.cubic_c2*wqq2: (CTE.cubic_c1*qq+CTE.cubic_d1*wqq2))/rad;
        }
        if(tkernel==KERNEL_Wendland){  //-Wendland kernel.
          const float wqq1=1.f-0.5f*qq;
          fac=CTE.wendland_bwen*qq*wqq1*wqq1*wqq1/rad;
        } 
        frx=fac*drx; fry=fac*dry; frz=fac*drz;
      }

      float dvx=velp1.x-velrhop2.x, dvy=velp1.y-velrhop2.y, dvz=velp1.z-velrhop2.z;
      {//===== Viscosity ===== 
        const float dot=drx*dvx + dry*dvy + drz*dvz;
        const float dot_rr2=dot/(rr2+CTE.eta2);
        visc=max(dot_rr2,visc);  // <----- Reduction to only one value. 
      }
      //===== Density derivative =====
      word rcode=code[p2];
      const float massp2=(CODE_GetType(rcode)!=CODE_TYPE_FLOATING? CTE.massf: ftomassp[CODE_GetTypeValue(rcode)]);
      arp1+=massp2*(dvx*frx+dvy*fry+dvz*frz);
    }
  }
}
//------------------------------------------------------------------------------
/// Particle interaction Bound-Fluid.
//------------------------------------------------------------------------------
template<TpKernel tkernel,unsigned hdiv> __global__ void KerFtInteractionForcesBound
  (unsigned n,uint4 nc,const unsigned *cellpart,const int2 *begincell
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp
  ,const word *code,const float *ftomassp
  ,float *viscdt,float *ar)
{
  unsigned p1=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p1<n){
    float arp1=0,visc=0;
    float4 r=velrhop[p1];
    float3 velp1=make_float3(r.x,r.y,r.z);
    r=pospres[p1];
    float3 posp1=make_float3(r.x,r.y,r.z);
    const int cel=cellpart[p1];
    //-Gets limits of interaction.
    int cx=cel%nc.x;
    int cz=int(cel/(nc.w));
    int cy=int((cel%(nc.w))/nc.x);

    //-Code for hdiv 1 or 2.
    int cxini=cx-min(cx,hdiv);
    int cxfin=cx+min(nc.x-cx-1,hdiv)+1;
    int yini=cy-min(cy,hdiv);
    int yfin=cy+min(nc.y-cy-1,hdiv)+1;
    int zini=cz-min(cz,hdiv);
    int zfin=cz+min(nc.z-cz-1,hdiv)+1;

    //-Interaction of Boundary with Fluid.
    for(int z=zini;z<zfin;z++){
      int zmod=(nc.w)*z+(nc.w*nc.z+1);//-Adds Nct+1 that is the first cell of fluid.
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        unsigned pini,pfin=0;
        for(int x=cxini;x<cxfin;x++){
          int2 cbeg=begincell[x+ymod];
          if(cbeg.y){
            if(!pfin)pini=cbeg.x;
            pfin=cbeg.y;
          }
        }
        if(pfin)KerFtInteractionForcesBoundBox<tkernel>(p1,pini,pfin,pospres,velrhop,idp,code,ftomassp,posp1,velp1,arp1,visc);
      }
    }
    //-Stores results.
    if(arp1 || visc){
      ar[p1]+=arp1;
      if(visc>viscdt[p1])viscdt[p1]=visc;
    }
  }
}

//---------------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles (Fluid-Fluid/Bound).
//---------------------------------------------------------------------------------------
template<TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph> __device__ __forceinline__ void KerFtInteractionForcesFluidBox
  (bool bound,unsigned p1,const unsigned &pini,const unsigned &pfin,const float4 *pospres,const float4 *velrhop,const unsigned *idp,const tsymatrix3f *tau
  ,const word *code,const float *ftomassp,float massftp1
  ,float mass_p2,float3 posp1,float3 velp1,float3 devsp1,float rhopp1,const tsymatrix3f &taup1
  ,float3 &acep1,float3 &vcor,float &arp1,float &visc,tsymatrix3f &csphp1,float &deltap1)
{
  for(int p2=pini;p2<pfin;p2++){
    float4 pospres2=pospres[p2];
    float drx=posp1.x-pospres2.x;
    float dry=posp1.y-pospres2.y;
    float drz=posp1.z-pospres2.z;
    float rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.fourh2 && rr2>=1e-18f){
      const float4 velrhop2=velrhop[p2];
      const float prrhop2=pospres2.w/(velrhop2.w*velrhop2.w);
      float prs=devsp1.x+prrhop2;
      float wab,frx,fry,frz;
      {//===== Kernel =====
        const float rad=sqrt(rr2);
        const float qq=rad/CTE.h;
        float fac;
        if(tkernel==KERNEL_Cubic){     //-Cubic kernel.
          const bool radgt=qq>1;
          const float wqq1=(radgt? 2.0f-qq: qq);
          const float wqq2=wqq1*wqq1;
          const float wqq3=wqq2*wqq1;
          wab=(radgt? CTE.cubic_a24*wqq3: CTE.cubic_a2*(1.0f-1.5f*wqq2+0.75f*wqq3));
          fac=(radgt? CTE.cubic_c2*wqq2: (CTE.cubic_c1*qq+CTE.cubic_d1*wqq2))/rad;
          //-Tensile correction
          float fab=wab*CTE.cubic_odwdeltap;
          fab*=fab; fab*=fab; //fab=fab^4
          prs+=fab*(devsp1.y+ prrhop2*(pospres2.w>0? 0.01f: -0.2f) );
        }
        if(tkernel==KERNEL_Wendland){  //-Wendland kernel.
          const float wqq=2.f*qq+1.f;
          const float wqq1=1.f-0.5f*qq;
          const float wqq2=wqq1*wqq1;
          wab=CTE.wendland_awen*wqq*wqq2*wqq2;
          fac=CTE.wendland_bwen*qq*wqq2*wqq1/rad;
        } 
        frx=fac*drx; fry=fac*dry; frz=fac*drz;
      }

      //===== Gets mass according to the type of particle ===== 
      word rcode=code[p2];
      const float massp2=(CODE_GetType(rcode)!=CODE_TYPE_FLOATING? mass_p2: ftomassp[CODE_GetTypeValue(rcode)]);

      {//===== Aceleration ===== 
        const float p_vpm=-prs*massp2*massftp1;
        acep1.x+=p_vpm*frx; acep1.y+=p_vpm*fry; acep1.z+=p_vpm*frz;
      }

      //===== Density derivative =====
      const float dvx=velp1.x-velrhop2.x, dvy=velp1.y-velrhop2.y, dvz=velp1.z-velrhop2.z;
      arp1+=massp2*(dvx*frx+dvy*fry+dvz*frz);

      const float csoun=velrhop2.w*CTE.overrhop0;  //const float csound=CTE.cs0*powf(rrhop*OVERRHOPCERO,3);
      const float cbar=(devsp1.z+ CTE.cs0*(csoun*csoun*csoun) )*0.5f;
      //===== DeltaSPH =====
      if(tdelta==DELTA_DBC || tdelta==DELTA_DBCExt){
        const float rhop1over2=rhopp1/velrhop2.w;
        const float visc_densi=CTE.delta2h*cbar*(rhop1over2-1)/(rr2+CTE.eta2);
        const float dot3=(drx*frx+dry*fry+drz*frz);
        const float delta=visc_densi*dot3*massp2;
        deltap1=(bound? FLT_MAX: deltap1+delta);
      }

      float robar=(rhopp1+velrhop2.w)*0.5f;
      {//===== Viscosity ===== 
        const float dot=drx*dvx + dry*dvy + drz*dvz;
        const float dot_rr2=dot/(rr2+CTE.eta2);
        //-Artificial viscosity.
        if(tvisco==VISCO_Artificial && dot<0){
          const float amubar=CTE.h*dot_rr2;   //amubar=CTE.h*dot/(rr2+CTE.eta2);
          const float pi_visc=(-CTE.visco*cbar*amubar/robar)*massp2*massftp1;
          acep1.x-=pi_visc*frx; acep1.y-=pi_visc*fry; acep1.z-=pi_visc*frz;
        }
        //-Laminar+SPS viscosity.
        if(tvisco==VISCO_LaminarSPS){ 
          const float temp=2.0f*CTE.visco/((rr2+CTE.eta2)*robar);
          const float vtemp=massp2*temp*(drx*frx+dry*fry+drz*frz);  
          acep1.x+=vtemp*dvx; acep1.y+=vtemp*dvy; acep1.z+=vtemp*dvz;
          // SPS turbulence model.
          tsymatrix3f tausum=taup1;
          if(tau){ //-Only with fluid.
            tausum=tau[p2];
            tausum.xx+=taup1.xx;
            tausum.xy+=taup1.xy;
            tausum.xz+=taup1.xz;
            tausum.yy+=taup1.yy;
            tausum.yz+=taup1.yz;
            tausum.zz+=taup1.zz;
          }
          acep1.x+=massp2*massftp1*(tausum.xx*frx+tausum.xy*fry+tausum.xz*frz);
          acep1.y+=massp2*massftp1*(tausum.xy*frx+tausum.yy*fry+tausum.yz*frz);
          acep1.z+=massp2*massftp1*(tausum.xz*frx+tausum.yz*fry+tausum.zz*frz);
          // CSPH terms.
          const float volp2=-massp2/velrhop2.w;
          float dv=dvx*volp2; csphp1.xx+=dv*frx; csphp1.xy+=dv*fry; csphp1.xz+=dv*frz;
                dv=dvy*volp2; csphp1.xy+=dv*frx; csphp1.yy+=dv*fry; csphp1.yz+=dv*frz;
                dv=dvz*volp2; csphp1.xz+=dv*frx; csphp1.yz+=dv*fry; csphp1.zz+=dv*frz;
        }
        visc=max(dot_rr2,visc);  //ViscDt=max(dot/(rr2+Eta2),ViscDt); // <----- Reduction to only one value. 
      }

      //===== XSPH correction =====
      if(xsph){
        const float wab_rhobar=massp2*(wab/robar);
        vcor.x-=wab_rhobar * dvx;
        vcor.y-=wab_rhobar * dvy;
        vcor.z-=wab_rhobar * dvz;
      }
    }
  }
}
//-------------------------------------------------------------------
/// Particle interaction Fluid-Fluid & Fluid-Bound.
//-------------------------------------------------------------------
template<TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph,unsigned hdiv> __global__ void KerFtInteractionForcesFluid
  (unsigned n,unsigned pinit,uint4 nc,unsigned cellfluid,const unsigned *cellpart,const int2 *begincell
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp,const tsymatrix3f *tau
  ,const word *code,const float *ftomassp
  ,float *viscdt,float *ar,float3 *ace,float3 *velxcor,tsymatrix3f *csph,float *delta)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    unsigned p1=p+pinit;     //-Number of particle.
    float arp1=0,visc=0,deltap1=0;
    float3 acep1=make_float3(0,0,0);
    float3 vcor;
    if(xsph)vcor=acep1;
    float4 r=velrhop[p1];
    float3 velp1=make_float3(r.x,r.y,r.z);
    float rhopp1=r.w;
    r=pospres[p1];
    float3 posp1=make_float3(r.x,r.y,r.z);
    const float csoun=rhopp1*CTE.overrhop0;
    float3 devsp1; devsp1.x=r.w/(rhopp1*rhopp1); devsp1.y=devsp1.x*(r.w>0? 0.01f: -0.2f); devsp1.z=CTE.cs0*(csoun*csoun*csoun);
    tsymatrix3f taup1=tau[p1];
    tsymatrix3f csphp1={0,0,0,0,0,0};
    //-Gets mass according to type of particle.
    float massftp1=1.0f;
    word rcode=code[p1];
    if(CODE_GetType(rcode)==CODE_TYPE_FLOATING)massftp1=ftomassp[CODE_GetTypeValue(rcode)];
    const int cel=cellpart[p1]-cellfluid; //-Substracts where the cells of fluid start.
    //-Gets limits of interaction.
    int cx=cel%nc.x;
    int cz=int(cel/(nc.w));
    int cy=int((cel%(nc.w))/nc.x);

    //-Code for hdiv 1 or 2.
    int cxini=cx-min(cx,hdiv);
    int cxfin=cx+min(nc.x-cx-1,hdiv)+1;
    int yini=cy-min(cy,hdiv);
    int yfin=cy+min(nc.y-cy-1,hdiv)+1;
    int zini=cz-min(cz,hdiv);
    int zfin=cz+min(nc.z-cz-1,hdiv)+1;

    //-Interaction with Fluids.
    for(int z=zini;z<zfin;z++){
      int zmod=(nc.w)*z+cellfluid;     //-Adds when the cells of fluid start.
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        unsigned pini,pfin=0;
        for(int x=cxini;x<cxfin;x++){
          int2 cbeg=begincell[x+ymod];
          if(cbeg.y){
            if(!pfin)pini=cbeg.x;
            pfin=cbeg.y;
          }
        }
        if(pfin)KerFtInteractionForcesFluidBox<tdelta,tkernel,tvisco,xsph> (false,p1,pini,pfin,pospres,velrhop,idp,tau,code,ftomassp,massftp1,CTE.massf,posp1,velp1,devsp1,rhopp1,taup1,acep1,vcor,arp1,visc,csphp1,deltap1);
      }
    }
    //-Interaction with Boundary.
    for(int z=zini;z<zfin;z++){
      int zmod=(nc.w)*z;
      for(int y=yini;y<yfin;y++){
        int ymod=zmod+nc.x*y;
        unsigned pini,pfin=0;
        for(int x=cxini;x<cxfin;x++){
          int2 cbeg=begincell[x+ymod];
          if(cbeg.y){
            if(!pfin)pini=cbeg.x;
            pfin=cbeg.y;
          }
        }
        if(pfin)KerFtInteractionForcesFluidBox<tdelta,tkernel,tvisco,xsph> (true,p1,pini,pfin,pospres,velrhop,idp,NULL,code,ftomassp,massftp1,CTE.massb,posp1,velp1,devsp1,rhopp1,taup1,acep1,vcor,arp1,visc,csphp1,deltap1);
      }
    }
    //-Stores resutls.
    if(arp1 || acep1.x || acep1.y || acep1.z || visc){
      if(tdelta==DELTA_DBC && deltap1!=FLT_MAX)arp1+=deltap1;
      if(tdelta==DELTA_DBCExt){
        float rdelta=delta[p1];
        delta[p1]=(rdelta==FLT_MAX || deltap1==FLT_MAX? FLT_MAX: rdelta+deltap1);
      }
      ar[p1]+=arp1;
      float3 r=ace[p1]; r.x+=acep1.x; r.y+=acep1.y; r.z+=acep1.z; ace[p1]=r;
      if(xsph){
        r=velxcor[p1]; r.x+=vcor.x; r.y+=vcor.y; r.z+=vcor.z; velxcor[p1]=r;
      }
      if(visc>viscdt[p1])viscdt[p1]=visc;
      if(tvisco==VISCO_LaminarSPS)csph[p1]=csphp1;
    }
  }
}

//==============================================================================
/// Interaction to compute forces.
//==============================================================================
template<TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph> void FtInteractionForces(TpCellMode cellmode,unsigned bsbound,unsigned bsfluid
  ,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells,const unsigned *cellpart,const int2 *begincell
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp,const tsymatrix3f* tau
  ,const word *code,const float *ftomassp
  ,float *viscdt,float* ar,float3 *ace,float3 *velxcor,tsymatrix3f* csph,float *delta,bool simulate2d)
{
  const unsigned npf=np-npb;
  uint4 nc=make_uint4(ncells.x,ncells.y,ncells.z,ncells.x*ncells.y);
  //-Interaction Fluid-Fluid & Fluid-Bound.
  if(npf){
    dim3 sgridf=GetGridSize(npf,bsfluid);
    //JDgKerPrint info; byte* ik=info.GetInfoPointer(sgridf,bsfluid);
    if(cellmode==CELLMODE_H)KerFtInteractionForcesFluid<tdelta,tkernel,tvisco,xsph,2> <<<sgridf,bsfluid>>> (npf,npb,nc,nc.w*nc.z+1,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta);
    else                    KerFtInteractionForcesFluid<tdelta,tkernel,tvisco,xsph,1> <<<sgridf,bsfluid>>> (npf,npb,nc,nc.w*nc.z+1,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta);
    //info.PrintValuesFull(true); //info.PrintValuesInfo();
  }
  //-Interaction Boundary-Fluid.
  if(npbok){
    dim3 sgridb=GetGridSize(npbok,bsbound);
    if(cellmode==CELLMODE_H)KerFtInteractionForcesBound<tkernel,2> <<<sgridb,bsbound>>> (npbok,nc,cellpart,begincell,pospres,velrhop,idp,code,ftomassp,viscdt,ar);
    else                    KerFtInteractionForcesBound<tkernel,1> <<<sgridb,bsbound>>> (npbok,nc,cellpart,begincell,pospres,velrhop,idp,code,ftomassp,viscdt,ar);
  }
  //-For 2D simulations, second component (.y) is removed.
  if(simulate2d && npf){//-
    dim3 sgrid=GetGridSize(npf,SPHBSIZE);
    KerResety <<<sgrid,SPHBSIZE>>> (npf,npb,ace);
  }
}
//==============================================================================
void FtInteraction_Forces(TpDeltaSph tdelta,TpKernel tkernel,TpVisco tvisco,bool xsph,TpCellMode cellmode,unsigned bsbound,unsigned bsfluid
  ,unsigned np,unsigned npb,unsigned npbok,tuint3 ncells,const unsigned *cellpart,const int2 *begincell
  ,const float4 *pospres,const float4 *velrhop,const unsigned *idp,const tsymatrix3f* tau
  ,const word *code,const float *ftomassp
  ,float *viscdt,float* ar,float3 *ace,float3 *velxcor,tsymatrix3f* csph,float *delta,bool simulate2d)
{
  if(tdelta==DELTA_None){ const TpDeltaSph tdel=DELTA_None;
    if(tkernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)FtInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    FtInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)FtInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    FtInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
    else if(tkernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)FtInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    FtInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)FtInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    FtInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
  }
  else if(tdelta==DELTA_DBC){ const TpDeltaSph tdel=DELTA_DBC;
    if(tkernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)FtInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    FtInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)FtInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    FtInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
    else if(tkernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)FtInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    FtInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)FtInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    FtInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
  }
  else if(tdelta==DELTA_DBCExt){ const TpDeltaSph tdel=DELTA_DBCExt;
    if(tkernel==KERNEL_Cubic){ const TpKernel tker=KERNEL_Cubic;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)FtInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    FtInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)FtInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    FtInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
    else if(tkernel==KERNEL_Wendland){ const TpKernel tker=KERNEL_Wendland;
      if(tvisco==VISCO_Artificial){ const TpVisco tvis=VISCO_Artificial;
        if(xsph)FtInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    FtInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
      else{ const TpVisco tvis=VISCO_LaminarSPS;
        if(xsph)FtInteractionForces<tdel,tker,tvis,true> (cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
        else    FtInteractionForces<tdel,tker,tvis,false>(cellmode,bsbound,bsfluid,np,npb,npbok,ncells,cellpart,begincell,pospres,velrhop,idp,tau,code,ftomassp,viscdt,ar,ace,velxcor,csph,delta,simulate2d);
      }
    }
  }
}


//------------------------------------------------------------------------------
/// Stores in result the minimum id of the found floating particles.
//------------------------------------------------------------------------------
__global__ void KerFtFindFirstFloating
  (unsigned n,const unsigned *list,unsigned listpini
  ,const word *code,const unsigned *idp,unsigned *result)
{
  unsigned pp=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p1=(list? list[pp]: pp+listpini);       //-Number of particle.
    if(CODE_GetType(code[p1])==CODE_TYPE_FLOATING)atomicMin(result,idp[p1]);
  }
}

//==============================================================================
/// Interaction to compute forces with periodic zone.
//==============================================================================
unsigned FtFindFirstFloating
  (unsigned np,unsigned npb,const unsigned *list,unsigned listfini
  ,const word *code,const unsigned *idp)
{
  unsigned idf=UINT_MAX;
  const unsigned npf=np-npb;
  if(npf){
    dim3 sgridf=GetGridSize(npf,SPHBSIZE);
    unsigned *idfloat=NULL;
    cudaMalloc((void**)&idfloat,sizeof(unsigned));
    cudaMemset(idfloat,255,sizeof(unsigned));
    KerFtFindFirstFloating <<<sgridf,SPHBSIZE>>> (npf,(list? list+npb: NULL),listfini,code,idp,idfloat);
    cudaMemcpy(&idf,idfloat,sizeof(unsigned),cudaMemcpyDeviceToHost);
    cudaFree(idfloat); 
  }
  return(idf);
}

//------------------------------------------------------
/// Adds variable forces to particle sets.
//------------------------------------------------------
__global__ void KerAddVarAccAng(unsigned n,unsigned pini,word codesel,float3 acclin,float3 accang,float3 centre
  ,const word *code,const float3 *pos,float3 *ace)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
  if(p<n){
    p+=pini;
    //Check if the current particle is part of the particle set by its Mk
    if(CODE_GetValue(code[p])==codesel){
      float3 curracc=ace[p]; //-Gets the current particles acceleration value.
      //-Adds linear acceleration.
      curracc.x+=acclin.x;
      curracc.y+=acclin.y;
      curracc.z+=acclin.z;
      //-Adds angular acceleration.
      float3 dis=pos[p]; //-Gets the current particles position.
      dis.x-=centre.x;
      dis.y-=centre.y;
      dis.z-=centre.z;
      curracc.x+=accang.y*(dis.z)-accang.z*(dis.y);
      curracc.y+=accang.z*(dis.x)-accang.x*(dis.z);
      curracc.z+=accang.x*(dis.y)-accang.y*(dis.x);
      //-Stores the new acceleration value.
      ace[p]=curracc;
    }
  }
}

//------------------------------------------------------
// Adds variable forces to particle sets.
//------------------------------------------------------
__global__ void KerAddVarAccLin(unsigned n,unsigned pini,word codesel,float3 acclin
  ,const word *code,float3 *ace)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
  if(p<n){
    p+=pini;
    //Check if the current particle is part of the particle set by its Mk
    if(CODE_GetValue(code[p])==codesel){
      float3 curracc=ace[p]; //-Gets the current particles acceleration value.
      //-Adds linear acceleration.
      curracc.x+=acclin.x;
      curracc.y+=acclin.y;
      curracc.z+=acclin.z;
      //-Stores the new acceleration value.
      ace[p]=curracc;
    }
  }
}

//==================================================================================================
/// Adds variable acceleration forces for particle MK groups that have an input file.
//==================================================================================================
void AddVarAcc(unsigned n,unsigned pini,word codesel,tfloat3 acclin,tfloat3 accang,tfloat3 centre
  ,const word *code,const float3 *pos,float3 *ace)
{
  if(n){
    dim3 sgrid=GetGridSize(n,SPHBSIZE);
    const bool withaccang=(accang.x!=0 || accang.y!=0 || accang.z!=0);
    if(withaccang)KerAddVarAccAng <<<sgrid,SPHBSIZE>>> (n,pini,codesel,Float3(acclin),Float3(accang),Float3(centre),code,pos,ace);
    else          KerAddVarAccLin <<<sgrid,SPHBSIZE>>> (n,pini,codesel,Float3(acclin),code,ace);
  }
}


}



