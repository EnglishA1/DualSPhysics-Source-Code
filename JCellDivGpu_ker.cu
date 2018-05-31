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

/// \file JCellDivGpu_ker.cu \brief Implements functions and CUDA kernels to compute operations of the Neighbour List.

#include "JCellDivGpu_ker.h"
#include "Types.h"
#include <float.h>
#include <cmath>

#pragma warning(disable : 4267) //Cancels "warning C4267: conversion from 'size_t' to 'int', possible loss of data"
#pragma warning(disable : 4244) //Cancels "warning C4244: conversion from 'unsigned __int64' to 'unsigned int', possible loss of data"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "JLog2.h"

namespace cudiv{

//------------------------------------------------------------------------------
/// Reduction of values in shared memory for a warp of KerPosLimitsRedu.
//------------------------------------------------------------------------------
template <unsigned blockSize> __device__ __forceinline__ void KerUintLimitsWarpRedu(volatile unsigned* sp1,volatile unsigned* sp2,const unsigned &tid){
  if(blockSize>=64){
    const unsigned tid2=tid+32;
    sp1[tid]=min(sp1[tid],sp1[tid2]);
    sp2[tid]=max(sp2[tid],sp2[tid2]);
  }
  if(blockSize>=32){
    const unsigned tid2=tid+16;
    sp1[tid]=min(sp1[tid],sp1[tid2]);
    sp2[tid]=max(sp2[tid],sp2[tid2]);
  }
  if(blockSize>=16){
    const unsigned tid2=tid+8;
    sp1[tid]=min(sp1[tid],sp1[tid2]);
    sp2[tid]=max(sp2[tid],sp2[tid2]);
  }
  if(blockSize>=8){
    const unsigned tid2=tid+4;
    sp1[tid]=min(sp1[tid],sp1[tid2]);
    sp2[tid]=max(sp2[tid],sp2[tid2]);
  }
  if(blockSize>=4){
    const unsigned tid2=tid+2;
    sp1[tid]=min(sp1[tid],sp1[tid2]);
    sp2[tid]=max(sp2[tid],sp2[tid2]);
  }
  if(blockSize>=2){
    const unsigned tid2=tid+1;
    sp1[tid]=min(sp1[tid],sp1[tid2]);
    sp2[tid]=max(sp2[tid],sp2[tid2]);
  }
}

//------------------------------------------------------------------------------
/// Reduction of values in shared memory for KerPosLimits.
//------------------------------------------------------------------------------
template <unsigned blockSize> __device__ __forceinline__ void KerUintLimitsRedu(unsigned* sp1,unsigned* sp2,const unsigned &tid,unsigned* results){
  __syncthreads();
  if(blockSize>=512){ 
    if(tid<256){
      sp1[tid]=min(sp1[tid],sp1[tid+256]);
      sp2[tid]=max(sp2[tid],sp2[tid+256]);
    }
    __syncthreads(); 
  }
  if(blockSize>=256){ 
    if(tid<128){
      sp1[tid]=min(sp1[tid],sp1[tid+128]);
      sp2[tid]=max(sp2[tid],sp2[tid+128]);
    }
    __syncthreads(); 
  }
  if(blockSize>=128){ 
    if(tid<64){
      sp1[tid]=min(sp1[tid],sp1[tid+64]);
      sp2[tid]=max(sp2[tid],sp2[tid+64]);
    }
    __syncthreads(); 
  }
  if(tid<32)KerUintLimitsWarpRedu<blockSize>(sp1,sp2,tid);
  if(tid==0){
    const unsigned nblocks=gridDim.x*gridDim.y;
    unsigned cr=blockIdx.y*gridDim.x+blockIdx.x;
    results[cr]=sp1[0]; cr+=nblocks;
    results[cr]=sp2[0];
  }
}

//==============================================================================
/// Reorders values usign RadixSort of thrust.
//==============================================================================
void Sort(unsigned* keys,unsigned* values,unsigned size,bool stable){
  if(size){
    thrust::device_ptr<unsigned> dev_keysg(keys);
    thrust::device_ptr<unsigned> dev_valuesg(values);
    if(!stable)thrust::stable_sort_by_key(dev_keysg,dev_keysg+size,dev_valuesg);
    else thrust::stable_sort_by_key(dev_keysg,dev_keysg+size,dev_valuesg);
  }
}

//==============================================================================
/// Returns dimensions of gridsize according to parameters.
//==============================================================================
dim3 GetGridSize(unsigned n,unsigned blocksize){
  dim3 sgrid;//=dim3(1,2,3);
  unsigned nb=unsigned(n+blocksize-1)/blocksize; //-Total number of blocks to be launched.
  sgrid.x=(nb<=65535? nb: unsigned(sqrt(float(nb))));
  sgrid.y=(nb<=65535? 1: unsigned((nb+sgrid.x-1)/sgrid.x));
  sgrid.z=1;
  return(sgrid);
}

//------------------------------------------------------------------------------
/// Reduction of values in shared memory for a warp of KerPosLimitsRedu.
//------------------------------------------------------------------------------
template <unsigned blockSize> __device__ __forceinline__ void KerPosLimitsWarpRedu(volatile float* spx1,volatile float* spy1,volatile float* spz1,volatile float* spx2,volatile float* spy2,volatile float* spz2,const unsigned &tid){
  if(blockSize>=64){
    const unsigned tid2=tid+32;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
  if(blockSize>=32){
    const unsigned tid2=tid+16;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
  if(blockSize>=16){
    const unsigned tid2=tid+8;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
  if(blockSize>=8){
    const unsigned tid2=tid+4;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
  if(blockSize>=4){
    const unsigned tid2=tid+2;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
  if(blockSize>=2){
    const unsigned tid2=tid+1;
    spx1[tid]=min(spx1[tid],spx1[tid2]); spy1[tid]=min(spy1[tid],spy1[tid2]); spz1[tid]=min(spz1[tid],spz1[tid2]);
    spx2[tid]=max(spx2[tid],spx2[tid2]); spy2[tid]=max(spy2[tid],spy2[tid2]); spz2[tid]=max(spz2[tid],spz2[tid2]);
  }
}

//------------------------------------------------------------------------------
/// Reduction of values in shared memory for KerPosLimits.
//------------------------------------------------------------------------------
template <unsigned blockSize> __device__ __forceinline__ void KerPosLimitsRedu(float* spx1,float* spy1,float* spz1,float* spx2,float* spy2,float* spz2,const unsigned &tid,float* results){
  __syncthreads();
  if(blockSize>=512){ 
    if(tid<256){
      spx1[tid]=min(spx1[tid],spx1[tid+256]); spy1[tid]=min(spy1[tid],spy1[tid+256]); spz1[tid]=min(spz1[tid],spz1[tid+256]);  
      spx2[tid]=max(spx2[tid],spx2[tid+256]); spy2[tid]=max(spy2[tid],spy2[tid+256]); spz2[tid]=max(spz2[tid],spz2[tid+256]);  
    }
    __syncthreads(); 
  }
  if(blockSize>=256){ 
    if(tid<128){
      spx1[tid]=min(spx1[tid],spx1[tid+128]); spy1[tid]=min(spy1[tid],spy1[tid+128]); spz1[tid]=min(spz1[tid],spz1[tid+128]);  
      spx2[tid]=max(spx2[tid],spx2[tid+128]); spy2[tid]=max(spy2[tid],spy2[tid+128]); spz2[tid]=max(spz2[tid],spz2[tid+128]);  
    }
    __syncthreads(); 
  }
  if(blockSize>=128){ 
    if(tid<64){
      spx1[tid]=min(spx1[tid],spx1[tid+64]); spy1[tid]=min(spy1[tid],spy1[tid+64]); spz1[tid]=min(spz1[tid],spz1[tid+64]);  
      spx2[tid]=max(spx2[tid],spx2[tid+64]); spy2[tid]=max(spy2[tid],spy2[tid+64]); spz2[tid]=max(spz2[tid],spz2[tid+64]);  
    }
    __syncthreads(); 
  }
  if(tid<32)KerPosLimitsWarpRedu<blockSize>(spx1,spy1,spz1,spx2,spy2,spz2,tid);
  if(tid==0){
    const unsigned nblocks=gridDim.x*gridDim.y;
    unsigned cr=blockIdx.y*gridDim.x+blockIdx.x;
    results[cr]=spx1[0]; cr+=nblocks;
    results[cr]=spy1[0]; cr+=nblocks;
    results[cr]=spz1[0]; cr+=nblocks;
    results[cr]=spx2[0]; cr+=nblocks;
    results[cr]=spy2[0]; cr+=nblocks;
    results[cr]=spz2[0];
  }
}


//------------------------------------------------------------------------------
/// Computes minimum and maximum position starting from the results of KerPosLimit.
//------------------------------------------------------------------------------
template <unsigned int blockSize> __global__ void KerReduPosLimits(unsigned n,float* data,float *results)
{
  extern __shared__ float spx1[];
  float *spy1=spx1+blockDim.x;
  float *spz1=spy1+blockDim.x;
  float *spx2=spz1+blockDim.x;
  float *spy2=spx2+blockDim.x;
  float *spz2=spy2+blockDim.x;
  const unsigned tid=threadIdx.x;
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of value.
  //-Loads values in shared memory.
  unsigned p2=p;
  spx1[tid]=(p<n? data[p2]: FLT_MAX);  p2+=n;
  spy1[tid]=(p<n? data[p2]: FLT_MAX);  p2+=n;
  spz1[tid]=(p<n? data[p2]: FLT_MAX);  p2+=n;
  spx2[tid]=(p<n? data[p2]: -FLT_MAX); p2+=n;
  spy2[tid]=(p<n? data[p2]: -FLT_MAX); p2+=n;
  spz2[tid]=(p<n? data[p2]: -FLT_MAX);
  __syncthreads();
  //-Reduction of values in shared memory.
  KerPosLimitsRedu<blockSize>(spx1,spy1,spz1,spx2,spy2,spz2,tid,results);
}

//==============================================================================
/// Reduction of limits of position starting from results[].
/// In results[], each block stores xmin,ymin,zmin,xmax,ymax,zmax grouping per block.
//==============================================================================
void ReduPosLimits(unsigned nblocks,float *aux,tfloat3 &pmin,tfloat3 &pmax,JLog2 *log){
  unsigned n=nblocks;
  const unsigned smemSize=DIVBSIZE*sizeof(float)*6;
  dim3 sgrid=GetGridSize(n,DIVBSIZE);
  unsigned n_blocks=sgrid.x*sgrid.y;
  //printf("n:%d  n_blocks:%d]\n",n,n_blocks);
  float *dat=aux;
  float *res=aux+(n_blocks*6);
  while(n>1){
    //printf("##>ReduMaxF n:%d  n_blocks:%d]\n",n,n_blocks);
    //printf("##>ReduMaxF>sgrid=(%d,%d,%d)\n",sgrid.x,sgrid.y,sgrid.z);
    KerReduPosLimits<DIVBSIZE><<<sgrid,DIVBSIZE,smemSize>>>(n,dat,res);
    //CheckErrorCuda("#>ReduMaxF KerReduMaxF failed.");
    n=n_blocks;
    sgrid=GetGridSize(n,DIVBSIZE);  
    n_blocks=sgrid.x*sgrid.y;
    float* x=dat; dat=res; res=x;
  }
  float resf[6];
  cudaMemcpy(resf,dat,sizeof(float)*6,cudaMemcpyDeviceToHost);
  //CheckErrorCuda("#>ReduMaxF cudaMemcpy failed.");
  pmin=TFloat3(resf[0],resf[1],resf[2]);
  pmax=TFloat3(resf[3],resf[4],resf[5]);
}


//------------------------------------------------------------------------------
/// Computes minimum and maximum positions of valid particles.
/// Ignores the particles with check[p]!=0.
/// If the particles is out the limits of position, check[p]=CHECK_OUTPOS.
/// If rhop!=NULL and the particle is out the allowed range (rhopmin,rhopmax)
/// then check[p]=CHECK_OUTRHOP.
/// In results[], each block stores xmin,ymin,zmin,xmax,ymax,zmax grouping per block.
//------------------------------------------------------------------------------
template <unsigned int blockSize,byte periactive> __global__ void KerLimitsPos(unsigned n,unsigned pini,const float3 *pos,const float *rhop,word *code,float3 posmin,float3 difmax,float rhopmin,float rhopmax,float *results)
{
  extern __shared__ float spx1[];
  float *spy1=spx1+blockDim.x;
  float *spz1=spy1+blockDim.x;
  float *spx2=spz1+blockDim.x;
  float *spy2=spx2+blockDim.x;
  float *spz2=spy2+blockDim.x;
  const unsigned tid=threadIdx.x;
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  //-Loads values in shared memory.
  if(p<n){
    const unsigned pp=p+pini;
    float3 rpos=pos[pp];
    float dx=rpos.x-posmin.x,dy=rpos.y-posmin.y,dz=rpos.z-posmin.z;
    bool partin=((periactive&1) || (dx>=0 && dx<difmax.x)) && ((periactive&2) || (dy>=0 && dy<difmax.y)) && ((periactive&4) || (dz>=0 && dz<difmax.z));
    word outcode=CODE_GetOutValue(code[pp]);
    if(!outcode)outcode=(partin? outcode: CODE_OUT_POS);
    if(!outcode && rhop){
      float rrhop=rhop[pp];
      outcode=((rrhop >= rhopmin && rrhop <= rhopmax)? outcode: CODE_OUT_RHOP);
    }
    if(!outcode){
      spx1[tid]=rpos.x; spy1[tid]=rpos.y; spz1[tid]=rpos.z;
      spx2[tid]=rpos.x; spy2[tid]=rpos.y; spz2[tid]=rpos.z;
    }
    else{
      code[pp]|=outcode;
      spx1[tid]=FLT_MAX;  spy1[tid]=FLT_MAX;  spz1[tid]=FLT_MAX;
      spx2[tid]=-FLT_MAX; spy2[tid]=-FLT_MAX; spz2[tid]=-FLT_MAX;
    }
  }
  else{
    spx1[tid]=FLT_MAX;  spy1[tid]=FLT_MAX;  spz1[tid]=FLT_MAX;
    spx2[tid]=-FLT_MAX; spy2[tid]=-FLT_MAX; spz2[tid]=-FLT_MAX;
  }
  __syncthreads();
  //-Reduction of values in shared memory.
  KerPosLimitsRedu<blockSize>(spx1,spy1,spz1,spx2,spy2,spz2,tid,results);
}


//==============================================================================
/// Computes minimum and maximum positions of valid particles.
/// Ignores the particles with check[p]!=0.
/// If the particles is out the limits of position, check[p]=CHECK_OUTPOS.
/// If rhop!=NULL and the particle is out the allowed range (rhopmin,rhopmax)
/// then check[p]=CHECK_OUTRHOP.
/// In results[], each block stores xmin,ymin,zmin,xmax,ymax,zmax grouping per block.
/// If there are no valid particles, the minimum will be higher than the maximum.
//==============================================================================
void LimitsPos(byte periactive,unsigned np,unsigned pini,const float3 *pos,const float *rhop,word *code,tfloat3 posmin,tfloat3 difmax,float rhopmin,float rhopmax,float *aux,tfloat3 &pmin,tfloat3 &pmax,JLog2 *log){
  if(!np){         //-If there are no particles, the process is cancelled.
    pmin=TFloat3(FLT_MAX,FLT_MAX,FLT_MAX);
    pmax=TFloat3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
    return;
  }
  //printf("[ReduMaxF ndata:%d  BLOCKSIZE:%d]\n",ndata,BLOCKSIZE);
  const unsigned smemSize=DIVBSIZE*sizeof(float)*6;
  dim3 sgrid=GetGridSize(np,DIVBSIZE);
  unsigned nblocks=sgrid.x*sgrid.y;
  if(!periactive)       KerLimitsPos<DIVBSIZE,0><<<sgrid,DIVBSIZE,smemSize>>>(np,pini,pos,rhop,code,Float3(posmin),Float3(difmax),rhopmin,rhopmax,aux);
  else if(periactive==1)KerLimitsPos<DIVBSIZE,1><<<sgrid,DIVBSIZE,smemSize>>>(np,pini,pos,rhop,code,Float3(posmin),Float3(difmax),rhopmin,rhopmax,aux);
  else if(periactive==2)KerLimitsPos<DIVBSIZE,2><<<sgrid,DIVBSIZE,smemSize>>>(np,pini,pos,rhop,code,Float3(posmin),Float3(difmax),rhopmin,rhopmax,aux);
  else if(periactive==3)KerLimitsPos<DIVBSIZE,3><<<sgrid,DIVBSIZE,smemSize>>>(np,pini,pos,rhop,code,Float3(posmin),Float3(difmax),rhopmin,rhopmax,aux);
  else if(periactive==4)KerLimitsPos<DIVBSIZE,4><<<sgrid,DIVBSIZE,smemSize>>>(np,pini,pos,rhop,code,Float3(posmin),Float3(difmax),rhopmin,rhopmax,aux);
  else if(periactive==5)KerLimitsPos<DIVBSIZE,5><<<sgrid,DIVBSIZE,smemSize>>>(np,pini,pos,rhop,code,Float3(posmin),Float3(difmax),rhopmin,rhopmax,aux);
  else if(periactive==6)KerLimitsPos<DIVBSIZE,6><<<sgrid,DIVBSIZE,smemSize>>>(np,pini,pos,rhop,code,Float3(posmin),Float3(difmax),rhopmin,rhopmax,aux);
  else if(periactive==7)KerLimitsPos<DIVBSIZE,7><<<sgrid,DIVBSIZE,smemSize>>>(np,pini,pos,rhop,code,Float3(posmin),Float3(difmax),rhopmin,rhopmax,aux);
  ReduPosLimits(nblocks,aux,pmin,pmax,log);
#ifdef DG_LimitsPos
  char cad[1024];
  //sprintf(cad,"LimitsPos_%s> n:%u  pini:%u",(rhop? "Fluid": "Bound"),np,pini); log->Print(cad);
  float3 *posh=new float3[np];
  byte *checkh=new byte[np];
  cudaMemcpy(posh,pos+pini,sizeof(float3)*np,cudaMemcpyDeviceToHost);
  cudaMemcpy(checkh,check+pini,sizeof(byte)*np,cudaMemcpyDeviceToHost);
  tfloat3 pminh=TFloat3(FLT_MAX);
  tfloat3 pmaxh=TFloat3(-FLT_MAX);
  for(unsigned p=0;p<np;p++)if(!checkh[p]){
    float px=posh[p].x,py=posh[p].y,pz=posh[p].z;
    if(pminh.x>px)pminh.x=px;  if(pminh.y>py)pminh.y=py;  if(pminh.z>pz)pminh.z=pz;
    if(pmaxh.x<px)pmaxh.x=px;  if(pmaxh.y<py)pmaxh.y=py;  if(pmaxh.z<pz)pmaxh.z=pz;
  }
  delete[] posh;
  delete[] checkh;
  if(pmin.x!=pminh.x||pmin.y!=pminh.y||pmin.z!=pminh.z||pmax.x!=pmaxh.x||pmax.y!=pmaxh.y||pmax.z!=pmaxh.z){
    sprintf(cad,"LimitsPos> GPU pmin= (%G,%G,%G)  pmax= (%G,%G,%G)",pmin.x,pmin.y,pmin.z,pmax.x,pmax.y,pmax.z); log->Print(cad);
    sprintf(cad,"LimitsPos> CPU pminh=(%G,%G,%G)  pmaxh=(%G,%G,%G)",pminh.x,pminh.y,pminh.z,pmaxh.x,pmaxh.y,pmaxh.z); log->Print(cad);
    throw "Error en LimitsPos()...";
  }
#endif
}


//------------------------------------------------------------------------------
/// Computes initial and final particle of each cell.
//------------------------------------------------------------------------------
__global__ void KerCalcBeginEndCell(unsigned n,unsigned pini,const unsigned *cellpart,int2 *begcell)
{
  extern __shared__ unsigned scell[];    // [blockDim.x+1}
  const unsigned pt=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  const unsigned p=pt+pini;
  unsigned cel;
  if(pt<n){
    cel=cellpart[p];
    scell[threadIdx.x+1]=cel;
    if(pt && !threadIdx.x)scell[0]=cellpart[p-1];
  }
  __syncthreads();
  if(pt<n){
    if(!pt || cel!=scell[threadIdx.x]){
      begcell[cel].x=p;
      if(pt)begcell[scell[threadIdx.x]].y=p;
    }
    if(pt==n-1)begcell[cel].y=p+1;
  }
}

//==============================================================================
/// Computes initial and final particle of each cell.
//==============================================================================
void CalcBeginEndCell(bool full,unsigned npb,unsigned np,unsigned sizebegcell,unsigned cellfluid,const unsigned *cellpart,int2 *begcell){
  if(full)cudaMemset(begcell,0,sizeof(int2)*sizebegcell);
  else cudaMemset(begcell+cellfluid,0,sizeof(int2)*(sizebegcell-cellfluid));
  const unsigned pini=(full? 0: npb);
  const unsigned n=np-pini;
  if(n){
    dim3 sgrid=GetGridSize(n,DIVBSIZE);
    KerCalcBeginEndCell <<<sgrid,DIVBSIZE,sizeof(unsigned)*(DIVBSIZE+1)>>> (n,pini,cellpart,begcell);
  }
}

//------------------------------------------------------------------------------
/// Reorders particle data according to idsort[].
//------------------------------------------------------------------------------
__global__ void KerSortDataParticles(unsigned n,unsigned pini,const unsigned *sortpart,const unsigned *idp,const float3 *pos,const float3 *vel,const float *rhop,const word *code,const float *press,const float *viscop, const unsigned *idpm,unsigned *idp2,float3 *pos2,float3 *vel2,float *rhop2,word *code2,float *press2,float *viscop2,unsigned *idpm2)
{
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned oldpos=(p<pini? p: sortpart[p]);
    idp2[p]=idp[oldpos];
    pos2[p]=pos[oldpos];
    vel2[p]=vel[oldpos];
    rhop2[p]=rhop[oldpos];
    code2[p]=code[oldpos];
	//multi
	press2[p]=press[oldpos];
	viscop2[p]=viscop[oldpos];
	idpm2[p]=idpm[oldpos];

  }
}
//------------------------------------------------------------------------------
__global__ void KerSortDataParticles(unsigned n,unsigned pini,const unsigned *sortpart,const float *a,const float3 *b,float *a2,float3 *b2)
{
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned oldpos=(p<pini? p: sortpart[p]);
    a2[p]=a[oldpos];
    b2[p]=b[oldpos];
  }
}
//------------------------------------------------------------------------------
__global__ void KerSortDataParticles(unsigned n,unsigned pini,const unsigned *sortpart,const float *a,const float3 *b,const float3 *c,float *a2,float3 *b2,float3 *c2)
{
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned oldpos=(p<pini? p: sortpart[p]);
    a2[p]=a[oldpos];
    b2[p]=b[oldpos];
    c2[p]=c[oldpos];
  }
}
//------------------------------------------------------------------------------
__global__ void KerSortDataParticles(unsigned n,unsigned pini,const unsigned *sortpart,const tsymatrix3f *a,tsymatrix3f *a2)
{
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned oldpos=(p<pini? p: sortpart[p]);
    a2[p]=a[oldpos];
  }
}


//==============================================================================
/// Reorders particle data according to sortpart.
//==============================================================================
void SortDataParticles(unsigned n,unsigned pini,const unsigned *sortpart,const unsigned *idp,const float3 *pos,const float3 *vel,const float *rhop,const word *code,const float *press,const float *viscop,const unsigned *idpm, unsigned *idp2,float3 *pos2,float3 *vel2,float *rhop2,word *code2,float *press2,float *viscop2,unsigned *idpm2){
  if(n){
    dim3 sgrid=GetGridSize(n,DIVBSIZE);
    KerSortDataParticles <<<sgrid,DIVBSIZE>>>(n,pini,sortpart,idp,pos,vel,rhop,code,press,viscop,idpm,idp2,pos2,vel2,rhop2,code2,press2,viscop2,idpm2);
  }
}
//==============================================================================
void SortDataParticles(unsigned n,unsigned pini,const unsigned *sortpart,const float *a,const float3 *b,float *a2,float3 *b2){
  if(n){
    dim3 sgrid=GetGridSize(n,DIVBSIZE);
    KerSortDataParticles <<<sgrid,DIVBSIZE>>>(n,pini,sortpart,a,b,a2,b2);
  }
}
//==============================================================================
void SortDataParticles(unsigned n,unsigned pini,const unsigned *sortpart,const float *a,const float3 *b,const float3 *c,float *a2,float3 *b2,float3 *c2){
  if(n){
    dim3 sgrid=GetGridSize(n,DIVBSIZE);
    KerSortDataParticles <<<sgrid,DIVBSIZE>>>(n,pini,sortpart,a,b,c,a2,b2,c2);
  }
}
//==============================================================================
void SortDataParticles(unsigned n,unsigned pini,const unsigned *sortpart,const tsymatrix3f *a,tsymatrix3f *a2){
  if(n){
    dim3 sgrid=GetGridSize(n,DIVBSIZE);
    KerSortDataParticles <<<sgrid,DIVBSIZE>>>(n,pini,sortpart,a,a2);
  }
}


//------------------------------------------------------------------------------
/// Computes minimum and maximum values starting from data[].
//------------------------------------------------------------------------------
template <unsigned int blockSize> __global__ void KerReduUintLimits(unsigned n,unsigned* data,unsigned *results)
{
  extern __shared__ unsigned sp1[];
  unsigned *sp2=sp1+blockDim.x;
  const unsigned tid=threadIdx.x;
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of value.
  //-Loads values in shared memory.
  unsigned p2=p;
  sp1[tid]=(p<n? data[p2]: UINT_MAX);  p2+=n;
  sp2[tid]=(p<n? data[p2]: 0);
  __syncthreads();
  //-Reduction of values in shared memory.
  KerUintLimitsRedu<blockSize>(sp1,sp2,tid,results);
}

//==============================================================================
/// Reduction of the limits of unsgined values starting from results[].
/// In results[], each block stores vmin,vmax grouping per block.
//==============================================================================
void ReduUintLimits(unsigned nblocks,unsigned *aux,unsigned &vmin,unsigned &vmax,JLog2 *log){
  unsigned n=nblocks;
  const unsigned smemSize=DIVBSIZE*sizeof(unsigned)*2;
  dim3 sgrid=GetGridSize(n,DIVBSIZE);
  unsigned n_blocks=sgrid.x*sgrid.y;
  //printf("n:%d  n_blocks:%d]\n",n,n_blocks);
  unsigned *dat=aux;
  unsigned *res=aux+(n_blocks*2);
  while(n>1){
    //printf("##>ReduMaxF n:%d  n_blocks:%d]\n",n,n_blocks);
    //printf("##>ReduMaxF>sgrid=(%d,%d,%d)\n",sgrid.x,sgrid.y,sgrid.z);
    KerReduUintLimits<DIVBSIZE><<<sgrid,DIVBSIZE,smemSize>>>(n,dat,res);
    //CheckErrorCuda("#>ReduMaxF KerReduMaxF failed.");
    n=n_blocks;
    sgrid=GetGridSize(n,DIVBSIZE);  
    n_blocks=sgrid.x*sgrid.y;
    unsigned* x=dat; dat=res; res=x;
  }
  unsigned resf[2];
  cudaMemcpy(resf,dat,sizeof(unsigned)*2,cudaMemcpyDeviceToHost);
  //CheckErrorCuda("#>ReduMaxF cudaMemcpy failed.");
  vmin=resf[0];
  vmax=resf[1];
}


//------------------------------------------------------------------------------
/// Reduction of values in shared memory for a warp of KerReduUintSum.
//------------------------------------------------------------------------------
template <unsigned blockSize> __device__ __forceinline__ void KerUintSumWarpRedu(volatile unsigned* sp1,const unsigned &tid){
  if(blockSize>=64)sp1[tid]+=sp1[tid+32];
  if(blockSize>=32)sp1[tid]+=sp1[tid+16];
  if(blockSize>=16)sp1[tid]+=sp1[tid+ 8];
  if(blockSize>= 8)sp1[tid]+=sp1[tid+ 4];
  if(blockSize>= 4)sp1[tid]+=sp1[tid+ 2];
  if(blockSize>= 2)sp1[tid]+=sp1[tid+ 1];
}

//------------------------------------------------------------------------------
/// Reduction of values in shared memory of KerReduUintSum.
//------------------------------------------------------------------------------
template <unsigned blockSize> __device__ __forceinline__ void KerUintSumRedu(unsigned* sp1,const unsigned &tid,unsigned* results){
  __syncthreads();
  if(blockSize>=512){ if(tid<256)sp1[tid]+=sp1[tid+256]; __syncthreads(); }
  if(blockSize>=256){ if(tid<128)sp1[tid]+=sp1[tid+128]; __syncthreads(); }
  if(blockSize>=128){ if(tid<64) sp1[tid]+=sp1[tid+64];  __syncthreads(); }
  if(tid<32)KerUintSumWarpRedu<blockSize>(sp1,tid);
  if(tid==0)results[blockIdx.y*gridDim.x+blockIdx.x]=sp1[0];
}

//------------------------------------------------------------------------------
/// Returns the summation of values contained in data[].
//------------------------------------------------------------------------------
template <unsigned int blockSize> __global__ void KerReduUintSum(unsigned n,unsigned* data,unsigned *results)
{
  extern __shared__ unsigned sp1[];
  const unsigned tid=threadIdx.x;
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of value.
  //-Loads values in shared memory.
  sp1[tid]=(p<n? data[p]: 0);
  __syncthreads();
  //-Reduction of values in shared memory.
  KerUintSumRedu<blockSize>(sp1,tid,results);
}

//==============================================================================
/// Returns the summation of values contained in aux[].
//==============================================================================
unsigned ReduUintSum(unsigned nblocks,unsigned *aux,JLog2 *log){
  unsigned n=nblocks;
  const unsigned smemSize=DIVBSIZE*sizeof(unsigned);
  dim3 sgrid=GetGridSize(n,DIVBSIZE);
  unsigned n_blocks=sgrid.x*sgrid.y;
  //printf("n:%d  n_blocks:%d]\n",n,n_blocks);
  unsigned *dat=aux;
  unsigned *res=aux+(n_blocks);
  while(n>1){
    //printf("##>ReduMaxF n:%d  n_blocks:%d]\n",n,n_blocks);
    //printf("##>ReduMaxF>sgrid=(%d,%d,%d)\n",sgrid.x,sgrid.y,sgrid.z);
    KerReduUintSum<DIVBSIZE><<<sgrid,DIVBSIZE,smemSize>>>(n,dat,res);
    //CheckErrorCuda("#>ReduMaxF KerReduMaxF failed.");
    n=n_blocks;
    sgrid=GetGridSize(n,DIVBSIZE);  
    n_blocks=sgrid.x*sgrid.y;
    unsigned* x=dat; dat=res; res=x;
  }
  unsigned resf;
  cudaMemcpy(&resf,dat,sizeof(unsigned),cudaMemcpyDeviceToHost);
  //CheckErrorCuda("#>ReduMaxF cudaMemcpy failed.");
  return(resf);
}

//------------------------------------------------------------------------------
/// Returns range of particles in the range of cells.
//------------------------------------------------------------------------------
template <unsigned blockSize> __global__ void KerGetRangeParticlesCells(unsigned ncel,unsigned ini,const int2 *begcell,unsigned *results)
{ //torder{ORDER_XYZ=1,ORDER_YZX=2,ORDER_XZY=3} 
  extern __shared__ unsigned sp1[];
  unsigned *sp2=sp1+blockDim.x;
  const unsigned tid=threadIdx.x;
  const unsigned cel=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of cell.
  //-Loads values in shared memory.
  if(cel<ncel){
    int2 rg=begcell[cel+ini];
    sp1[tid]=(rg.x<rg.y? unsigned(rg.x): UINT_MAX);
    sp2[tid]=(rg.x<rg.y? unsigned(rg.y): 0);
  }
  else{
    sp1[tid]=UINT_MAX;
    sp2[tid]=0;
  }
  __syncthreads();
  //-Reduction of values in shared memory.
  KerUintLimitsRedu<blockSize>(sp1,sp2,tid,results);
}

//==============================================================================
/// Returns range of particles in the range of cells.
//==============================================================================
void GetRangeParticlesCells(unsigned celini,unsigned celfin,const int2 *begcell,unsigned *aux,unsigned &pmin,unsigned &pmax,JLog2 *log){
  unsigned ncel=celfin-celini;
  if(!ncel){       //-If there are no cells, the process is cancelled.
    pmin=UINT_MAX; pmax=0;
    return;
  }
  const unsigned smemSize=DIVBSIZE*sizeof(unsigned)*2;
  dim3 sgrid=GetGridSize(ncel,DIVBSIZE);
  unsigned nblocks=sgrid.x*sgrid.y;
  KerGetRangeParticlesCells<DIVBSIZE><<<sgrid,DIVBSIZE,smemSize>>>(ncel,celini,begcell,aux);
  ReduUintLimits(nblocks,aux,pmin,pmax,log);
#ifdef DG_GetRangeParticlesCells
  char cad[1024];
  sprintf(cad,"GetRangeParticlesCells> ncel:%u  celini:%u",ncel,celini); log->Print(cad);
  int2 *begcellh=new int2[ncel];
  cudaMemcpy(begcellh,begcell+celini,sizeof(int2)*ncel,cudaMemcpyDeviceToHost);
  unsigned pminh=UINT_MAX;
  unsigned pmaxh=0;
  for(unsigned p=0;p<ncel;p++){
    unsigned x=unsigned(begcellh[p].x),y=unsigned(begcellh[p].y);
    if(x<y){
      if(pminh>x)pminh=x;
      if(pmaxh<y)pmaxh=y;
    }
  }
  delete[] begcellh;
  if(pmin!=pminh||pmax!=pmaxh){
    sprintf(cad,"GetRangeParticlesCells> GPU pmin= (%u)  pmax= (%u)",pmin,pmax); log->Print(cad);
    sprintf(cad,"GetRangeParticlesCells> CPU pminh=(%u)  pmaxh=(%u)",pminh,pmaxh); log->Print(cad);
    throw "Error en GetRangeParticlesCells()...";
  }
#endif
}

//------------------------------------------------------------------------------
/// Returns number of particles in the range of cells.
//------------------------------------------------------------------------------
template <unsigned blockSize> __global__ void KerGetParticlesCells(unsigned ncel,unsigned ini,const int2 *begcell,unsigned *results)
{ //torder{ORDER_XYZ=1,ORDER_YZX=2,ORDER_XZY=3} 
  extern __shared__ unsigned sp1[];
  const unsigned tid=threadIdx.x;
  const unsigned cel=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of cell.
  //-Loads values in shared memory.
  if(cel<ncel){
    int2 rg=begcell[cel+ini];
    sp1[tid]=(rg.y>rg.x? unsigned(rg.y-rg.x): 0);
  }
  else sp1[tid]=0;
  __syncthreads();
  //-Reduction of values in shared memory.
  KerUintSumRedu<blockSize>(sp1,tid,results);
}

//==============================================================================
/// Returns number of particles in the range of cells.
//==============================================================================
unsigned GetParticlesCells(unsigned celini,unsigned celfin,const int2 *begcell,unsigned *aux,JLog2 *log){
  unsigned ncel=celfin-celini;
  if(!ncel)return(0); //-If there are no cells, the process is cancelled.
  const unsigned smemSize=DIVBSIZE*sizeof(unsigned);
  dim3 sgrid=GetGridSize(ncel,DIVBSIZE);
  unsigned nblocks=sgrid.x*sgrid.y;
  KerGetParticlesCells<DIVBSIZE><<<sgrid,DIVBSIZE,smemSize>>>(ncel,celini,begcell,aux);
  unsigned sum=ReduUintSum(nblocks,aux,log);
#ifdef DG_GetParticlesCells
  char cad[1024];
  //sprintf(cad,"GetParticlesCells> ncel:%u  celini:%u",ncel,celini); log->PrintDbg(cad);
  int2 *begcellh=new int2[ncel];
  cudaMemcpy(begcellh,begcell+celini,sizeof(int2)*ncel,cudaMemcpyDeviceToHost);
  unsigned sumh=0;
  for(unsigned p=0;p<ncel;p++){
    unsigned x=unsigned(begcellh[p].x),y=unsigned(begcellh[p].y);
    if(y>x)sumh+=(y-x);
  }
  delete[] begcellh;
  if(sum!=sumh){
    sprintf(cad,"GetParticlesCells> GPU sum= (%u)",sum); log->PrintDbg(cad);
    sprintf(cad,"GetParticlesCells> CPU sumh=(%u)",sumh); log->PrintDbg(cad);
    throw "Error en GetParticlesCells()...";
  }
#endif
  return(sum);
}


}


