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

/// \file JPeriodicGpu_ker.cu \brief Implements functions and CUDA kernels to obtain particles that interact with periodic edges.

#include "JPeriodicGpu_ker.h"
#pragma warning(disable : 4267) //Cancels "warning C4267: conversion from 'size_t' to 'int', possible loss of data"
#pragma warning(disable : 4244) //Cancels "warning C4244: conversion from 'unsigned __int64' to 'unsigned int', possible loss of data"
#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace cuperi{

//==============================================================================
/// Returns dimensions of gridsize according to parameters.
//==============================================================================
dim3 GetGridSize(unsigned n,unsigned blocksize){
  dim3 sgrid;//=dim3(1,2,3);
  unsigned nb=unsigned(n+blocksize-1)/blocksize;  //-Total number of blocks to be launched.
  sgrid.x=(nb<=65535? nb: unsigned(sqrt(float(nb))));
  sgrid.y=(nb<=65535? 1: unsigned((nb+sgrid.x-1)/sgrid.x));
  sgrid.z=1;
  return(sgrid);
}

//------------------------------------------------------------------------------
/// Checks position of all particles for all periodic conditions.
//------------------------------------------------------------------------------
template<bool xrun,bool yrun,bool zrun> __global__ void KerCheckPositionAll(unsigned n,unsigned pini,float3 *pos,float xmin,float xmax,float3 xperinc,float ymin,float ymax,float3 yperinc,float zmin,float zmax,float3 zperinc){
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned p2=p+pini;
    bool modif=false;
    float3 rpos=pos[p2];
    if(xrun){
      if(rpos.x<xmin){  rpos=make_float3(rpos.x-xperinc.x,rpos.y-xperinc.y,rpos.z-xperinc.z); modif=true; }
      if(rpos.x>=xmax){ rpos=make_float3(rpos.x+xperinc.x,rpos.y+xperinc.y,rpos.z+xperinc.z); modif=true; }
    }
    if(yrun){
      if(rpos.y<ymin){  rpos=make_float3(rpos.x-yperinc.x,rpos.y-yperinc.y,rpos.z-yperinc.z); modif=true; }
      if(rpos.y>=ymax){ rpos=make_float3(rpos.x+yperinc.x,rpos.y+yperinc.y,rpos.z+yperinc.z); modif=true; }
    }
    if(zrun){
      if(rpos.z<zmin){  rpos=make_float3(rpos.x-zperinc.x,rpos.y-zperinc.y,rpos.z-zperinc.z); modif=true; }
      if(rpos.z>=zmax){ rpos=make_float3(rpos.x+zperinc.x,rpos.y+zperinc.y,rpos.z+zperinc.z); modif=true; }
    }
    if(modif)pos[p2]=rpos;
  }
}

//==============================================================================
/// Checks position of all particles for all periodic conditions.
//==============================================================================
void CheckPositionAll(unsigned n,unsigned pini,float3 *pos,bool xrun,float xmin,float xmax,tfloat3 xperinc,bool yrun,float ymin,float ymax,tfloat3 yperinc,bool zrun,float zmin,float zmax,tfloat3 zperinc)
{
  if(n){
    dim3 sgrid=GetGridSize(n,PERIODICBSIZE);
    if(xrun){ const bool runx=true;
      if(yrun){ const bool runy=true;
        if(zrun)KerCheckPositionAll<runx,runy,true> <<<sgrid,PERIODICBSIZE>>> (n,pini,pos,xmin,xmax,Float3(xperinc),ymin,ymax,Float3(yperinc),zmin,zmax,Float3(zperinc));
        else KerCheckPositionAll<runx,runy,false> <<<sgrid,PERIODICBSIZE>>> (n,pini,pos,xmin,xmax,Float3(xperinc),ymin,ymax,Float3(yperinc),zmin,zmax,Float3(zperinc));
      }
      else{ const bool runy=false;
        if(zrun)KerCheckPositionAll<runx,runy,true> <<<sgrid,PERIODICBSIZE>>> (n,pini,pos,xmin,xmax,Float3(xperinc),ymin,ymax,Float3(yperinc),zmin,zmax,Float3(zperinc));
        else KerCheckPositionAll<runx,runy,false> <<<sgrid,PERIODICBSIZE>>> (n,pini,pos,xmin,xmax,Float3(xperinc),ymin,ymax,Float3(yperinc),zmin,zmax,Float3(zperinc));
      }
    }
    else{ const bool runx=false;
      if(yrun){ const bool runy=true;
        if(zrun)KerCheckPositionAll<runx,runy,true> <<<sgrid,PERIODICBSIZE>>> (n,pini,pos,xmin,xmax,Float3(xperinc),ymin,ymax,Float3(yperinc),zmin,zmax,Float3(zperinc));
        else KerCheckPositionAll<runx,runy,false> <<<sgrid,PERIODICBSIZE>>> (n,pini,pos,xmin,xmax,Float3(xperinc),ymin,ymax,Float3(yperinc),zmin,zmax,Float3(zperinc));
      }
      else{ const bool runy=false;
        if(zrun)KerCheckPositionAll<runx,runy,true> <<<sgrid,PERIODICBSIZE>>> (n,pini,pos,xmin,xmax,Float3(xperinc),ymin,ymax,Float3(yperinc),zmin,zmax,Float3(zperinc));
        else KerCheckPositionAll<runx,runy,false> <<<sgrid,PERIODICBSIZE>>> (n,pini,pos,xmin,xmax,Float3(xperinc),ymin,ymax,Float3(yperinc),zmin,zmax,Float3(zperinc));
      }
    }
  }
}

//------------------------------------------------------------------------------
/// Gets list of particles of a region of cells.
//------------------------------------------------------------------------------
template<unsigned axis,bool bordermax> __global__ void KerGetListParticlesCells(unsigned n,unsigned pini,int ncx,int nsheet,unsigned cellini,const unsigned* cellpart,const float4* pospres,int celmin,int celmax,float posmin,unsigned countlist,unsigned sizelist,unsigned* list){
  extern __shared__ unsigned slist[];
  if(!threadIdx.x)slist[0]=0;
  __syncthreads();
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned p2=p+pini;
    const int cel=cellpart[p2]-cellini;
    bool sel=false;
    //-Checks if particle is within the range of cells and position.
    if(axis==0){   //-X-Axis.
      const int cx=cel%ncx;
      sel=(celmin<=cx && cx<=celmax);
      if(sel && bordermax)sel=(posmin<=pospres[p2].x);
    }
    if(axis==1){   //-Y-Axis.
      const int cy=int((cel%nsheet)/ncx);
      sel=(celmin<=cy && cy<=celmax);
      if(sel && bordermax)sel=(posmin<=pospres[p2].y);
    }
    if(axis==2){   //-Z-Axis.
      const int cz=int(cel/nsheet);
      sel=(celmin<=cz && cz<=celmax);
      if(sel && bordermax)sel=(posmin<=pospres[p2].z);
    }
    if(sel){
      unsigned cp=atomicAdd(slist,1);
      slist[cp+2]=p2;
    }
  }
  __syncthreads();
  const unsigned ns=slist[0];
  if(!threadIdx.x && ns)slist[1]=atomicAdd((list+sizelist),ns);
  __syncthreads();
  if(threadIdx.x<ns){
    unsigned cp=slist[1]+threadIdx.x;
    if(cp<sizelist)list[cp]=slist[threadIdx.x+2];
  }
}

//==============================================================================
/// Adds particles to a list and returns the total number of particles in the list,
/// this value can be higher than the size of the list.
/// - sizelist: is the maximum number of particles that the list can contain and
///   position where the number of stored particles is recorded.
/// - countlist: is the number of particles previously stored.
/// - with bordermax, also checks the minimum position.
//==============================================================================
unsigned GetListParticlesCells(unsigned axis,bool bordermax,unsigned n,unsigned pini,tuint3 ncells,unsigned cellini,const unsigned* cellpart,const float4* pospres,tuint3 celmin,tuint3 celmax,tfloat3 posmin,unsigned countlist,unsigned sizelist,unsigned *list){
  if(!countlist)cudaMemset(list+sizelist,0,sizeof(unsigned));
  dim3 sgrid=GetGridSize(n,PERIODICBSIZE);
  const unsigned smem=(PERIODICBSIZE+2)*sizeof(unsigned);
  if(bordermax){ const bool bormax=true;
    if(!axis)       KerGetListParticlesCells<0,bormax> <<<sgrid,PERIODICBSIZE,smem>>> (n,pini,ncells.x,ncells.x*ncells.y,cellini,cellpart,pospres,celmin.x,celmax.x,posmin.x,countlist,sizelist,list);
    else if(axis==1)KerGetListParticlesCells<1,bormax> <<<sgrid,PERIODICBSIZE,smem>>> (n,pini,ncells.x,ncells.x*ncells.y,cellini,cellpart,pospres,celmin.y,celmax.y,posmin.y,countlist,sizelist,list);
    else if(axis==2)KerGetListParticlesCells<2,bormax> <<<sgrid,PERIODICBSIZE,smem>>> (n,pini,ncells.x,ncells.x*ncells.y,cellini,cellpart,pospres,celmin.z,celmax.z,posmin.z,countlist,sizelist,list);
  }
  else{ const bool bormax=false;
    if(!axis)       KerGetListParticlesCells<0,bormax> <<<sgrid,PERIODICBSIZE,smem>>> (n,pini,ncells.x,ncells.x*ncells.y,cellini,cellpart,pospres,celmin.x,celmax.x,posmin.x,countlist,sizelist,list);
    else if(axis==1)KerGetListParticlesCells<1,bormax> <<<sgrid,PERIODICBSIZE,smem>>> (n,pini,ncells.x,ncells.x*ncells.y,cellini,cellpart,pospres,celmin.y,celmax.y,posmin.y,countlist,sizelist,list);
    else if(axis==2)KerGetListParticlesCells<2,bormax> <<<sgrid,PERIODICBSIZE,smem>>> (n,pini,ncells.x,ncells.x*ncells.y,cellini,cellpart,pospres,celmin.z,celmax.z,posmin.z,countlist,sizelist,list);
  }
  unsigned count;
  cudaMemcpy(&count,list+sizelist,sizeof(unsigned),cudaMemcpyDeviceToHost);
  return(count);
}

//------------------------------------------------------------------------------
/// Copies particle data in the given range and computes cell for particles within the periodic zone. 
//------------------------------------------------------------------------------
template<int axis> __global__ void KerPreSortRange(unsigned np,unsigned npb,unsigned bini,unsigned fini,float3 posinc,float3 posmin,float3 difmax,unsigned nc1,unsigned nc2,float ovscell,const float4 *pospres,unsigned *ref,unsigned *cellpart,unsigned *sortpart,float4 *pospres2)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<np){
    const unsigned p2=p+(p<npb? bini: fini-npb);
    const unsigned boxini=(p<npb? 0: nc1*nc2);
    const unsigned boxout=nc1*nc2*2+(p<npb? 0: 1);
    ref[p]=p2;
    sortpart[p]=p;
    float4 rpos=pospres[p2];
    rpos.x+=posinc.x; rpos.y+=posinc.y; rpos.z+=posinc.z;
    pospres2[p]=rpos;
    float dx=rpos.x-posmin.x,dy=rpos.y-posmin.y,dz=rpos.z-posmin.z;
    bool ok=(dx>=0 && dy>=0 && dz>=0 && dx<difmax.x && dy<difmax.y && dz<difmax.z);
    unsigned c1,c2;
    if(axis==0){ c1=(ok? unsigned(dy*ovscell): nc1); c2=unsigned(dz*ovscell); }
    if(axis==1){ c1=(ok? unsigned(dx*ovscell): nc1); c2=unsigned(dz*ovscell); }
    if(axis==2){ c1=(ok? unsigned(dx*ovscell): nc1); c2=unsigned(dy*ovscell); }
    cellpart[p]=(c1<nc1 && c2<nc2? boxini+c2*nc1+c1: boxout);
  }
}


//==============================================================================
/// Copies particle data in the given range and computes cell for particles within the periodic zone. 
//==============================================================================
void PreSortRange(unsigned axis,unsigned np,unsigned npb,unsigned bini,unsigned fini,tfloat3 posinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,float ovscell,const float4 *pospres,unsigned *ref,unsigned *cellpart,unsigned *sortpart,float4 *pospres2){
  if(np){
    dim3 sgrid=GetGridSize(np,PERIODICBSIZE);
    if(axis==0)KerPreSortRange<0> <<<sgrid,PERIODICBSIZE>>> (np,npb,bini,fini,Float3(posinc),Float3(posmin),Float3(difmax),nc1,nc2,ovscell,pospres,ref,cellpart,sortpart,pospres2);
    else if(axis==1)KerPreSortRange<1> <<<sgrid,PERIODICBSIZE>>> (np,npb,bini,fini,Float3(posinc),Float3(posmin),Float3(difmax),nc1,nc2,ovscell,pospres,ref,cellpart,sortpart,pospres2);
    else if(axis==2)KerPreSortRange<2> <<<sgrid,PERIODICBSIZE>>> (np,npb,bini,fini,Float3(posinc),Float3(posmin),Float3(difmax),nc1,nc2,ovscell,pospres,ref,cellpart,sortpart,pospres2);
  }
}

//------------------------------------------------------------------------------
/// Copies particle data in the list range and computes cell for particles within the periodic zone. 
//------------------------------------------------------------------------------
template<int axis> __global__ void KerPreSortList(unsigned np,unsigned npb,unsigned listini,const unsigned* list,float3 posinc,float3 posmin,float3 difmax,unsigned nc1,unsigned nc2,float ovscell,const float4 *pospres,unsigned *ref,unsigned *cellpart,unsigned *sortpart,float4 *pospres2)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<np){
    const unsigned p2=list[p+listini];
    const unsigned boxini=(p<npb? 0: nc1*nc2);
    const unsigned boxout=nc1*nc2*2+(p<npb? 0: 1);
    ref[p]=p2;
    sortpart[p]=p;
    float4 rpos=pospres[p2];
    rpos.x+=posinc.x; rpos.y+=posinc.y; rpos.z+=posinc.z;
    pospres2[p]=rpos;
    float dx=rpos.x-posmin.x,dy=rpos.y-posmin.y,dz=rpos.z-posmin.z;
    bool ok=(dx>=0 && dy>=0 && dz>=0 && dx<difmax.x && dy<difmax.y && dz<difmax.z);
    unsigned c1,c2;
    if(axis==0){ c1=(ok? unsigned(dy*ovscell): nc1); c2=unsigned(dz*ovscell); }
    if(axis==1){ c1=(ok? unsigned(dx*ovscell): nc1); c2=unsigned(dz*ovscell); }
    if(axis==2){ c1=(ok? unsigned(dx*ovscell): nc1); c2=unsigned(dy*ovscell); }
    cellpart[p]=(c1<nc1 && c2<nc2? boxini+c2*nc1+c1: boxout);
  }
}

//==============================================================================
/// Copies particle data in the list range and computes cell for particles within the periodic zone. 
//==============================================================================
void PreSortList(unsigned axis,unsigned np,unsigned npb,unsigned listini,const unsigned* list,tfloat3 posinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,float ovscell,const float4 *pospres,unsigned *ref,unsigned *cellpart,unsigned *sortpart,float4 *pospres2){
  if(np){
    dim3 sgrid=GetGridSize(np,PERIODICBSIZE);
    if(axis==0)KerPreSortList<0> <<<sgrid,PERIODICBSIZE>>> (np,npb,listini,list,Float3(posinc),Float3(posmin),Float3(difmax),nc1,nc2,ovscell,pospres,ref,cellpart,sortpart,pospres2);
    else if(axis==1)KerPreSortList<1> <<<sgrid,PERIODICBSIZE>>> (np,npb,listini,list,Float3(posinc),Float3(posmin),Float3(difmax),nc1,nc2,ovscell,pospres,ref,cellpart,sortpart,pospres2);
    else if(axis==2)KerPreSortList<2> <<<sgrid,PERIODICBSIZE>>> (np,npb,listini,list,Float3(posinc),Float3(posmin),Float3(difmax),nc1,nc2,ovscell,pospres,ref,cellpart,sortpart,pospres2);
  }
}


//------------------------------------------------------------------------------
/// Duplicates selected particles for periodic conditions XY, XZ or YZ. 
//------------------------------------------------------------------------------
template<bool modexy,int axis> __global__ void KerPreSortTwo(unsigned np,const unsigned* newparts,unsigned pini,float twposmin,float twinc,float3 posmin,float3 difmax,unsigned nc1,unsigned nc2,float ovscell,float4 *pospres,unsigned *ref,unsigned *cellpart,unsigned *sortpart)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<np){
    const unsigned p1=newparts[p];
    const unsigned p2=pini+p;
    float4 rpos=pospres[p1];
    if(modexy)rpos.y+=(rpos.y<twposmin? twinc: -twinc);    //-PMODE_XY
    else rpos.z+=(rpos.z<twposmin? twinc: -twinc);         //-PMODE_XZ or PMODE_YZ
    ref[p2]=ref[p1];
    sortpart[p2]=p2;
    pospres[p2]=rpos;
    //-Computes cell.
    const unsigned boxfluid=nc1*nc2;
    const bool isbound=(cellpart[p1]<boxfluid);
    const unsigned boxini=(isbound? 0: boxfluid);
    const unsigned boxout=boxfluid+boxfluid+(isbound? 0: 1);
    float dx=rpos.x-posmin.x,dy=rpos.y-posmin.y,dz=rpos.z-posmin.z;
    bool ok=(dx>=0 && dy>=0 && dz>=0 && dx<difmax.x && dy<difmax.y && dz<difmax.z);
    unsigned c1,c2;
    if(axis==0){ c1=(ok? unsigned(dy*ovscell): nc1); c2=unsigned(dz*ovscell); }
    if(axis==1){ c1=(ok? unsigned(dx*ovscell): nc1); c2=unsigned(dz*ovscell); }
    if(axis==2){ c1=(ok? unsigned(dx*ovscell): nc1); c2=unsigned(dy*ovscell); }
    cellpart[p2]=(c1<nc1 && c2<nc2? boxini+c2*nc1+c1: boxout);
  }
}

//==============================================================================
/// Duplicates selected particles for periodic conditions XY, XZ or YZ. 
//==============================================================================
void PreSortTwo(bool modexy,unsigned axis,unsigned np,const unsigned* newparts,unsigned pini,float twposmin,float twinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,float ovscell,float4 *pospres,unsigned *ref,unsigned *cellpart,unsigned *sortpart){
  if(np){
    dim3 sgrid=GetGridSize(np,PERIODICBSIZE);
    if(modexy){ const bool tmodexy=true;
      if(axis==0)     KerPreSortTwo<tmodexy,0> <<<sgrid,PERIODICBSIZE>>> (np,newparts,pini,twposmin,twinc,Float3(posmin),Float3(difmax),nc1,nc2,ovscell,pospres,ref,cellpart,sortpart);
      else if(axis==1)KerPreSortTwo<tmodexy,1> <<<sgrid,PERIODICBSIZE>>> (np,newparts,pini,twposmin,twinc,Float3(posmin),Float3(difmax),nc1,nc2,ovscell,pospres,ref,cellpart,sortpart);
      else if(axis==2)KerPreSortTwo<tmodexy,2> <<<sgrid,PERIODICBSIZE>>> (np,newparts,pini,twposmin,twinc,Float3(posmin),Float3(difmax),nc1,nc2,ovscell,pospres,ref,cellpart,sortpart);
    }
    else{ const bool tmodexy=false;
      if(axis==0)     KerPreSortTwo<tmodexy,0> <<<sgrid,PERIODICBSIZE>>> (np,newparts,pini,twposmin,twinc,Float3(posmin),Float3(difmax),nc1,nc2,ovscell,pospres,ref,cellpart,sortpart);
      else if(axis==1)KerPreSortTwo<tmodexy,1> <<<sgrid,PERIODICBSIZE>>> (np,newparts,pini,twposmin,twinc,Float3(posmin),Float3(difmax),nc1,nc2,ovscell,pospres,ref,cellpart,sortpart);
      else if(axis==2)KerPreSortTwo<tmodexy,2> <<<sgrid,PERIODICBSIZE>>> (np,newparts,pini,twposmin,twinc,Float3(posmin),Float3(difmax),nc1,nc2,ovscell,pospres,ref,cellpart,sortpart);
    }
  }
}

//------------------------------------------------------------------------------
/// Selects particles to be duplicated. 
//------------------------------------------------------------------------------
template<bool modexy> __global__ void KerSelecPartsTwo(unsigned np,float twposmin,float twposmax,unsigned boxfluid,unsigned boxout,const float4 *pospres,const unsigned *cellpart,unsigned *newparts)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<np){
    const unsigned cell=cellpart[p];
    const float rpos=(modexy? pospres[p].y: pospres[p].z); //-PMODE_XY or (PMODE_XZ & PMODE_YZ)
    if(cell<boxout && (rpos<twposmin || rpos>=twposmax)){
      unsigned cp=atomicAdd(newparts,1);
      newparts[cp+2]=p;
      if(cell<boxfluid)atomicAdd((newparts+1),1);
    }
  }
}

//==============================================================================
/// Selects particles to be duplicated.
//==============================================================================
void SelecPartsTwo(bool modexy,unsigned np,float twposmin,float twposmax,unsigned boxfluid,unsigned boxout,const float4 *pospres,const unsigned *cellpart,unsigned *newparts,unsigned &newnp,unsigned &newnbound){
  newnp=newnbound=0;
  if(np){
    cudaMemset(newparts,0,sizeof(unsigned)*2);
    dim3 sgrid=GetGridSize(np,PERIODICBSIZE);
    if(modexy)KerSelecPartsTwo<true> <<<sgrid,PERIODICBSIZE>>> (np,twposmin,twposmax,boxfluid,boxout,pospres,cellpart,newparts);
    else KerSelecPartsTwo<false> <<<sgrid,PERIODICBSIZE>>> (np,twposmin,twposmax,boxfluid,boxout,pospres,cellpart,newparts);
    unsigned count[2];
    cudaMemcpy(count,newparts,sizeof(unsigned)*2,cudaMemcpyDeviceToHost);
    newnp=count[0];
    newnbound=count[1];
  }
}

//==============================================================================
/// Reorders values using RadixSort of thrust.
//==============================================================================
void SortZone(unsigned* keys,unsigned* values,unsigned size,bool stable){
  if(size){
    thrust::device_ptr<unsigned> dev_keysg(keys);
    thrust::device_ptr<unsigned> dev_valuesg(values);
    if(!stable)thrust::sort_by_key(dev_keysg,dev_keysg+size,dev_valuesg);
    else thrust::stable_sort_by_key(dev_keysg,dev_keysg+size,dev_valuesg);
  }
}

//==============================================================================
/// Reorders values using RadixSort of thrust.
//==============================================================================
void SortValues(unsigned* values,unsigned size){
  if(size){
    thrust::device_ptr<unsigned> dev_valuesg(values);
    thrust::sort(dev_valuesg,dev_valuesg+size);
  }
}

//------------------------------------------------------------------------------
/// Computes initial and final particle of each cell.
//------------------------------------------------------------------------------
__global__ void KerCalcBeginEndCellZone(unsigned n,const unsigned *cellpart,int2 *begcell)
{
  extern __shared__ unsigned scell[];    // [blockDim.x+1}
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  unsigned cel;
  if(p<n){
    cel=cellpart[p];
    scell[threadIdx.x+1]=cel;
    if(p && !threadIdx.x)scell[0]=cellpart[p-1];
  }
  __syncthreads();
  if(p<n){
    if(!p || cel!=scell[threadIdx.x]){
      begcell[cel].x=p;
      if(p)begcell[scell[threadIdx.x]].y=p;
    }
    if(p==n-1)begcell[cel].y=p+1;
  }
}

//==============================================================================
/// Computes initial and final particle of each cell.
//==============================================================================
void CalcBeginEndCellZone(unsigned np,unsigned sizebegcell,const unsigned *cellpart,int2 *begcell){
  cudaMemset(begcell,0,sizeof(int2)*sizebegcell);
  if(np){
    dim3 sgrid=GetGridSize(np,PERIODICBSIZE);
    KerCalcBeginEndCellZone <<<sgrid,PERIODICBSIZE,sizeof(unsigned)*(PERIODICBSIZE+1)>>> (np,cellpart,begcell);
  }
}

//------------------------------------------------------------------------------
/// Reorders particle data according to sortpart.
//------------------------------------------------------------------------------
__global__ void KerSortPosParticles(unsigned n,const unsigned *sortpart,const float4 *pospres,const unsigned *ref,float4 *pospres2,unsigned *ref2)
{
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned oldpos=sortpart[p];
    pospres2[p]=pospres[oldpos];
    ref2[p]=ref[oldpos];
  }
}

//==============================================================================
/// Reorders particle data according to sortpart.
//==============================================================================
void SortPosParticles(unsigned n,const unsigned *sortpart,const float4 *pospres,const unsigned *ref,float4 *pospres2,unsigned *ref2){
  if(n){
    dim3 sgrid=GetGridSize(n,PERIODICBSIZE);
    KerSortPosParticles <<<sgrid,PERIODICBSIZE>>>(n,sortpart,pospres,ref,pospres2,ref2);
  }
}

//------------------------------------------------------------------------------
/// Reorders ordered particle data.
//------------------------------------------------------------------------------
template<bool sorttau>__global__ void KerSortLoadParticles(unsigned n,const unsigned *ref,const word *code,const unsigned *idp,const float4 *velrhop,const tsymatrix3f *tau,word *code2,unsigned *idp2,float4 *velrhop2,tsymatrix3f *tau2)
{
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned refpos=ref[p];
    code2[p]=code[refpos];
    idp2[p]=idp[refpos];
    velrhop2[p]=velrhop[refpos];
    if(sorttau)tau2[p]=tau[refpos];
  }
}

//==============================================================================
/// Reorders ordered particle data.
//==============================================================================
void SortLoadParticles(unsigned n,const unsigned *ref,const word *code,const unsigned *idp,const float4 *velrhop,const tsymatrix3f *tau,word *code2,unsigned *idp2,float4 *velrhop2,tsymatrix3f *tau2){
  if(n){
    dim3 sgrid=GetGridSize(n,PERIODICBSIZE);
    if(tau && tau2)KerSortLoadParticles<true>  <<<sgrid,PERIODICBSIZE>>>(n,ref,code,idp,velrhop,tau,code2,idp2,velrhop2,tau2);
    else           KerSortLoadParticles<false> <<<sgrid,PERIODICBSIZE>>>(n,ref,code,idp,velrhop,tau,code2,idp2,velrhop2,tau2);
  }
}

//------------------------------------------------------------------------------
/// Reorders ordered particle data.
//------------------------------------------------------------------------------
__global__ void KerSortLoadParticles(unsigned n,const unsigned *ref,const word *code,word *code2)
{
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned refpos=ref[p];
    code2[p]=code[refpos];
  }
}

//==============================================================================
/// Reorders ordered particle data.
//==============================================================================
void SortLoadParticles(unsigned n,const unsigned *ref,const word *code,word *code2){
  if(n){
    dim3 sgrid=GetGridSize(n,PERIODICBSIZE);
    KerSortLoadParticles <<<sgrid,PERIODICBSIZE>>>(n,ref,code,code2);
  }
}

//------------------------------------------------------------------------------
/// Computes index of cell of the particles of Inter for interaction in the periodic zone.
//------------------------------------------------------------------------------
template<int axis> __global__ void KerCalcCellInterList(unsigned n,const unsigned *list,unsigned hdiv,unsigned ncx,unsigned nsheet,unsigned boxfluid,const unsigned *cellpart,unsigned *cellpart2)
{
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    unsigned cel=cellpart[list[p]];
    if(cel>=boxfluid)cel-=boxfluid;
    //-Gets coordinates of cells of domain of the particle.
    int cx=cel%ncx;
    int cz=int(cel/nsheet);
    int cy=int((cel%nsheet)/ncx);
    //-Computes coordinates for the periodic zone.
    unsigned c1,c2;
    if(axis==0){ c1=cy; c2=cz; }
    if(axis==1){ c1=cx; c2=cz; }
    if(axis==2){ c1=cx; c2=cy; }
    c1+=hdiv; c2+=hdiv;
    cellpart2[p]=unsigned((c1<<16)|(c2&0xffff));
  }
}

//==============================================================================
/// Computes index of cell of the particles of Inter for interaction in the periodic zone.
//==============================================================================
void CalcCellInterList(unsigned axis,unsigned np,const unsigned *list,unsigned hdiv,unsigned ncx,unsigned nsheet,unsigned boxfluid,const unsigned *cellpart,unsigned *cellpart2){
  if(np){
    dim3 sgrid=GetGridSize(np,PERIODICBSIZE);
    if(axis==0)KerCalcCellInterList<0> <<<sgrid,PERIODICBSIZE>>> (np,list,hdiv,ncx,nsheet,boxfluid,cellpart,cellpart2);
    else if(axis==1)KerCalcCellInterList<1> <<<sgrid,PERIODICBSIZE>>> (np,list,hdiv,ncx,nsheet,boxfluid,cellpart,cellpart2);
    else if(axis==2)KerCalcCellInterList<2> <<<sgrid,PERIODICBSIZE>>> (np,list,hdiv,ncx,nsheet,boxfluid,cellpart,cellpart2);
  }
}

//------------------------------------------------------------------------------
/// Computes index of cell of the particles of Inter for interaction in the periodic zone.
//------------------------------------------------------------------------------
template<int axis> __global__ void KerCalcCellInterRange(unsigned n,unsigned npb,unsigned bini,unsigned fini,unsigned hdiv,unsigned ncx,unsigned nsheet,unsigned boxfluid,const unsigned *cellpart,unsigned *cellpart2)
{
  const unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    unsigned p1=p+(p<npb? bini: fini-npb);
    unsigned cel=cellpart[p1];
    if(cel>=boxfluid)cel-=boxfluid;
    //-Gets coordinates of cells of domain of the particle.
    int cx=cel%ncx;
    int cz=int(cel/nsheet);
    int cy=int((cel%nsheet)/ncx);
    //-Computes coordinates for the periodic zone.
    unsigned c1,c2;
    if(axis==0){ c1=cy; c2=cz; }
    if(axis==1){ c1=cx; c2=cz; }
    if(axis==2){ c1=cx; c2=cy; }
    c1+=hdiv; c2+=hdiv;
    cellpart2[p]=unsigned((c1<<16)|(c2&0xffff));
  }
}

//==============================================================================
/// Computes index of cell of the particles of Inter for interaction in the periodic zone.
//==============================================================================
void CalcCellInterRange(unsigned axis,unsigned np,unsigned npb,unsigned bini,unsigned fini,unsigned hdiv,unsigned ncx,unsigned nsheet,unsigned boxfluid,const unsigned *cellpart,unsigned *cellpart2){
  if(np){
    dim3 sgrid=GetGridSize(np,PERIODICBSIZE);
    if(axis==0)KerCalcCellInterRange<0> <<<sgrid,PERIODICBSIZE>>> (np,npb,bini,fini,hdiv,ncx,nsheet,boxfluid,cellpart,cellpart2);
    else if(axis==1)KerCalcCellInterRange<1> <<<sgrid,PERIODICBSIZE>>> (np,npb,bini,fini,hdiv,ncx,nsheet,boxfluid,cellpart,cellpart2);
    else if(axis==2)KerCalcCellInterRange<2> <<<sgrid,PERIODICBSIZE>>> (np,npb,bini,fini,hdiv,ncx,nsheet,boxfluid,cellpart,cellpart2);
  }
}


}


