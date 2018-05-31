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

/// \file JCellDivGpuSingle_ker.cu \brief Implements functions and CUDA kernels to compute operations of the Neighbour List.

#include "JCellDivGpuSingle_ker.h"
#include "Types.h"
#include <float.h>
#include "JLog2.h"

namespace cudiv{

//------------------------------------------------------------------------------
/// Loads idsort[] and cell[] to reorder particles wiht radixsort
//------------------------------------------------------------------------------
__global__ void KerPreSortFull(unsigned np,unsigned npb,const float3 *pos,const word *code,float3 posmin,float3 difmax,uint3 ncells,float ovscell,unsigned *cellpart,unsigned *sortpart)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<np){
    sortpart[p]=p;
    const unsigned nsheet=ncells.x*ncells.y;
    const unsigned cellignore=nsheet*ncells.z; //- cellignore==nct
    const unsigned cellfluid=cellignore+1;
    const unsigned cellboundout=cellfluid+cellignore;
    const unsigned cellfluidout=cellboundout+1;
    float3 rpos=pos[p];
    if(p<npb){     //-Boundary particles.
      float dx=rpos.x-posmin.x,dy=rpos.y-posmin.y,dz=rpos.z-posmin.z;
      unsigned cx=unsigned(dx*ovscell),cy=unsigned(dy*ovscell),cz=unsigned(dz*ovscell);
      if(cx>=ncells.x)cx=ncells.x-1; if(cy>=ncells.y)cx=ncells.y-1; if(cz>=ncells.z)cx=ncells.z-1;
      cellpart[p]=(!CODE_GetOutValue(code[p])? ((dx>=0 && dy>=0 && dz>=0 && dx<difmax.x && dy<difmax.y && dz<difmax.z)? cx+cy*ncells.x+cz*nsheet: cellignore): cellboundout);
    }
    else{          //-Fluid particles.
      unsigned cx=unsigned((rpos.x-posmin.x)*ovscell),cy=unsigned((rpos.y-posmin.y)*ovscell),cz=unsigned((rpos.z-posmin.z)*ovscell);
      if(cx>=ncells.x)cx=ncells.x-1; if(cy>=ncells.y)cx=ncells.y-1; if(cz>=ncells.z)cx=ncells.z-1;
      cellpart[p]=(!CODE_GetOutValue(code[p])? cellfluid+cx+cy*ncells.x+cz*nsheet: cellfluidout);
    }
  }
}
//------------------------------------------------------------------------------
__global__ void KerPreSortFluid(unsigned n,unsigned npb,const float3 *pos,const word *code,float3 posmin,uint3 ncells,float ovscell,unsigned *cellpart,unsigned *sortpart)
{
  unsigned p=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    p+=npb;
    sortpart[p]=p;
    const unsigned nsheet=ncells.x*ncells.y;
    const unsigned cellfluid=nsheet*ncells.z+1;
    const unsigned cellfluidout=cellfluid+cellfluid;
    float3 rpos=pos[p];
    //-Fluid particles.
    unsigned cx=unsigned((rpos.x-posmin.x)*ovscell),cy=unsigned((rpos.y-posmin.y)*ovscell),cz=unsigned((rpos.z-posmin.z)*ovscell);
    if(cx>=ncells.x)cx=ncells.x-1; if(cy>=ncells.y)cx=ncells.y-1; if(cz>=ncells.z)cx=ncells.z-1;
    cellpart[p]=(!CODE_GetOutValue(code[p])? cellfluid+cx+cy*ncells.x+cz*nsheet: cellfluidout);
  }
}

//==============================================================================
/// Computes cell of each particle (CellPart[]) starting from its position, 
/// all excluded particles were already labelled in check[] by a value different from zero.
/// Assigns consecutive values to SortPart[].
/// With full=true, also processes the boundary particles.
//==============================================================================
void PreSort(bool full,unsigned np,unsigned npb,const float3 *pos,const word *code,tfloat3 posmin,tfloat3 difmax,tuint3 ncells,float ovscell,unsigned *cellpart,unsigned *sortpart,JLog2 *log){
  unsigned n=(full? np: np-npb);
  if(n){
    dim3 sgrid=GetGridSize(n,DIVBSIZE);
    if(full)KerPreSortFull<<<sgrid,DIVBSIZE>>>(n,npb,pos,code,Float3(posmin),Float3(difmax),make_uint3(ncells.x,ncells.y,ncells.z),ovscell,cellpart,sortpart);
    else KerPreSortFluid<<<sgrid,DIVBSIZE>>>(n,npb,pos,code,Float3(posmin),make_uint3(ncells.x,ncells.y,ncells.z),ovscell,cellpart,sortpart);
  }
}



}



