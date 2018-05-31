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

/// \file JPeriodicGpu_ker.h \brief Declares functions and CUDA kernels to obtain particles that interact with periodic edges.

#ifndef _JPeriodicGpu_ker_
#define _JPeriodicGpu_ker_

#include "Types.h"
#include <cuda_runtime_api.h>

#define PERIODICBSIZE 256

/// Implements a set of functions and CUDA kernels to obtain particles that interact with periodic edges.
namespace cuperi{

inline float3 Float3(const tfloat3& v){ float3 p={v.x,v.y,v.z}; return(p); }
inline float3 Float3(float x,float y,float z){ float3 p={x,y,z}; return(p); }
inline tfloat3 ToTFloat3(const float3& v){ return(TFloat3(v.x,v.y,v.z)); }

dim3 GetGridSize(unsigned n,unsigned blocksize);

void CheckPositionAll(unsigned n,unsigned pini,float3 *pos,bool xrun,float xmin,float xmax,tfloat3 xperinc,bool yrun,float ymin,float ymax,tfloat3 yperinc,bool zrun,float zmin,float zmax,tfloat3 zperinc);

unsigned GetListParticlesCells(unsigned axis,bool bordermax,unsigned n,unsigned pini,tuint3 ncells,unsigned cellini,const unsigned* cellpart,const float4* pospres,tuint3 celmin,tuint3 celmax,tfloat3 posmin,unsigned countlist,unsigned sizelist,unsigned *list);
void PreSortRange(unsigned axis,unsigned np,unsigned npb,unsigned bini,unsigned fini,tfloat3 posinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,float ovscell,const float4 *pospres,unsigned *ref,unsigned *cellpart,unsigned *sortpart,float4 *pospres2);
void PreSortList(unsigned axis,unsigned np,unsigned npb,unsigned listini,const unsigned* list,tfloat3 posinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,float ovscell,const float4 *pospres,unsigned *ref,unsigned *cellpart,unsigned *sortpart,float4 *pospres2);

void SelecPartsTwo(bool modexy,unsigned np,float twposmin,float twposmax,unsigned boxfluid,unsigned boxout,const float4 *pospres,const unsigned *cellpart,unsigned *newparts,unsigned &newnp,unsigned &newnbound);
void PreSortTwo(bool modexy,unsigned axis,unsigned np,const unsigned* newparts,unsigned pini,float twposmin,float twinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,float ovscell,float4 *pospres,unsigned *ref,unsigned *cellpart,unsigned *sortpart);

void SortZone(unsigned* keys,unsigned* values,unsigned size,bool stable);
void SortValues(unsigned* values,unsigned size);
void CalcBeginEndCellZone(unsigned np,unsigned sizebegcell,const unsigned *cellpart,int2 *begcell);
void SortPosParticles(unsigned n,const unsigned *sortpart,const float4 *pospres,const unsigned *ref,float4 *pospres2,unsigned *ref2);
void SortLoadParticles(unsigned n,const unsigned *ref,const word *code,const unsigned *idp,const float4 *velrhop,const tsymatrix3f *tau,word *code2,unsigned *idp2,float4 *velrhop2,tsymatrix3f *tau2);
void SortLoadParticles(unsigned n,const unsigned *ref,const word *code,word *code2);
void CalcCellInterList(unsigned axis,unsigned np,const unsigned *list,unsigned hdiv,unsigned ncx,unsigned nsheet,unsigned boxfluid,const unsigned *cellpart,unsigned *cellpart2);
void CalcCellInterRange(unsigned axis,unsigned np,unsigned npb,unsigned bini,unsigned fini,unsigned hdiv,unsigned ncx,unsigned nsheet,unsigned boxfluid,const unsigned *cellpart,unsigned *cellpart2);


}

#endif



