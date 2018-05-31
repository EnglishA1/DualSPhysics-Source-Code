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

/// \file JCellDivGpu_ker.h \brief Declares functions and CUDA kernels to compute operations of the Neighbour List.

#ifndef _JCellDivGpu_ker_
#define _JCellDivGpu_ker_

#include "TypesDef.h"
#include <cuda_runtime_api.h>

//#define DG_LimitsPos //-In LimitsPos() checks that the result is correct.
//#define DG_GetRangeParticlesCells //-In GetParticlesCellRange() checks that the result is correct.

class JLog2;

#define DIVBSIZE 256

/// Implements a set of functions and CUDA kernels to compute operations of the Neighbour List.
namespace cudiv{

inline float3 Float3(const tfloat3& v){ float3 p={v.x,v.y,v.z}; return(p); }
inline float3 Float3(float x,float y,float z){ float3 p={x,y,z}; return(p); }
inline tfloat3 ToTFloat3(const float3& v){ return(TFloat3(v.x,v.y,v.z)); }

void Sort(unsigned* keys,unsigned* values,unsigned size,bool stable);

dim3 GetGridSize(unsigned n,unsigned blocksize);
void ReduPosLimits(unsigned nblocks,float *aux,tfloat3 &pmin,tfloat3 &pmax,JLog2 *log);


inline unsigned LimitsPosSize(unsigned ndata){ unsigned n=6,s=((ndata/DIVBSIZE)+1); return((s*n + ((s/DIVBSIZE)+1)*n) + DIVBSIZE); }
void LimitsPos(byte periactive,unsigned np,unsigned pini,const float3 *pos,const float *rhop,word *code,tfloat3 posmin,tfloat3 difmax,float rhopmin,float rhopmax,float *aux,tfloat3 &pmin,tfloat3 &pmax,JLog2 *log);
void CalcBeginEndCell(bool full,unsigned npb,unsigned np,unsigned sizebegcell,unsigned cellfluid,const unsigned *cellpart,int2 *begcell);

void SortDataParticles(unsigned np,unsigned pini,const unsigned *sortpart,const unsigned *idp,const float3 *pos,const float3 *vel,const float *rhop,const word *code,const float *press,const float *viscop,const unsigned *idpm, unsigned *idp2,float3 *pos2,float3 *vel2,float *rhop2,word *code2,float *press2,float *viscop2,unsigned *idpm2);


void SortDataParticles(unsigned np,unsigned pini,const unsigned *sortpart,const float *a,const float3 *b,float *a2,float3 *b2);
void SortDataParticles(unsigned np,unsigned pini,const unsigned *sortpart,const float *a,const float3 *b,const float3 *c,float *a2,float3 *b2,float3 *c2);
void SortDataParticles(unsigned np,unsigned pini,const unsigned *sortpart,const tsymatrix3f *a,tsymatrix3f *a2);

inline unsigned GetRangeParticlesCellsSizeAux(unsigned celini,unsigned celfin){ unsigned n=2,s=(((celfin-celini)/DIVBSIZE)+1); return((s*n + ((s/DIVBSIZE)+1)*n) + DIVBSIZE); } 
void GetRangeParticlesCells(unsigned celini,unsigned celfin,const int2 *begcell,unsigned *aux,unsigned &pmin,unsigned &pmax,JLog2 *log);

inline unsigned GetParticlesCellsSizeAux(unsigned celini,unsigned celfin){ unsigned n=1,s=(((celfin-celini)/DIVBSIZE)+1); return((s*n + ((s/DIVBSIZE)+1)*n) + DIVBSIZE); }  
unsigned GetParticlesCells(unsigned celini,unsigned celfin,const int2 *begcell,unsigned *aux,JLog2 *log);

unsigned GetListParticlesCells(tuint3 celmin,tuint3 celmax,const int2 *begcell,unsigned countlist,unsigned sizelist,unsigned *list);

}

#endif



