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

/// \file JGpuArrays.h \brief Declares the class \ref JGpuArrays.

#ifndef _JGpuArrays_
#define _JGpuArrays_

#pragma warning(disable : 4996) //Cancels sprintf() deprecated.

#include "JObjectGpu.h"
#include "TypesDef.h"


//##############################################################################
//# JGpuArraysSize
//##############################################################################
/// \brief Defines the type of elements of the arrays managed in \ref JGpuArrays. 

class JGpuArraysSize : protected JObjectGpu
{
protected:
  const unsigned ElementSize;
  unsigned ArraySize;

  static const unsigned MAXPOINTERS=30;
  void* Pointers[MAXPOINTERS];
  unsigned Count;
  unsigned CountUsed;

  unsigned CountMax,CountUsedMax;
  
  void FreeMemory();
  unsigned FindPointerUsed(void *pointer)const;

public:
  JGpuArraysSize(unsigned elementsize);
  ~JGpuArraysSize();
  void Reset();
  
  void SetArrayCount(unsigned count);
  unsigned GetArrayCount()const{ return(Count); }
  unsigned GetArrayCountUsed()const{ return(CountUsed); }
  
  unsigned GetArrayCountMax()const{ return(CountMax); }
  unsigned GetArrayCountUsedMax()const{ return(CountUsedMax); }

  void SetArraySize(unsigned size);
  unsigned GetArraySize()const{ return(ArraySize); }

  long long GetAllocMemoryGpu()const{ return((long long)(Count)*ElementSize*ArraySize); };

  void* Reserve();
  void Free(void *pointer);
};

//##############################################################################
//# JGpuArrays
//##############################################################################
/// \brief Manages arrays with memory allocated in GPU. 

class JGpuArrays : protected JObjectGpu
{
public:
  typedef enum{ SIZE_1B=1,SIZE_2B=2,SIZE_4B=4,SIZE_8B=8,SIZE_12B=12,SIZE_16B=16,SIZE_24B=24,SIZE_32B=32 }TpArraySize;  //-Type of arrays.

protected:
  JGpuArraysSize *Arrays1b;
  JGpuArraysSize *Arrays2b;
  JGpuArraysSize *Arrays4b;
  JGpuArraysSize *Arrays8b;
  JGpuArraysSize *Arrays12b;
  JGpuArraysSize *Arrays16b;
  JGpuArraysSize *Arrays24b;
  JGpuArraysSize *Arrays32b;
  
  JGpuArraysSize* GetArrays(TpArraySize tsize)const{ return(tsize==SIZE_32B? Arrays32b: (tsize==SIZE_24B? Arrays24b: (tsize==SIZE_16B? Arrays16b: (tsize==SIZE_12B? Arrays12b: (tsize==SIZE_8B? Arrays8b: (tsize==SIZE_4B? Arrays4b: (tsize==SIZE_2B? Arrays2b: Arrays1b))))))); }

public:
  JGpuArrays();
  ~JGpuArrays();
  void Reset();
  long long GetAllocMemoryGpu()const;
  
  void SetArrayCount(TpArraySize tsize,unsigned count){ GetArrays(tsize)->SetArrayCount(count); }
  void AddArrayCount(TpArraySize tsize,unsigned count=1){ SetArrayCount(tsize,GetArrayCount(tsize)+count); }
  unsigned GetArrayCount(TpArraySize tsize)const{ return(GetArrays(tsize)->GetArrayCount()); }
  unsigned GetArrayCountUsed(TpArraySize tsize)const{ return(GetArrays(tsize)->GetArrayCountUsed()); }

  void SetArraySize(unsigned size);
  unsigned GetArraySize()const{ return(Arrays1b->GetArraySize()); }

  byte* ReserveByte(){ return((byte*)Arrays1b->Reserve()); }
  word* ReserveWord(){ return((word*)Arrays2b->Reserve()); }
  unsigned* ReserveUint(){ return((unsigned*)Arrays4b->Reserve()); }
  float* ReserveFloat(){ return((float*)Arrays4b->Reserve()); }
  float3* ReserveFloat3(){ return((float3*)Arrays12b->Reserve()); }
  float4* ReserveFloat4(){ return((float4*)Arrays16b->Reserve()); }
  double* ReserveDouble(){ return((double*)Arrays8b->Reserve()); }
  double2* ReserveDouble2(){ return((double2*)Arrays16b->Reserve()); }
  double3* ReserveDouble3(){ return((double3*)Arrays24b->Reserve()); }
  double4* ReserveDouble4(){ return((double4*)Arrays32b->Reserve()); }
  tsymatrix3f* ReserveSymatrix3f(){ return((tsymatrix3f*)Arrays24b->Reserve()); }

  void Free(byte *pointer){ Arrays1b->Free(pointer); }
  void Free(word *pointer){ Arrays2b->Free(pointer); }
  void Free(unsigned *pointer){ Arrays4b->Free(pointer); }
  void Free(float *pointer){ Arrays4b->Free(pointer); }
  void Free(float3 *pointer){ Arrays12b->Free(pointer); }
  void Free(float4 *pointer){ Arrays16b->Free(pointer); }
  void Free(double *pointer){ Arrays8b->Free(pointer); }
  void Free(double2 *pointer){ Arrays16b->Free(pointer); }
  void Free(double3 *pointer){ Arrays24b->Free(pointer); }
  void Free(double4 *pointer){ Arrays32b->Free(pointer); }
  void Free(tsymatrix3f *pointer){ Arrays24b->Free(pointer); }
};


#endif



