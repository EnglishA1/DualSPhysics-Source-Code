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

/// \file JGpuArrays.cpp \brief Implements the class \ref JGpuArrays.

#include "JGpuArrays.h"
#include <cstdio>

using namespace std;

//==============================================================================
/// Constructor.
//==============================================================================
JGpuArraysSize::JGpuArraysSize(unsigned elementsize):ElementSize(elementsize){
  ClassName="JGpuArraysSize";
  for(unsigned c=0;c<MAXPOINTERS;c++)Pointers[c]=NULL;
  Count=0;
  CountMax=CountUsedMax=0;
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JGpuArraysSize::~JGpuArraysSize(){
  Reset();
}
 
//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JGpuArraysSize::Reset(){
  FreeMemory();
  ArraySize=0;
}

//==============================================================================
/// Releases allocated memory.
//==============================================================================
void JGpuArraysSize::FreeMemory(){
  for(unsigned c=0;c<Count;c++)if(Pointers[c]){ cudaFree(Pointers[c]); Pointers[c]=NULL; }
  CountUsed=Count=0;
}

//==============================================================================
/// Changes the number of allocated arrays. 
/// Assigns new arrays or releases the current ones without use.
/// If count is lower than the number of arrays in use, throws and exception
//==============================================================================
void JGpuArraysSize::SetArrayCount(unsigned count){
  const char met[]="SetArrayCount";
  if(count>MAXPOINTERS)RunException(met,"The number of requested arrays is higher than the maximum.");
  if(count<CountUsed)RunException(met,"Memory of arrays in use can not be released.");
  if(ArraySize){
    if(Count<count){         //-Generates new arrays. 
      for(unsigned c=Count;c<count;c++)cudaMalloc((void**)(Pointers+c),ElementSize*ArraySize);
      CheckCudaError(met,"GPU memory allocation failed.");
    }
    if(Count>count){         //-Releases arrays. 
      for(unsigned c=count;c<Count;c++){ cudaFree(Pointers[c]); Pointers[c]=NULL; }
    }
  }
  Count=count;
  CountMax=max(CountMax,Count);
}

//==============================================================================
/// Changes the number of elements of the arrays.
/// If there is some array in use, it throws and exception.
//==============================================================================
void JGpuArraysSize::SetArraySize(unsigned size){
  if(CountUsed)RunException("SetArraySize","The dimension of arrays can not be changed since there are some arrays in use.");
  if(ArraySize!=size){
    ArraySize=size;
    unsigned count=Count;
    FreeMemory();
    if(count)SetArrayCount(count);
  }
}

//==============================================================================
/// Asks to reserve an array.
//==============================================================================
void* JGpuArraysSize::Reserve(){
  if(CountUsed==Count||!ArraySize){
    char cad[128]; sprintf(cad,"There are no arrays available of %u bytes.",ElementSize);
    RunException("Reserve",cad);
  }
  CountUsed++;
  CountUsedMax=max(CountUsedMax,CountUsed);
  return(Pointers[CountUsed-1]);
}

//==============================================================================
/// Returns the position of a given pointer.
/// If it does not exist, it returns MAXPOINTERS.
//==============================================================================
unsigned JGpuArraysSize::FindPointerUsed(void *pointer)const{
  unsigned pos=0;
  for(;pos<CountUsed&&Pointers[pos]!=pointer;pos++);
  return(pos>=CountUsed? MAXPOINTERS: pos);
}

//==============================================================================
/// Releases the reserve of an array.
//==============================================================================
void JGpuArraysSize::Free(void *pointer){
  if(pointer){
    unsigned pos=FindPointerUsed(pointer);
    if(pos==MAXPOINTERS)RunException("Free","The given pointer was not reserved.");
    if(pos+1<CountUsed){
      void *aux=Pointers[CountUsed-1]; Pointers[CountUsed-1]=Pointers[pos]; Pointers[pos]=aux;
    }
    CountUsed--;
  }
}  


//##############################################################################
//==============================================================================
//##############################################################################
//==============================================================================
/// Constructor.
//==============================================================================
JGpuArrays::JGpuArrays(){
  ClassName="JGpuArrays";
  Arrays1b=new JGpuArraysSize(1);
  Arrays2b=new JGpuArraysSize(2);
  Arrays4b=new JGpuArraysSize(4);
  Arrays8b=new JGpuArraysSize(8);
  Arrays12b=new JGpuArraysSize(12);
  Arrays16b=new JGpuArraysSize(16);
  Arrays24b=new JGpuArraysSize(24);
  Arrays32b=new JGpuArraysSize(32);
}

//==============================================================================
/// Destructor.
//==============================================================================
JGpuArrays::~JGpuArrays(){
  //printf("____JGpuArrays> 1b:(%u/%u)  4b:(%u/%u)  12b:(%u/%u)  16b:(%u/%u)   (CountUsedMax/CountMax).\n",Arrays1b->GetArrayCountUsedMax(),Arrays1b->GetArrayCountMax(),Arrays4b->GetArrayCountUsedMax(),Arrays4b->GetArrayCountMax(),Arrays12b->GetArrayCountUsedMax(),Arrays12b->GetArrayCountMax(),Arrays16b->GetArrayCountUsedMax(),Arrays16b->GetArrayCountMax());
  delete Arrays1b;
  delete Arrays2b;
  delete Arrays4b;
  delete Arrays8b;
  delete Arrays12b;
  delete Arrays16b;
  delete Arrays24b;
  delete Arrays32b;
}
 
//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JGpuArrays::Reset(){
  Arrays1b->Reset(); 
  Arrays2b->Reset(); 
  Arrays4b->Reset(); 
  Arrays8b->Reset(); 
  Arrays12b->Reset();
  Arrays16b->Reset();
  Arrays24b->Reset();
  Arrays32b->Reset();
}
 
//==============================================================================
/// Returns the allocated memory in GPU.
//==============================================================================
long long JGpuArrays::GetAllocMemoryGpu()const{ 
  long long m=Arrays1b->GetAllocMemoryGpu();
  m+=Arrays2b->GetAllocMemoryGpu();
  m+=Arrays4b->GetAllocMemoryGpu();
  m+=Arrays8b->GetAllocMemoryGpu();
  m+=Arrays12b->GetAllocMemoryGpu();
  m+=Arrays16b->GetAllocMemoryGpu();
  m+=Arrays24b->GetAllocMemoryGpu();
  m+=Arrays32b->GetAllocMemoryGpu();
  return(m);
}

//==============================================================================
/// Changes the number of elements of the arrays.
/// If there is some array in use, it throws an exception.
//==============================================================================
void JGpuArrays::SetArraySize(unsigned size){ 
  Arrays1b->SetArraySize(size); 
  Arrays2b->SetArraySize(size); 
  Arrays4b->SetArraySize(size); 
  Arrays8b->SetArraySize(size); 
  Arrays12b->SetArraySize(size);
  Arrays16b->SetArraySize(size);
  Arrays24b->SetArraySize(size);
  Arrays32b->SetArraySize(size);
}


