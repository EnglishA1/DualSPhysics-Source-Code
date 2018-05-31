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

/// \file JCellDivGpuSingle.h \brief Declares the class \ref JCellDivGpuSingle.

#ifndef _JCellDivGpuSingle_
#define _JCellDivGpuSingle_

#pragma warning(disable : 4996) //Cancels sprintf() deprecated.

#include "JCellDivGpu.h"

//class TimersGpu;

//##############################################################################
//# JCellDivGpuSingle
//##############################################################################
/// \brief Defines the class responsible of computing the Neighbour List in Single-GPU.

class JCellDivGpuSingle : public JCellDivGpu
{
protected:
  bool AllocMemory(unsigned np,unsigned nct);
  void ConfigInitCellMode(TpCellMode cellmode,unsigned np,unsigned npb,bool rhopout,float rhopmin,float rhopmax);

  void CalcCellDomain(const float3* posg,const float* rhopg,word* codeg);
  void MergeMapCellBoundFluid(const tuint3 &celbmin,const tuint3 &celbmax,const tuint3 &celfmin,const tuint3 &celfmax,tuint3 &celmin,tuint3 &celmax)const;
  void PreSort(const float3* posg,const word* codeg);

public:
  const bool UseFluidDomain;

  JCellDivGpuSingle(bool stable,JLog2 *log,std::string dirout,byte periactive,bool laminarsps,bool usefluiddomain,const tfloat3 &mapposmin,const tfloat3 &mapposmax,float dosh,unsigned casenbound,unsigned casenfixed,unsigned casenpb,TpCellOrder order);
  void ConfigInit(TpCellMode cellmode,unsigned np,unsigned npb,bool rhopout,float rhopmin,float rhopmax);
  void Divide(bool boundchanged,const float3* posg,const float* rhopg,word* codeg,TimersGpu timers);

  long long GetAllocMemoryCpu()const{ return(JCellDivGpu::GetAllocMemoryCpu()); }
  long long GetAllocMemoryGpu()const{ return(JCellDivGpu::GetAllocMemoryGpu()); }
  long long GetAllocMemoryGpuNp()const{ return(JCellDivGpu::GetAllocMemoryGpuNp()); };
  long long GetAllocMemoryGpuNct()const{ return(JCellDivGpu::GetAllocMemoryGpuNct()); };
};

#endif




