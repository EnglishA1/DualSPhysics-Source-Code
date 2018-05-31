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

/// \file JCellDivCpuSingle.h \brief Declares the class \ref JCellDivCpuSingle.

#ifndef _JCellDivCpuSingle_
#define _JCellDivCpuSingle_

#pragma warning(disable : 4996) //Cancels sprintf() deprecated.

#include "JCellDivCpu.h"

//##############################################################################
//# JCellDivCpuSingle
//##############################################################################
/// \brief Defines the class responsible of computing the Neighbour List in Single-Core.


class JCellDivCpuSingle : public JCellDivCpu
{
protected:
  void AllocMemory(unsigned np,unsigned nct){ ResizeMemoryNp(np); ResizeMemoryNct(nct); }

  void CalcCellDomain(const unsigned* idp,const tfloat3* pos,const float* rhop,word* code);
  void MergeMapCellBoundFluid(const tuint3 &celbmin,const tuint3 &celbmax,const tuint3 &celfmin,const tuint3 &celfmax,tuint3 &celmin,tuint3 &celmax)const;
  void PreSort(const tfloat3* pos,const word* code);
  void PreSortBound(const tfloat3* pos,const word* check,const tfloat3 &dposmin,const tfloat3 &dposdif);
  void PreSortFluid(const tfloat3* pos,const word* code,const tfloat3 &dposmin,const tfloat3 &dposdif);
  void MakeSortFull();
  void MakeSortFluid();

public:
  const bool UseFluidDomain;

  JCellDivCpuSingle(bool stable,JLog2 *log,std::string dirout,byte periactive,bool laminarsps,bool usefluiddomain,const tfloat3 &mapposmin,const tfloat3 &mapposmax,float dosh,unsigned casenbound,unsigned casenfixed,unsigned casenpb,TpCellOrder order);
  void ConfigInit(TpCellMode cellmode,unsigned np,unsigned npb,bool rhopout,float rhopmin,float rhopmax);
  void Divide(bool boundchanged,const unsigned* idp,const tfloat3* pos,const float* rhop,word* code,TimersCpu timers);

  long long GetAllocMemory()const{ return(JCellDivCpu::GetAllocMemory()); }

  //void DgSaveCsvBeginCell(std::string filename,int numfile);
};

#endif




