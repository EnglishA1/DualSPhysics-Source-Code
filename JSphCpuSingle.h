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

/// \file JSphCpuSingle.h \brief Declares the class \ref JSphCpuSingle.

#ifndef _JSphCpuSingle_
#define _JSphCpuSingle_

#include "Types.h"
#include "JSphCpu.h"
#include <string>

class JCellDivCpuSingle;
class JPartsLoad;
class JPeriodicCpu;

//##############################################################################
//# JSphCpuSingle
//##############################################################################
/// \brief Defines the attributes and functions used only in Single-Core implementation.

class JSphCpuSingle : public JSphCpu
{
protected:
  JCellDivCpuSingle* CellDivSingle;
  JPartsLoad* PartsLoaded;
  JPeriodicCpu* PeriZone;

  void FreeMemory();
  void AllocMemory();
  long long GetAllocMemoryCpu()const;
  void UpdateMaxValues();
  void LoadConfig(JCfgRun *cfg);
  void LoadCaseParticles();
  void ConfigDomain();

  void PeriInit();
  void PeriInteraction(TpInter tinter);
  void PeriInteractionShepard();
  void PeriCheckPosition();

  void RunCellDivide(bool symvars);

  void Interaction_Forces(TpInter tinter);

  float ComputeStep(bool rhopbound){ return(TStep==STEP_Verlet? ComputeStep_Ver(rhopbound): ComputeStep_Sym(rhopbound)); }
  float ComputeStep_Ver(bool rhopbound); 
  float ComputeStep_Sym(bool rhopbound); 

  void RunFloating(float dt2,bool predictor); 
  void RunShepard();
  
  void SaveData();
  void FinishRun(bool stop);

public:
  JSphCpuSingle();
  ~JSphCpuSingle();
  void Run(std::string appname,JCfgRun *cfg,JLog2 *log);
};

#endif


