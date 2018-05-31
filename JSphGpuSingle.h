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

/// \file JSphGpuSingle.h \brief Declares the class \ref JSphGpuSingle.

#ifndef _JSphGpuSingle_
#define _JSphGpuSingle_

#include "Types.h"
#include "JSphGpu.h"
#include <string>

class JCellDivGpuSingle;
class JPartsLoad;
class JPeriodicGpu;
class JPartsLoad4;

//##############################################################################
//# JSphGpuSingle
//##############################################################################
/// \brief Defines the attributes and functions used only in Single-GPU implementation.

class JSphGpuSingle : public JSphGpu
{
protected:
  JCellDivGpuSingle* CellDivSingle;
  JPartsLoad* PartsLoaded;

  JPartsLoad4* PartsLoaded4;
  
  JPeriodicGpu* PeriZone;

  long long GetAllocMemoryCpu()const;  
  long long GetAllocMemoryGpu()const;  
  long long GetMemoryGpuNp()const;
  long long GetMemoryGpuNct()const;
  void UpdateMaxValues();
  void LoadConfig(JCfgRun *cfg);
  void LoadCaseParticles();
  void ConfigDomain();
  void ConfigRunMode(const std::string &preinfo);

  void PeriInit();
  void PeriCheckPosition();
  void PeriInteraction(TpInter tinter);
  void PeriInteractionShepard();

  void RunCellDivide(bool symvars);
  
  void Interaction_Forces(TpInter tinter);

  float ComputeStep(bool rhopbound){ return(TStep==STEP_Verlet? ComputeStep_Ver(rhopbound): ComputeStep_Sym(rhopbound)); }
  float ComputeStep_Ver(bool rhopbound);
  float ComputeStep_Sym(bool rhopbound);

  void RunFloating(float dt2,bool predictor);
  void RunShepard();

  void SaveData();
  void FinishRun(bool stop);

//****print array form GPU****
  //float
  void CudaArrayPrintFloat(unsigned n, float *Auxg,unsigned *idp)
  {
	  float *auxc=new float[n];
	  cudaMemcpy(auxc,Auxg,sizeof(float)*n,cudaMemcpyDeviceToHost);
	  for(unsigned p=0;p<Np;p++)printf("Array[%d]=%f\n",p,auxc[Idp[p]]);
	  delete[] auxc;
	  getchar();
  }
  
  //float3
  void CudaArrayPrintFloat3(unsigned n, float3 *Auxg)
  {
	  float3 *auxc=new float3[n];
	  cudaMemcpy(auxc,Auxg,sizeof(float3)*n,cudaMemcpyDeviceToHost);
	  for(unsigned p=0;p<Np;p++)printf("Array[%d].x=%f		Array[%d].y=%f		Array[%d].z=%f \n",p,auxc[p].x,auxc[p].y,auxc[p].z);
	  delete[] auxc;
	  getchar();
  }

  //float4
  void CudaArrayPrintFloat4(unsigned n, float4 *Auxg)
  {
	  float4 *auxc=new float4[n];
	  cudaMemcpy(auxc,Auxg,sizeof(float4)*n,cudaMemcpyDeviceToHost);
	  for(unsigned p=0;p<Np;p++)printf("Array[%d].x=%f Array[%d].y=%f Array[%d].z=%f Array[%d].w=%f \n",p,auxc[p].x,p,auxc[p].y,p,auxc[p].z,p,auxc[p].w);
	  delete[] auxc;
	  getchar();
  }
//end


  //float3
  void CudaArrayPrintTsym3f(unsigned n, tsymatrix3f *Auxg)
  {
	  tsymatrix3f *auxc=new tsymatrix3f[n];
	  cudaMemcpy(auxc,Auxg,sizeof(tsymatrix3f)*n,cudaMemcpyDeviceToHost);
	  for(unsigned p=0;p<Np;p++)printf("Array[%d].xx=%f Array[%d].xy=%f Array[%d].xz=%f Array[%d].yy=%f Array[%d].yz=%f Array[%d].zz=%f \n",p,auxc[p].xx,p,auxc[p].xy,p,auxc[p].xz,p,auxc[p].yy,p,auxc[p].yz,p,auxc[p].zz);
	  delete[] auxc;
	  getchar();
  }
//end

public:
  JSphGpuSingle();
  ~JSphGpuSingle();
  void Run(std::string appname,JCfgRun *cfg,JLog2 *log);

};

#endif


