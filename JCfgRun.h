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

/// \file JCfgRun.h \brief Declares the class \ref JCfgRun.

#ifndef _JCfgRun_
#define _JCfgRun_

#pragma warning(disable : 4996) //Cancels sprintf() deprecated.

#include "Types.h"
#include "Functions.h"
#include "JObject.h"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdlib>

//##############################################################################
//# JCfgRun
//##############################################################################
/// \brief Defines the class responsible of collecting the execution parameters by command line.

class JCfgRun : protected JObject
{
public:
protected:
  bool SvDef;
  int DirsDef;
  static void LoadFloat6(std::string txopt,float def,tfloat3 &v1,tfloat3 &v2);

public:
  bool PrintInfo;
  bool Cpu;
  bool Gpu;
  int GpuId;
  bool GpuFree;
  bool Stable;

  int OmpThreads;
  TpOmpMode OmpMode;

  TpCellOrder CellOrder;
  TpCellMode  CellMode;           ///<Modes of cells division.
  TpStep TStep;
  int VerletSteps;
  TpKernel TKernel;
  TpVisco TVisco;
  float Visco;
  float TimeMax,TimePart;
  int ShepardSteps;
  float DeltaSph;
  bool SvRes,SvTimers,SvDomainVtk;
  bool Sv_Binx,Sv_Csv,Sv_Info,Sv_Vtk,Sv_Pvtk;
  std::string CaseName,RunName,DirOut;
  std::string PartBeginDir;
  unsigned PartBegin,PartBeginFirst;
  bool RhopOutModif;              ///<Indicates whether \ref RhopOutMin or RhopOutMax is changed.
  float RhopOutMin,RhopOutMax;    ///<Limits for \ref RhopOut density correction.

  float FtPause;

  byte DomainMode; //0:Without configuration, 1:Particles, 2:Fixed
  tfloat3 DomainParticlesMin,DomainParticlesMax;
  tfloat3 DomainParticlesPrcMin,DomainParticlesPrcMax;
  tfloat3 DomainFixedMin,DomainFixedMax;

  std::string PtxasFile;          ///<File with ptxas information.

  JCfgRun();
  void Reset();
  void VisuInfo()const;
  void VisuConfig()const;
  void LoadArgv(int argc,char** argv);
  void LoadFile(std::string fname,int lv);
  void LoadOpts(std::string *optlis,int optn,int lv,std::string file);
  void ErrorParm(const std::string &opt,int optc,int lv,const std::string &file)const;
};

#endif


