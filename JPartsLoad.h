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

/// \file JPartsLoad.h \brief Declares the class \ref JPartsLoad.

#ifndef _JPartsLoad_
#define _JPartsLoad_



#pragma warning(disable : 4996) //Cancels sprintf() deprecated.

#include "TypesDef.h"
#include "JObject.h"
#include <cstring>

//##############################################################################
//# JPartsLoad
//##############################################################################
/// \brief Manages the initial load of particle data.

class JPartsLoad : protected JObject
{
public:
  typedef enum{ LOUT_None=0,LOUT_Out=1,LOUT_AllOut=2 }TpLoadOut; //-Controls the load of excluded particles (out).

protected:
  TpLoadOut LoadOut;  
  bool Simulate2D;
  bool PartBi2;
  tfloat3 MapPosMin,MapPosMax;

  unsigned PartBegin;
  float PartBeginTimeStep;

  unsigned Size;
  unsigned Count;            //-Total number of loaded particles if FullOut Count=Size=CaseNp).
  unsigned CountOut;         //-Total number of loaded excluded particles.
  unsigned *Idp;
  tfloat3 *Pos,*Vel;
  float* Rhop;

  void AllocMemory(unsigned size);
  static void CalcPosLimits(unsigned n,const tfloat3 *pos,tfloat3 &posmin,tfloat3 &posmax);

public:
  JPartsLoad();
  ~JPartsLoad();
  void Reset();

  void LoadParticles(TpLoadOut loadout,const std::string &casedir,const std::string &casename,unsigned partbegin,const std::string &casedirbegin,unsigned casenp,unsigned casenbound,unsigned casenfixed,unsigned casenmoving,unsigned casenfloat);
  unsigned GetCount()const{ return(Count); }
  unsigned GetCountOk()const{ return(Count-CountOut); }
  unsigned GetCountOut()const{ return(CountOut); }

  void GetLimits(tfloat3 bordermin,tfloat3 bordermax,tfloat3 &posmin,tfloat3 &posmax)const;
  bool GetSimulate2D()const{ return(Simulate2D); }
  float GetPartBeginTimeStep()const{ return(PartBeginTimeStep); }

  const unsigned* GetIdp(){ return(Idp); }
  const tfloat3* GetPos(){ return(Pos); }
  const tfloat3* GetVel(){ return(Vel); }
  const float* GetRhop(){ return(Rhop); }

  long long GetAllocMemory()const;
  bool LoadPartBi2()const{ return(PartBi2); }
};

#endif


