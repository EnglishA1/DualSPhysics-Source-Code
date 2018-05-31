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

/// \file JPeriodicGpu.h \brief Declares the class \ref JPeriodicGpu.

#ifndef _JPeriodicGpu_
#define _JPeriodicGpu_

#pragma warning(disable : 4996) //Cancels sprintf() deprecated.

#include "TypesDef.h"
#include "Types.h"
#include "JObjectGpu.h"
#include "JLog2.h"

//#define DBG_JPeriodicGpu 1 

class JCellDivGpu;

//##############################################################################
//# JPeriodicGpu
//##############################################################################
/// \brief Manages the interactions between periodic edges in GPU. 

class JPeriodicGpu : protected JObjectGpu
{
public:
  typedef enum{ PMODE_YZ=6,PMODE_XZ=5,PMODE_XY=4,PMODE_Z=3,PMODE_Y=2,PMODE_X=1,PMODE_null=0 }TpPeriMode; //-Modes of periodicity.
  typedef enum{ PERI_Xmin=1,PERI_Xmax=2,PERI_Ymin=3,PERI_Ymax=4,PERI_Zmin=5,PERI_Zmax=6 }TpPeriZone; //-Type of periodic zone.

protected:
  const bool Stable;
  JLog2 *Log;
  std::string DirOut;
  const JCellDivGpu* CellDiv;

  const bool LaminarSPS;
  const tfloat3 MapPosMin;
  const tfloat3 MapPosMax;
  const unsigned Hdiv; 
  const float Width;         //-Interaction distance (2h).
  const float OvScell;

  static const unsigned SizeModes=3;
  bool Axes[3];
  unsigned ModesCount;
  TpPeriMode Modes[SizeModes];
  tfloat3 PeriInc[SizeModes];

  //-Variables that depend on the periodic zone.
  bool ComputeForces;        //-Indicates if force interaction or Shepard interaction.
  TpPeriMode PerMode;
  tfloat3 PerInc;
  TpPeriZone PerZone;
  bool PerBorderMin;
  bool WithData;             //-Indicates if there are data for interaction.

  tfloat3 ZonePosMin,ZonePosMax;       //-Domain of the periodic zone ( position).
  tuint3 DataCellMin,DataCellMax;      //-Source domain of particles (cells).
  tuint3 InterCellMin,InterCellMax;    //-Domain of interaction with periodic zone (cells).

  //-Variables to create a list of particles in the periodic edges.
  unsigned SizeBorList;                //-Space available for particles of the periodic edges.
  unsigned* BorList;                   //-Stores position in memory of the particles of the edges and the number of particles of each set [p*npb1,p*npf1,p*npb2,p*npf2,...,aux] in the last 4 positions 
  unsigned BorNp1,BorBoundNp1,BorFluidNp1,BorNp2,BorBoundNp2,BorFluidNp2; //-Stores the number of particles of each set.
  bool UseBorRange;                    //-Instead of using BorList[], uses the initial particle of each set.
  unsigned BorBoundIni1,BorFluidIni1,BorBoundIni2,BorFluidIni2;

  unsigned SizeNct;                    //-Number of cells for which memory is allocated.
  unsigned Nc1,Nc2,Nct;
  int2 *BeginEndCell;                  //-Array with the beginning and end of each cell. [nct*Bound,nct*Fluid,BoundOut,FluidOut]

  unsigned SizeNp;                     //-Number of particles for which memory is allocated.
  unsigned Np,Npb;

  //-Variables with particle data for execution [SizeNp].
  unsigned *Ref,*CellPart,*SortPart;
  unsigned *Idp;
  word *Code;
  float4 *PosPres,*VelRhop; 
  tsymatrix3f *Tau;

  unsigned Nrun;

  void Reset();
  unsigned ValueMemoryInc(unsigned v)const{ unsigned x=unsigned(0.1f*v); return(v+(1000>x? 1000: x)); }
  void ResizeMemoryList(unsigned size);
  void ResizeMemoryNct(unsigned nct);
  void ResizeMemoryNp(unsigned np);
  unsigned SizeBeginEndCell(unsigned nct)const{ return(nct+nct+2); } //-[nct*Bound,nct*Fluid,BoundOut,FluidOut]

  void ConfigMode(TpPeriMode mode,tfloat3 perinc);
  inline void SelecZone(unsigned czone);
  template<TpPeriZone tzone> void CalcInterCell(tuint3 &celmin,tuint3 &celmax)const;

  void ClearBorderList();
  void ConfigRangeBorderList(unsigned celini,unsigned celfin,unsigned &bini,unsigned &fini,unsigned &bnp,unsigned &fnp);
  void CalcBorderList(const float4 *pospres);

  void Prepare(bool forces,unsigned czone,const float4* pospres,const word* code,const float4* velrhop,const unsigned* idp,const tsymatrix3f* tau);
  void LoadParticles(const float4* pospres,const word* code,const float4* velrhop,const unsigned* idp,const tsymatrix3f* tau);

  static std::string PeriToStr(TpPeriZone peri);
  void DgSaveVtkBorList(const float4* pospres,const unsigned* idp);
  void DgSaveVtkZone();
  void DgSaveVtkInter(const float4 *pospres,const unsigned *idp);

public:
  JPeriodicGpu(bool stable,JLog2 *log,std::string dirout,const tfloat3 &mapposmin,const tfloat3 &mapposmax,const JCellDivGpu* celldiv);
  ~JPeriodicGpu();
  long long GetAllocMemoryGpu()const{ return(GetAllocMemoryGpuNp()+GetAllocMemoryGpuNct()); };  
  long long GetAllocMemoryGpuNp()const;  
  long long GetAllocMemoryGpuNct()const;  

  void ConfigX(tfloat3 perinc){ ConfigMode(PMODE_X,perinc); }
  void ConfigY(tfloat3 perinc){ ConfigMode(PMODE_Y,perinc); }
  void ConfigZ(tfloat3 perinc){ ConfigMode(PMODE_Z,perinc); }
  void ConfigXY(){ ConfigMode(PMODE_XY,TFloat3(MapPosMin.x-MapPosMax.x,0,0)); ConfigMode(PMODE_Y,TFloat3(0,MapPosMin.y-MapPosMax.y,0)); }
  void ConfigXZ(){ ConfigMode(PMODE_XZ,TFloat3(MapPosMin.x-MapPosMax.x,0,0)); ConfigMode(PMODE_Z,TFloat3(0,0,MapPosMin.z-MapPosMax.z)); }
  void ConfigYZ(){ ConfigMode(PMODE_YZ,TFloat3(0,MapPosMin.y-MapPosMax.y,0)); ConfigMode(PMODE_Z,TFloat3(0,0,MapPosMin.z-MapPosMax.z)); }

  unsigned GetZonesCount()const{ return(ModesCount*2); }
  void PrepareForces(unsigned czone,const float4* pospres,const word* code,const float4* velrhop,const unsigned* idp,const tsymatrix3f* tau){ Prepare(true,czone,pospres,code,velrhop,idp,tau); }
  void PrepareShepard(unsigned czone,const float4* posvol,const word* code){ Prepare(false,czone,posvol,code,NULL,NULL,NULL); }

  unsigned GetListNp()const{ return(PerBorderMin? BorNp1: BorNp2); }
  unsigned GetListNpb()const{ return(PerBorderMin? BorBoundNp1: BorBoundNp2); }
  unsigned GetListBoundIni()const{ return(PerBorderMin? BorBoundIni1: BorBoundIni2); }
  unsigned GetListFluidIni()const{ return(PerBorderMin? BorFluidIni1: BorFluidIni2); }
  const unsigned* GetList()const{ return(UseBorRange? NULL: (PerBorderMin? BorList: BorList+BorNp1)); }
  const unsigned* GetListCellPart()const{ return(CellPart); }

  const int2* GetBeginEndCell()const{ return(BeginEndCell); }
  const float4* GetPosPres()const{ return(PosPres); } 
  const float4* GetVelRhop()const{ return(VelRhop); } 
  const word* GetCode()const{ return(Code); } 
  const unsigned* GetIdp()const{ return(Idp); } 
  const tsymatrix3f* GetTau()const{ return(Tau); } 

  unsigned GetNc1()const{ return(Nc1); }
  unsigned GetBoxFluid()const{ return(Nct); }

  void CheckPositionAll(bool boundchanged,float3* posg);
};

#endif



