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

/// \file JPeriodicCpu.h \brief Declares the class \ref JPeriodicCpu.

#ifndef _JPeriodicCpu_
#define _JPeriodicCpu_

#pragma warning(disable : 4996) //Cancels sprintf() deprecated.

#include "TypesDef.h"
#include "Types.h"
#include "JObject.h"
#include "JLog2.h"

//#define DBG_JPeriodicCpu 1 

class JCellDivCpu;

//##############################################################################
//# JPeriodicCpu
//##############################################################################
/// \brief Manages the interactions between periodic edges in CPU. 

class JPeriodicCpu : protected JObject
{
public:
  typedef enum{ PMODE_YZ=6,PMODE_XZ=5,PMODE_XY=4,PMODE_Z=3,PMODE_Y=2,PMODE_X=1,PMODE_null=0 }TpPeriMode; //-Modes of periodicity.
  typedef enum{ PERI_Xmin=1,PERI_Xmax=2,PERI_Ymin=3,PERI_Ymax=4,PERI_Zmin=5,PERI_Zmax=6 }TpPeriZone; //-Type of periodic zone.

protected:
  JLog2 *Log;
  std::string DirOut;
  const JCellDivCpu* CellDiv;

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

  unsigned *PartsInCell;
  unsigned *BeginCell;                 //-Array with the beginning of each cell. [nct*Bound,nct*Fluid,BoundOut,FluidOut,END]

  unsigned SizeNp;                     //-Number of particles for which memory is allocated.
  unsigned Np,Npb;

  //-Variables with particle data for execution [SizeNp].
  unsigned *Ref,*CellPart,*SortPart;
  unsigned *Idp;  
  word *Code;
  tfloat3 *Pos,*Vel; 
  float *Rhop;
  float *Csound,*PrRhop,*Tensil;
  tsymatrix3f *Tau;

  unsigned Nrun;

  void Reset();
  unsigned ValueMemoryInc(unsigned v)const{ unsigned x=unsigned(0.1f*v); return(v+(1000>x? 1000: x)); }
  void ResizeMemoryList(unsigned size);
  void ResizeMemoryNp(unsigned np);
  void ResizeMemoryNct(unsigned nct);
  unsigned SizeBeginCell(unsigned nct)const{ return(nct+nct+2+1); } //-[nct*Bound,nct*Fluid,BoundOut,FluidOut,END]

  void ConfigMode(TpPeriMode mode,tfloat3 perinc);
  inline void SelecZone(unsigned czone);
  template<TpPeriZone tzone> void CalcInterCell(tuint3 &celmin,tuint3 &celmax)const;

  void ClearBorderList();
  void ConfigRangeBorderList(unsigned celini,unsigned celfin,unsigned &bini,unsigned &fini,unsigned &bnp,unsigned &fnp);
  template<unsigned axismax> unsigned GetListParticlesCells_(tuint3 ncells,unsigned celini,const unsigned* begincell,const tfloat3* pos,tuint3 celmin,tuint3 celmax,tfloat3 posmin,unsigned countlist,unsigned sizelist,unsigned *list) const;
  inline unsigned GetListParticlesCells(unsigned axismax,tuint3 ncells,unsigned celini,const unsigned* begincell,const tfloat3* pos,tuint3 celmin,tuint3 celmax,tfloat3 posmin,unsigned countlist,unsigned sizelist,unsigned *list) const;
  void CalcBorderList(const tfloat3 *pos);

  template<int axis> void PreSortRange_(unsigned np,unsigned npb,unsigned bini,unsigned fini,tfloat3 posinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,float ovscell,const tfloat3 *pos,unsigned *ref,unsigned *cellpart,tfloat3 *pos2)const;
  inline void PreSortRange(unsigned axis,unsigned np,unsigned npb,unsigned bini,unsigned fini,tfloat3 posinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,float ovscell,const tfloat3 *pos,unsigned *ref,unsigned *cellpart,tfloat3 *pos2)const;
  template<int axis> void PreSortList_(unsigned np,unsigned npb,const unsigned *list,tfloat3 posinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,float ovscell,const tfloat3 *pos,unsigned *ref,unsigned *cellpart,tfloat3 *pos2)const;
  inline void PreSortList(unsigned axis,unsigned np,unsigned npb,const unsigned *list,tfloat3 posinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,float ovscell,const tfloat3 *pos,unsigned *ref,unsigned *cellpart,tfloat3 *pos2)const;
  void MakeSortPart();
  void SortPosParticles(unsigned np,const unsigned *sortpart,const tfloat3 *pos,const unsigned *ref,tfloat3 *pos2,unsigned *ref2)const;
  void SortLoadParticles(unsigned np,const unsigned *ref,const float *rhop,const word *code,const unsigned *idp,const tfloat3 *vel,const float *csound,const float *prrhop,const float *tensil,const tsymatrix3f *tau,float *rhop2,word *code2,unsigned *idp2,tfloat3 *vel2,float *csound2,float *prrhop2,float *tensil2,tsymatrix3f *tau2)const;
  void SortLoadParticles(unsigned np,const unsigned *ref,const float *rhop,const word *code,float *rhop2,word *code2)const;

  template<int axis> void CalcCellInterRange_(unsigned np,unsigned npb,unsigned bini,unsigned fini,unsigned hdiv,unsigned ncx,unsigned nsheet,unsigned boxfluid,const unsigned *cellpart,unsigned *cellpart2)const;
  inline void CalcCellInterRange(unsigned axis,unsigned np,unsigned npb,unsigned bini,unsigned fini,unsigned hdiv,unsigned ncx,unsigned nsheet,unsigned boxfluid,const unsigned *cellpart,unsigned *cellpart2)const;
  template<int axis> void CalcCellInterList_(unsigned np,const unsigned *list,unsigned hdiv,unsigned ncx,unsigned nsheet,unsigned boxfluid,const unsigned *cellpart,unsigned *cellpart2)const;
  inline void CalcCellInterList(unsigned axis,unsigned np,const unsigned *list,unsigned hdiv,unsigned ncx,unsigned nsheet,unsigned boxfluid,const unsigned *cellpart,unsigned *cellpart2)const;

  template<bool modexy,int axis> void PreSortTwo_(unsigned np,float twposmin,float twposmax,float twinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,tfloat3 *pos,unsigned *ref,unsigned *cellpart,unsigned &newnp,unsigned &newnbound)const;
  inline void PreSortTwo(bool modexy,int axis,unsigned np,float twposmin,float twposmax,float twinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,tfloat3 *pos,unsigned *ref,unsigned *cellpart,unsigned &newnp,unsigned &newnbound)const;

  void Prepare(bool forces,unsigned czone,const tfloat3* pos,const float* rhop,const word* code,const unsigned* idp,const tfloat3* vel,const float *csound,const float *prrhop,const float *tensil,const tsymatrix3f* tau);
  void LoadParticles(const tfloat3* pos,const float* rhop,const word* code,const unsigned* idp,const tfloat3* vel,const float *csound,const float *prrhop,const float *tensil,const tsymatrix3f *tau);

  template<TpPeriZone tzone> inline void CheckPosition_(bool boundchanged,tfloat3* pos,const unsigned* idp);

  static std::string PeriToStr(TpPeriZone peri);
  void DgSaveVtkBorList(const tfloat3* pos,const unsigned* idp);
  void DgSaveVtkZone();
  void DgSaveVtkInter(const tfloat3 *pos,const unsigned *idp);

public:
  JPeriodicCpu(JLog2 *log,std::string dirout,const tfloat3 &mapposmin,const tfloat3 &mapposmax,const JCellDivCpu* celldiv);
  ~JPeriodicCpu();
  long long GetAllocMemory()const;  

  void ConfigX(tfloat3 perinc){ ConfigMode(PMODE_X,perinc); }
  void ConfigY(tfloat3 perinc){ ConfigMode(PMODE_Y,perinc); }
  void ConfigZ(tfloat3 perinc){ ConfigMode(PMODE_Z,perinc); }
  void ConfigXY(){ ConfigMode(PMODE_XY,TFloat3(MapPosMin.x-MapPosMax.x,0,0)); ConfigMode(PMODE_Y,TFloat3(0,MapPosMin.y-MapPosMax.y,0)); }
  void ConfigXZ(){ ConfigMode(PMODE_XZ,TFloat3(MapPosMin.x-MapPosMax.x,0,0)); ConfigMode(PMODE_Z,TFloat3(0,0,MapPosMin.z-MapPosMax.z)); }
  void ConfigYZ(){ ConfigMode(PMODE_YZ,TFloat3(0,MapPosMin.y-MapPosMax.y,0)); ConfigMode(PMODE_Z,TFloat3(0,0,MapPosMin.z-MapPosMax.z)); }

  unsigned GetZonesCount()const{ return(ModesCount*2); }
  void PrepareForces(unsigned czone,const tfloat3* pos,const float* rhop,const word* code,const unsigned* idp,const tfloat3* vel,const float* csound,const float* prrhop,const float* tensil,const tsymatrix3f* tau){ Prepare(true,czone,pos,rhop,code,idp,vel,csound,prrhop,tensil,tau); }
  void PrepareShepard(unsigned czone,const tfloat3* pos,const float* rhop,const word* code){ Prepare(false,czone,pos,rhop,code,NULL,NULL,NULL,NULL,NULL,NULL); }

  unsigned GetListNp()const{ return(PerBorderMin? BorNp1: BorNp2); }
  unsigned GetListNpb()const{ return(PerBorderMin? BorBoundNp1: BorBoundNp2); }
  unsigned GetListBoundIni()const{ return(PerBorderMin? BorBoundIni1: BorBoundIni2); }
  unsigned GetListFluidIni()const{ return(PerBorderMin? BorFluidIni1: BorFluidIni2); }
  const unsigned* GetList()const{ return(UseBorRange? NULL: (PerBorderMin? BorList: BorList+BorNp1)); }
  const unsigned* GetListCellPart()const{ return(CellPart); }

  const word* GetCode()const{ return(Code); } 
  const unsigned* GetIdp()const{ return(Idp); } 
  const tfloat3* GetPos()const{ return(Pos); } 
  const tfloat3* GetVel()const{ return(Vel); } 
  const float* GetRhop()const{ return(Rhop); } 
  const float* GetCsound()const{ return(Csound); } 
  const float* GetPrRhop()const{ return(PrRhop); } 
  const float* GetTensil()const{ return(Tensil); } 
  const tsymatrix3f* GetTau()const{ return(Tau); } 

  unsigned GetNc1()const{ return(Nc1); }
  unsigned CellBegin(unsigned box,byte kind)const{ return(BeginCell[(kind==1? box: box+Nct)]); }

  void CheckPosition(unsigned czone,bool boundchanged,tfloat3* pos,const unsigned* idp);

};

#endif



