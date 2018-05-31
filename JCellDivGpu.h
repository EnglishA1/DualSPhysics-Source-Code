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

/// \file JCellDivGpu.h \brief Declares the class \ref JCellDivGpu.

#ifndef _JCellDivGpu_
#define _JCellDivGpu_


#pragma warning(disable : 4996) //Cancels sprintf() deprecated.

#include "Types.h"
#include "JObjectGpu.h"
#include "JSphTimersGpu.h"
#include "JLog2.h"
#include <cmath>
#include <cstring>
#include <sstream>
#include <iostream>
#include <fstream>

//#define DG_JCellDivGpu //-In JCellDivGpu checks that the result of PreSort is valid.

//##############################################################################
//# JCellDivGpu
//##############################################################################
/// \brief Defines the class responsible of computing the Neighbour List in GPU.

class JCellDivGpu : protected JObjectGpu
{
protected:
  const bool Stable;         ///<Ensures the same results when repeated the same simulation.
  JLog2 *Log;                ///<Declares an object of the class \ref JLog2.
  std::string DirOut;        ///<Name of the output directory.
  bool RhopOut;              ///<Indicates whether the density correction RhopOut is activated or not.
  float RhopOutMin;          ///<Minimum value for the density correction RhopOut.
  float RhopOutMax;          ///<Minimum value for the density correction RhopOut.

  //-Constant values during simulation.
  TpCellMode CellMode;       ///<Modes of cells division.
  unsigned Hdiv;             ///<Value to divide 2h.
  float Scell;               ///<Size of the cells.
  float OvScell;             ///<Value of 1/ref\ Scell.
  tuint3 MapCells;           ///<Maximum number of cells in each direction (X,Y,Z).

  //-Variables with allocated memory as a function of particles in GPU.
  unsigned SizeNp;           ///<Size of \ref CellPart and \ref SortPart.
  unsigned *CellPart;        ///<Cell of each particle [Np].
  unsigned *SortPart;        ///<Array to reorder particles. 
  unsigned SizeAuxMem;       ///<Size of \ref AuxMem.
  float *AuxMem;             ///<Auxiliar array used for "reduction" task. 

  //-Variables with allocated memory as a function of cells in GPU.
  unsigned SizeNct;          ///<Size of \ref BeginEndCell.
  int2 *BeginEndCell;        ///<Position of the first and last particle of each cell. [BoundOk(nct),BoundIgnore(1),Fluid(nct),BoundOut(1),FluidOut(1),Displaced(1)]

  long long MemAllocGpuNp;  ///<Amount of allocated memory in GPU for particles.
  long long MemAllocGpuNct; ///<Amount of allocated memory in GPU for cells.

  unsigned Ndiv;             ///<Number of times that \ref Divide was applied.            
  unsigned NdivFull;         ///<Number of times that \ref Divide was appiled to all particles.
  unsigned Nptot;            ///<Number of particles (including excluded) at that step. 
  unsigned Np;               ///<Number of particles at that step. 
  unsigned Npb;              ///<Number of particles of the boundary block.
  unsigned NpbOut;           ///<Number of particles of the boundary block that were excluded.
  unsigned NpfOut;           ///<Number of fluid particles that were excluded.
  unsigned NpfOutRhop;       ///<Number of fluid particles that were excluded due to its value of density out of the desirable limits.
  unsigned NpfOutMove;       ///<Number of fluid particles that were excluded since the particle moves faster than the allowed vecocity.
  unsigned NpbIgnore;        ///<Number of boundary particles that are not going to interact with fluid particles.

  tuint3 CellDomainMin;      ///<First cell to be considered for particle interactions at that step.
  tuint3 CellDomainMax;      ///<Last cell to be considered for particle interactions at that step.
  unsigned Ncx;              ///<Number of cells in X direction at that step with particle interactions.
  unsigned Ncy;              ///<Number of cells in Y direction at that step with particle interactions.
  unsigned Ncz;              ///<Number of cells in Z direction at that step with particle interactions.
  unsigned Nsheet;           ///<Number of cells in X direction * Y direction at that ste with particle interactionsp.
  unsigned Nct;              ///<Number of total cells at that step with particle interactions.
  unsigned Nctt;             ///<\ref Nct + special cells, Nctt=SizeBeginCell()-1
  unsigned BoxIgnore;        ///<Index of \ref BeginCell with the first cell with ignored boundary particles. 
  unsigned BoxFluid;         ///<Index of \ref BeginCell with the first fluid particles. 
  unsigned BoxBoundOut;      ///<Index of \ref BeginCell with the excluded boundary particle.
  unsigned BoxFluidOut;      ///<Index of \ref BeginCell with the excluded fluid particle.
  
  bool BoundLimitOk;         ///<Indicates that limits of boundaries are already computed in \ref BoundLimitCellMin and \ref BoundLimitCellMax.
  tuint3 BoundLimitCellMin;  ///<Cell with the boundary with the minimum positions (X,Y,Z).
  tuint3 BoundLimitCellMax;  ///<Cell with the boundary with the maximum positions (X,Y,Z).

  bool BoundDivideOk;        ///<Indicates that limits of boundaries were already computed in \ref BoundDivideCellMin and \ref BoundDivideCellMax.
  tuint3 BoundDivideCellMin; ///<Value of \ref CellDomainMin when \ref Divide was applied to boundary particles.
  tuint3 BoundDivideCellMax; ///<Value of \ref CellDomainMax when \ref Divide was applied to boundary particles.

  bool DivideFull;           ///<Indicates of \ref Divide was applied to all particles and not only fluid particles.

  void Reset();
  
  //-Manages the dynamic memory allocation.
  void FreeBasicMemoryNct();
  void FreeBasicMemoryAll();
  void AllocBasicMemoryNp(unsigned np);
  bool CheckMemoryNp(bool returnerror);
  void AllocBasicMemoryNct(unsigned nct);
  bool CheckMemoryNct(bool returnerror);

  unsigned SizeBeginEndCell(unsigned nct)const{ return((nct*2)+4); } //-[BoundOk(nct),BoundIgnore(1),Fluid(nct),BoundOut(1),FluidOut(1),Displaced(1)]

  long long GetAllocMemoryCpu()const{ return(0); }
  long long GetAllocMemoryGpuNp()const{ return(MemAllocGpuNp); };
  long long GetAllocMemoryGpuNct()const{ return(MemAllocGpuNct); };
  long long GetAllocMemoryGpu()const{ return(GetAllocMemoryGpuNp()+GetAllocMemoryGpuNct()); };

  void ConfigInitCellMode(TpCellMode cellmode,float dosh,unsigned np,unsigned npb,bool rhopout,float rhopmin,float rhopmax);

  void VisuBoundaryOut(unsigned p,unsigned id,tfloat3 pos,word code)const;
  tuint3 GetMapCell(const tfloat3 &pos)const;
  void CalcCellDomainBound(unsigned n,unsigned pini,const float3* posg,word* codeg,tuint3 &cellmin,tuint3 &cellmax);
  void CalcCellDomainFluid(unsigned n,unsigned pini,const float3* posg,const float* rhopg,word* codeg,tuint3 &cellmin,tuint3 &cellmax);

  void CellBeginEnd(unsigned cell,unsigned ndata,unsigned* data)const;
  int2 CellBeginEnd(unsigned cell)const;
  unsigned CellSize(unsigned cell)const{ int2 v=CellBeginEnd(cell); return(unsigned(v.y-v.x)); }

public:
  const byte PeriActive;     ///<Activate the use of periodic boundary conditions.
  const TpCellOrder Order;   ///<Direction (X,Y,Z) used to order particles. 
  const unsigned CaseNbound; ///<Number of boundary particles.
  const unsigned CaseNfixed; ///<Number of fixed boundary particles. 
  const unsigned CaseNpb;    ///<Number of particles of the boundary block
  const tfloat3 MapPosMin;   ///<Minimum limits of the map for the case.
  const tfloat3 MapPosMax;   ///<Maximum limits of the map for the case.
  const tfloat3 MapPosDif;   ///<\ref MapPosMax - \ref MapPosMin.
  const float Dosh;          ///<2*h (being h the smoothing length).
  const bool Floating;       ///<Indicates if there are floating bodies.
  const bool LaminarSPS;     ///<Indicates if Laminar + SPS viscosity treatment is being used.

  JCellDivGpu(bool stable,JLog2 *log,std::string dirout,byte periactive,bool laminarsps,const tfloat3 &mapposmin,const tfloat3 &mapposmax,float dosh,unsigned casenbound,unsigned casenfixed,unsigned casenpb,TpCellOrder order);
  ~JCellDivGpu();

  void SortBasicArrays(const unsigned *idp,const float3 *pos,const float3 *vel,const float *rhop,const word *code,const float *press,const float *viscop,const unsigned *idpm, unsigned *idp2,float3 *pos2,float3 *vel2,float *rhop2,word *code2,float *press2,float *viscop2,unsigned *idpm2);

  void SortDataArrays(const float *a,const float3 *b,float *a2,float3 *b2);
  void SortDataArrays(const float *a,const float3 *b,const float3 *c,float *a2,float3 *b2,float3 *c2);
  void SortDataArrays(const tsymatrix3f *a,tsymatrix3f *a2);

  void CheckParticlesOut(unsigned npfout,const unsigned *idp,const tfloat3 *pos,const float *rhop,const word *code);
  float* GetAuxMem(unsigned size);

  TpCellMode GetCellMode()const{ return(CellMode); }
  unsigned GetHdiv()const{ return(Hdiv); }
  float GetScell()const{ return(Scell); }
  tuint3 GetMapCells()const{ return(MapCells); };

  unsigned GetNct()const{ return(Nct); }
  unsigned GetNcx()const{ return(Ncx); }
  unsigned GetNcy()const{ return(Ncy); }
  unsigned GetNcz()const{ return(Ncz); }
  tuint3 GetNcells()const{ return(TUint3(Ncx,Ncy,Ncz)); }
  unsigned GetBoxFluid()const{ return(BoxFluid); }

  tuint3 GetCellDomainMin()const{ return(CellDomainMin); }
  tuint3 GetCellDomainMax()const{ return(CellDomainMax); }
  tfloat3 GetDomainLimits(bool limitmin,unsigned slicecellmin=0)const;

  unsigned GetNp()const{ return(Np); }
  unsigned GetNpb()const{ return(Npb); }
  unsigned GetNpbIgnore()const{ return(NpbIgnore); }
  unsigned GetNpOut()const{ return(NpbOut+NpfOut); }

  unsigned GetNpfOutPos()const{ return(NpfOut-(NpfOutMove+NpfOutRhop)); }
  unsigned GetNpfOutMove()const{ return(NpfOutMove); }
  unsigned GetNpfOutRhop()const{ return(NpfOutRhop); }

  const unsigned* GetCellPart()const{ return(CellPart); }
  const int2* GetBeginCell(){ return(BeginEndCell); }

  uint2 GetRangeParticlesCells(bool fluid,unsigned celini,unsigned celfin)const;
  unsigned GetParticlesCells(unsigned celini,unsigned celfin);
};

#endif


