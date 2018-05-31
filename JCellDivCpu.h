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

/// \file JCellDivCpu.h \brief Declares the class \ref JCellDivCpu.

#ifndef _JCellDivCpu_
#define _JCellDivCpu_


#pragma warning(disable : 4996) //Cancels sprintf() deprecated.

#include "Types.h"
#include "JObject.h"
#include "JSphTimersCpu.h"
#include "JLog2.h"
#include <cmath>
#include <cstring>
#include <sstream>
#include <iostream>
#include <fstream>

//#define DBG_JCellDivCpu 1 

//##############################################################################
//# JCellDivCpu
//##############################################################################
/// \brief Defines the class responsible of computing the Neighbour List in CPU.

class JCellDivCpu : protected JObject
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

  //-Variables with allocated memory as a function of particles.
  unsigned SizeNp;           ///<Size of \ref CellPart and \ref SortPart.
  unsigned *CellPart;        ///<Cell of each particle [Np].
  unsigned *SortPart;        ///<Array to reorder particles. 

  //-Variables with allocated memory as a function of cells.
  unsigned SizeNct;          ///<Size of \ref PartsInCell and \ref BeginCell.
  unsigned *PartsInCell;     ///<Number of particles of each cell.
  unsigned *BeginCell;       ///<Position of the first particle of each cell (nct*Bound+BoundIgnore+nct*Fluid+FluidOut+END).

  //-Variables to reordered particles.
  byte *VSort;               ///<Memory to reorder particles [sizeof(tfloat3)*Np].
  int *VSortInt;             ///<To order arrays of type int (pointing at VSort).
  word *VSortWord;           ///<To order arrays of type word (pointing at VSort).
  float *VSortFloat;         ///<To order arrays of type float (pointing at VSort).
  tfloat3 *VSortFloat3;      ///<To order arrays of type tfloat3 (pointing at VSort).
  tsymatrix3f *VSortMatrix3f;///<To order arrays of type tsymatrix3f (pointing at VSort).

  long long MemAllocNp;      ///<Amount of allocated memory for particles.
  long long MemAllocNct;     ///<Amount of allocated memory for cells.

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

  void ResizeMemoryNp(unsigned np);  ///<Resizes \ref CellPart, \ref SortPart and \ref VSort (dynamic memory allocation).
  void ResizeMemoryNct(unsigned nct);///<Resizes \ref PartsInCell and \ref BeginCell (dynamic memory allocation).

  unsigned SizeBeginCell(unsigned nct)const{ return((nct*2)+4+1); } //-BoundOk(nct),BoundIgnore(1),Fluid(nct),BoundOut(1),FluidOut(1),Displaced(1),End(1).
  
  long long GetAllocMemoryNp()const{ return(MemAllocNp); };
  long long GetAllocMemoryNct()const{ return(MemAllocNct); };
  long long GetAllocMemory()const{ return(GetAllocMemoryNp()+GetAllocMemoryNct()); };

  void VisuBoundaryOut(unsigned p,unsigned id,tfloat3 pos,word code)const;
  tuint3 GetMapCell(const tfloat3 &pos)const;
  void CalcCellDomainBound(unsigned n,unsigned pini,const unsigned* idp,const tfloat3* pos,word* code,tuint3 &cellmin,tuint3 &cellmax);
  void CalcCellDomainFluid(unsigned n,unsigned pini,const unsigned* idp,const tfloat3* pos,const float* rhopg,word* code,tuint3 &cellmin,tuint3 &cellmax);

  unsigned CellSize(unsigned box)const{ return(BeginCell[box+1]-BeginCell[box]); }

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

  JCellDivCpu(bool stable,JLog2 *log,std::string dirout,byte periactive,bool laminarsps,const tfloat3 &mapposmin,const tfloat3 &mapposmax,float dosh,unsigned casenbound,unsigned casenfixed,unsigned casenpb,TpCellOrder order);
  ~JCellDivCpu();

  void SortParticles(unsigned *vec);
  void SortParticles(word *vec);
  void SortParticles(float *vec);
  void SortParticles(tfloat3 *vec);
  void SortParticles(tsymatrix3f *vec);  

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
  unsigned GetNpfOut()const{ return(NpfOut); }

  unsigned GetNpfOutPos()const{ return(NpfOut-(NpfOutMove+NpfOutRhop)); }
  unsigned GetNpfOutMove()const{ return(NpfOutMove); }
  unsigned GetNpfOutRhop()const{ return(NpfOutRhop); }


  bool CellNoEmpty(unsigned box,byte kind)const;
  unsigned CellBegin(unsigned box,byte kind)const;
  unsigned CellSize(unsigned box,byte kind)const;

  const unsigned* GetCellPart()const{ return(CellPart); }
  const unsigned* GetBeginCell()const{ return(BeginCell); }
};

#endif


