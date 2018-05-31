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

/// \file JCellDivGpuSingle.cpp \brief Implements the class \ref JCellDivGpuSingle.

#include "JCellDivGpuSingle.h"
#include "JCellDivGpuSingle_ker.h"
#include "Functions.h"

using namespace std;

//==============================================================================
/// Constructor.
//==============================================================================
JCellDivGpuSingle::JCellDivGpuSingle(bool stable,JLog2 *log,std::string dirout,byte periactive,bool laminarsps,bool usefluiddomain,const tfloat3 &mapposmin,const tfloat3 &mapposmax,float dosh,unsigned casenbound,unsigned casenfixed,unsigned casenpb,TpCellOrder order):JCellDivGpu(stable,log,dirout,periactive,laminarsps,mapposmin,mapposmax,dosh,casenbound,casenfixed,casenpb,order),UseFluidDomain(usefluiddomain){
  ClassName="JCellDivGpuSingle";
}

//==============================================================================
/// Assigns memory according to number of particles and number of cells.
/// Returns true if memory can not be allocated.
//==============================================================================
bool JCellDivGpuSingle::AllocMemory(unsigned np,unsigned nct){
  const char met[]="AllocMemory";
  AllocBasicMemoryNp(np);
  bool err=CheckMemoryNp(true);
  if(!err){
    AllocBasicMemoryNct(nct);
    err=CheckMemoryNct(true);
  }
  if(err)FreeBasicMemoryAll();
  return(err);
}

//==============================================================================
/// Configures cell size and the use of cell neighbours according to cellmode.
/// Reset() in case of error.
//==============================================================================
void JCellDivGpuSingle::ConfigInitCellMode(TpCellMode cellmode,unsigned np,unsigned npb,bool rhopout,float rhopmin,float rhopmax){
  //Log->Print(string("ConfigInitCellMode> ")+fun::VarStr("cellmode",string(GetNameCellMode(cellmode))));
  Reset();
  Nptot=Np=np; Npb=npb;
  RhopOut=rhopout; RhopOutMin=rhopmin; RhopOutMax=rhopmax;
  CellMode=cellmode;
  Hdiv=(CellMode==CELLMODE_H? 2: 1);
  Scell=(CellMode==CELLMODE_H? Dosh/2: (CellMode==CELLMODE_2H? Dosh: Dosh*2));
  OvScell=1.f/Scell;
  MapCells=TUint3(unsigned(ceil(MapPosDif.x/Scell)),unsigned(ceil(MapPosDif.y/Scell)),unsigned(ceil(MapPosDif.z/Scell)));
  bool err=AllocMemory(np,MapCells.x*MapCells.y*MapCells.z);
  if(err)Reset();
}

//==============================================================================
/// Configures cell size and the use of cell neighbours 
/// If cellmode is CELLMODE_None selects CELLMODE_H or CELLMODE_2H according to available memory.
/// In case any is not valid, it throws an exception.
//==============================================================================
void JCellDivGpuSingle::ConfigInit(TpCellMode cellmode,unsigned np,unsigned npb,bool rhopout,float rhopmin,float rhopmax){
  const char met[]="ConfigInit";
  if(cellmode!=CELLMODE_None)ConfigInitCellMode(cellmode,np,npb,rhopout,rhopmin,rhopmax);
  else{
    ConfigInitCellMode(CELLMODE_H,np,npb,rhopout,rhopmin,rhopmax);
    if(CellMode==CELLMODE_None)ConfigInitCellMode(CELLMODE_2H,np,npb,rhopout,rhopmin,rhopmax);
  }
  if(CellMode==CELLMODE_None)RunException(met,"The requested gpu memory could not be allocated .");
}

//==============================================================================
/// Computes limits of the domain in cells adjusting to the fluid CellDomainMin/Max). 
/// Labels excluded particles in code[].
//==============================================================================
void JCellDivGpuSingle::CalcCellDomain(const float3* posg,const float* rhopg,word* codeg){
  //-Computes domain of the boundary.
  tuint3 celbmin,celbmax;
  if(!BoundLimitOk){
    CalcCellDomainBound(Npb,0,posg,codeg,celbmin,celbmax);
    BoundLimitOk=true; BoundLimitCellMin=celbmin; BoundLimitCellMax=celbmax;
  } 
  else{ celbmin=BoundLimitCellMin; celbmax=BoundLimitCellMax; }
  //-Computes domain of the fluid.
  tuint3 celfmin,celfmax;
  CalcCellDomainFluid(Np-Npb,Npb,posg,rhopg,codeg,celfmin,celfmax);
  //-Computes domain adjusting to the boundary and the fluid(with 2h halo).
  MergeMapCellBoundFluid(celbmin,celbmax,celfmin,celfmax,CellDomainMin,CellDomainMax);
}

//==============================================================================
/// Combines cell limits of boundary and fluid with map limits.
/// With UseFluidDomain=TRUE, it uses domain of the fluid plus 2h if there is boundary
/// if not, it uses the domain with fluid and boundary.
/// If there is no boundary CellDomainMin=CellDomainMax=(0,0,0).
//==============================================================================
void JCellDivGpuSingle::MergeMapCellBoundFluid(const tuint3 &celbmin,const tuint3 &celbmax,const tuint3 &celfmin,const tuint3 &celfmax,tuint3 &celmin,tuint3 &celmax)const{
  //char cad[256]; sprintf(cad,"celb=(%u,%u,%u)-(%u,%u,%u)  Npb:%u",celbmin.x,celbmin.y,celbmin.z,celbmax.x,celbmax.y,celbmax.z,Npb); Log->Print(cad);
  if(UseFluidDomain){
    celmin=TUint3(max(min(celbmin.x,celfmin.x),(celfmin.x>=Hdiv? celfmin.x-Hdiv: 0)),max(min(celbmin.y,celfmin.y),(celfmin.y>=Hdiv? celfmin.y-Hdiv: 0)),max(min(celbmin.z,celfmin.z),(celfmin.z>=Hdiv? celfmin.z-Hdiv: 0)));
    celmax=TUint3(min(max(celbmax.x,celfmax.x),celfmax.x+Hdiv),min(max(celbmax.y,celfmax.y),celfmax.y+Hdiv),min(max(celbmax.z,celfmax.z),celfmax.z+Hdiv));
  }
  else{
    celmin=MinValues(celbmin,celfmin);
    celmax=MaxValues(celbmax,celfmax);
  }
  if(celmax.x>=MapCells.x)celmax.x=MapCells.x-1;
  if(celmax.y>=MapCells.y)celmax.y=MapCells.y-1;
  if(celmax.z>=MapCells.z)celmax.z=MapCells.z-1;
  if(celmin.x>celmax.x||celmin.y>celmax.y||celmin.z>celmax.z){ celmin=celmax=TUint3(0,0,0); }
}

//==============================================================================
/// Computes the number of cells starting from CellDomainMin/Max). 
/// Computes cell of each particle CellPart[]) starting from its position,
/// all excluded particles were already labeled in code[] by CalcCellDomain().
/// Assigns consecutive values to SortPart[].
//==============================================================================
void JCellDivGpuSingle::PreSort(const float3* posg,const word* codeg){
  //-Computes number of cells.
  Ncx=CellDomainMax.x-CellDomainMin.x+1;
  Ncy=CellDomainMax.y-CellDomainMin.y+1;
  Ncz=CellDomainMax.z-CellDomainMin.z+1;
  //printf("======  ncx:%u ncy:%u ncz:%u\n",Ncx,Ncy,Ncz);
  Nsheet=Ncx*Ncy; Nct=Nsheet*Ncz;
  BoxIgnore=Nct; BoxFluid=BoxIgnore+1; BoxBoundOut=BoxFluid+Nct; BoxFluidOut=BoxBoundOut+1;
  tfloat3 dposmin=MapPosMin+TFloat3(Scell*CellDomainMin.x,Scell*CellDomainMin.y,Scell*CellDomainMin.z);
  tfloat3 dposdif=TFloat3(Scell*Ncx,Scell*Ncy,Scell*Ncz);
  cudiv::PreSort(DivideFull,Np,Npb,posg,codeg,dposmin,dposdif,TUint3(Ncx,Ncy,Ncz),OvScell,CellPart,SortPart,Log);
#ifdef DG_JCellDivGpu
  tfloat3 *posh=new tfloat3[Np];  
  unsigned *num=new unsigned[Np];
  unsigned * cellparth=new unsigned[Np];
  cudaMemcpy(posh,posg,sizeof(float3)*Np,cudaMemcpyDeviceToHost);
  for(unsigned p=0;p<Np;p++)num[p]=p;
  cudaMemcpy(cellparth,CellPart,sizeof(unsigned)*(Np),cudaMemcpyDeviceToHost);
  unsigned nctot2=Nctot*2;
  char cad[512];
  for(unsigned p=0;p<Np;p++){
    if(cellparth[p]>=nctot2){
      sprintf(cad,"PreSort> Invalid value of CellPart. cellpart[%u]=%u",p,cellparth[p]); Log->Print(cad);
    }  
    else if(p>=Npb&&cellparth[p]<Nctot){
      sprintf(cad,"PreSort> Invalid value of fluid for CellPart. cellpart[%u]=%u",p,cellparth[p]); Log->Print(cad);
    }  
  }
  //JBuffer buf(1024*1024,1024*512);
  //buf.InStr("POINTSDATA"); buf.InUint(Np); buf.InFloat3Vec(Np,posh);
  //buf.InStr("CellPart:unsigned_int");      buf.InUintVec(Np,cellparth);
  //buf.InStr("Num:unsigned_int");           buf.InUintVec(Np,num);
  //buf.InStr("END"); 
  //JFormatFiles2::PointsToVtk(DirOut+"_CellPart.vtk",&buf);

  delete[] posh;
  delete[] num;
  delete[] cellparth;
#endif
}

//==============================================================================
/// Division of particles in cells.
//==============================================================================
void JCellDivGpuSingle::Divide(bool boundchanged,const float3* posg,const float* rhopg,word* codeg,TimersGpu timers){
  const char met[]="Divide";
  TmgStart(timers,TMG_NlLimits);
  //-If the position of the boundary changes, it is necessary to recompute limits and reorder particles. 
  if(boundchanged){
    BoundLimitOk=BoundDivideOk=false;
    BoundLimitCellMin=BoundLimitCellMax=TUint3(0);
    BoundDivideCellMin=BoundDivideCellMax=TUint3(0);
  }
  Nptot=Np;
  NpbOut=NpfOut=NpfOutRhop=NpfOutMove=NpbIgnore=0;
  //-Computes limits of the domain.
  CalcCellDomain(posg,rhopg,codeg);
  TmgStop(timers,TMG_NlLimits);
  //-Determines if divide affects to all particles.
  TmgStart(timers,TMG_NlPreSort);
  if(!BoundDivideOk||BoundDivideCellMin!=CellDomainMin||BoundDivideCellMax!=CellDomainMax){
    DivideFull=true;
    BoundDivideOk=true; BoundDivideCellMin=CellDomainMin; BoundDivideCellMax=CellDomainMax;
  }
  else DivideFull=false;
//  if(DivideFull)Log->PrintDbg("--> DivideFull=TRUE"); else Log->PrintDbg("--> DivideFull=FALSE");
//  Log->PrintDbg(string("--> CellDomain:%s")+fun::Uint3RangeStr(CellDomainMin,CellDomainMax));

  //-Computes cell of each particle (CellPart).
  PreSort(posg,codeg);
  TmgStop(timers,TMG_NlPreSort);
  //-Order CellPart & SortPart as function of cell.
  TmgStart(timers,TMG_NlRadixSort);
  if(DivideFull)cudiv::Sort(CellPart,SortPart,Np,Stable);
  else cudiv::Sort(CellPart+Npb,SortPart+Npb,Np-Npb,Stable);
  TmgStop(timers,TMG_NlRadixSort);
  //-Computes initial and last particle of each cell (BeginEndCell).
  TmgStart(timers,TMG_NlCellBegin);
  cudiv::CalcBeginEndCell(DivideFull,Npb,Np,SizeBeginEndCell(Nct),BoxIgnore+1,CellPart,BeginEndCell);
  //-Updates number of particles.
  NpbIgnore=CellSize(BoxIgnore);
  unsigned beginendcell[4];
  CellBeginEnd(BoxBoundOut,4,beginendcell);
  NpbOut=beginendcell[1]-beginendcell[0];
  NpfOut=beginendcell[3]-beginendcell[2];
  //printf("---> Nct:%u  BoxBoundOut:%u  SizeBeginEndCell:%u\n",Nct,BoxBoundOut,SizeBeginEndCell(Nct));
  //printf("---> NpbIgnore:%u  NpbOut:%u  NpfOut:%u\n",NpbIgnore,NpbOut,NpfOut);
  Np=Nptot-NpbOut-NpfOut;
  Ndiv++;
  if(DivideFull)NdivFull++;
  TmgStop(timers,TMG_NlCellBegin);
  CheckCudaError(met,"Error in Neighbour List construction.");
}



