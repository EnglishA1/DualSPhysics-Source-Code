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

/// \file JCellDivCpu.cpp \brief Implements the class \ref JCellDivCpu.

#include "Types.h"
#include "JCellDivCpu.h"
#include "Functions.h"
#include "JFormatFiles2.h"
#include <float.h>

using namespace std;

//==============================================================================
/// Constructor.
//==============================================================================
JCellDivCpu::JCellDivCpu(bool stable,JLog2 *log,std::string dirout,byte periactive,bool laminarsps,const tfloat3 &mapposmin,const tfloat3 &mapposmax,float dosh,unsigned casenbound,unsigned casenfixed,unsigned casenpb,TpCellOrder order):Stable(stable),PeriActive(periactive),LaminarSPS(laminarsps),MapPosMin(mapposmin),MapPosMax(mapposmax),Dosh(dosh),CaseNbound(casenbound),CaseNfixed(casenfixed),CaseNpb(casenpb),Order(order),MapPosDif(mapposmax-mapposmin),Floating(CaseNpb!=CaseNbound){
  ClassName="JCellDivCpu";
  Log=log;
  DirOut=dirout;
  CellPart=NULL;    SortPart=NULL;
  PartsInCell=NULL; BeginCell=NULL;
  VSort=NULL;
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JCellDivCpu::~JCellDivCpu(){
  //char cad[1024]; sprintf(cad,"---> DivideFull:%u/%u",NdivFull,Ndiv); Log->PrintDbg(cad);
  Reset();
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JCellDivCpu::Reset(){
  ResizeMemoryNp(0);
  ResizeMemoryNct(0);
  RhopOut=false; RhopOutMin=RhopOutMax=0;
  Ndiv=NdivFull=0;
  Nptot=Np=Npb=0;
  NpfOut=NpfOutRhop=NpfOutMove=NpbIgnore=0;
  CellDomainMin=TUint3(1);
  CellDomainMax=TUint3(0);
  Ncx=Ncy=Ncz=Nsheet=Nct=Nctt=0;
  CellMode=CELLMODE_None;
  Hdiv=0;
  Scell=OvScell=0;
  MapCells=TUint3(0,0,0);
  BoundLimitOk=BoundDivideOk=false;
  BoundLimitCellMin=BoundLimitCellMax=TUint3(0);
  BoundDivideCellMin=BoundDivideCellMax=TUint3(0);
  DivideFull=false;
}

//==============================================================================
/// Changes the allocated memory space for arrays of Np.
//==============================================================================
void JCellDivCpu::ResizeMemoryNp(unsigned np){
  delete[] CellPart;    CellPart=NULL;
  delete[] SortPart;    SortPart=NULL;
  delete[] VSort;       VSort=NULL;
  MemAllocNp=0;
  if(np){
    try{
      CellPart=new unsigned[np];
      SortPart=new unsigned[np];
      if(LaminarSPS)VSort=new byte[sizeof(tsymatrix3f)*np];
      else VSort=new byte[sizeof(tfloat3)*np];
    }
    catch(const std::bad_alloc){
      RunException("ResizeMemoryNp","The requested memory could not be allocated.");
    }
  }
  VSortInt=(int*)VSort;      VSortWord=(word*)VSort;  
  VSortFloat=(float*)VSort;  VSortFloat3=(tfloat3*)VSort;   
  VSortMatrix3f=(LaminarSPS? (tsymatrix3f*)VSort: NULL);  
  SizeNp=np;
  MemAllocNp=sizeof(unsigned)*SizeNp*2+sizeof(tfloat3)*SizeNp;
}

//==============================================================================
/// Changes the allocated memory space for arrays of Nct.
//==============================================================================
void JCellDivCpu::ResizeMemoryNct(unsigned nct){
  delete[] PartsInCell;   PartsInCell=NULL;
  delete[] BeginCell;     BeginCell=NULL; 
  MemAllocNct=0;
  //char cad[512]; sprintf(cad,"---- JCellDivCpu::ResizeMemoryNct nct:%u",nct); Log->Print(cad);
  if(nct){
    const unsigned nc=SizeBeginCell(nct);
    try{
      PartsInCell=new unsigned[nc-1];
      BeginCell=new unsigned[nc];
    }
    catch(const std::bad_alloc){
      RunException("ResizeMemoryNct","The requested memory could not be allocated.");
    }
  }
  SizeNct=nct;
  MemAllocNct=sizeof(unsigned)*(SizeBeginCell(SizeNct)*2-1);
}

//==============================================================================
/// Displays the information of a boundary particle that was excluded.
//==============================================================================
void JCellDivCpu::VisuBoundaryOut(unsigned p,unsigned id,tfloat3 pos,word code)const{
  string info="particle boundary out> type:";
  word tp=CODE_GetType(code);
  if(tp==CODE_TYPE_FIXED)info=info+"Fixed";
  else if(tp==CODE_TYPE_MOVING)info=info+"Moving";
  else if(tp==CODE_TYPE_FLOATING)info=info+"Floating";
  info=info+" cause:";
  word out=CODE_GetOutValue(code);
  if(out==CODE_OUT_MOVE)info=info+"Speed";
  else if(out==CODE_OUT_POS)info=info+"Position";
  else info=info+"???";
  char cad[512];
  sprintf(cad," p:%u id:%u pos:(%f,%f,%f)",p,id,pos.x,pos.y,pos.z);
  Log->PrintDbg(info+cad);
}

//==============================================================================
/// Returns coordinates of cell starting from a position.
//==============================================================================
tuint3 JCellDivCpu::GetMapCell(const tfloat3 &pos)const{
  float dx=pos.x-MapPosMin.x,dy=pos.y-MapPosMin.y,dz=pos.z-MapPosMin.z;
  unsigned cx=unsigned(dx*OvScell),cy=unsigned(dy*OvScell),cz=unsigned(dz*OvScell);
  return(TUint3(cx,cy,cz));
}

//==============================================================================
/// Computes minimum and maximum positions of a given range of particles Bound.
/// Ignores the particles with check[p]!=0.
/// If the particles is out the limits of position, check[p]=CHECK_OUTPOS.
//==============================================================================
void JCellDivCpu::CalcCellDomainBound(unsigned n,unsigned pini,const unsigned* idp,const tfloat3* pos,word* code,tuint3 &cellmin,tuint3 &cellmax){
  tfloat3 pmin=TFloat3(FLT_MAX,FLT_MAX,FLT_MAX);
  tfloat3 pmax=TFloat3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
  const bool perix=(PeriActive&1)!=0,periy=(PeriActive&2)!=0,periz=(PeriActive&4)!=0;
  unsigned nerr=0;
  const unsigned pfin=pini+n;
  for(unsigned p=pini;p<pfin;p++){
    const float px=pos[p].x,py=pos[p].y,pz=pos[p].z;
    float dx=px-MapPosMin.x,dy=py-MapPosMin.y,dz=pz-MapPosMin.z;
    bool partin=(perix||(dx>=0&&dx<MapPosDif.x))&&(periy||(dy>=0&&dy<MapPosDif.y))&&(periz||(dz>=0&&dz<MapPosDif.z)); //-Particle inside the domain.
    bool ok=!CODE_GetOutValue(code[p]);
    if(partin&&ok){
      if(pmin.x>px)pmin.x=px;
      if(pmin.y>py)pmin.y=py;
      if(pmin.z>pz)pmin.z=pz;
      if(pmax.x<px)pmax.x=px;
      if(pmax.y<py)pmax.y=py;
      if(pmax.z<pz)pmax.z=pz;
    }
    else{
      if(ok)code[p]=CODE_SetOutPos(code[p]);
      if(nerr<10)VisuBoundaryOut(p,idp[p],OrderDecodeValue(Order,pos[p]),code[p]);
      nerr++;
    }
  }
  if(nerr)RunException("CalcCellDomainBound","Some boundary particle was found outside the domain.");
  pmax=pmax+TFloat3(Scell*0.01f);
  cellmin=(pmin.x>pmax.x? MapCells: GetMapCell(pmin));
  cellmax=(pmin.x>pmax.x? TUint3(0): GetMapCell(pmax));
  //char cad[512]; sprintf(cad,"CalcDomainBound> pos:(%s)-(%s) cell:(%s)-(%s)",fun::Float3gStr(pmin).c_str(),fun::Float3gStr(pmax).c_str(),fun::Uint3Str(cellmin).c_str(),fun::Uint3Str(cellmax).c_str()); Log->PrintDbg(cad);
}

//==============================================================================
/// Computes minimum and maximum positions of a given range of particles Fluid.
/// Ignores the particles with check[p]!=0.
/// If the particles is out the limits of position, check[p]=CHECK_OUTPOS.
/// If rhop!=NULL and the particle is out the allowed range (rhopmin,rhopmax)
/// then check[p]=CHECK_OUTRHOP.
//==============================================================================
void JCellDivCpu::CalcCellDomainFluid(unsigned n,unsigned pini,const unsigned* idp,const tfloat3* pos,const float* rhop,word* code,tuint3 &cellmin,tuint3 &cellmax){
  tfloat3 pmin=TFloat3(FLT_MAX,FLT_MAX,FLT_MAX);
  tfloat3 pmax=TFloat3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
  const bool perix=(PeriActive&1)!=0,periy=(PeriActive&2)!=0,periz=(PeriActive&4)!=0;
  unsigned nerr=0;
  const unsigned pfin=pini+n;
  for(unsigned p=pini;p<pfin;p++)if(!CODE_GetOutValue(code[p])){
    const float px=pos[p].x,py=pos[p].y,pz=pos[p].z;
    float dx=px-MapPosMin.x,dy=py-MapPosMin.y,dz=pz-MapPosMin.z;
    bool partin=(perix||(dx>=0&&dx<MapPosDif.x))&&(periy||(dy>=0&&dy<MapPosDif.y))&&(periz||(dz>=0&&dz<MapPosDif.z)); //-Particle inside the domain.
    bool rhopok=(RhopOutMin<=rhop[p]&&rhop[p]<=RhopOutMax);
    if(partin&&rhopok){
      if(pmin.x>px)pmin.x=px;
      if(pmin.y>py)pmin.y=py;
      if(pmin.z>pz)pmin.z=pz;
      if(pmax.x<px)pmax.x=px;
      if(pmax.y<py)pmax.y=py;
      if(pmax.z<pz)pmax.z=pz;
    }
    else{
      code[p]=(rhopok? CODE_SetOutPos(code[p]): CODE_SetOutRhop(code[p]));
      if(!rhopok)NpfOutRhop++;
      if(!partin&&CODE_GetType(code[p])==CODE_TYPE_FLOATING){
        if(nerr<10)VisuBoundaryOut(p,idp[p],OrderDecodeValue(Order,pos[p]),code[p]);
        nerr++;
      }
    }
  }
  else if(CODE_GetOutValue(code[p])==CODE_OUT_MOVE)NpfOutMove++; 
    
  if(nerr)RunException("CalcCellDomainFluid","Some floating particle was found outside the domain.");
  pmax=pmax+TFloat3(Scell*0.01f); 
  cellmin=(pmin.x>pmax.x? MapCells: GetMapCell(pmin));
  cellmax=(pmin.x>pmax.x? TUint3(0): GetMapCell(pmax));
  //char cad[512]; sprintf(cad,"CalcDomainFluid> pos:(%s)-(%s) cell:(%s)-(%s)",fun::Float3gStr(pmin).c_str(),fun::Float3gStr(pmax).c_str(),fun::Uint3Str(cellmin).c_str(),fun::Uint3Str(cellmax).c_str()); Log->Print(cad);
}

//==============================================================================
/// Reorders data of all particles.
//==============================================================================
void JCellDivCpu::SortParticles(unsigned *vec){
  if(DivideFull){
    for(unsigned p=0;p<Nptot;p++)VSortInt[p]=vec[SortPart[p]];
    memcpy(vec,VSortInt,sizeof(unsigned)*Nptot);
  }
  else{
    for(unsigned p=Npb;p<Nptot;p++)VSortInt[p]=vec[SortPart[p]];
    memcpy(vec+Npb,VSortInt+Npb,sizeof(unsigned)*(Nptot-Npb));
  }
}
//==============================================================================
void JCellDivCpu::SortParticles(word *vec){
  if(DivideFull){
    for(unsigned p=0;p<Nptot;p++)VSortWord[p]=vec[SortPart[p]];
    memcpy(vec,VSortWord,sizeof(word)*Nptot);
  }
  else{
    for(unsigned p=Npb;p<Nptot;p++)VSortWord[p]=vec[SortPart[p]];
    memcpy(vec+Npb,VSortWord+Npb,sizeof(word)*(Nptot-Npb));
  }
}
//==============================================================================
void JCellDivCpu::SortParticles(float *vec){
  if(DivideFull){
    for(unsigned p=0;p<Nptot;p++)VSortFloat[p]=vec[SortPart[p]];
    memcpy(vec,VSortFloat,sizeof(float)*Nptot);
  }
  else{
    for(unsigned p=Npb;p<Nptot;p++)VSortFloat[p]=vec[SortPart[p]];
    memcpy(vec+Npb,VSortFloat+Npb,sizeof(float)*(Nptot-Npb));
  }
}
//==============================================================================
void JCellDivCpu::SortParticles(tfloat3 *vec){
  if(DivideFull){
    for(unsigned p=0;p<Nptot;p++)VSortFloat3[p]=vec[SortPart[p]];
    memcpy(vec,VSortFloat3,sizeof(tfloat3)*Nptot);
  }
  else{
    for(unsigned p=Npb;p<Nptot;p++)VSortFloat3[p]=vec[SortPart[p]];
    memcpy(vec+Npb,VSortFloat3+Npb,sizeof(tfloat3)*(Nptot-Npb));
  }
}

//==============================================================================
void JCellDivCpu::SortParticles(tsymatrix3f *vec){  //ALEX_SPS
  if(DivideFull){
    for(unsigned p=0;p<Nptot;p++)VSortMatrix3f[p]=vec[SortPart[p]];
    memcpy(vec,VSortMatrix3f,sizeof(tsymatrix3f)*Nptot);
  }
  else{
    for(unsigned p=Npb;p<Nptot;p++)VSortMatrix3f[p]=vec[SortPart[p]];
    memcpy(vec+Npb,VSortMatrix3f+Npb,sizeof(tsymatrix3f)*(Nptot-Npb));
  }
}

//==============================================================================
/// Shows the current limits of the simulation.
//==============================================================================
tfloat3 JCellDivCpu::GetDomainLimits(bool limitmin,unsigned slicecellmin)const{
  tuint3 celmin=GetCellDomainMin(),celmax=GetCellDomainMax();
  if(celmin.x>celmax.x)celmin.x=celmax.x=0; else celmax.x++;
  if(celmin.y>celmax.y)celmin.y=celmax.y=0; else celmax.y++;
  if(celmin.z>celmax.z)celmin.z=celmax.z=slicecellmin; else celmax.z++;
  tfloat3 pmin=MapPosMin+TFloat3(Scell*celmin.x,Scell*celmin.y,Scell*celmin.z);
  tfloat3 pmax=MapPosMin+TFloat3(Scell*celmax.x,Scell*celmax.y,Scell*celmax.z);
  return(limitmin? pmin: pmax);
}

//==============================================================================
/// Indicates if the cell is empty or not.
//==============================================================================
bool JCellDivCpu::CellNoEmpty(unsigned box,byte kind)const{
#ifdef DBG_JCellDivCpu
  if(box>=Nct)RunException("CellNoEmpty","No valid cell.");
#endif
  if(kind==2)box+=BoxFluid;
  return(BeginCell[box]<BeginCell[box+1]);
}

//==============================================================================
/// Returns the first particle of a cell.
//==============================================================================
unsigned JCellDivCpu::CellBegin(unsigned box,byte kind)const{
#ifdef DBG_JCellDivCpu
  if(box>Nct)RunException("CellBegin","No valid cell.");
#endif
  return(BeginCell[(kind==1? box: box+BoxFluid)]);
}

//==============================================================================
/// Returns the number of particles of a cell.
//==============================================================================
unsigned JCellDivCpu::CellSize(unsigned box,byte kind)const{
#ifdef DBG_JCellDivCpu
  if(box>Nct)RunException("CellSize","No valid cell.");
#endif
  if(kind==2)box+=BoxFluid;
  return(BeginCell[box+1]-BeginCell[box]);
}




