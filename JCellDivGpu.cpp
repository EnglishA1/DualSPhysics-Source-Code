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

/// \file JCellDivGpu.cpp \brief Implements the class \ref JCellDivGpu.

#include "JCellDivGpu.h"
#include "JCellDivGpu_ker.h"
#include "Functions.h"
#include "JFormatFiles2.h"

using namespace std;

//==============================================================================
/// Constructor.
//==============================================================================
JCellDivGpu::JCellDivGpu(bool stable,JLog2 *log,std::string dirout,byte periactive,bool laminarsps,const tfloat3 &mapposmin,const tfloat3 &mapposmax,float dosh,unsigned casenbound,unsigned casenfixed,unsigned casenpb,TpCellOrder order):Stable(stable),PeriActive(periactive),LaminarSPS(laminarsps),MapPosMin(mapposmin),MapPosMax(mapposmax),Dosh(dosh),CaseNbound(casenbound),CaseNfixed(casenfixed),CaseNpb(casenpb),Order(order),MapPosDif(mapposmax-mapposmin),Floating(CaseNpb!=CaseNbound){
  ClassName="JCellDivGpu";
  Log=log;
  DirOut=dirout;
  CellPart=NULL;  SortPart=NULL;  AuxMem=NULL;
  BeginEndCell=NULL;
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JCellDivGpu::~JCellDivGpu(){
  //char cad[1024]; sprintf(cad,"---> DivideFull:%u/%u",NdivFull,Ndiv); Log->PrintDbg(cad);
  Reset();
}
 
//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JCellDivGpu::Reset(){
  FreeBasicMemoryAll();
  RhopOut=false; RhopOutMin=RhopOutMax=0;
  Ndiv=NdivFull=0;
  Nptot=Np=Npb=0;
  NpbOut=NpfOut=NpfOutRhop=NpfOutMove=NpbIgnore=0;
  CellDomainMin=CellDomainMax=TUint3(0);
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
/// Releases allocated memory for cells.
//==============================================================================
void JCellDivGpu::FreeBasicMemoryNct(){
  cudaFree(BeginEndCell); BeginEndCell=NULL;
  SizeNct=0;
  MemAllocGpuNct=0;
}

//==============================================================================
/// Releases allocated memory for particles and cells.
//==============================================================================
void JCellDivGpu::FreeBasicMemoryAll(){
  FreeBasicMemoryNct();
  cudaFree(CellPart);     CellPart=NULL;
  cudaFree(SortPart);     SortPart=NULL;
  cudaFree(AuxMem);       AuxMem=NULL; 
  SizeNp=SizeAuxMem=0;
  MemAllocGpuNp=0;
}

//==============================================================================
/// Assigns basic memory according to the number of particles. 
//==============================================================================
void JCellDivGpu::AllocBasicMemoryNp(unsigned np){
  FreeBasicMemoryAll();
  //-Allocates memory for particles.
  SizeNp=np;
  size_t m=sizeof(unsigned)*SizeNp;
  cudaMalloc((void**)&CellPart,m); MemAllocGpuNp+=m;
  cudaMalloc((void**)&SortPart,m); MemAllocGpuNp+=m;
  SizeAuxMem=cudiv::LimitsPosSize(SizeNp);
  m=sizeof(float)*SizeAuxMem;
  cudaMalloc((void**)&AuxMem,m); MemAllocGpuNp+=m;
}

//==============================================================================
/// Checks memory allocation for particles. 
/// Returns true if memory can not be allocated.
//==============================================================================
bool JCellDivGpu::CheckMemoryNp(bool returnerror){
  bool err=false;
  cudaError_t cuerr=cudaGetLastError();
  if(cuerr!=cudaSuccess){
    err=true;
    Log->Print(string("**CellDiv: GPU memory allocation failed for ")+fun::UintStr(SizeNp)+" particles.");
    if(cuerr!=cudaErrorMemoryAllocation||!returnerror)RunExceptionCuda("CheckMemoryNp","GPU memory allocation failed.",cuerr);
  }
  //-Shows the requested memory.
  char cad[128];
  sprintf(cad,"**CellDiv: Requested GPU memory for %u particles: %.1f MB.",SizeNp,double(MemAllocGpuNp)/(1024*1024)); Log->Print(cad);
  return(err);
}

//==============================================================================
/// Assigns memory according to the number of cells. 
//==============================================================================
void JCellDivGpu::AllocBasicMemoryNct(unsigned nct){
  FreeBasicMemoryNct();
  //-Allocates memory for cells.
  SizeNct=nct;
  size_t m=sizeof(int2)*SizeBeginEndCell(SizeNct);
  cudaMalloc((void**)&BeginEndCell,m); MemAllocGpuNct+=m;
}

//==============================================================================
/// Checks memory allocation for particles. 
/// Returns true if memory can not be allocated.
//==============================================================================
bool JCellDivGpu::CheckMemoryNct(bool returnerror){
  bool err=false;
  cudaError_t cuerr=cudaGetLastError();
  if(cuerr!=cudaSuccess){
    err=true;
    Log->Print(string("**CellDiv: GPU memory allocation failed for ")+fun::UintStr(SizeNct)+" cells (CellMode="+GetNameCellMode(CellMode)+").");
    if(cuerr!=cudaErrorMemoryAllocation||!returnerror)RunExceptionCuda("AllocMemoryNct","GPU memory allocation failed.",cuerr);
  }
  //-Shows the requested memory.
  char cad[128];
  sprintf(cad,"**CellDiv: Requested gpu memory for %u cells (CellMode=%s): %.1f MB.",SizeNct,GetNameCellMode(CellMode),double(MemAllocGpuNct)/(1024*1024)); Log->Print(cad);
  return(err);
}

//==============================================================================
/// Displays the information of a boundary particle that was excluded.
//==============================================================================
void JCellDivGpu::VisuBoundaryOut(unsigned p,unsigned id,tfloat3 pos,word code)const{
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
tuint3 JCellDivGpu::GetMapCell(const tfloat3 &pos)const{
  float dx=pos.x-MapPosMin.x,dy=pos.y-MapPosMin.y,dz=pos.z-MapPosMin.z;
  unsigned cx=unsigned(dx*OvScell),cy=unsigned(dy*OvScell),cz=unsigned(dz*OvScell);
  return(TUint3(cx,cy,cz));
}

//==============================================================================
/// Computes minimum and maximum positions of a given range of particles Bound.
/// Ignores the particles with check[p]!=0.
/// If the particles is out the limits of position, check[p]=CHECK_OUTPOS.
//==============================================================================
void JCellDivGpu::CalcCellDomainBound(unsigned n,unsigned pini,const float3* posg,word* codeg,tuint3 &cellmin,tuint3 &cellmax){
  tfloat3 pmin,pmax;
  cudiv::LimitsPos(PeriActive,n,pini,posg,NULL,codeg,MapPosMin,MapPosDif,0,0,AuxMem,pmin,pmax,Log);
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
void JCellDivGpu::CalcCellDomainFluid(unsigned n,unsigned pini,const float3* posg,const float* rhopg,word* codeg,tuint3 &cellmin,tuint3 &cellmax){
  tfloat3 pmin,pmax;
  cudiv::LimitsPos(PeriActive,n,pini,posg,(RhopOut? rhopg: NULL),codeg,MapPosMin,MapPosDif,RhopOutMin,RhopOutMax,AuxMem,pmin,pmax,Log);
  pmax=pmax+TFloat3(Scell*0.01f);
  cellmin=(pmin.x>pmax.x? MapCells: GetMapCell(pmin));
  cellmax=(pmin.x>pmax.x? TUint3(0): GetMapCell(pmax));
  //char cad[512]; sprintf(cad,"CalcDomainFluid> pos:(%s)-(%s) cell:(%s)-(%s)",fun::Float3gStr(pmin).c_str(),fun::Float3gStr(pmax).c_str(),fun::Uint3Str(cellmin).c_str(),fun::Uint3Str(cellmax).c_str()); Log->Print(cad);
}

//==============================================================================
/// Returns the first particle of a cell.
//==============================================================================
void JCellDivGpu::CellBeginEnd(unsigned cell,unsigned ndata,unsigned* data)const{
  cudaMemcpy(data,BeginEndCell+cell,sizeof(int)*ndata,cudaMemcpyDeviceToHost);
}

//==============================================================================
/// Returns the first particle of a cell.
//==============================================================================
int2 JCellDivGpu::CellBeginEnd(unsigned cell)const{
  int2 v;
  cudaMemcpy(&v,BeginEndCell+cell,sizeof(int2),cudaMemcpyDeviceToHost);
  return(v);
}

//==============================================================================
/// Reorders basic arrays according to SortPart. 
//==============================================================================
void JCellDivGpu::SortBasicArrays(const unsigned *idp,const float3 *pos,const float3 *vel,const float *rhop,const word *code,const float *press,const float *viscop,const unsigned *idpm, unsigned *idp2,float3 *pos2,float3 *vel2,float *rhop2,word *code2,float *press2,float *viscop2,unsigned *idpm2){
  const unsigned pini=(DivideFull? 0: Npb);
  cudiv::SortDataParticles(Nptot,pini,SortPart,idp,pos,vel,rhop,code,press,viscop,idpm,idp2,pos2,vel2,rhop2,code2,press2,viscop2,idpm2);
}

//==============================================================================
/// Reorders arrays of particle data according to SortPart. 
//==============================================================================
void JCellDivGpu::SortDataArrays(const float *a,const float3 *b,float *a2,float3 *b2){
  const unsigned pini=(DivideFull? 0: Npb);
  cudiv::SortDataParticles(Nptot,pini,SortPart,a,b,a2,b2);
}
//==============================================================================
void JCellDivGpu::SortDataArrays(const float *a,const float3 *b,const float3 *c,float *a2,float3 *b2,float3 *c2){
  const unsigned pini=(DivideFull? 0: Npb);
  cudiv::SortDataParticles(Nptot,pini,SortPart,a,b,c,a2,b2,c2);
}
//==============================================================================
void JCellDivGpu::SortDataArrays(const tsymatrix3f *a,tsymatrix3f *a2){
  const unsigned pini=(DivideFull? 0: Npb);
  cudiv::SortDataParticles(Nptot,pini,SortPart,a,a2);
}

//==============================================================================
/// Ends process of CellDive:
/// Checking that all excluded particles are fluids 
/// and computing the number of excluded by pos, rhop or mov.
/// The components of the data are already in the original order.
//==============================================================================
void JCellDivGpu::CheckParticlesOut(unsigned npout,const unsigned *idp,const tfloat3 *pos,const float *rhop,const word *code){
  unsigned nerr=0;
  for(unsigned p=0;p<npout;p++){
    word type=CODE_GetType(code[p]);
    if(nerr<10&&type==CODE_TYPE_FIXED||type==CODE_TYPE_MOVING||type==CODE_TYPE_FLOATING){ //-There are some excluded boundary particle.
      nerr++;
      VisuBoundaryOut(p,idp[p],pos[p],code[p]);
    }
    word out=CODE_GetOutValue(code[p]);
    if(out==CODE_OUT_RHOP)NpfOutRhop++;
    else if(out==CODE_OUT_MOVE)NpfOutMove++;
  }
  if(nerr)RunException("CheckParticlesOut","A boundary particle was excluded.");
}

//==============================================================================
/// Returns a pointer with the auxiliar allocated memory in GPU,
/// that was only used as intermediate during some tasks.
/// So that this memory can be used for other uses.
/// This memory is resized according to the number of particles thus 
/// its size and direction can vary.
//==============================================================================
float* JCellDivGpu::GetAuxMem(unsigned size){
  //printf("GetAuxMem> size:%u  SizeAuxMem:%u\n",size,SizeAuxMem);
  if(size>SizeAuxMem)RunException("GetAuxMem","The requested memory is not available.");
  return(AuxMem);
}

//==============================================================================
/// Returns current limts of the domain.
//==============================================================================
tfloat3 JCellDivGpu::GetDomainLimits(bool limitmin,unsigned slicecellmin)const{
  tuint3 celmin=GetCellDomainMin(),celmax=GetCellDomainMax();
  if(celmin.x>celmax.x)celmin.x=celmax.x=0; else celmax.x++;
  if(celmin.y>celmax.y)celmin.y=celmax.y=0; else celmax.y++;
  if(celmin.z>celmax.z)celmin.z=celmax.z=slicecellmin; else celmax.z++;
  tfloat3 pmin=MapPosMin+TFloat3(Scell*celmin.x,Scell*celmin.y,Scell*celmin.z);
  tfloat3 pmax=MapPosMin+TFloat3(Scell*celmax.x,Scell*celmax.y,Scell*celmax.z);
  return(limitmin? pmin: pmax);
}

//==============================================================================
/// Returns range of particles in the range of cells.
//==============================================================================
uint2 JCellDivGpu::GetRangeParticlesCells(bool fluid,unsigned celini,unsigned celfin)const{
  if(fluid){ celini+=BoxFluid; celfin+=BoxFluid; }
  unsigned pmin=UINT_MAX,pmax=0;
  if(celini<celfin){
    bool memorynew=false;
    unsigned *auxg=NULL;
    unsigned size=cudiv::GetRangeParticlesCellsSizeAux(celini,celfin);
    if(size<=SizeAuxMem)auxg=(unsigned*)AuxMem;
    else{
      memorynew=true;
      cudaMalloc((void**)&auxg,sizeof(unsigned)*size);
    } 
    cudiv::GetRangeParticlesCells(celini,celfin,BeginEndCell,auxg,pmin,pmax,Log);
    if(memorynew)cudaFree(auxg);
  }
  uint2 rg; rg.x=pmin; rg.y=pmax;
  return(rg);
}

//==============================================================================
/// Returns number of particles in the range of cells.
//==============================================================================
unsigned JCellDivGpu::GetParticlesCells(unsigned celini,unsigned celfin){
  unsigned count=0;
  if(celini<celfin){
    bool memorynew=false;
    unsigned *auxg=NULL;
    unsigned size=cudiv::GetParticlesCellsSizeAux(celini,celfin);
    if(size<=SizeAuxMem)auxg=(unsigned*)AuxMem;
    else{
      memorynew=true;
      cudaMalloc((void**)&auxg,sizeof(unsigned)*size);
    } 
    count=cudiv::GetParticlesCells(celini,celfin,BeginEndCell,auxg,Log);
    if(memorynew)cudaFree(auxg);
  }
  return(count);
}



