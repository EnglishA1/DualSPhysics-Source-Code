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

/// \file JPeriodicGpu.cpp \brief Implements the class \ref JPeriodicGpu.

#include "JPeriodicGpu.h"
#include "JPeriodicGpu_ker.h"
#include "JCellDivGpu.h"
#include "Functions.h"
#include "JFormatFiles2.h"

using namespace std;

//==============================================================================
/// Constructor.
//==============================================================================
JPeriodicGpu::JPeriodicGpu(bool stable,JLog2 *log,std::string dirout,const tfloat3 &mapposmin,const tfloat3 &mapposmax,const JCellDivGpu* celldiv):Stable(stable),MapPosMin(mapposmin),MapPosMax(mapposmax),CellDiv(celldiv),Width(celldiv->Dosh),Hdiv(celldiv->GetHdiv()),OvScell(1.f/celldiv->GetScell()),LaminarSPS(celldiv->LaminarSPS){
  ClassName="JPeriodicGpu";
  Log=log;
  DirOut=dirout;
  BorList=NULL;
  BeginEndCell=NULL;
  Ref=NULL; CellPart=NULL; SortPart=NULL;
  Idp=NULL; Code=NULL; PosPres=NULL; VelRhop=NULL; Tau=NULL;
  Reset();
}

//==============================================================================
/// Destructor.
//==============================================================================
JPeriodicGpu::~JPeriodicGpu(){
  Reset();
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JPeriodicGpu::Reset(){
  Axes[0]=Axes[1]=Axes[2]=false;
  ResizeMemoryList(0);
  ModesCount=0;
  Nrun=0;
  WithData=false;
}

//==============================================================================
/// Returns the allocated memory in GPU for the particles.
//==============================================================================
long long JPeriodicGpu::GetAllocMemoryGpuNp()const{  
  long long s=0;
  if(BorList)s+=sizeof(unsigned)*(SizeBorList+1);
  if(Ref)s+=sizeof(unsigned)*(SizeNp+2);
  if(CellPart)s+=sizeof(unsigned)*SizeNp;
  if(SortPart)s+=sizeof(unsigned)*SizeNp;
  if(Idp)s+=sizeof(unsigned)*(SizeNp+2);
  if(Code)s+=sizeof(word)*SizeNp;
  if(PosPres)s+=sizeof(float4)*SizeNp;
  if(VelRhop)s+=sizeof(float4)*SizeNp;
  if(Tau)s+=sizeof(tsymatrix3f)*SizeNp;
  return(s);
}

//==============================================================================
/// Returns the allocated memory in GPU for the cells.
//==============================================================================
long long JPeriodicGpu::GetAllocMemoryGpuNct()const{  
  unsigned s=0;
  if(BeginEndCell)s+=sizeof(int2)*SizeBeginEndCell(SizeNct);
  return(s);
}

//==============================================================================
/// Resizes BorList.
//==============================================================================
void JPeriodicGpu::ResizeMemoryList(unsigned size){
  ResizeMemoryNp(0);
  ResizeMemoryNct(0);
  cudaFree(BorList); BorList=NULL;
  if(size)cudaMalloc((void**)&BorList,sizeof(unsigned)*(size+1));
  SizeBorList=size;
}

//==============================================================================
/// Resizes arrays of Nct.
//==============================================================================
void JPeriodicGpu::ResizeMemoryNct(unsigned nct){
  ResizeMemoryNp(0);
  cudaFree(BeginEndCell); BeginEndCell=NULL;
  if(nct)cudaMalloc((void**)&BeginEndCell,sizeof(int2)*SizeBeginEndCell(nct));
  SizeNct=nct;
}

//==============================================================================
/// Resizes arrays of Np.
//==============================================================================
void JPeriodicGpu::ResizeMemoryNp(unsigned np){
  cudaFree(Ref);      Ref=NULL; 
  cudaFree(CellPart); CellPart=NULL; 
  cudaFree(SortPart); SortPart=NULL; 
  cudaFree(Idp);      Idp=NULL; 
  cudaFree(Code);     Code=NULL; 
  cudaFree(PosPres);  PosPres=NULL; 
  cudaFree(VelRhop);  VelRhop=NULL; 
  cudaFree(Tau);      Tau=NULL; 
  if(np){
    cudaMalloc((void**)&Ref,sizeof(unsigned)*(np+2)); //-Se intercambia con Idp.
    cudaMalloc((void**)&CellPart,sizeof(unsigned)*np);
    cudaMalloc((void**)&SortPart,sizeof(unsigned)*np);
    cudaMalloc((void**)&Idp,sizeof(unsigned)*(np+2)); //-Se usa tb con cuperi::SelecPartsTwo()
    cudaMalloc((void**)&Code,sizeof(word)*np);
    cudaMalloc((void**)&PosPres,sizeof(float4)*np);
    cudaMalloc((void**)&VelRhop,sizeof(float4)*np);
    if(LaminarSPS)cudaMalloc((void**)&Tau,sizeof(tsymatrix3f)*np);
  }
  SizeNp=np;
}

//==============================================================================
/// Configures modes of periodicity.
//==============================================================================
void JPeriodicGpu::ConfigMode(TpPeriMode mode,tfloat3 perinc){
  const char met[]="ConfigMode";
  const int axis=(mode==PMODE_X||mode==PMODE_XY||mode==PMODE_XZ? 0: (mode==PMODE_Y||mode==PMODE_YZ? 1: 2));
  //-Checks that the specific mode does not exist.
  if(Axes[axis])RunException(met,"The specified mode is already configured.");
  //-Checks minimum dimension in periodic axis.
  float w=(!axis? MapPosMax.x-MapPosMin.x: (axis==1? MapPosMax.y-MapPosMin.y: MapPosMax.z-MapPosMin.z));
  if(w<=Width)RunException(met,"The size in periodic axes must be over 2h.");
  //-Adss mode.
  Modes[ModesCount]=mode;
  PeriInc[ModesCount]=perinc;
  ModesCount++;
}

//==============================================================================
/// Selects zone of particles for the periodic condition.
//==============================================================================
void JPeriodicGpu::SelecZone(unsigned czone){
  //-Selects periodic zone.
  if(czone>ModesCount*2)RunException("SelecZone","Error...");
  const unsigned cm=unsigned(czone/2),cz=czone%2;
  PerMode=Modes[cm];
  PerInc=PeriInc[cm];
  if(PerMode==PMODE_X||PerMode==PMODE_XY||PerMode==PMODE_XZ)PerZone=(cz? PERI_Xmax: PERI_Xmin);
  else if(PerMode==PMODE_Y||PerMode==PMODE_YZ)PerZone=(cz? PERI_Ymax: PERI_Ymin);
  else if(PerMode==PMODE_Z)PerZone=(cz? PERI_Zmax: PERI_Zmin);
  PerBorderMin=(PerZone==PERI_Xmin||PerZone==PERI_Ymin||PerZone==PERI_Zmin);
}

//==============================================================================
/// Computes and returns the interaction domain with a periodic zone (in cells).
//==============================================================================
template<JPeriodicGpu::TpPeriZone tzone> void JPeriodicGpu::CalcInterCell(tuint3 &celmin,tuint3 &celmax)const{
  celmin=TUint3(0);
  celmax=CellDiv->GetMapCells()-TUint3(1);
  if(tzone==PERI_Xmin)celmax.x=Hdiv-1; 
  if(tzone==PERI_Xmax)celmin.x=celmax.x-Hdiv;
  if(tzone==PERI_Ymin)celmax.y=Hdiv-1; 
  if(tzone==PERI_Ymax)celmin.y=celmax.y-Hdiv;
  if(tzone==PERI_Zmin)celmax.z=Hdiv-1; 
  if(tzone==PERI_Zmax)celmin.z=celmax.z-Hdiv;
  celmin=MaxValues(celmin,CellDiv->GetCellDomainMin());
  celmax=MinValues(celmax,CellDiv->GetCellDomainMax());
}

//==============================================================================
/// Initialises variables of BorderList.
//==============================================================================
void JPeriodicGpu::ClearBorderList(){
  BorNp1=BorBoundNp1=BorFluidNp1=BorNp2=BorBoundNp2=BorFluidNp2=0;
  UseBorRange=false;
  BorBoundIni1=BorFluidIni1=BorBoundIni2=BorFluidIni2=0;
}

//==============================================================================
/// Creates list of particles of a periodic edge. 
//==============================================================================
void JPeriodicGpu::ConfigRangeBorderList(unsigned celini,unsigned celfin,unsigned &bini,unsigned &fini,unsigned &bnp,unsigned &fnp){
  uint2 bound={0,0};
  if(ComputeForces)bound=CellDiv->GetRangeParticlesCells(false,celini,celfin);
  uint2 fluid=CellDiv->GetRangeParticlesCells(true,celini,celfin);
  //char cad[128]; sprintf(cad,"CalcBorderList1> b:(%u - %u) f:(%u - %u)",bound.x,bound.y,fluid.x,fluid.y); Log->PrintDbg(cad);
  if(bound.y<=bound.x)bound.x=bound.y=0;
  if(fluid.y<=fluid.x)fluid.x=fluid.y=0;
  bini=bound.x; bnp=bound.y-bound.x;
  fini=fluid.x; fnp=fluid.y-fluid.x;
}

//==============================================================================
/// Creates list of particles in periodic edges. 
//==============================================================================
void JPeriodicGpu::CalcBorderList(const float4 *pospres){
  ClearBorderList();  
  if(PerZone==PERI_Zmin){    //-Particles are located in a consecutive way.
    UseBorRange=true;
    const unsigned nsheet=CellDiv->GetNcx()*CellDiv->GetNcy();
    ConfigRangeBorderList(nsheet*InterCellMin.z,nsheet*(InterCellMax.z+1),BorBoundIni1,BorFluidIni1,BorBoundNp1,BorFluidNp1);
    ConfigRangeBorderList(nsheet*DataCellMin.z,nsheet*(DataCellMax.z+1),BorBoundIni2,BorFluidIni2,BorBoundNp2,BorFluidNp2);
    //if(1){ char cad[256]; sprintf(cad,"++> RANGE-> bini1:%u fini1:%u  bini2:%u fini2:%u",BorBoundIni1,BorFluidIni1,BorBoundIni2,BorFluidIni2); Log->PrintDbg(cad); }
  }
  else{
    if(!BorList)ResizeMemoryList(10);
    const unsigned np=CellDiv->GetNp();
    const unsigned npb=CellDiv->GetNpb();
    const unsigned npbok=npb-CellDiv->GetNpbIgnore();
    const tuint3 ncells=CellDiv->GetNcells();
    bool run=true;
    const unsigned axis=(PerZone==PERI_Xmin||PerZone==PERI_Xmax? 0: (PerZone==PERI_Ymin||PerZone==PERI_Ymax? 1: 2));
    const tfloat3 interposmin=MapPosMax-TFloat3(Width);
    while(run){
      //Log->PrintDbg("++> RunLoop...");
      unsigned count=0,count2;
      //-Gets list of left edge.
      count2=(!ComputeForces? count: cuperi::GetListParticlesCells(axis,false,npbok,0,ncells,0,CellDiv->GetCellPart(),pospres,InterCellMin,InterCellMax,interposmin,count,SizeBorList,BorList));
      BorBoundNp1=count2-count; count=count2;   
      count2=cuperi::GetListParticlesCells(axis,false,np-npb,npb,ncells,CellDiv->GetBoxFluid(),CellDiv->GetCellPart(),pospres,InterCellMin,InterCellMax,interposmin,count,SizeBorList,BorList);
      BorFluidNp1=count2-count; count=count2;       
    //-Gets list of right edge.
      count2=(!ComputeForces? count: cuperi::GetListParticlesCells(axis,true,npbok,0,ncells,0,CellDiv->GetCellPart(),pospres,DataCellMin,DataCellMax,interposmin,count,SizeBorList,BorList));
      BorBoundNp2=count2-count; count=count2;   
      count2=cuperi::GetListParticlesCells(axis,true,np-npb,npb,ncells,CellDiv->GetBoxFluid(),CellDiv->GetCellPart(),pospres,DataCellMin,DataCellMax,interposmin,count,SizeBorList,BorList);
      BorFluidNp2=count2-count; count=count2;       
      //-Repeates the process if size of BorList[] is not enough.
      if(count>SizeBorList){
        ResizeMemoryList(ValueMemoryInc(count));
        run=true;
      }
      else run=false;
    }
    //-Orders list created with atomic functions since its order can be different for several executions.
    if(Stable){
      cuperi::SortValues(BorList,BorBoundNp1+BorFluidNp1);
      cuperi::SortValues(BorList+(BorBoundNp1+BorFluidNp1),BorBoundNp2+BorFluidNp2);
    }
  }
  BorNp1=BorBoundNp1+BorFluidNp1;
  BorNp2=BorBoundNp2+BorFluidNp2;
  //if(1){ char cad[256]; sprintf(cad,"++> bnp1:%u fnp1:%u  bnp2:%u fnp2:%u",BorBoundNp1,BorFluidNp1,BorBoundNp2,BorFluidNp2); Log->PrintDbg(cad); }
}

//==============================================================================
/// Prepares zone of particles for the periodic condition.
//==============================================================================
void JPeriodicGpu::Prepare(bool forces,unsigned czone,const float4* pospres,const word* code,const float4* velrhop,const unsigned* idp,const tsymatrix3f* tau){
  ComputeForces=forces;
  //-Selects zone.
  SelecZone(czone);
  //-Determines domain of the zone.
  //if(1){ char cad[512]; sprintf(cad,"Prepare> MapPos:(%s)-(%s)",fun::Float3gStr(MapPosMin).c_str(),fun::Float3gStr(MapPosMax).c_str()); Log->PrintDbg(cad); }
  ZonePosMin=MapPosMin-TFloat3(Width);
  ZonePosMax=MapPosMax+TFloat3(Width);
  if(PerZone==PERI_Xmin)ZonePosMax.x=MapPosMin.x;
  if(PerZone==PERI_Xmax)ZonePosMin.x=MapPosMax.x;
  if(PerZone==PERI_Ymin)ZonePosMax.y=MapPosMin.y;
  if(PerZone==PERI_Ymax)ZonePosMin.y=MapPosMax.y;
  if(PerZone==PERI_Zmin)ZonePosMax.z=MapPosMin.z;
  if(PerZone==PERI_Zmax)ZonePosMin.z=MapPosMax.z;
  //if(1){ char cad[512]; sprintf(cad,"Prepare> ZonePos:(%s)-(%s)",fun::Float3gStr(ZonePosMin).c_str(),fun::Float3gStr(ZonePosMax).c_str()); Log->PrintDbg(cad); }
  //-Computes interaction domain with periodic zone (in cells).
  if(PerZone==PERI_Xmin)CalcInterCell<PERI_Xmin>(InterCellMin,InterCellMax);
  if(PerZone==PERI_Xmax)CalcInterCell<PERI_Xmax>(InterCellMin,InterCellMax);
  if(PerZone==PERI_Ymin)CalcInterCell<PERI_Ymin>(InterCellMin,InterCellMax);
  if(PerZone==PERI_Ymax)CalcInterCell<PERI_Ymax>(InterCellMin,InterCellMax);
  if(PerZone==PERI_Zmin)CalcInterCell<PERI_Zmin>(InterCellMin,InterCellMax);
  if(PerZone==PERI_Zmax)CalcInterCell<PERI_Zmax>(InterCellMin,InterCellMax);
  //if(1){ char cad[512]; sprintf(cad,"Prepare> InterCell:(%s)-(%s)",fun::Uint3Str(InterCellMin).c_str(),fun::Uint3Str(InterCellMax).c_str()); Log->PrintDbg(cad); }
  WithData=(InterCellMin.x<=InterCellMax.x&&InterCellMin.y<=InterCellMax.y&&InterCellMin.z<=InterCellMax.z);
  //if(InterCellMin.x>InterCellMax.x||InterCellMin.y>InterCellMax.y||InterCellMin.z>InterCellMax.z){ InterCellMin=TUint3(1); InterCellMax=TUint3(0); }
  //-Determines source domain of particles (domain of cells where particles are searched).
  if(WithData){
    DataCellMin=TUint3(0);
    DataCellMax=CellDiv->GetMapCells()-TUint3(1);
    if(PerZone==PERI_Xmin)DataCellMin.x=(Hdiv>DataCellMax.x? 0: DataCellMax.x-Hdiv); 
    if(PerZone==PERI_Xmax)DataCellMax.x=Hdiv-1;
    if(PerZone==PERI_Ymin)DataCellMin.y=(Hdiv>DataCellMax.y? 0: DataCellMax.y-Hdiv); 
    if(PerZone==PERI_Ymax)DataCellMax.y=Hdiv-1;
    if(PerZone==PERI_Zmin)DataCellMin.z=(Hdiv>DataCellMax.z? 0: DataCellMax.z-Hdiv); 
    if(PerZone==PERI_Zmax)DataCellMax.z=Hdiv-1;
    DataCellMin=MaxValues(DataCellMin,CellDiv->GetCellDomainMin());
    DataCellMax=MinValues(DataCellMax,CellDiv->GetCellDomainMax());
    WithData=(DataCellMin.x<=DataCellMax.x&&DataCellMin.y<=DataCellMax.y&&DataCellMin.z<=DataCellMax.z);
    //if(DataCellMin.x>DataCellMax.x||DataCellMin.y>DataCellMax.y||DataCellMin.z>DataCellMax.z){ DataCellMin=TUint3(1); DataCellMax=TUint3(0); }
  }
  if(WithData){
    if(PerBorderMin)CalcBorderList(pospres);
    //if(PerBorderMin)DgSaveVtkBorList(pospres,idp); //-Generates VTK with points (debug).

    //-Computes cells of zone and allocates memory.
    const tuint3 ncells=CellDiv->GetNcells()+TUint3(Hdiv+Hdiv);
    Nc1=Nc2=0;
    if(PerZone==PERI_Xmin||PerZone==PERI_Xmax){ Nc1=ncells.y; Nc2=ncells.z; }
    if(PerZone==PERI_Ymin||PerZone==PERI_Ymax){ Nc1=ncells.x; Nc2=ncells.z; }
    if(PerZone==PERI_Zmin||PerZone==PERI_Zmax){ Nc1=ncells.x; Nc2=ncells.y; }
    Nct=Nc1*Nc2; //Nctt=Nct+Nct+2; BoxOut=Nct*2;
    //if(1){ char cad[512]; sprintf(cad,"Prepare> Nc1:%u Nc2:%u Nct:%u BoxOut:%u",Nc1,Nc2,Nct,BoxOut); Log->PrintDbg(cad); }
    if(Nct>SizeNct)ResizeMemoryNct(ValueMemoryInc(Nct));

    //-Computes maximum number of particles in zone.
    unsigned npmax=max(BorNp1,BorNp2);
    if(PerMode==PMODE_XY||PerMode==PMODE_XZ||PerMode==PMODE_YZ)npmax*=2;
    //-Allocates memory for particles.
    if(npmax>SizeNp)ResizeMemoryNp(ValueMemoryInc(npmax));
    //-Loads particles of DataCellMin/Max.
    LoadParticles(pospres,code,velrhop,idp,tau);
  }
  else{ Nc1=Nc2=Nct=0; Np=Npb=0; DataCellMin=TUint3(1); DataCellMax=TUint3(0); }
}

//==============================================================================
/// Loads particles of the domain DataCellMin/Max that belong to ZonePosMin/Max.
//==============================================================================
void JPeriodicGpu::LoadParticles(const float4* pospres,const word* code,const float4* velrhop,const unsigned* idp,const tsymatrix3f* tau){
  const tfloat3 perinc=(PerBorderMin? PerInc: PerInc*TFloat3(-1));
  const unsigned axis=(PerZone==PERI_Xmin||PerZone==PERI_Xmax? 0: (PerZone==PERI_Ymin||PerZone==PERI_Ymax? 1: 2));
  const tfloat3 zoneposdif=ZonePosMax-ZonePosMin;
  Npb=(PerBorderMin? BorBoundNp2: BorBoundNp1);
  Np=(PerBorderMin? BorNp2: BorNp1);
  //if(1){ char cad[512]; sprintf(cad,"LoadParticles> ZonePos:(%s)-(%s)",fun::Float3gStr(ZonePosMin).c_str(),fun::Float3gStr(ZonePosMax).c_str()); Log->PrintDbg(cad); }
  //if(1){ char cad[256]; sprintf(cad,"LoadParticles> np:%u  npb:%u",Np,Npb); Log->PrintDbg(cad); }
  //-Copis particle data located in DataCellMin/Max.
  if(UseBorRange){
    //Log->PrintDbg("++> LoadParticles_Range...");                                  
    if(PerBorderMin)cuperi::PreSortRange(axis,Np,Npb,BorBoundIni2,BorFluidIni2,perinc,ZonePosMin,zoneposdif,Nc1,Nc2,OvScell,pospres,Ref,CellPart,SortPart,PosPres);
    else cuperi::PreSortRange(axis,Np,Npb,BorBoundIni1,BorFluidIni1,perinc,ZonePosMin,zoneposdif,Nc1,Nc2,OvScell,pospres,Ref,CellPart,SortPart,PosPres);
  }
  else{
    //Log->PrintDbg("++> LoadParticles_List...");
    unsigned listini=(PerBorderMin? BorNp1: 0);
    cuperi::PreSortList(axis,Np,Npb,listini,BorList,perinc,ZonePosMin,zoneposdif,Nc1,Nc2,OvScell,pospres,Ref,CellPart,SortPart,PosPres);
  }

  //-Adds particles for periodic configuration XY, XZ or YZ.
  if(PerMode==PMODE_XY||PerMode==PMODE_XZ||PerMode==PMODE_YZ){
    float twposmin,twposmax,twinc;
    const bool modexy=(PerMode==PMODE_XY);
    if(modexy){ twposmin=ZonePosMin.y+Width+Width; twposmax=ZonePosMax.y-Width-Width; twinc=MapPosMax.y-MapPosMin.y; }
    else{       twposmin=ZonePosMin.z+Width+Width; twposmax=ZonePosMax.z-Width-Width; twinc=MapPosMax.z-MapPosMin.z; }
    unsigned newnp,newnbound;
    cuperi::SelecPartsTwo(modexy,Np,twposmin,twposmax,Nct,Nct*2,PosPres,CellPart,Idp,newnp,newnbound);
    if(Stable)cuperi::SortValues(Idp+2,newnp);   //-Orders list that was created with atomic functions and its order can be different for serveral executions.
    //if(1){ char cad[128]; sprintf(cad,"tw>> nbound:%u nfluid:%u",newnbound,newnp-newnbound); Log->PrintDbg(cad); }
    cuperi::PreSortTwo(modexy,axis,newnp,Idp+2,Np,twposmin,twinc,ZonePosMin,zoneposdif,Nc1,Nc2,OvScell,PosPres,Ref,CellPart,SortPart);
    Np+=newnp; Npb+=newnbound;
  }

  //-Orders CellPart and SortPart as function of the cell.
  cuperi::SortZone(CellPart,SortPart,Np,Stable);
  //-Computes initial and final particles of each cell (BeginEndCell).
  cuperi::CalcBeginEndCellZone(Np,SizeBeginEndCell(Nct),CellPart,BeginEndCell);
  //-Counts excluded particles and discards them.
  int2 vnout[2];
  cudaMemcpy(&vnout,BeginEndCell+(SizeBeginEndCell(Nct)-2),sizeof(int2)*2,cudaMemcpyDeviceToHost);
  const unsigned noutb=vnout[0].y-vnout[0].x,noutf=vnout[1].y-vnout[1].x;
  //if(1){ char cad[256]; sprintf(cad,"++> nout:%u+%u",noutb,noutf); Log->PrintDbg(cad); }
  Np-=(noutb+noutf); Npb-=noutb;
  //-Orders particle data.
  cuperi::SortPosParticles(Np,SortPart,PosPres,Ref,VelRhop,Idp);
  swap(PosPres,VelRhop);
  swap(Ref,Idp);
  if(ComputeForces)cuperi::SortLoadParticles(Np,Ref,code,idp,velrhop,tau,Code,Idp,VelRhop,Tau);
  else cuperi::SortLoadParticles(Np,Ref,code,Code);
  //if(1)DgSaveVtkZone();

  //-Computes index of cell of the particles of Inter for interaction in the periodic zone.
  const unsigned ncx=CellDiv->GetNcx();
  const unsigned nsheet=ncx*CellDiv->GetNcy();
  if(UseBorRange){
    if(PerBorderMin)cuperi::CalcCellInterRange(axis,BorNp1,BorBoundNp1,BorBoundIni1,BorFluidIni1,Hdiv,ncx,nsheet,CellDiv->GetBoxFluid(),CellDiv->GetCellPart(),CellPart);
    else            cuperi::CalcCellInterRange(axis,BorNp2,BorBoundNp2,BorBoundIni2,BorFluidIni2,Hdiv,ncx,nsheet,CellDiv->GetBoxFluid(),CellDiv->GetCellPart(),CellPart);
  }
  else{
    if(PerBorderMin)cuperi::CalcCellInterList(axis,BorNp1,BorList,Hdiv,ncx,nsheet,CellDiv->GetBoxFluid(),CellDiv->GetCellPart(),CellPart);
    else            cuperi::CalcCellInterList(axis,BorNp2,BorList+BorNp1,Hdiv,ncx,nsheet,CellDiv->GetBoxFluid(),CellDiv->GetCellPart(),CellPart);
  }
  //if(1)DgSaveVtkInter(pospres,idp);
  //RunException("LoadParticles","Stop...");
  if(ComputeForces&&!PerBorderMin)Nrun++;
}

//==============================================================================
/// Checks position of all particles for all periodic conditions.
//==============================================================================
void JPeriodicGpu::CheckPositionAll(bool boundchanged,float3* posg){
  const unsigned pini=(boundchanged? 0: CellDiv->GetNpb());
  const unsigned n=CellDiv->GetNp()-pini;
  bool xrun=false,yrun=false,zrun=false;
  float xmin,xmax,ymin,ymax,zmin,zmax;
  tfloat3 xperinc,yperinc,zperinc;
  for(unsigned cm=0;cm<ModesCount;cm++){
    TpPeriMode permode=Modes[cm];
    if(permode==PMODE_X||permode==PMODE_XY||permode==PMODE_XZ){
      xrun=true; xmin=MapPosMin.x; xmax=MapPosMax.x; xperinc=PeriInc[cm];
    }else if(permode==PMODE_Y||permode==PMODE_YZ){
      yrun=true; ymin=MapPosMin.y; ymax=MapPosMax.y; yperinc=PeriInc[cm];
    }else if(permode==PMODE_Z){
      zrun=true; zmin=MapPosMin.z; zmax=MapPosMax.z; zperinc=PeriInc[cm];
    }
  }
  cuperi::CheckPositionAll(n,pini,posg,xrun,xmin,xmax,xperinc,yrun,ymin,ymax,yperinc,zrun,zmin,zmax,zperinc);
}

//==============================================================================
/// Returns type of periodic zone in text.
//==============================================================================
string JPeriodicGpu::PeriToStr(TpPeriZone peri){
  switch(peri){
    case PERI_Xmin: return("Xmin");
    case PERI_Xmax: return("Xmax");
    case PERI_Ymin: return("Ymin");
    case PERI_Ymax: return("Ymax");
    case PERI_Zmin: return("Zmin");
    case PERI_Zmax: return("Zmax");
  }
  return("???");
}

//==============================================================================
/// Stores points selected in BorList (Debug).
//==============================================================================
void JPeriodicGpu::DgSaveVtkBorList(const float4* pospres,const unsigned* idp){
  //-Recovers data of GPU.
  const unsigned np=CellDiv->GetNp(); 
  float4* hpos=new float4[np];
  unsigned* hidp=NULL;
  if(idp)hidp=new unsigned[np];
  cudaMemcpy(hpos,pospres,sizeof(float4)*np,cudaMemcpyDeviceToHost);
  if(idp)cudaMemcpy(hidp,idp,sizeof(unsigned)*np,cudaMemcpyDeviceToHost);
  //-Generates files.
  for(int cf=0;cf<2;cf++){
    unsigned bnp=(!cf? BorBoundNp1: BorBoundNp2);
    unsigned fnp=(!cf? BorFluidNp1: BorFluidNp2);
    const unsigned n=bnp+fnp;
    tfloat3* vpos=new tfloat3[n];
    unsigned* vtype=new unsigned[n];
    unsigned* vidp=NULL;
    if(idp)vidp=new unsigned[n];
    if(UseBorRange){
      const unsigned bini=(!cf? BorBoundIni1: BorBoundIni2);
      const unsigned fini=(!cf? BorFluidIni1: BorFluidIni2);
      for(unsigned p=0;p<n;p++){
        unsigned pp=(p<bnp? bini+p: fini+p-bnp);
        float4 ps=hpos[pp];
        tfloat3 ps2=TFloat3(ps.x,ps.y,ps.z);
        vpos[p]=OrderDecodeValue(CellDiv->Order,ps2);
        vtype[p]=(p<bnp? 0: 1);
        if(vidp)vidp[p]=hidp[pp];
      }
    }
    else{
      const unsigned pini=(!cf? 0: BorBoundNp1+BorFluidNp1);
      unsigned* hlist=new unsigned[n];
      cudaMemcpy(hlist,BorList+pini,sizeof(unsigned)*n,cudaMemcpyDeviceToHost);
      for(unsigned p=0;p<n;p++){
        unsigned pp=hlist[p];
        float4 ps=hpos[pp];
        tfloat3 ps2=TFloat3(ps.x,ps.y,ps.z);
        vpos[p]=OrderDecodeValue(CellDiv->Order,ps2);
        vtype[p]=(p<bnp? 0: 1);
        if(vidp)vidp[p]=hidp[pp];
      }
      delete[] hlist;
    }
    //-Generates file.
    JFormatFiles2::StScalarData fields[3];
    unsigned nfields=0;
    if(vtype){ fields[nfields]=JFormatFiles2::DefineField("Type",JFormatFiles2::UInt32,1,vtype); nfields++; }
    if(vidp){  fields[nfields]=JFormatFiles2::DefineField("Idp",JFormatFiles2::UInt32,1,vidp);   nfields++; }
    string file=DirOut+(!cf? "InterZone_min.vtk": "InterZone_max.vtk");
    if(!ComputeForces)file=DirOut+(!cf? "ShInterZone_min.vtk": "ShInterZone_max.vtk");
    JFormatFiles2::SaveVtk(file,n,vpos,nfields,fields);
    //-Releases memory.
    delete[] vpos;
    delete[] vidp;
    delete[] vtype;
  }
  delete[] hpos;
  delete[] hidp;
}

//==============================================================================
/// Stores particles selected in the periodic zone (Debug).
//==============================================================================
void JPeriodicGpu::DgSaveVtkZone(){
  Log->PrintDbg("DgSaveVtkZone");
  //-Recovers data of GPU.
  float4* hpos=new float4[Np];
  unsigned* hcel=new unsigned[Np];
  float4* hvel=(ComputeForces? new float4[Np]: NULL);
  unsigned* hidp=(ComputeForces? new unsigned[Np]: NULL);
  cudaMemcpy(hpos,PosPres,sizeof(float4)*Np,cudaMemcpyDeviceToHost);
  cudaMemcpy(hcel,CellPart,sizeof(unsigned)*Np,cudaMemcpyDeviceToHost);
  if(hvel)cudaMemcpy(hvel,VelRhop,sizeof(float4)*Np,cudaMemcpyDeviceToHost);
  if(hidp)cudaMemcpy(hidp,Idp,sizeof(unsigned)*Np,cudaMemcpyDeviceToHost);
  //-Generates files.
  const unsigned boxfluid=Nc1*Nc2;
  const unsigned boxout=boxfluid*2;
  tfloat3* pos=new tfloat3[Np];
  tfloat3* vel=(ComputeForces? new tfloat3[Np]: NULL);
  float* rhop=(ComputeForces? new float[Np]: NULL);
  unsigned* type=new unsigned[Np];
  unsigned* celx=new unsigned[Np];
  unsigned* cely=new unsigned[Np];
  for(unsigned p=0;p<Np;p++){
    pos[p]=OrderDecodeValue(CellDiv->Order,TFloat3(hpos[p].x,hpos[p].y,hpos[p].z));
    if(vel)vel[p]=OrderDecodeValue(CellDiv->Order,TFloat3(hvel[p].x,hvel[p].y,hvel[p].z));
    if(rhop)rhop[p]=hvel[p].w;
    unsigned cel=hcel[p];
    type[p]=(cel<boxfluid? 0: (cel<boxout? 1: cel-boxout+2));
    if(cel>=boxfluid)cel-=boxfluid;
    celx[p]=unsigned(cel%Nc1);
    cely[p]=unsigned(cel/Nc1);
  }
  //-Generates file.
  JFormatFiles2::StScalarData fields[7];
  unsigned nfields=0;
  if(type){ fields[nfields]=JFormatFiles2::DefineField("Type",JFormatFiles2::UInt32,1,type);  nfields++; }
  if(hidp){ fields[nfields]=JFormatFiles2::DefineField("Idp",JFormatFiles2::UInt32,1,hidp);   nfields++; }
  if(vel){  fields[nfields]=JFormatFiles2::DefineField("Vel",JFormatFiles2::Float32,3,vel);   nfields++; }
  if(rhop){ fields[nfields]=JFormatFiles2::DefineField("Rhop",JFormatFiles2::Float32,1,rhop); nfields++; }
  if(celx){ fields[nfields]=JFormatFiles2::DefineField("CelX",JFormatFiles2::UInt32,1,celx);  nfields++; }
  if(cely){ fields[nfields]=JFormatFiles2::DefineField("CelY",JFormatFiles2::UInt32,1,cely);  nfields++; }
  string fname=(ComputeForces? "Pzone_": "ShPzone_");
  string file=fun::FileNameSec(DirOut+fname+PeriToStr(PerZone)+".vtk",Nrun);
  JFormatFiles2::SaveVtk(file,Np,pos,nfields,fields);
  //-Releases memory.
  delete[] hpos;
  delete[] hcel;
  delete[] hvel;
  delete[] hidp;
  delete[] pos;
  delete[] vel;
  delete[] rhop;
  delete[] type;
  delete[] celx;
  delete[] cely;
}

//==============================================================================
/// Stores selected particles Inter (Debug).
//==============================================================================
void JPeriodicGpu::DgSaveVtkInter(const float4 *pospres,const unsigned *idp){
  Log->PrintDbg("DgSaveVtkInter");
  //-Recovers data of GPU.
  const unsigned nptot=CellDiv->GetNp();
  const unsigned np=(PerBorderMin? BorNp1: BorNp2);
  const unsigned npb=(PerBorderMin? BorBoundNp1: BorBoundNp2);
  float4* hpos=new float4[nptot];
  unsigned* hidp=(idp? new unsigned[nptot]: NULL);
  unsigned* hcel=new unsigned[np];
  cudaMemcpy(hpos,pospres,sizeof(float4)*nptot,cudaMemcpyDeviceToHost);
  if(idp)cudaMemcpy(hidp,idp,sizeof(unsigned)*nptot,cudaMemcpyDeviceToHost);
  cudaMemcpy(hcel,CellPart,sizeof(unsigned)*np,cudaMemcpyDeviceToHost);
  //-Generates files.
  tfloat3* vpos=new tfloat3[np];
  unsigned* vidp=(idp? new unsigned[np]: NULL);
  unsigned* celx=new unsigned[np];
  unsigned* cely=new unsigned[np];
  unsigned* type=new unsigned[np];
  if(UseBorRange){
    const unsigned bini=(PerBorderMin? BorBoundIni1: BorBoundIni2);
    const unsigned fini=(PerBorderMin? BorFluidIni1: BorFluidIni2);
    for(unsigned p=0;p<np;p++){
      unsigned pp=(p<npb? bini+p: fini+p-npb);
      float4 ps=hpos[pp];
      tfloat3 ps2=TFloat3(ps.x,ps.y,ps.z);
      vpos[p]=OrderDecodeValue(CellDiv->Order,ps2);
      type[p]=(p<npb? 0: 1);
      if(vidp)vidp[p]=hidp[pp];
    }
  }
  else{
    const unsigned pini=(PerBorderMin? 0: BorNp1);
    unsigned* hlist=new unsigned[np];
    cudaMemcpy(hlist,BorList+pini,sizeof(unsigned)*np,cudaMemcpyDeviceToHost);
    for(unsigned p=0;p<np;p++){
      unsigned pp=hlist[p];
      float4 ps=hpos[pp];
      tfloat3 ps2=TFloat3(ps.x,ps.y,ps.z);
      vpos[p]=OrderDecodeValue(CellDiv->Order,ps2);
      type[p]=(p<npb? 0: 1);
      if(vidp)vidp[p]=hidp[pp];
    }
    delete[] hlist;
  }
  for(unsigned p=0;p<np;p++){
    unsigned cel=hcel[p];
    celx[p]=cel>>16;
    cely[p]=cel&0xffff;
  }
  //-Generates file.
  JFormatFiles2::StScalarData fields[5];
  unsigned nfields=0;
  if(type){ fields[nfields]=JFormatFiles2::DefineField("Type",JFormatFiles2::UInt32,1,type); nfields++; }
  if(vidp){ fields[nfields]=JFormatFiles2::DefineField("Idp",JFormatFiles2::UInt32,1,vidp);  nfields++; }
  if(celx){ fields[nfields]=JFormatFiles2::DefineField("CelX",JFormatFiles2::UInt32,1,celx); nfields++; }
  if(cely){ fields[nfields]=JFormatFiles2::DefineField("CelY",JFormatFiles2::UInt32,1,cely); nfields++; }
  string fname=(ComputeForces? "Pinter_": "ShPinter_");
  string file=fun::FileNameSec(DirOut+fname+PeriToStr(PerZone)+".vtk",Nrun);
  JFormatFiles2::SaveVtk(file,np,vpos,nfields,fields);
  //-Releases memory.
  delete[] hpos;
  delete[] hidp;
  delete[] hcel;
  delete[] vidp;
  delete[] vpos;
  delete[] celx;
  delete[] cely;
  delete[] type;
}

