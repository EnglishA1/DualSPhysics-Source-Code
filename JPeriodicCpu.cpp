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

/// \file JPeriodicCpu.cpp \brief Implements the class \ref JPeriodicCpu.

#include "JPeriodicCpu.h"
#include "JCellDivCpu.h"
#include "Functions.h"
#include "JFormatFiles2.h"

using namespace std;

//==============================================================================
/// Constructor.
//==============================================================================
JPeriodicCpu::JPeriodicCpu(JLog2 *log,std::string dirout,const tfloat3 &mapposmin,const tfloat3 &mapposmax,const JCellDivCpu* celldiv):MapPosMin(mapposmin),MapPosMax(mapposmax),CellDiv(celldiv),Width(celldiv->Dosh),Hdiv(celldiv->GetHdiv()),OvScell(1.f/celldiv->GetScell()),LaminarSPS(celldiv->LaminarSPS){
  ClassName="JPeriodicCpu";
  Log=log;
  DirOut=dirout;
  BorList=NULL;
  PartsInCell=NULL; BeginCell=NULL;
  Ref=NULL; CellPart=NULL; SortPart=NULL;
  Idp=NULL; Code=NULL; Pos=NULL; Vel=NULL; Rhop=NULL;
  Csound=NULL; PrRhop=NULL; Tensil=NULL; Tau=NULL;
  Reset();
}

//==============================================================================
/// Destructor
//==============================================================================
JPeriodicCpu::~JPeriodicCpu(){
  Reset();
}

//==============================================================================
/// Initialisation of variables.
//==============================================================================
void JPeriodicCpu::Reset(){
  Axes[0]=Axes[1]=Axes[2]=false;
  ResizeMemoryList(0);
  ModesCount=0;
  Nrun=0;
  WithData=false;
}

//==============================================================================
/// Returns the allocated memory in CPU.
//==============================================================================
long long JPeriodicCpu::GetAllocMemory()const{  
  long long s=0;
  //-As function of SizeNp.
  if(BorList)s+=sizeof(unsigned)*(SizeBorList);
  if(Ref)s+=sizeof(unsigned)*SizeNp;
  if(CellPart)s+=sizeof(unsigned)*SizeNp;
  if(SortPart)s+=sizeof(unsigned)*SizeNp;
  if(Idp)s+=sizeof(unsigned)*(SizeNp);
  if(Code)s+=sizeof(word)*(SizeNp);
  if(Pos)s+=sizeof(tfloat3)*SizeNp;
  if(Vel)s+=sizeof(tfloat3)*SizeNp;
  if(Rhop)s+=sizeof(float)*SizeNp;
  if(Csound)s+=sizeof(float)*SizeNp;
  if(PrRhop)s+=sizeof(float)*SizeNp;
  if(Tensil)s+=sizeof(float)*SizeNp;
  if(Tau)s+=sizeof(tsymatrix3f)*SizeNp;
  //-As function of SizeNct.
  const unsigned nctt=SizeBeginCell(SizeNct);
  if(PartsInCell)s+=sizeof(unsigned)*(nctt-1);
  if(BeginCell)s+=sizeof(unsigned)*nctt;
  return(s);
}

//==============================================================================
/// Resizes BorList.
//==============================================================================
void JPeriodicCpu::ResizeMemoryList(unsigned size){
  ResizeMemoryNp(0);
  ResizeMemoryNct(0);
  delete[] BorList; BorList=NULL;
  if(size)BorList=new unsigned[size];
  SizeBorList=size;
}

//==============================================================================
/// Resizes arrays of Nct.
//==============================================================================
void JPeriodicCpu::ResizeMemoryNct(unsigned nct){
  ResizeMemoryNp(0);
  SizeNct=0;
  delete[] PartsInCell;  PartsInCell=NULL; 
  delete[] BeginCell;    BeginCell=NULL; 
  if(nct){
    const unsigned nctt=SizeBeginCell(nct);
    try{
      PartsInCell=new unsigned[nctt-1];
      BeginCell=new unsigned[nctt];
    }
    catch(const std::bad_alloc){
      RunException("ResizeMemoryNct","The requested memory could not be allocated.");
    }
  }
  SizeNct=nct;
}

//==============================================================================
/// Resizes arrays of Np.
//==============================================================================
void JPeriodicCpu::ResizeMemoryNp(unsigned np){
  delete[] Ref;      Ref=NULL;
  delete[] CellPart; CellPart=NULL;
  delete[] SortPart; SortPart=NULL;
  delete[] Idp;      Idp=NULL;
  delete[] Code;     Code=NULL;
  delete[] Pos;      Pos=NULL;
  delete[] Vel;      Vel=NULL;
  delete[] Rhop;     Rhop=NULL;
  delete[] Csound;   Csound=NULL;
  delete[] PrRhop;   PrRhop=NULL;
  delete[] Tensil;   Tensil=NULL;
  delete[] Tau;      Tau=NULL;
  if(np>0){
    try{
      Ref=new unsigned[np];
      CellPart=new unsigned[np];
      SortPart=new unsigned[np];
      Idp=new unsigned[np];
      Code=new word[np];
      Pos=new tfloat3[np];
      Vel=new tfloat3[np];
      Rhop=new float[np];
      Csound=new float[np];
      PrRhop=new float[np];
      Tensil=new float[np];
      if(LaminarSPS)Tau=new tsymatrix3f[np];
    }
    catch(const std::bad_alloc){
      RunException("ResizeMemoryNp","The requested memory could not be allocated.");
    }
  }
  SizeNp=np;
}

//==============================================================================
/// Configures modes of periodicity.
//==============================================================================
void JPeriodicCpu::ConfigMode(TpPeriMode mode,tfloat3 perinc){
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
void JPeriodicCpu::SelecZone(unsigned czone){
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
template<JPeriodicCpu::TpPeriZone tzone> void JPeriodicCpu::CalcInterCell(tuint3 &celmin,tuint3 &celmax)const{
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
void JPeriodicCpu::ClearBorderList(){
  BorNp1=BorBoundNp1=BorFluidNp1=BorNp2=BorBoundNp2=BorFluidNp2=0;
  UseBorRange=false;
  BorBoundIni1=BorFluidIni1=BorBoundIni2=BorFluidIni2=0;
}

//==============================================================================
/// Creates list of particles of a periodic edge. 
//==============================================================================
void JPeriodicCpu::ConfigRangeBorderList(unsigned celini,unsigned celfin,unsigned &bini,unsigned &fini,unsigned &bnp,unsigned &fnp){
  unsigned boundfin=CellDiv->CellBegin(celfin-1,1)+CellDiv->CellSize(celfin-1,1);
  unsigned boundini=CellDiv->CellBegin(celini,1);
  unsigned fluidfin=CellDiv->CellBegin(celfin-1,2)+CellDiv->CellSize(celfin-1,2);
  unsigned fluidini=CellDiv->CellBegin(celini,2);
  //char cad[128]; sprintf(cad,"CalcBorderList1> b:(%u - %u) f:(%u - %u)",boundini,boundfin,fluidini,fluidfin); Log->PrintDbg(cad);
  if(boundfin<=boundini)boundini=boundfin=0;
  if(fluidfin<=fluidini)fluidini=fluidfin=0;
  bini=boundini; bnp=boundfin-boundini;
  fini=fluidini; fnp=fluidfin-fluidini;
}


//==============================================================================
/// Adds particles to a list and returns the total amount of particles in the list,
/// this value can be higher than the size of the list.
/// - sizelist: Maximum number of particles in the list.
/// - countlist: Number of particles previously stored.
/// - with axismax=(0, 1 o 2) checks that position is higher than posmin.
//==============================================================================
template<unsigned axismax> unsigned JPeriodicCpu::GetListParticlesCells_(tuint3 ncells,unsigned celini,const unsigned* begincell,const tfloat3* pos,tuint3 celmin,tuint3 celmax,tfloat3 posmin,unsigned countlist,unsigned sizelist,unsigned *list)const{
  unsigned cnew=countlist;
  const unsigned ncx=ncells.x;
  const unsigned nsheet=ncx*ncells.y;
  for(unsigned cz=celmin.z;cz<=celmax.z;cz++){
    const unsigned celbase=celini+cz*nsheet;
    for(unsigned cy=celmin.y;cy<=celmax.y;cy++){
      const unsigned celbase2=celbase+cy*ncx;
      const unsigned pini=begincell[celbase2+celmin.x];
      const unsigned pfin=begincell[celbase2+celmax.x+1];
      for(unsigned p=pini;p<pfin;p++){
        bool sel=true;
        if(axismax==0)sel=(posmin.x<=pos[p].x);
        if(axismax==1)sel=(posmin.y<=pos[p].y);
        if(axismax==2)sel=(posmin.z<=pos[p].z);
        if(sel){
          if(cnew<sizelist)list[cnew]=p; cnew++;
        }
      }
    }
  }
  return(cnew);
}
//==============================================================================
unsigned JPeriodicCpu::GetListParticlesCells(unsigned axismax,tuint3 ncells,unsigned celini,const unsigned* begincell,const tfloat3* pos,tuint3 celmin,tuint3 celmax,tfloat3 posmin,unsigned countlist,unsigned sizelist,unsigned *list)const{
  unsigned num=0;
  if(axismax==0)     num=GetListParticlesCells_<0>(ncells,celini,begincell,pos,celmin,celmax,posmin,countlist,sizelist,list);
  else if(axismax==1)num=GetListParticlesCells_<1>(ncells,celini,begincell,pos,celmin,celmax,posmin,countlist,sizelist,list);
  else if(axismax==2)num=GetListParticlesCells_<2>(ncells,celini,begincell,pos,celmin,celmax,posmin,countlist,sizelist,list);
  else               num=GetListParticlesCells_<9>(ncells,celini,begincell,pos,celmin,celmax,posmin,countlist,sizelist,list);
  return(num);
}

//==============================================================================
/// Creates list of particles in periodic edges. 
//==============================================================================
void JPeriodicCpu::CalcBorderList(const tfloat3 *pos){
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
      count2=(!ComputeForces? count: GetListParticlesCells(9,ncells,0,CellDiv->GetBeginCell(),pos,InterCellMin,InterCellMax,interposmin,count,SizeBorList,BorList));
      BorBoundNp1=count2-count; count=count2;   
      count2=GetListParticlesCells(9,ncells,CellDiv->GetBoxFluid(),CellDiv->GetBeginCell(),pos,InterCellMin,InterCellMax,interposmin,count,SizeBorList,BorList);
      BorFluidNp1=count2-count; count=count2;       
    //-Gets list of right edge.
      count2=(!ComputeForces? count: GetListParticlesCells(axis,ncells,0,CellDiv->GetBeginCell(),pos,DataCellMin,DataCellMax,interposmin,count,SizeBorList,BorList));
      BorBoundNp2=count2-count; count=count2;   
      count2=GetListParticlesCells(axis,ncells,CellDiv->GetBoxFluid(),CellDiv->GetBeginCell(),pos,DataCellMin,DataCellMax,interposmin,count,SizeBorList,BorList);
      BorFluidNp2=count2-count; count=count2;       
      //-Repeates the process if size of BorList[] is not enough.
      if(count>SizeBorList){
        ResizeMemoryList(ValueMemoryInc(count));
        run=true;
      }
      else run=false;
    }
  }
  BorNp1=BorBoundNp1+BorFluidNp1;
  BorNp2=BorBoundNp2+BorFluidNp2;
  //if(1){ char cad[256]; sprintf(cad,"++> bnp1:%u fnp1:%u  bnp2:%u fnp2:%u",BorBoundNp1,BorFluidNp1,BorBoundNp2,BorFluidNp2); Log->PrintDbg(cad); }
}

//==============================================================================
/// Prepares zone of particles for the periodic condition.
//==============================================================================
void JPeriodicCpu::Prepare(bool forces,unsigned czone,const tfloat3* pos,const float* rhop,const word* code,const unsigned* idp,const tfloat3* vel,const float *csound,const float *prrhop,const float *tensil,const tsymatrix3f* tau){
  ComputeForces=forces;
  //-Selects zone.
  SelecZone(czone);
//  Log->PrintDbg("***JPeriodicCpu::Prepare");
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
    if(PerBorderMin)CalcBorderList(pos);
    //if(PerBorderMin)DgSaveVtkBorList(pos,idp); //-Generates VTK with points (debug).

    //-Computes cells of zone and allocates memory.
    const tuint3 ncells=CellDiv->GetNcells()+TUint3(Hdiv+Hdiv);
    Nc1=Nc2=0;
    if(PerZone==PERI_Xmin||PerZone==PERI_Xmax){ Nc1=ncells.y; Nc2=ncells.z; }
    if(PerZone==PERI_Ymin||PerZone==PERI_Ymax){ Nc1=ncells.x; Nc2=ncells.z; }
    if(PerZone==PERI_Zmin||PerZone==PERI_Zmax){ Nc1=ncells.x; Nc2=ncells.y; }
    Nct=Nc1*Nc2; //Nctt=Nct+Nct+1;  BoxOut=Nctt-1;
    //if(1){ char cad[512]; sprintf(cad,"Prepare> Nc1:%u Nc2:%u Nct:%u BoxOut:%u",Nc1,Nc2,Nct,BoxOut); Log->PrintDbg(cad); }
    if(Nct>SizeNct)ResizeMemoryNct(ValueMemoryInc(Nct));

    //-Computes maximum number of particles in zone.
    unsigned npmax=max(BorNp1,BorNp2);
    if(PerMode==PMODE_XY||PerMode==PMODE_XZ||PerMode==PMODE_YZ)npmax*=2;
    //-Allocates memory for particles.
    if(npmax>SizeNp)ResizeMemoryNp(ValueMemoryInc(npmax));
    //-Loads particles of DataCellMin/Max.
    LoadParticles(pos,rhop,code,idp,vel,csound,prrhop,tensil,tau);
  }
  else{ Nc1=Nc2=Nct=0; Np=Npb=0; DataCellMin=TUint3(1); DataCellMax=TUint3(0); }
}

//==============================================================================
/// Copies particle data in the given ranges and computes cell for particles inside the periodic zone.
//==============================================================================
template<int axis> void JPeriodicCpu::PreSortRange_(unsigned np,unsigned npb,unsigned bini,unsigned fini,tfloat3 posinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,float ovscell,const tfloat3 *pos,unsigned *ref,unsigned *cellpart,tfloat3 *pos2)const{
  const int n=int(np);
#ifdef _WITHOMP
  #pragma omp parallel for schedule (static)
#endif
  for(int p=0;p<n;p++){
    const bool bound=(unsigned(p)<npb);
    const unsigned p2=unsigned(p)+(bound? bini: fini-npb);
    const unsigned boxini=(bound? 0: nc1*nc2);
    const unsigned boxout=nc1*nc2*2+(bound? 0: 1);
    ref[p]=p2;
    tfloat3 rpos=pos[p2]+posinc;
    pos2[p]=rpos;
    const float dx=rpos.x-posmin.x,dy=rpos.y-posmin.y,dz=rpos.z-posmin.z;
    bool ok=(dx>=0 && dy>=0 && dz>=0 && dx<difmax.x && dy<difmax.y && dz<difmax.z);
    unsigned c1,c2;
    if(axis==0){ c1=(ok? unsigned(dy*ovscell): nc1); c2=unsigned(dz*ovscell); }
    if(axis==1){ c1=(ok? unsigned(dx*ovscell): nc1); c2=unsigned(dz*ovscell); }
    if(axis==2){ c1=(ok? unsigned(dx*ovscell): nc1); c2=unsigned(dy*ovscell); }
    cellpart[p]=(c1<nc1&&c2<nc2? boxini+c2*nc1+c1: boxout);
  }
}
//==============================================================================
void JPeriodicCpu::PreSortRange(unsigned axis,unsigned np,unsigned npb,unsigned bini,unsigned fini,tfloat3 posinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,float ovscell,const tfloat3 *pos,unsigned *ref,unsigned *cellpart,tfloat3 *pos2)const{
  if(axis==0)     PreSortRange_<0>(np,npb,bini,fini,posinc,posmin,difmax,nc1,nc2,ovscell,pos,ref,cellpart,pos2);
  else if(axis==1)PreSortRange_<1>(np,npb,bini,fini,posinc,posmin,difmax,nc1,nc2,ovscell,pos,ref,cellpart,pos2);
  else if(axis==2)PreSortRange_<2>(np,npb,bini,fini,posinc,posmin,difmax,nc1,nc2,ovscell,pos,ref,cellpart,pos2);
}

//==============================================================================
/// Copies particle data in the given list and computes cell for particles inside the periodic zone.
//==============================================================================
template<int axis> void JPeriodicCpu::PreSortList_(unsigned np,unsigned npb,const unsigned *list,tfloat3 posinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,float ovscell,const tfloat3 *pos,unsigned *ref,unsigned *cellpart,tfloat3 *pos2)const{
  const int n=int(np);
#ifdef _WITHOMP
  #pragma omp parallel for schedule (static) if(n>1000)
#endif
  for(int p=0;p<n;p++){
    const bool bound=(unsigned(p)<npb);
    const unsigned p2=list[p];
    const unsigned boxini=(bound? 0: nc1*nc2);
    const unsigned boxout=nc1*nc2*2+(bound? 0: 1);
    ref[p]=p2;
    tfloat3 rpos=pos[p2]+posinc;
    pos2[p]=rpos;
    const float dx=rpos.x-posmin.x,dy=rpos.y-posmin.y,dz=rpos.z-posmin.z;
    bool ok=(dx>=0 && dy>=0 && dz>=0 && dx<difmax.x && dy<difmax.y && dz<difmax.z);
    unsigned c1,c2;
    if(axis==0){ c1=(ok? unsigned(dy*ovscell): nc1); c2=unsigned(dz*ovscell); }
    if(axis==1){ c1=(ok? unsigned(dx*ovscell): nc1); c2=unsigned(dz*ovscell); }
    if(axis==2){ c1=(ok? unsigned(dx*ovscell): nc1); c2=unsigned(dy*ovscell); }
    cellpart[p]=(c1<nc1&&c2<nc2? boxini+c2*nc1+c1: boxout);
  }
}
//==============================================================================
void JPeriodicCpu::PreSortList(unsigned axis,unsigned np,unsigned npb,const unsigned *list,tfloat3 posinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,float ovscell,const tfloat3 *pos,unsigned *ref,unsigned *cellpart,tfloat3 *pos2)const{
  if(axis==0)     PreSortList_<0>(np,npb,list,posinc,posmin,difmax,nc1,nc2,ovscell,pos,ref,cellpart,pos2);
  else if(axis==1)PreSortList_<1>(np,npb,list,posinc,posmin,difmax,nc1,nc2,ovscell,pos,ref,cellpart,pos2);
  else if(axis==2)PreSortList_<2>(np,npb,list,posinc,posmin,difmax,nc1,nc2,ovscell,pos,ref,cellpart,pos2);
}

//==============================================================================
/// Gemerates array of reordering SortPart[] and stores first particle of each cell in BeginCell[].
//==============================================================================
void JPeriodicCpu::MakeSortPart(){
  const unsigned nctot=Nct+Nct+2;
  //-Counts particles in each cell.
  memset(PartsInCell,0,sizeof(unsigned)*nctot);
  for(unsigned p=0;p<Np;p++)PartsInCell[CellPart[p]]++;
  //-Adjusts initial positions of cells.
  BeginCell[0]=0;
  for(unsigned box=0;box<nctot;box++)BeginCell[box+1]=BeginCell[box]+PartsInCell[box];
  //-Locates particles in their cells.
  memset(PartsInCell,0,sizeof(unsigned)*nctot);
  for(unsigned p=0;p<Np;p++){
    unsigned box=CellPart[p];
    SortPart[BeginCell[box]+PartsInCell[box]]=p;
    PartsInCell[box]++;
  }
}

//==============================================================================
/// Reorders particle data according to sortpart.
//==============================================================================
void JPeriodicCpu::SortPosParticles(unsigned np,const unsigned *sortpart,const tfloat3 *pos,const unsigned *ref,tfloat3 *pos2,unsigned *ref2)const{
  const int n=int(np);
#ifdef _WITHOMP
  #pragma omp parallel for schedule (static) if(n>2000)
#endif
  for(int p=0;p<n;p++){
    const unsigned oldpos=sortpart[p];
    pos2[p]=pos[oldpos];
    ref2[p]=ref[oldpos];
  }
}

//==============================================================================
/// Ordered load of particle data.
//==============================================================================
void JPeriodicCpu::SortLoadParticles(unsigned np,const unsigned *ref,const float *rhop,const word *code,const unsigned *idp,const tfloat3 *vel,const float *csound,const float *prrhop,const float *tensil,const tsymatrix3f *tau,float *rhop2,word *code2,unsigned *idp2,tfloat3 *vel2,float *csound2,float *prrhop2,float *tensil2,tsymatrix3f *tau2)const{
  const int n=int(np);
#ifdef _WITHOMP
  #pragma omp parallel for schedule (static) if(n>2000)
#endif
  for(int p=0;p<n;p++){
    const unsigned refpos=ref[p];
    rhop2[p]=rhop[refpos];
    code2[p]=code[refpos];
    idp2[p]=idp[refpos];
    vel2[p]=vel[refpos];
    csound2[p]=csound[refpos];
    prrhop2[p]=prrhop[refpos];
    if(tensil)tensil2[p]=tensil[refpos];
    if(tau)tau2[p]=tau[refpos];
  }
}

//==============================================================================
/// Ordered load of particle data.
//==============================================================================
void JPeriodicCpu::SortLoadParticles(unsigned np,const unsigned *ref,const float *rhop,const word *code,float *rhop2,word *code2)const{
  const int n=int(np);
#ifdef _WITHOMP
  #pragma omp parallel for schedule (static) if(n>2000)
#endif
  for(int p=0;p<n;p++){
    const unsigned refpos=ref[p];
    rhop2[p]=rhop[refpos];
    code2[p]=code[refpos];
  }
}

//==============================================================================
/// Computes index of cell of the particles of Inter for interaction in the periodic zone.
//==============================================================================
template<int axis> void JPeriodicCpu::CalcCellInterRange_(unsigned np,unsigned npb,unsigned bini,unsigned fini,unsigned hdiv,unsigned ncx,unsigned nsheet,unsigned boxfluid,const unsigned *cellpart,unsigned *cellpart2)const{
  const int n=int(np);
#ifdef _WITHOMP
  #pragma omp parallel for schedule (static) if(n>2000)
#endif
  for(int p=0;p<n;p++){
    unsigned p1=p+(unsigned(p)<npb? bini: fini-npb);
    unsigned cel=cellpart[p1];
    if(cel>=boxfluid)cel-=boxfluid;
    //-Gets coordinates of cells of domain of the particle.
    int cx=cel%ncx;
    int cz=int(cel/nsheet);
    int cy=int((cel%nsheet)/ncx);
    //-Computes coordinates for the periodic zone.
    unsigned c1,c2;
    if(axis==0){ c1=cy; c2=cz; }
    if(axis==1){ c1=cx; c2=cz; }
    if(axis==2){ c1=cx; c2=cy; }
    c1+=hdiv; c2+=hdiv;
    cellpart2[p]=unsigned((c1<<16)|(c2&0xffff));
  }
}
//==============================================================================
void JPeriodicCpu::CalcCellInterRange(unsigned axis,unsigned np,unsigned npb,unsigned bini,unsigned fini,unsigned hdiv,unsigned ncx,unsigned nsheet,unsigned boxfluid,const unsigned *cellpart,unsigned *cellpart2)const{
  if(axis==0)     CalcCellInterRange_<0>(np,npb,bini,fini,hdiv,ncx,nsheet,boxfluid,cellpart,cellpart2);
  else if(axis==1)CalcCellInterRange_<1>(np,npb,bini,fini,hdiv,ncx,nsheet,boxfluid,cellpart,cellpart2);
  else if(axis==2)CalcCellInterRange_<2>(np,npb,bini,fini,hdiv,ncx,nsheet,boxfluid,cellpart,cellpart2);
}

//==============================================================================
/// Computes index of cell of the particles of Inter for interaction in the periodic zone.
//==============================================================================
template<int axis> void JPeriodicCpu::CalcCellInterList_(unsigned np,const unsigned *list,unsigned hdiv,unsigned ncx,unsigned nsheet,unsigned boxfluid,const unsigned *cellpart,unsigned *cellpart2)const{
  const int n=int(np);
#ifdef _WITHOMP
  #pragma omp parallel for schedule (static) if(n>2000)
#endif
  for(int p=0;p<n;p++){
    unsigned cel=cellpart[list[p]];
    if(cel>=boxfluid)cel-=boxfluid;
    //-Gets coordinates of cells of domain of the particle.
    int cx=cel%ncx;
    int cz=int(cel/nsheet);
    int cy=int((cel%nsheet)/ncx);
    //-Computes coordinates for the periodic zone.
    unsigned c1,c2;
    if(axis==0){ c1=cy; c2=cz; }
    if(axis==1){ c1=cx; c2=cz; }
    if(axis==2){ c1=cx; c2=cy; }
    c1+=hdiv; c2+=hdiv;
    cellpart2[p]=unsigned((c1<<16)|(c2&0xffff));
  }
}
//==============================================================================
void JPeriodicCpu::CalcCellInterList(unsigned axis,unsigned np,const unsigned *list,unsigned hdiv,unsigned ncx,unsigned nsheet,unsigned boxfluid,const unsigned *cellpart,unsigned *cellpart2)const{
  if(axis==0)     CalcCellInterList_<0>(np,list,hdiv,ncx,nsheet,boxfluid,cellpart,cellpart2);
  else if(axis==1)CalcCellInterList_<1>(np,list,hdiv,ncx,nsheet,boxfluid,cellpart,cellpart2);
  else if(axis==2)CalcCellInterList_<2>(np,list,hdiv,ncx,nsheet,boxfluid,cellpart,cellpart2);
}

//==============================================================================
/// Selects the particles to be duplicated.
//==============================================================================
template<bool modexy,int axis> void JPeriodicCpu::PreSortTwo_(unsigned np,float twposmin,float twposmax,float twinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,tfloat3 *pos,unsigned *ref,unsigned *cellpart,unsigned &newnp,unsigned &newnbound)const{
  const unsigned boxfluid=nc1*nc2;
  const unsigned boxout=boxfluid*2;
  newnbound=0;
  unsigned cnew=np;
  for(unsigned p=0;p<np;p++){
    const unsigned cell=cellpart[p];
    const float rpos=(modexy? pos[p].y: pos[p].z);    //-PMODE_XY or (PMODE_XZ & PMODE_YZ)
    if(cell<boxout&&(rpos<twposmin||rpos>=twposmax)){
      const bool bound=(cell<boxfluid);
      ref[cnew]=ref[p];
      tfloat3 ps=pos[p];
      if(modexy)ps.y+=(ps.y<twposmin? twinc: -twinc); //-PMODE_XY
      else ps.z+=(ps.z<twposmin? twinc: -twinc);      //-PMODE_XZ or PMODE_YZ
      pos[cnew]=ps;
      //-Computes cell.
      const float dx=ps.x-posmin.x,dy=ps.y-posmin.y,dz=ps.z-posmin.z;
      bool ok=(dx>=0 && dy>=0 && dz>=0 && dx<difmax.x && dy<difmax.y && dz<difmax.z);
      unsigned c1,c2;
      if(axis==0){ c1=(ok? unsigned(dy*OvScell): nc1); c2=unsigned(dz*OvScell); }
      if(axis==1){ c1=(ok? unsigned(dx*OvScell): nc1); c2=unsigned(dz*OvScell); }
      if(axis==2){ c1=(ok? unsigned(dx*OvScell): nc1); c2=unsigned(dy*OvScell); }
      cellpart[cnew]=(c1<nc1&&c2<nc2? (bound? 0: boxfluid)+c2*nc1+c1: (bound? boxout: boxout+1));
      if(bound)newnbound++;
      cnew++;
    }
  }
  newnp=cnew-np;
}
//==============================================================================
void JPeriodicCpu::PreSortTwo(bool modexy,int axis,unsigned np,float twposmin,float twposmax,float twinc,tfloat3 posmin,tfloat3 difmax,unsigned nc1,unsigned nc2,tfloat3 *pos,unsigned *ref,unsigned *cellpart,unsigned &newnp,unsigned &newnbound)const{
  if(modexy){ const bool tmodexy=true;
    if(axis==0)     PreSortTwo_<tmodexy,0>(np,twposmin,twposmax,twinc,posmin,difmax,nc1,nc2,pos,ref,cellpart,newnp,newnbound);
    else if(axis==1)PreSortTwo_<tmodexy,1>(np,twposmin,twposmax,twinc,posmin,difmax,nc1,nc2,pos,ref,cellpart,newnp,newnbound);
    else if(axis==2)PreSortTwo_<tmodexy,2>(np,twposmin,twposmax,twinc,posmin,difmax,nc1,nc2,pos,ref,cellpart,newnp,newnbound);
  }else{ const bool tmodexy=false;
    if(axis==0)     PreSortTwo_<tmodexy,0>(np,twposmin,twposmax,twinc,posmin,difmax,nc1,nc2,pos,ref,cellpart,newnp,newnbound);
    else if(axis==1)PreSortTwo_<tmodexy,1>(np,twposmin,twposmax,twinc,posmin,difmax,nc1,nc2,pos,ref,cellpart,newnp,newnbound);
    else if(axis==2)PreSortTwo_<tmodexy,2>(np,twposmin,twposmax,twinc,posmin,difmax,nc1,nc2,pos,ref,cellpart,newnp,newnbound);
  }
}

//==============================================================================
/// Loads particles of the domain DataCellMin/Max that belong to ZonePosMin/Max.
//==============================================================================
void JPeriodicCpu::LoadParticles(const tfloat3* pos,const float* rhop,const word* code,const unsigned* idp,const tfloat3* vel,const float *csound,const float *prrhop,const float *tensil,const tsymatrix3f *tau){
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
    if(PerBorderMin)PreSortRange(axis,Np,Npb,BorBoundIni2,BorFluidIni2,perinc,ZonePosMin,zoneposdif,Nc1,Nc2,OvScell,pos,Ref,CellPart,Pos);
    else            PreSortRange(axis,Np,Npb,BorBoundIni1,BorFluidIni1,perinc,ZonePosMin,zoneposdif,Nc1,Nc2,OvScell,pos,Ref,CellPart,Pos);
  }
  else{
    //Log->PrintDbg("++> LoadParticles_List...");
    PreSortList(axis,Np,Npb,BorList+(PerBorderMin? BorNp1: 0),perinc,ZonePosMin,zoneposdif,Nc1,Nc2,OvScell,pos,Ref,CellPart,Pos);
  }

  //-Adds particles for periodic configuration XY, XZ or YZ.
  if(PerMode==PMODE_XY||PerMode==PMODE_XZ||PerMode==PMODE_YZ){
    float twposmin,twposmax,twinc;
    const bool modexy=(PerMode==PMODE_XY);
    if(modexy){ twposmin=ZonePosMin.y+Width+Width; twposmax=ZonePosMax.y-Width-Width; twinc=MapPosMax.y-MapPosMin.y; }
    else{       twposmin=ZonePosMin.z+Width+Width; twposmax=ZonePosMax.z-Width-Width; twinc=MapPosMax.z-MapPosMin.z; }
    unsigned newnp,newnbound;
    PreSortTwo(modexy,axis,Np,twposmin,twposmax,twinc,ZonePosMin,zoneposdif,Nc1,Nc2,Pos,Ref,CellPart,newnp,newnbound);
    //if(1){ char cad[128]; sprintf(cad,"tw>> nbound:%u nfluid:%u",newnbound,newnp-newnbound); Log->PrintDbg(cad); }
    Np+=newnp; Npb+=newnbound;
  }

  //-Computes BeginCell[] and array of reordering SortPart[].
  MakeSortPart();
  //-Counts excluded particles and discards them.
  {
    const unsigned boxout=Nct+Nct;
    const unsigned noutb=BeginCell[boxout+1]-BeginCell[boxout];
    const unsigned noutf=BeginCell[boxout+2]-BeginCell[boxout+1];
    //if(1){ char cad[256]; sprintf(cad,"++> nout:%u+%u",noutb,noutf); Log->PrintDbg(cad); }
    Np-=(noutb+noutf); Npb-=noutb;
  }
  //-Orders particle data.
  SortPosParticles(Np,SortPart,Pos,Ref,Vel,Idp);
  swap(Pos,Vel);
  swap(Ref,Idp);
  if(ComputeForces)SortLoadParticles(Np,Ref,rhop,code,idp,vel,csound,prrhop,tensil,tau,Rhop,Code,Idp,Vel,Csound,PrRhop,Tensil,Tau);
  else SortLoadParticles(Np,Ref,rhop,code,Rhop,Code);
  //if(1)DgSaveVtkZone();

  //-Computes index of cell of the particles of Inter for interaction in the periodic zone.
  const unsigned ncx=CellDiv->GetNcx();
  const unsigned nsheet=ncx*CellDiv->GetNcy();
  if(UseBorRange){
    if(PerBorderMin)CalcCellInterRange(axis,BorNp1,BorBoundNp1,BorBoundIni1,BorFluidIni1,Hdiv,ncx,nsheet,CellDiv->GetBoxFluid(),CellDiv->GetCellPart(),CellPart);
    else            CalcCellInterRange(axis,BorNp2,BorBoundNp2,BorBoundIni2,BorFluidIni2,Hdiv,ncx,nsheet,CellDiv->GetBoxFluid(),CellDiv->GetCellPart(),CellPart);
  }
  else{
    if(PerBorderMin)CalcCellInterList(axis,BorNp1,BorList,Hdiv,ncx,nsheet,CellDiv->GetBoxFluid(),CellDiv->GetCellPart(),CellPart);
    else            CalcCellInterList(axis,BorNp2,BorList+BorNp1,Hdiv,ncx,nsheet,CellDiv->GetBoxFluid(),CellDiv->GetCellPart(),CellPart);
  }
  //if(1)DgSaveVtkInter(pos,idp);
  //RunException("LoadParticles","Stop...");
  if(ComputeForces&&!PerBorderMin)Nrun++;
}

//==============================================================================
/// Checks position of  particles for periodic conditions in the edge 
/// of a cell for lower limits and of 2 cells for upper limits 2 cells for.
//==============================================================================
void JPeriodicCpu::CheckPosition(unsigned czone,bool boundchanged,tfloat3* pos,const unsigned* idp){
  SelecZone(czone);
  switch(PerZone){
    case PERI_Xmin:  CheckPosition_<PERI_Xmin>(boundchanged,pos,idp);  break;
    case PERI_Xmax:  CheckPosition_<PERI_Xmax>(boundchanged,pos,idp);  break;
    case PERI_Ymin:  CheckPosition_<PERI_Ymin>(boundchanged,pos,idp);  break;
    case PERI_Ymax:  CheckPosition_<PERI_Ymax>(boundchanged,pos,idp);  break;
    case PERI_Zmin:  CheckPosition_<PERI_Zmin>(boundchanged,pos,idp);  break;
    case PERI_Zmax:  CheckPosition_<PERI_Zmax>(boundchanged,pos,idp);  break;
  }
}
//==============================================================================
template<JPeriodicCpu::TpPeriZone tzone> void JPeriodicCpu::CheckPosition_(bool boundchanged,tfloat3* pos,const unsigned* idp){
  //-Computes domin of seaching.
  tuint3 cellmin,cellmax;
  switch(tzone){
    case PERI_Xmin:  CalcInterCell<PERI_Xmin>(cellmin,cellmax); cellmax.x=0;  break;
    case PERI_Xmax:  CalcInterCell<PERI_Xmax>(cellmin,cellmax); cellmin.x++;  break;
    case PERI_Ymin:  CalcInterCell<PERI_Ymin>(cellmin,cellmax); cellmax.y=0;  break;
    case PERI_Ymax:  CalcInterCell<PERI_Ymax>(cellmin,cellmax); cellmin.y++;  break;
    case PERI_Zmin:  CalcInterCell<PERI_Zmin>(cellmin,cellmax); cellmax.z=0;  break;
    case PERI_Zmax:  CalcInterCell<PERI_Zmax>(cellmin,cellmax); cellmin.z++;  break;
  }
  //-Processes particles of domain of cells close to the edge.
  if(cellmin.x<=cellmax.x&&cellmin.y<=cellmax.y&&cellmin.z<=cellmax.z){
    const unsigned ncx=CellDiv->GetNcx();
    const unsigned nsheet=ncx*CellDiv->GetNcy();
    for(byte kind=(boundchanged? 1: 2);kind<=2;kind++){
      for(unsigned cz=cellmin.z;cz<=cellmax.z;cz++)for(unsigned cy=cellmin.y;cy<=cellmax.y;cy++){
        const unsigned base=cz*nsheet+cy*ncx;
        unsigned pini=CellDiv->CellBegin(base+cellmin.x,kind);
        unsigned pfin=CellDiv->CellBegin(base+cellmax.x+1,kind);
        switch(tzone){
          case PERI_Xmin:
          case PERI_Xmax:  
            for(unsigned p=pini;p<pfin;p++){
              if(pos[p].x<MapPosMin.x) pos[p]=pos[p]-PerInc;
              if(pos[p].x>=MapPosMax.x)pos[p]=pos[p]+PerInc;
            }
          break;
          case PERI_Ymin:
          case PERI_Ymax:  
            for(unsigned p=pini;p<pfin;p++){
              if(pos[p].y<MapPosMin.y) pos[p]=pos[p]-PerInc;
              if(pos[p].y>=MapPosMax.y)pos[p]=pos[p]+PerInc;
            }
          break;
          case PERI_Zmin:
          case PERI_Zmax:  
            for(unsigned p=pini;p<pfin;p++){
              if(pos[p].z<MapPosMin.z) pos[p]=pos[p]-PerInc;
              if(pos[p].z>=MapPosMax.z)pos[p]=pos[p]+PerInc;
            }
          break;
        }
      }
    }
  }
}

//==============================================================================
/// Returns type of periodic zone in text.
//==============================================================================
string JPeriodicCpu::PeriToStr(TpPeriZone peri){
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
void JPeriodicCpu::DgSaveVtkBorList(const tfloat3* pos,const unsigned* idp){

  const unsigned np=CellDiv->GetNp(); 
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
        vpos[p]=OrderDecodeValue(CellDiv->Order,pos[pp]);
        vtype[p]=(p<bnp? 0: 1);
        if(vidp)vidp[p]=idp[pp];
      }
    }
    else{
      const unsigned pini=(!cf? 0: BorNp1);
      for(unsigned p=0;p<n;p++){
        unsigned pp=BorList[pini+p];
        vpos[p]=OrderDecodeValue(CellDiv->Order,pos[pp]);
        vtype[p]=(p<bnp? 0: 1);
        if(vidp)vidp[p]=idp[pp];
      }
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
}


//==============================================================================
/// Stores particles selected in the periodic zone (Debug).
//==============================================================================
void JPeriodicCpu::DgSaveVtkZone(){
  Log->PrintDbg("DgSaveVtkZone");
  //-Generates files.
  const unsigned boxfluid=Nc1*Nc2;
  const unsigned boxout=boxfluid*2;
  tfloat3* vpos=new tfloat3[Np];
  tfloat3* vvel=(ComputeForces? new tfloat3[Np]: NULL);
  unsigned* type=new unsigned[Np];
  unsigned* celx=new unsigned[Np];
  unsigned* cely=new unsigned[Np];
  for(unsigned p=0;p<Np;p++){
    vpos[p]=OrderDecodeValue(CellDiv->Order,Pos[p]);
    if(vvel)vvel[p]=OrderDecodeValue(CellDiv->Order,Vel[p]);
    unsigned cel=CellPart[SortPart[p]];
    type[p]=(cel<boxfluid? 0: (cel<boxout? 1: cel-boxout+2));
    if(cel>=boxfluid)cel-=boxfluid;
    celx[p]=unsigned(cel%Nc1);
    cely[p]=unsigned(cel/Nc1);
  }
  //-Generates file.
  JFormatFiles2::StScalarData fields[7];
  unsigned nfields=0;
  if(type){ fields[nfields]=JFormatFiles2::DefineField("Type",JFormatFiles2::UInt32,1,type);  nfields++; }
  if(ComputeForces){ fields[nfields]=JFormatFiles2::DefineField("Idp",JFormatFiles2::UInt32,1,Idp);   nfields++; }
  if(vvel){  fields[nfields]=JFormatFiles2::DefineField("Vel",JFormatFiles2::Float32,3,vvel);   nfields++; }
  if(ComputeForces){ fields[nfields]=JFormatFiles2::DefineField("Rhop",JFormatFiles2::Float32,1,Rhop); nfields++; }
  if(celx){ fields[nfields]=JFormatFiles2::DefineField("CelX",JFormatFiles2::UInt32,1,celx);  nfields++; }
  if(cely){ fields[nfields]=JFormatFiles2::DefineField("CelY",JFormatFiles2::UInt32,1,cely);  nfields++; }
  string fname=(ComputeForces? "Pzone_": "ShPzone_");
  string file=fun::FileNameSec(DirOut+fname+PeriToStr(PerZone)+".vtk",Nrun);
  JFormatFiles2::SaveVtk(file,Np,vpos,nfields,fields);
  //-Releases memory.
  delete[] vpos;
  delete[] vvel;
  delete[] type;
  delete[] celx;
  delete[] cely;
}

//==============================================================================
/// Stores selected particles Inter (Debug).
//==============================================================================
void JPeriodicCpu::DgSaveVtkInter(const tfloat3 *pos,const unsigned *idp){
  Log->PrintDbg("DgSaveVtkInter");

  const unsigned np=(PerBorderMin? BorNp1: BorNp2);
  const unsigned npb=(PerBorderMin? BorBoundNp1: BorBoundNp2);
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
      vpos[p]=OrderDecodeValue(CellDiv->Order,pos[pp]);
      type[p]=(p<npb? 0: 1);
      if(vidp)vidp[p]=idp[pp];
    }
  }
  else{
    const unsigned pini=(PerBorderMin? 0: BorNp1);
    for(unsigned p=0;p<np;p++){
      unsigned pp=BorList[pini+p];
      vpos[p]=OrderDecodeValue(CellDiv->Order,pos[pp]);
      type[p]=(p<npb? 0: 1);
      if(vidp)vidp[p]=idp[pp];
    }
  }
  for(unsigned p=0;p<np;p++){
    unsigned cel=CellPart[p];
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
  delete[] vpos;
  delete[] vidp;
  delete[] celx;
  delete[] cely;
  delete[] type;
}





