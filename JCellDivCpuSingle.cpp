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

/// \file JCellDivCpuSingle.cpp \brief Implements the class \ref JCellDivCpuSingle.

#include "JCellDivCpuSingle.h"
#include "Functions.h"

using namespace std;

//==============================================================================
// Constructor.
//==============================================================================
JCellDivCpuSingle::JCellDivCpuSingle(bool stable,JLog2 *log,std::string dirout,byte periactive,bool laminarsps,bool usefluiddomain,const tfloat3 &mapposmin,const tfloat3 &mapposmax,float dosh,unsigned casenbound,unsigned casenfixed,unsigned casenpb,TpCellOrder order):JCellDivCpu(stable,log,dirout,periactive,laminarsps,mapposmin,mapposmax,dosh,casenbound,casenfixed,casenpb,order),UseFluidDomain(usefluiddomain){
  ClassName="JCellDivCpuSingle";
}

//==============================================================================
/// Configures object for initial use.
//==============================================================================
void JCellDivCpuSingle::ConfigInit(TpCellMode cellmode,unsigned np,unsigned npb,bool rhopout,float rhopmin,float rhopmax){  
  Reset();
  Nptot=Np=np; Npb=npb;
  RhopOut=rhopout; RhopOutMin=rhopmin; RhopOutMax=rhopmax;
  CellMode=cellmode;
  Hdiv=(CellMode==CELLMODE_H? 2: 1);
  Scell=(CellMode==CELLMODE_H? Dosh/2: (CellMode==CELLMODE_2H? Dosh: Dosh*2));
  OvScell=1.f/Scell;
  MapCells=TUint3(unsigned(ceil(MapPosDif.x/Scell)),unsigned(ceil(MapPosDif.y/Scell)),unsigned(ceil(MapPosDif.z/Scell)));
  //-Allocates memory.
  AllocMemory(np,MapCells.x*MapCells.y*MapCells.z);
}

//==============================================================================
/// Computes limits of the domain in cells adjusting to the fluid CellDomainMin/Max). 
/// Labels excluded particles in code[].
//==============================================================================
void JCellDivCpuSingle::CalcCellDomain(const unsigned* idp,const tfloat3* pos,const float* rhop,word* code){
  //-Computes domain of the boundary.
  tuint3 celbmin,celbmax;
  if(!BoundLimitOk){
    CalcCellDomainBound(Npb,0,idp,pos,code,celbmin,celbmax);
    BoundLimitOk=true; BoundLimitCellMin=celbmin; BoundLimitCellMax=celbmax;
  } 
  else{ celbmin=BoundLimitCellMin; celbmax=BoundLimitCellMax; }
  //-Computes domain of the fluid.
  tuint3 celfmin,celfmax;
  CalcCellDomainFluid(Np-Npb,Npb,idp,pos,rhop,code,celfmin,celfmax);
    //-Computes domain adjusting to the boundary and the fluid(with 2h halo). 
  MergeMapCellBoundFluid(celbmin,celbmax,celfmin,celfmax,CellDomainMin,CellDomainMax);
}

//==============================================================================
/// Combines cell limits of boundary and fluid with map limits.
/// With UseFluidDomain=TRUE, it uses domain of the fluid plus 2h if there is boundary
/// if not, it uses the domain with fluid and boundary.
/// If there is no boundary CellDomainMin=CellDomainMax=(0,0,0).
//==============================================================================
void JCellDivCpuSingle::MergeMapCellBoundFluid(const tuint3 &celbmin,const tuint3 &celbmax,const tuint3 &celfmin,const tuint3 &celfmax,tuint3 &celmin,tuint3 &celmax)const{
  //char cad[256];
  //sprintf(cad,"celb=(%u,%u,%u)-(%u,%u,%u)  Npb:%u",celbmin.x,celbmin.y,celbmin.z,celbmax.x,celbmax.y,celbmax.z,Npb); Log->Print(cad);
  //sprintf(cad,"celf=(%u,%u,%u)-(%u,%u,%u)  Np:%u",celfmin.x,celfmin.y,celfmin.z,celfmax.x,celfmax.y,celfmax.z,Np); Log->Print(cad);
  if(UseFluidDomain){
    celmin=TUint3(max(min(celbmin.x,celfmin.x),(celfmin.x>=Hdiv? celfmin.x-Hdiv: 0)),max(min(celbmin.y,celfmin.y),(celfmin.y>=Hdiv? celfmin.y-Hdiv: 0)),max(min(celbmin.z,celfmin.z),(celfmin.z>=Hdiv? celfmin.z-Hdiv: 0)));
    celmax=TUint3(min(max(celbmax.x,celfmax.x),celfmax.x+Hdiv),min(max(celbmax.y,celfmax.y),celfmax.y+Hdiv),min(max(celbmax.z,celfmax.z),celfmax.z+Hdiv));
    //sprintf(cad,"cel1=(%u,%u,%u)-(%u,%u,%u)",celmin.x,celmin.y,celmin.z,celmax.x,celmax.y,celmax.z); Log->Print(cad);
  }
  else{
    celmin=MinValues(celbmin,celfmin);
    celmax=MaxValues(celbmax,celfmax);
    //sprintf(cad,"cel2=(%u,%u,%u)-(%u,%u,%u)",celmin.x,celmin.y,celmin.z,celmax.x,celmax.y,celmax.z); Log->Print(cad);
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
void JCellDivCpuSingle::PreSort(const tfloat3* pos,const word* code){
  //-Computes number of cells.
  Ncx=CellDomainMax.x-CellDomainMin.x+1;
  Ncy=CellDomainMax.y-CellDomainMin.y+1;
  Ncz=CellDomainMax.z-CellDomainMin.z+1;
  //if(1){ char cad[1024]; sprintf(cad,"---> %u>  ncx:%u ncy:%u ncz:%u\n",Ndiv,Ncx,Ncy,Ncz); Log->Print(cad); }
  Nsheet=Ncx*Ncy; Nct=Nsheet*Ncz; Nctt=SizeBeginCell(Nct)-1;
  BoxIgnore=Nct; BoxFluid=BoxIgnore+1; BoxBoundOut=BoxFluid+Nct; BoxFluidOut=BoxBoundOut+1;
  const tfloat3 dposmin=MapPosMin+TFloat3(Scell*CellDomainMin.x,Scell*CellDomainMin.y,Scell*CellDomainMin.z);
  const tfloat3 dposdif=TFloat3(Scell*Ncx,Scell*Ncy,Scell*Ncz);

  //-Loads SortPart[] with the current p in the arrays of data where the particle that should be in that position is.
  //-Loads BeginCell[] with the first particles of each cell.
  if(DivideFull){
    memset(PartsInCell,0,sizeof(unsigned)*Nctt);
    PreSortBound(pos,code,dposmin,dposdif);
    PreSortFluid(pos,code,dposmin,dposdif);
    MakeSortFull();
  }
  else{
    memset(PartsInCell+BoxFluid,0,sizeof(unsigned)*(Nctt-BoxFluid));
    PreSortFluid(pos,code,dposmin,dposdif);
    MakeSortFluid();
  }
}

//==============================================================================
/// Computes cell of each boundary particle and counts particles per cell.
//==============================================================================
void JCellDivCpuSingle::PreSortBound(const tfloat3* pos,const word* code,const tfloat3 &dposmin,const tfloat3 &dposdif){
  for(unsigned p=0;p<Npb;p++){
    const float px=pos[p].x,py=pos[p].y,pz=pos[p].z;
    const float dx=px-dposmin.x,dy=py-dposmin.y,dz=pz-dposmin.z;
    unsigned cx=unsigned(dx*OvScell),cy=unsigned(dy*OvScell),cz=unsigned(dz*OvScell);
    if(cx>=Ncx)cx=Ncx-1; if(cy>=Ncy)cy=Ncy-1; if(cz>=Ncz)cz=Ncz-1;
    unsigned box=(!CODE_GetOutValue(code[p])? ((dx>=0 && dy>=0 && dz>=0 && dx<dposdif.x && dy<dposdif.y && dz<dposdif.z)? cx+cy*Ncx+cz*Nsheet: BoxIgnore): BoxBoundOut);
    CellPart[p]=box;
    PartsInCell[box]++;
  }
}

//==============================================================================
/// Computes cell of each fluid particle and counts particles per cell.
//==============================================================================
void JCellDivCpuSingle::PreSortFluid(const tfloat3* pos,const word* code,const tfloat3 &dposmin,const tfloat3 &dposdif){
  for(unsigned p=Npb;p<Np;p++){
    const float px=pos[p].x,py=pos[p].y,pz=pos[p].z;
    const float dx=px-dposmin.x,dy=py-dposmin.y,dz=pz-dposmin.z;
    unsigned cx=unsigned(dx*OvScell),cy=unsigned(dy*OvScell),cz=unsigned(dz*OvScell);
    if(cx>=Ncx)cx=Ncx-1; if(cy>=Ncy)cy=Ncy-1; if(cz>=Ncz)cz=Ncz-1;
    unsigned box=(!CODE_GetOutValue(code[p])? BoxFluid+cx+cy*Ncx+cz*Nsheet: BoxFluidOut);
    CellPart[p]=box;
    PartsInCell[box]++;
  }
}

//==============================================================================
/// Computes SortPart[] (where is the particle that should be in that position).
/// No problem when there are excluded boundary particles.
//==============================================================================
void JCellDivCpuSingle::MakeSortFull(){
  //-Adjusts initial positions of cells.
  BeginCell[0]=0;
  for(unsigned box=0;box<Nctt;box++)BeginCell[box+1]=BeginCell[box]+PartsInCell[box];
  //-Places particles in their boxes.
  memset(PartsInCell,0,sizeof(unsigned)*Nctt);
  for(unsigned p=0;p<Np;p++){
    unsigned box=CellPart[p];
    SortPart[BeginCell[box]+PartsInCell[box]]=p;
    PartsInCell[box]++;
  }
  //-Orders values of CellPart[].
  SortParticles(CellPart);
}

//==============================================================================
/// Computes SortPart[] (where is the particle that should be in that position).
/// There are not excluded boundary particles.
//==============================================================================
void JCellDivCpuSingle::MakeSortFluid(){
  //-Adjusts initial positions of cells.
  for(unsigned box=BoxFluid;box<Nctt;box++)BeginCell[box+1]=BeginCell[box]+PartsInCell[box];
  //-Places particles in their boxes.
  memset(PartsInCell+BoxFluid,0,sizeof(unsigned)*(Nctt-BoxFluid));
  for(unsigned p=Npb;p<Np;p++){
    unsigned box=CellPart[p];
    SortPart[BeginCell[box]+PartsInCell[box]]=p;
    PartsInCell[box]++;
  }
  //-Orders values of CellPart[].
  SortParticles(CellPart);
}

//==============================================================================
/// Division of particles in cells.
//==============================================================================
void JCellDivCpuSingle::Divide(bool boundchanged,const unsigned* idp,const tfloat3* pos,const float* rhop,word* code,TimersCpu timers){
  const char met[]="Divide";
  TmcStart(timers,TMC_NlLimits);
  //-If the position of the boundary changes, it is necessary to recompute limits and reorder particles. 
  if(boundchanged){
    BoundLimitOk=BoundDivideOk=false;
    BoundLimitCellMin=BoundLimitCellMax=TUint3(0);
    BoundDivideCellMin=BoundDivideCellMax=TUint3(0);
  }
  Nptot=Np;
  NpbOut=NpfOut=NpfOutRhop=NpfOutMove=NpbIgnore=0;
  //-Computes limits of the domain.
  CalcCellDomain(idp,pos,rhop,code);
  TmcStop(timers,TMC_NlLimits);

  //-Determines if divide affects to all particles.
  TmcStart(timers,TMC_NlMakeSort);
  if(!BoundDivideOk||BoundDivideCellMin!=CellDomainMin||BoundDivideCellMax!=CellDomainMax){
    DivideFull=true;
    BoundDivideOk=true; BoundDivideCellMin=CellDomainMin; BoundDivideCellMax=CellDomainMax;
  }
  else DivideFull=false;
//  if(DivideFull)Log->PrintDbg("--> DivideFull=TRUE"); else Log->PrintDbg("--> DivideFull=FALSE");
  //Log->PrintDbg(string("--> ")+fun::UintStr(Ndiv)+string("> CellDomain:")+fun::Uint3RangeStr(CellDomainMin,CellDomainMax));

  PreSort(pos,code);
  //-Updates number of particles.
  NpbIgnore=CellSize(BoxIgnore);
  //NpbOut=CellSize(BoxBoundOut); //-Always zero, if not it throws exception inCalcCellDomainBound().
  NpfOut=CellSize(BoxFluidOut);
  //printf("---> Nct:%u  BoxBoundOut:%u  SizeBeginEndCell:%u\n",Nct,BoxBoundOut,SizeBeginEndCell(Nct));
  //printf("---> NpbIgnore:%u  NpbOut:%u  NpfOut:%u\n",NpbIgnore,NpbOut,NpfOut);
  Np=Nptot-NpbOut-NpfOut;
  //DgSaveCsvBeginCell("BeginCell.csv",Ndiv);
  Ndiv++;
  if(DivideFull)NdivFull++;
  TmcStop(timers,TMC_NlMakeSort);

}


//==============================================================================
/// Stores CSV file with particle data.
//==============================================================================
/*void JCellDivCpuSingle::DgSaveCsvBeginCell(std::string filename,int numfile){
  const char met[]="DgSaveCsvBeginCell";
  int mpirank=Log->GetMpiRank();
  if(mpirank>=0)filename=string("p")+fun::IntStr(mpirank)+"_"+filename;
  if(numfile>=0){
    string ext=fun::GetExtension(filename);
    filename=fun::GetWithoutExtension(filename);
    char cad[64];
    sprintf(cad,"%04d",numfile);
    filename=filename+cad+"."+ext;
  }
  filename=DirOut+filename;
  //-Generates CSV file.
  ofstream pf;
  pf.open(filename.c_str());
  if(pf){
    char cad[1024];
    sprintf(cad,"Ncx:%u;Ncy:%u;Ncz:%u;Nct:%u",Ncx,Ncy,Ncz,Nct);  pf << cad << endl;
    pf << "Cell;Cellxyz;Value" << endl;
    unsigned dif=0;
    for(unsigned c=0;c<Nctt;c++){
      if(c==BoxBoundOut)dif++;
      else{
        pf << fun::UintStr(c-dif);
        if(c==BoxIgnore)pf<< ";BoxIgnore";
        else if(c==BoxBoundOut)pf<< ";BoxBoundOut";
        else if(c==BoxFluidOut)pf<< ";BoxFluidOut";
        else if(c>BoxFluidOut)pf<< ";???";
        else{
          unsigned box=(c<BoxFluid? c: c-BoxFluid);
          const int cz=int(box/Nsheet);
          int bx=box-(cz*Nsheet);
          const int cy=int(bx/Ncx);
          const int cx=bx-(cy*Ncx);
          sprintf(cad,";%c_%u_%u_%u",(c<BoxFluid? 'B': 'F'),cx,cy,cz);
          pf<< cad;
        }
        pf << endl;
      }
    }
    if(pf.fail())RunException(met,"Failed writing to file.",filename);
    pf.close();
  }
  else RunException(met,"File could not be opened.",filename);
}*/




