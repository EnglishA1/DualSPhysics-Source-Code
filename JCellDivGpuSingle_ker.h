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

/// \file JCellDivGpuSingle_ker.h \brief Declares functions and CUDA kernels to compute operations of the Neighbour List.

#ifndef _JCellDivGpuSingle_ker_
#define _JCellDivGpuSingle_ker_

#include "JCellDivGpu_ker.h"

namespace cudiv{

void PreSort(bool full,unsigned np,unsigned npb,const float3 *pos,const word *code,tfloat3 posmin,tfloat3 difmax,tuint3 ncells,float ovscell,unsigned *cellpart,unsigned *sortpart,JLog2 *log);

}
#endif



