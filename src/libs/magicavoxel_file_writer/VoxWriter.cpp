// This is an independent project of an individual developer. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

// Copyright 2018 Stephane Cuillerdier @Aiekick

// Permission is hereby granted, free of charge, to any person obtaining a 
// copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without 
// limitation the rights to use, copy, modify, merge, publish, distribute, 
// sublicense, and/or sell copies of the Software, and to permit persons to 
// whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included 
// in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// This File is a helper for write a vox file after 0.99 release to support 
// the world mode editor
// just add all color with the color Index with AddColor
// And add all voxels with the method AddVoxel with the voxel in world position, and finally save the model
// that's all, the file was initially created for my Proecedural soft 
// "SdfMesher" cf :https://twitter.com/hashtag/sdfmesher?src=hash
// it support just my needs for the moment, but i put here because its a basis for more i thinck

#include "VoxWriter.h"
#include <stdio.h> 

//#define VERBOSE 

namespace vox
{
	DICTstring::DICTstring()
	{
		bufferSize = 0;
	}
	
	void DICTstring::write(FILE *fp)
	{
		bufferSize = (int32_t)buffer.size();
		fwrite(&bufferSize, sizeof(int32_t), 1, fp);
		fwrite(buffer.data(), sizeof(char), bufferSize, fp);	
	}
		
	size_t DICTstring::getSize()
	{
		bufferSize = (int32_t)buffer.size();
		return sizeof(int32_t) + sizeof(char) * bufferSize;
	}
		
	//////////////////////////////////////////////////////////////////

	DICTitem::DICTitem()
	{

	}

	DICTitem::DICTitem(std::string vKey, std::string vValue)
	{
		key.buffer = vKey;
		value.buffer = vValue;
	}

	void DICTitem::write(FILE *fp)
	{
		key.write(fp);
		value.write(fp);	
	}
		
	size_t DICTitem::getSize()
	{
		return key.getSize() + value.getSize();
	}

	//////////////////////////////////////////////////////////////////
	
	DICT::DICT()
	{
		count = 0;
	}

	void DICT::write(FILE *fp)
	{
		count = (int32_t)keys.size();
		fwrite(&count, sizeof(int32_t), 1, fp);
		for (int i=0;i<count;i++)
			keys[i].write(fp);
	}
		
	size_t DICT::getSize()
	{
		count = (int32_t)keys.size();
		size_t s = sizeof(int32_t);
		for (int i=0;i<count;i++)
			s += keys[i].getSize();
		return s;
	}
		
	void DICT::Add(std::string vKey, std::string vValue)
	{
		keys.push_back(DICTitem(vKey, vValue));
	}

	//////////////////////////////////////////////////////////////////
	
	nTRN::nTRN(int32_t countFrames)
	{
		nodeId = 0;
		reservedId = -1;
		childNodeId = 0;
		numFrames = 1;
		layerId = -1;
		numFrames = countFrames;
		while ((int32_t)frames.size() < numFrames)
			frames.push_back(DICT());
	}

	void nTRN::write(FILE *fp)
	{
		// chunk header
		int32_t id = GetMVID('n', 'T', 'R', 'N');
		fwrite(&id, sizeof(int32_t), 1, fp);
		size_t contentSize = getSize();
		fwrite(&contentSize, sizeof(int32_t), 1, fp);
		size_t childSize = 0;
		fwrite(&childSize, sizeof(int32_t), 1, fp);

		// datas's
		fwrite(&nodeId, sizeof(int32_t), 1, fp);
		nodeAttribs.write(fp);
		fwrite(&childNodeId, sizeof(int32_t), 1, fp);
		fwrite(&reservedId, sizeof(int32_t), 1, fp);
		fwrite(&layerId, sizeof(int32_t), 1, fp);
		fwrite(&numFrames, sizeof(int32_t), 1, fp);
		for (int i=0;i<numFrames;i++)
			frames[i].write(fp);
	}

	size_t nTRN::getSize()
	{
		size_t s = sizeof(int32_t) * 5 + nodeAttribs.getSize();
		for (int i = 0; i < numFrames; i++)
			s += frames[i].getSize();
		return s;
	}
	
	//////////////////////////////////////////////////////////////////
	
	nGRP::nGRP(int32_t vCount)
	{
		nodeId = 0;
		nodeChildrenNodes = vCount;
		while ((int32_t)childNodes.size() < nodeChildrenNodes)
			childNodes.push_back(0);
	}

	void nGRP::write(FILE *fp)
	{
		// chunk header
		int32_t id = GetMVID('n', 'G', 'R', 'P');
		fwrite(&id, sizeof(int32_t), 1, fp);
		size_t contentSize = getSize();
		fwrite(&contentSize, sizeof(int32_t), 1, fp);
		size_t childSize = 0;
		fwrite(&childSize, sizeof(int32_t), 1, fp);

		// datas's
		fwrite(&nodeId, sizeof(int32_t), 1, fp);
		nodeAttribs.write(fp);
		fwrite(&nodeChildrenNodes, sizeof(int32_t), 1, fp);
		fwrite(childNodes.data(), sizeof(int32_t), nodeChildrenNodes, fp);
	}

	size_t nGRP::getSize()
	{
		return sizeof(int32_t) * (2 + nodeChildrenNodes) + nodeAttribs.getSize();
	}
	
	//////////////////////////////////////////////////////////////////
	
	MODEL::MODEL()
	{
		modelId = 0;
	}
	
	void MODEL::write(FILE *fp)
	{
		fwrite(&modelId, sizeof(int32_t), 1, fp);
		modelAttribs.write(fp);
	}

	size_t MODEL::getSize()
	{
		return sizeof(int32_t) + modelAttribs.getSize();
	}
	
	//////////////////////////////////////////////////////////////////
	
	nSHP::nSHP(int32_t vCount)
	{
		nodeId = 0;
		numModels = vCount;
		while ((int32_t)models.size() < numModels)
			models.push_back(MODEL());
	}
	
	void nSHP::write(FILE *fp)
	{
		// chunk header
		int32_t id = GetMVID('n', 'S', 'H', 'P');
		fwrite(&id, sizeof(int32_t), 1, fp);
		size_t contentSize = getSize();
		fwrite(&contentSize, sizeof(int32_t), 1, fp);
		size_t childSize = 0;
		fwrite(&childSize, sizeof(int32_t), 1, fp);

		// datas's
		fwrite(&nodeId, sizeof(int32_t), 1, fp);
		nodeAttribs.write(fp);
		fwrite(&numModels, sizeof(int32_t), 1, fp);
		for (int i = 0; i < numModels; i++)
			models[i].write(fp);
	}

	size_t nSHP::getSize()
	{
		size_t s = sizeof(int32_t) * 2 + nodeAttribs.getSize();
		for (int i = 0; i < numModels; i++)
			s += models[i].getSize();
		return s;
	}
	
	//////////////////////////////////////////////////////////////////
	
	LAYR::LAYR()
	{
		nodeId = 0;
		reservedId = -1;
	}

	void LAYR::write(FILE *fp)
	{
		// chunk header
		int32_t id = GetMVID('L', 'A', 'Y', 'R');
		fwrite(&id, sizeof(int32_t), 1, fp);
		size_t contentSize = getSize();
		fwrite(&contentSize, sizeof(int32_t), 1, fp);
		size_t childSize = 0;
		fwrite(&childSize, sizeof(int32_t), 1, fp);

		// datas's
		fwrite(&nodeId, sizeof(int32_t), 1, fp);
		nodeAttribs.write(fp);
		fwrite(&reservedId, sizeof(int32_t), 1, fp);
	}

	size_t LAYR::getSize()
	{
		return sizeof(int32_t) * 2 + nodeAttribs.getSize();
	}

	//////////////////////////////////////////////////////////////////

	SIZE::SIZE()
	{
		sizex = 0;
		sizey = 0;
		sizez = 0;
	}
	
	void SIZE::write(FILE *fp)
	{
		// chunk header
		int32_t id = GetMVID('S', 'I', 'Z', 'E');
		fwrite(&id, sizeof(int32_t), 1, fp);
		size_t contentSize = getSize();
		fwrite(&contentSize, sizeof(int32_t), 1, fp);
		size_t childSize = 0;
		fwrite(&childSize, sizeof(int32_t), 1, fp);

		// datas's
		fwrite(&sizex, sizeof(int32_t), 1, fp);
		fwrite(&sizey, sizeof(int32_t), 1, fp);
		fwrite(&sizez, sizeof(int32_t), 1, fp);
	}

	size_t SIZE::getSize()
	{
		return sizeof(int32_t) * 3;
	}

	//////////////////////////////////////////////////////////////////

	XYZI::XYZI()
	{
		numVoxels = 0;
	}

	void XYZI::write(FILE *fp)
	{
		// chunk header
		int32_t id = GetMVID('X', 'Y', 'Z', 'I');
		fwrite(&id, sizeof(int32_t), 1, fp);
		size_t contentSize = getSize();
		fwrite(&contentSize, sizeof(int32_t), 1, fp);
		size_t childSize = 0;
		fwrite(&childSize, sizeof(int32_t), 1, fp);

		// datas's
		fwrite(&numVoxels, sizeof(int32_t), 1, fp);
		fwrite(voxels.data(), sizeof(uint8_t), voxels.size(), fp);
	}

	size_t XYZI::getSize()
	{
		numVoxels = (int32_t)voxels.size() / 4;
		return sizeof(int32_t) * (1 + numVoxels);
	}

	//////////////////////////////////////////////////////////////////
	
	RGBA::RGBA()
	{
		
	}

	void RGBA::write(FILE *fp)
	{
		// chunk header
		int32_t id = GetMVID('R', 'G', 'B', 'A');
		fwrite(&id, sizeof(int32_t), 1, fp);
		size_t contentSize = getSize();
		fwrite(&contentSize, sizeof(int32_t), 1, fp);
		size_t childSize = 0;
		fwrite(&childSize, sizeof(int32_t), 1, fp);

		// datas's
		fwrite(colors, sizeof(uint8_t), contentSize, fp);
	}

	size_t RGBA::getSize()
	{
		return sizeof(uint8_t) * 4 * 256;
	}

	//////////////////////////////////////////////////////////////////

	VoxCube::VoxCube()
	{
		id = 0;
		tx = 0;
		ty = 0;
		tz = 0;
	}

	void VoxCube::write(FILE *fp)
	{
		size.write(fp);
		xyzi.write(fp);
	}

	//////////////////////////////////////////////////////////////////

	VoxWriter* VoxWriter::Create(const std::string& vFilePathName, const uint32_t& vLimitX,
		const uint32_t& vLimitY, const uint32_t& vLimitZ, int32_t *vError)
	{
		VoxWriter * vox = new VoxWriter(vLimitX, vLimitY, vLimitZ);

		*vError = vox->IsOk(vFilePathName);

		if (*vError == 0)
		{
			return vox;
		}
		else
		{
			printf("Vox file creation failed, err : %s", GetErrnoMsg(*vError).c_str());

			SAFE_DELETE(vox);
		}

		return vox;
	}

	std::string VoxWriter::GetErrnoMsg(const int32_t& vError)
	{
		std::string res;

		switch (vError)
		{
		case 1:
			res = "Operation not permitted"; break;
		case 2:
			res = "No such file or directory"; break;
		case 3:
			res = "No such process"; break;
		case 4:
			res = "Interrupted function"; break;
		case 5:
			res = "I / O error"; break;
		case 6:
			res = "No such device or address"; break;
		case 7:
			res = "Argument list too long"; break;
		case 8:
			res = "Exec format error"; break;
		case 9:
			res = "Bad file number"; break;
		case 10:
			res = "No spawned processes"; break;
		case 11:
			res = "No more processes or not enough memory or maximum nesting level reached"; break;
		case 12:
			res = "Not enough memory"; break;
		case 13:
			res = "Permission denied"; break;
		case 14:
			res = "Bad address"; break;
		case 16:
			res = "Device or resource busy"; break;
		case 17:
			res = "File exists"; break;
		case 18:
			res = "Cross - device link"; break;
		case 19:
			res = "No such device"; break;
		case 20:
			res = "Not a director"; break;
		case 21:
			res = "Is a directory"; break;
		case 22:
			res = "Invalid argument"; break;
		case 23:
			res = "Too many files open in system"; break;
		case 24:
			res = "Too many open files"; break;
		case 25:
			res = "Inappropriate I / O control operation"; break;
		case 27:
			res = "File too large"; break;
		case 28:
			res = "No space left on device"; break;
		case 29:
			res = "Invalid seek"; break;
		case 30:
			res = "Read - only file system"; break;
		case 31:
			res = "Too many links"; break;
		case 32:
			res = "Broken pipe"; break;
		case 33:
			res = "Math argument"; break;
		case 34:
			res = "Result too large"; break;
		case 36:
			res = "Resource deadlock would occur"; break;
		case 38:
			res = "Filename too long"; break;
		case 39:
			res = "No locks available"; break;
		case 40:
			res = "Function not supported"; break;
		case 41:
			res = "Directory not empty"; break;
		case 42:
			res = "Illegal byte sequence"; break;
		case 80:
			res = "String was truncated"; break;
		}

		return res;
	}

	//////////////////////////////////////////////////////////////////
	// the limit of magicavoxel is 127 for one cube, is 127 voxels (indexs : 0 -> 126)
	// vMaxVoxelPerCubeX,Y,Z define the limit of one cube
	VoxWriter::VoxWriter(int32_t vMaxVoxelPerCubeX, int32_t vMaxVoxelPerCubeY, int32_t vMaxVoxelPerCubeZ)
	{
		MV_VERSION = 150; // the old version of MV not open another file than if version is 150 (answer by @ephtracy

		ID_VOX = GetID('V', 'O', 'X', ' ');
		ID_PACK = GetID('P', 'A', 'C', 'K');
		ID_MAIN = GetID('M', 'A', 'I', 'N');
		ID_SIZE = GetID('S', 'I', 'Z', 'E');
		ID_XYZI = GetID('X', 'Y', 'Z', 'I');
		ID_RGBA = GetID('R', 'G', 'B', 'A');
		ID_NTRN = GetID('n', 'T', 'R', 'N');
		ID_NGRP = GetID('n', 'G', 'R', 'P');
		ID_NSHP = GetID('n', 'S', 'H', 'P');

		maxCubeId = 0;

		minCubeX = (int)1e7;
		minCubeY = (int)1e7;
		minCubeZ = (int)1e7;
		
		// the limit of magicavoxel is 127 because the first is 1 not 0
		// so this is 0 to 126
		// index limit, size is 127
		m_MaxVoxelPerCubeX = ct::clamp<int32_t>(vMaxVoxelPerCubeX, 0, 126);
		m_MaxVoxelPerCubeY = ct::clamp<int32_t>(vMaxVoxelPerCubeY, 0, 126);
		m_MaxVoxelPerCubeZ = ct::clamp<int32_t>(vMaxVoxelPerCubeZ, 0, 126);

		maxVolume.lowerBound = 1e7f;
		maxVolume.upperBound = -1e7f;
	}

	VoxWriter::~VoxWriter()
	{

	}

    int32_t VoxWriter::IsOk(const std::string& vFilePathName)
	{
		if (OpenFileForWriting(vFilePathName))
		{
			CloseFile();
		}
		return lastError;
	}

	void VoxWriter::ClearVoxels()
	{

	}

	void VoxWriter::ClearColors()
	{

	}

	void VoxWriter::AddColor(const uint8_t& r, const uint8_t& g, const uint8_t& b, const uint8_t& a, const uint8_t& index)
	{
		while (colors.size() <= index)
			colors.push_back(0);
		colors[index] = GetID(r, g, b, a);
	}

	void VoxWriter::AddVoxel(const int32_t& vX, const int32_t& vY, const int32_t& vZ, const uint8_t& vColorIndex)
	{
		// cube pos
		int32_t ox = (int32_t)std::floor((double)vX / (double)m_MaxVoxelPerCubeX);
		int32_t oy = (int32_t)std::floor((double)vY / (double)m_MaxVoxelPerCubeY);
		int32_t oz = (int32_t)std::floor((double)vZ / (double)m_MaxVoxelPerCubeZ);
		
		minCubeX = ct::mini<int32_t>(minCubeX, ox);
		minCubeY = ct::mini<int32_t>(minCubeX, oy);
		minCubeZ = ct::mini<int32_t>(minCubeX, oz);

		auto cube = GetCube(ox, oy, oz);

		MergeVoxelInCube(vX, vY, vZ, vColorIndex, cube);
	}

	void VoxWriter::SaveToFile(const std::string& vFilePathName)
	{
		if (OpenFileForWriting(vFilePathName))
		{
			int32_t zero = 0;
			
			fwrite(&ID_VOX, sizeof(int32_t), 1, m_File);
			fwrite(&MV_VERSION, sizeof(int32_t), 1, m_File);
			
			// MAIN CHUNCK
			fwrite(&ID_MAIN, sizeof(int32_t), 1, m_File);
			fwrite(&zero, sizeof(int32_t), 1, m_File);

			long numBytesMainChunkPos = GetFilePos();
			fwrite(&zero, sizeof(int32_t), 1, m_File);

			long headerSize = GetFilePos();

			int count = (int)cubes.size();

			int nodeIds = 0;
			nTRN rootTransform(1);
			rootTransform.nodeId = nodeIds;
			rootTransform.childNodeId = ++nodeIds;

			nGRP rootGroup(count);
			rootGroup.nodeId = nodeIds; //
			rootGroup.nodeChildrenNodes = count;
			
			std::vector<nSHP> shapes;
			std::vector<nTRN> shapeTransforms;
			for (int i = 0; i < count; i++)
			{
				VoxCube *c = &cubes[i];
				
				c->write(m_File);
				
				nTRN trans(1);
				trans.nodeId = ++nodeIds; //
				rootGroup.childNodes[i] = nodeIds;
				trans.childNodeId = ++nodeIds;
				trans.layerId = 0;
				
				c->tx = (int)std::floor((c->tx - minCubeX + 0.5f) * m_MaxVoxelPerCubeX - maxVolume.lowerBound.x - maxVolume.Size().x * 0.5);
				c->ty = (int)std::floor((c->ty - minCubeY + 0.5f) * m_MaxVoxelPerCubeY - maxVolume.lowerBound.y - maxVolume.Size().y * 0.5);
				c->tz = (int)std::floor((c->tz - minCubeZ + 0.5f) * m_MaxVoxelPerCubeZ);
				
				// not an animation in my case so only first frame frames[0]
				trans.frames[0].Add("_t", ct::toStr(c->tx) + " " + ct::toStr(c->ty) + " " + ct::toStr(c->tz));
				
				shapeTransforms.push_back(trans);

				nSHP shape(1);
				shape.nodeId = nodeIds; //
				shape.models[0].modelId = i;
				shapes.push_back(shape);
			}

			rootTransform.write(m_File);
			rootGroup.write(m_File);
			
			// trn & shp
			for (int i = 0; i < count; i++)
			{
				shapeTransforms[i].write(m_File);
				shapes[i].write(m_File);
			}

			// no layr in my cases
			
			// layr
			/*for (int i = 0; i < 8; i++)
			{
				LAYR layr;
				layr.nodeId = i;
				layr.nodeAttribs.Add("_name", ct::toStr(i));
				layr.write(m_File);
			}*/

			// RGBA Palette
			if (colors.size() > 0)
			{
				RGBA palette;
				for (int32_t i = 0; i < 255; i++)
				{
					if (i < (int32_t)colors.size())
					{
						palette.colors[i] = colors[i];
					}
					else
					{
						palette.colors[i] = 0;
					}
				}

				palette.write(m_File);
			}

			const long mainChildChunkSize = GetFilePos() - headerSize;
			SetFilePos(numBytesMainChunkPos);
			uint32_t size = (uint32_t)mainChildChunkSize;
			fwrite(&size, sizeof(uint32_t), 1, m_File);
			
			CloseFile();
		}
	}

	uint32_t VoxWriter::GetID(const uint8_t& a, const uint8_t& b, const uint8_t& c, const uint8_t& d)
	{
		return (a) | (b << 8) | (c << 16) | (d << 24);
	}

	bool VoxWriter::OpenFileForWriting(const std::string& vFilePathName)
	{
#if _MSC_VER
		lastError = fopen_s(&m_File, vFilePathName.c_str(), "wb");
#else
        m_File = fopen(vFilePathName.c_str(), "wb");
        lastError = m_File ? 0 : errno;
#endif
		if (lastError != 0)
			return false;
		return true;
	}

	void VoxWriter::CloseFile()
	{
		fclose(m_File);
	}

	long VoxWriter::GetFilePos()
	{
		return ftell(m_File);
	}

	void VoxWriter::SetFilePos(const long& vPos)
	{
		//  SEEK_SET	Beginning of file
		//  SEEK_CUR	Current position of the file pointer
		//	SEEK_END	End of file
		fseek(m_File, vPos, SEEK_SET);
	}

	int32_t VoxWriter::GetCubeId(const int32_t& vX, const int32_t& vY, const int32_t& vZ)
	{
		if (cubesId.find(vX) != cubesId.end())
		{
			if (cubesId[vX].find(vY) != cubesId[vX].end())
			{
				if (cubesId[vX][vY].find(vZ) != cubesId[vX][vY].end())
				{
					return cubesId[vX][vY][vZ];
				}
			}
		}

		cubesId[vX][vY][vZ] = maxCubeId++;

		return cubesId[vX][vY][vZ];
	}

	// Wrap a position inside a particular cube dimension
	inline uint8_t Wrap(int v, int lim)
	{
		v = v % lim;
		if (v < 0)
		{
			v += lim;
		}
		return (uint8_t)v;
	}

	void VoxWriter::MergeVoxelInCube(const int32_t& vX, const int32_t& vY, const int32_t& vZ, const uint8_t& vColorIndex, VoxCube *vCube)
	{
		maxVolume.Combine(ct::dvec3((double)vX, (double)vY, (double)vZ));

		bool exist = false;

		if (voxelId.find(vX) != voxelId.end())
		{
			if (voxelId[vX].find(vY) != voxelId[vX].end())
			{
				if (voxelId[vX][vY].find(vZ) != voxelId[vX][vY].end())
				{
					exist = true;
				}
			}
		}

		if (exist == false)
		{
			vCube->xyzi.voxels.push_back(Wrap(vX, m_MaxVoxelPerCubeX)); // x
			vCube->xyzi.voxels.push_back(Wrap(vY, m_MaxVoxelPerCubeY)); // y
			vCube->xyzi.voxels.push_back(Wrap(vZ, m_MaxVoxelPerCubeZ)); // z

			// correspond a la loc de la couleur du voxel en question
			voxelId[vX][vY][vZ] = (int)vCube->xyzi.voxels.size();

			vCube->xyzi.voxels.push_back(vColorIndex); // color index
		}
	}

	VoxCube* VoxWriter::GetCube(const int32_t& vX, const int32_t& vY, const int32_t& vZ)
	{
		int32_t id = GetCubeId(vX, vY, vZ);

		if (id == cubes.size())
		{
			VoxCube c;

			c.id = id;

			c.tx = vX;
			c.ty = vY;
			c.tz = vZ;

			c.size.sizex = m_MaxVoxelPerCubeX + 1;
			c.size.sizey = m_MaxVoxelPerCubeY + 1;
			c.size.sizez = m_MaxVoxelPerCubeZ + 1;

			cubes.push_back(c);
		}

		if (id < (int32_t)cubes.size())
		{
			return &cubes[id];
		}

		return 0;
	}
}

