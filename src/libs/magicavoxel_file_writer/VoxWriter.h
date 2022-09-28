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
#ifndef __VOX_WRITER_H__
#define __VOX_WRITER_H__

#include <cstdint>
#include <cmath>
#include <string>
#include <map>
#include <vector>
#include <sstream>
// extracted and adapted from https://github.com/aiekick/cTools (LICENSE MIT)
// for make VoxWriter lib free
#define SAFE_DELETE(a) if (a != 0) delete a, a = 0

namespace ct
{
	template <typename T> ::std::string toStr(const T& DOUBLE)
	{
		::std::ostringstream os;
		os << DOUBLE;
		return os.str();
	}
	template <typename T> inline T mini(const T& a, T& b) { return a < b ? a : b; }
	template <typename T> inline T maxi(const T& a, T& b) { return a > b ? a : b; }
	template <typename T> inline T clamp(const T& n) { return n >= T(0) && n <= T(1) ? n : T(n > T(0)); } // clamp n => 0 to 1
	template <typename T> inline T clamp(const T&  n, const T&  b) { return n >= T(0) && n <= b ? n : T(n > T(0))*b; } // clamp n => 0 to b
	template <typename T> inline T clamp(const T&  n, const T&  a, const T& b) { return n >= a && n <= b ? n : n < a ? a : b; } // clamp n => a to b

	// specialized
	struct dvec3 
	{ 
		double x, y, z;
		dvec3() { x = 0.0, y = 0.0, z = 0.0; }
		dvec3(const double& vxyz) { x = vxyz, y = vxyz, z = vxyz; }
		dvec3(const double& vx, const double& vy, const double& vz)	{ x = vx, y = vy, z = vz; }
		void operator += (const double v) { x += v; y += v; z += v; }
		void operator -= (const double v) { x -= v; y -= v; z -= v; }
		void operator += (const dvec3 v) { x += v.x; y += v.y; z += v.z; }
		void operator -= (const dvec3 v) { x -= v.x; y -= v.y; z -= v.z; }
		void operator *= (double v) { x *= v; y *= v; z *= v; }
		void operator /= (double v) { x /= v; y /= v; z /= v; }
		void operator *= (dvec3 v) { x *= v.x; y *= v.y; z *= v.z; }
		void operator /= (dvec3 v) { x /= v.x; y /= v.y; z /= v.z; }
	};
	inline dvec3 operator + (const dvec3& v, const double& f) { return dvec3(v.x + f, v.y + f, v.z + f); }
	inline dvec3 operator + (const dvec3& v, dvec3 f) { return dvec3(v.x + f.x, v.y + f.y, v.z + f.z); }
	inline dvec3 operator - (const dvec3& v, const double& f) { return dvec3(v.x - f, v.y - f, v.z - f); }
	inline dvec3 operator - (const dvec3& v, dvec3 f) { return dvec3(v.x - f.x, v.y - f.y, v.z - f.z); }
	inline dvec3 operator * (const dvec3& v, const double& f) { return dvec3(v.x * f, v.y * f, v.z * f); }
	inline dvec3 operator * (const dvec3& v, dvec3 f) { return dvec3(v.x * f.x, v.y * f.y, v.z * f.z); }
	inline dvec3 operator / (const dvec3& v, const double& f) { return dvec3(v.x / f, v.y / f, v.z / f); }
	inline dvec3 operator / (dvec3& v, const double& f) { return dvec3(v.x / f, v.y / f, v.z / f); }
	inline dvec3 operator / (const double& f, dvec3& v) { return dvec3(f / v.x, f / v.y, f / v.z); }
	inline dvec3 operator / (const dvec3& v, dvec3 f) { return dvec3(v.x / f.x, v.y / f.y, v.z / f.z); }

	// specialized
	struct dAABBCC // copy of b2AABB struct
	{
		dvec3 lowerBound;	///< the lower left vertex
		dvec3 upperBound;	///< the upper right vertex

		dAABBCC() : lowerBound(0.0), upperBound(0.0) {}
		dAABBCC(dvec3 vlowerBound, dvec3 vUpperBound)
		{
			lowerBound.x = mini(vlowerBound.x, vUpperBound.x);
			lowerBound.y = mini(vlowerBound.y, vUpperBound.y);
			lowerBound.z = mini(vlowerBound.z, vUpperBound.z);
			upperBound.x = maxi(vlowerBound.x, vUpperBound.x);
			upperBound.y = maxi(vlowerBound.y, vUpperBound.y);
			upperBound.z = maxi(vlowerBound.z, vUpperBound.z);
		}
		/// Add a vector to this vector.
		void operator += (const dvec3& v){lowerBound += v; upperBound += v;}

		/// Subtract a vector from this vector.
		void operator -= (const dvec3& v){lowerBound -= v; upperBound -= v;}

		/// Multiply this vector by a scalar.
		void operator *= (double a){lowerBound *= a; upperBound *= a;}

		/// Divide this vector by a scalar.
		void operator /= (double a){lowerBound /= a; upperBound /= a;}

		/// Get the center of the AABB.
		dvec3 GetCenter() const { return (lowerBound + upperBound) * 0.5; }

		/// Get the extents of the AABB (half-widths).
		dvec3 GetExtents() const {return (upperBound - lowerBound) * 0.5;}

		/// Get the perimeter length
		double GetPerimeter() const
		{
			double wx = upperBound.x - lowerBound.x;
			double wy = upperBound.y - lowerBound.y;
			double wz = upperBound.z - lowerBound.z;
			return 2.0 * (wx + wy + wz);
		}

		/// Combine a point into this one.
		void Combine(dvec3 pt)
		{
			lowerBound.x = mini<double>(lowerBound.x, pt.x);
			lowerBound.y = mini<double>(lowerBound.y, pt.y);
			lowerBound.z = mini<double>(lowerBound.z, pt.z);
			upperBound.x = maxi<double>(upperBound.x, pt.x);
			upperBound.y = maxi<double>(upperBound.y, pt.y);
			upperBound.z = maxi<double>(upperBound.z, pt.z);
		}
		
		/// Does this aabb contain the provided vec2.
		bool ContainsPoint(const dvec3& pt) const
		{
			bool result = true;
			result = result && lowerBound.x <= pt.x;
			result = result && lowerBound.y <= pt.y;
			result = result && lowerBound.z <= pt.z;
			result = result && pt.x <= upperBound.x;
			result = result && pt.y <= upperBound.y;
			result = result && pt.z <= upperBound.z;
			return result;
		}

		bool Intersects(const dAABBCC& other)
		{
			bool result = true;
			result = result || lowerBound.x <= other.lowerBound.x;
			result = result || lowerBound.y <= other.lowerBound.y;
			result = result || lowerBound.z <= other.lowerBound.z;
			result = result || other.upperBound.x <= upperBound.x;
			result = result || other.upperBound.y <= upperBound.y;
			result = result || other.upperBound.z <= upperBound.z;
			return result;
		}

		const dvec3 Size()
		{
			return dvec3(upperBound - lowerBound);
		}
	};

	/// Add a float to a dAABBCC.
	inline dAABBCC operator + (const dAABBCC& v, float f){return dAABBCC(v.lowerBound + f, v.upperBound + f);}

	/// Add a dAABBCC to a dAABBCC.
	inline dAABBCC operator + (const dAABBCC& v, dAABBCC f){return dAABBCC(v.lowerBound + f.lowerBound, v.upperBound + f.upperBound);}

	/// Substract a float from a dAABBCC.
	inline dAABBCC operator - (const dAABBCC& v, float f){return dAABBCC(v.lowerBound - f, v.upperBound - f);}

	/// Substract a dAABBCC to a dAABBCC.
	inline dAABBCC operator - (const dAABBCC& v, dAABBCC f){return dAABBCC(v.lowerBound - f.lowerBound, v.upperBound - f.upperBound);}

	/// Multiply a float with a dAABBCC.
	inline dAABBCC operator * (const dAABBCC& v, float f){return dAABBCC(v.lowerBound * f, v.upperBound * f);}

	/// Multiply a dAABBCC with a dAABBCC.
	inline dAABBCC operator * (const dAABBCC& v, dAABBCC f){return dAABBCC(v.lowerBound * f.lowerBound, v.upperBound * f.upperBound);}

	/// Divide a dAABBCC by a float.
	inline dAABBCC operator / (const dAABBCC& v, float f){return dAABBCC(v.lowerBound / f, v.upperBound / f);}

	/// Divide a dAABBCC by a float.
	inline dAABBCC operator / (dAABBCC& v, float f){return dAABBCC(v.lowerBound / f, v.upperBound / f);}

	/// Divide a dAABBCC by a dAABBCC.
	inline dAABBCC operator / (const dAABBCC& v, dAABBCC f){return dAABBCC(v.lowerBound / f.lowerBound, v.upperBound / f.upperBound);}
}

namespace vox
{
	inline uint32_t GetMVID(uint8_t a, uint8_t b, uint8_t c, uint8_t d)
	{
		return (a) | (b << 8) | (c << 16) | (d << 24);
	}
		
	struct DICTstring
	{
		int32_t bufferSize;
		std::string buffer;
		
		DICTstring();
		
		void write(FILE *fp);
		size_t getSize();
	};

	struct DICTitem
	{
		DICTstring key;
		DICTstring value;
		
		DICTitem();
		DICTitem(std::string vKey, std::string vValue);
	
		void write(FILE *fp);
		size_t getSize();
	};

	struct DICT
	{
		int32_t count;
		std::vector<DICTitem> keys;

		DICT();
		void write(FILE *fp);
		size_t getSize();
		void Add(std::string vKey, std::string vValue);
	};

	struct nTRN
	{
		int32_t nodeId;
		DICT nodeAttribs;
		int32_t childNodeId;
		int32_t reservedId;
		int32_t layerId;
		int32_t numFrames;
		std::vector<DICT> frames;

		nTRN(int32_t countFrames);

		void write(FILE *fp);
		size_t getSize();
	};

	struct nGRP
	{
		int32_t nodeId;
		DICT nodeAttribs;
		int32_t nodeChildrenNodes;
		std::vector<int32_t> childNodes;

		nGRP(int32_t vCount);
		
		void write(FILE *fp);
		size_t getSize();
	};

	struct MODEL
	{
		int32_t modelId;
		DICT modelAttribs;

		MODEL();
		
		void write(FILE *fp);
		size_t getSize();
	};
	
	struct nSHP
	{
		int32_t nodeId;
		DICT nodeAttribs;
		int32_t numModels;
		std::vector<MODEL> models;

		nSHP(int32_t vCount);
		
		void write(FILE *fp);
		size_t getSize();
	};

	struct LAYR
	{
		int32_t nodeId;
		DICT nodeAttribs;
		int32_t reservedId;

		LAYR();
		void write(FILE *fp);
		size_t getSize();
	};

	struct SIZE
	{
		int32_t sizex;
		int32_t sizey;
		int32_t sizez;
		
		SIZE();
		
		void write(FILE *fp);
		size_t getSize();
	};

	struct XYZI
	{
		int32_t numVoxels;
		std::vector<uint8_t> voxels;

		XYZI();
		void write(FILE *fp);
		size_t getSize();
	};

	struct RGBA
	{
		int32_t colors[256];

		RGBA();
		void write(FILE *fp);
		size_t getSize();
	};

	struct VoxCube
	{
		int id;

		// size
		SIZE size;

		// translate
		int tx;
		int ty;
		int tz;

		XYZI xyzi;

		VoxCube();

		void write(FILE *fp);
	};

	class VoxWriter
	{
	public:
		static VoxWriter* Create(const std::string& vFilePathName, const uint32_t& vLimitX, 
			const uint32_t& vLimitY, const uint32_t& vLimitZ, int32_t *vError);
		static std::string GetErrnoMsg(const int32_t& vError);

	private:
		int32_t MV_VERSION;
		int32_t ID_VOX, ID_PACK, ID_MAIN, ID_SIZE, ID_XYZI,
			ID_RGBA, ID_NTRN, ID_NGRP, ID_NSHP;
		int32_t m_MaxVoxelPerCubeX;
		int32_t m_MaxVoxelPerCubeY;
		int32_t m_MaxVoxelPerCubeZ;
		FILE * m_File;
		ct::dAABBCC maxVolume;
		std::vector<uint32_t> colors;
		std::vector<VoxCube> cubes;
		int32_t maxCubeId;
		int32_t minCubeX;
		int32_t minCubeY;
		int32_t minCubeZ;
		std::map<int32_t, std::map<int32_t, std::map<int32_t, int32_t>>> cubesId;
		std::map<int32_t, std::map<int32_t, std::map<int32_t, int32_t>>> voxelId;

		int32_t lastError;

	public:
		VoxWriter(int32_t vMaxVoxelPerCubeX = 126, int32_t vMaxVoxelPerCubeY = 126, int32_t vMaxVoxelPerCubeZ = 126);
		~VoxWriter();
        int32_t IsOk(const std::string& vFilePathName);
		void ClearVoxels();
		void ClearColors();
		void AddColor(const uint8_t& r, const uint8_t& g, const uint8_t& b, const uint8_t& a, const uint8_t& index);
		void AddVoxel(const int32_t& vX, const int32_t& vY, const int32_t& vZ, const uint8_t& vColorIndex);
		void SaveToFile(const std::string& vFilePathName);
		
	private:
		uint32_t GetID(const uint8_t& a, const uint8_t& b, const uint8_t& c, const uint8_t& d);
		bool OpenFileForWriting(const std::string& vFilePathName);
		void CloseFile();
		long GetFilePos();
		void SetFilePos(const long& vPos);
		int32_t GetCubeId(const int32_t& vX, const int32_t& vY, const int32_t& vZ);
		VoxCube* GetCube(const int32_t& vX, const int32_t& vY, const int32_t& vZ);
		void MergeVoxelInCube(const int32_t& vX, const int32_t& vY, const int32_t& vZ, const uint8_t& vColorIndex, VoxCube *vCube);
	};
}
#endif //__VOX_WRITER_H__
