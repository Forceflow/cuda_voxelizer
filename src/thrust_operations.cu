#include "thrust_operations.cuh"

// thrust vectors (global)
thrust::host_vector<glm::vec3> trianglethrust_host;
thrust::device_vector<glm::vec3> trianglethrust_device;

// method 3: use a thrust vector
extern "C" void trianglesToGPU_thrust(const trimesh::TriMesh *mesh, float** triangles) {
	// fill host vector
	for (size_t i = 0; i < mesh->faces.size(); i++) {
		glm::vec3 v0 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][0]]);
		glm::vec3 v1 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][1]]);
		glm::vec3 v2 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][2]]);
		trianglethrust_host.push_back(v0);
		trianglethrust_host.push_back(v1);
		trianglethrust_host.push_back(v2);
	}
	trianglethrust_device = trianglethrust_host;
	*triangles = (float*)thrust::raw_pointer_cast(&(trianglethrust_device[0]));
}