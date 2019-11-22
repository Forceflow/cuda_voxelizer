#include "thrust_operations.cuh"

// thrust vectors (global) (see https://stackoverflow.com/questions/54742267/having-thrustdevice-vector-in-global-scope)
thrust::host_vector<glm::vec3>* trianglethrust_host;
thrust::device_vector<glm::vec3>* trianglethrust_device;

// method 3: use a thrust vector
float* meshToGPU_thrust(const trimesh::TriMesh *mesh) {
	Timer t; t.start(); // TIMER START
	// create vectors on heap 
	trianglethrust_host = new thrust::host_vector<glm::vec3>;
	trianglethrust_device = new thrust::device_vector<glm::vec3>;
	// fill host vector
	fprintf(stdout, "[Mesh] Copying %zu triangles to Thrust host vector \n", mesh->faces.size());
	for (size_t i = 0; i < mesh->faces.size(); i++) {
		glm::vec3 v0 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][0]]);
		glm::vec3 v1 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][1]]);
		glm::vec3 v2 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][2]]);
		trianglethrust_host->push_back(v0);
		trianglethrust_host->push_back(v1);
		trianglethrust_host->push_back(v2);
	}
	fprintf(stdout, "[Mesh] Copying Thrust host vector to Thrust device vector \n");
	*trianglethrust_device = *trianglethrust_host;
	t.stop(); fprintf(stdout, "[Mesh] Transfer time to GPU: %.1f ms \n", t.elapsed_time_milliseconds); // TIMER END
	return (float*) thrust::raw_pointer_cast(&((*trianglethrust_device)[0]));
}

void cleanup_thrust(){
	fprintf(stdout, "[Mesh] Freeing Thrust host and device vectors \n");
	if (trianglethrust_device) free(trianglethrust_device);
	if (trianglethrust_host) free(trianglethrust_host);
}