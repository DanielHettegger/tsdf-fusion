#include "tsdf_fusion.hcu"
#include "utils.hpp"

__constant__ float center_x = 250.0;
__constant__ float center_y = 250.0;
__constant__ float diastance_thresh = 170.0;

__device__ 
float calculate_weight(float depth, int px, int py){
  float pos[2] = {px - center_x, py - center_y};
  float center_dist = normf(2, pos);
  float center_dist_adapt = fmaxf(0.0, center_dist - diastance_thresh);
  float p = 1.316 - 0.00315 * center_dist_adapt;
  float k = 0.000305 + 0.000009285 * center_dist_adapt;
  return 1.0 / expf(center_dist_adapt * k) / p;
}

// CUDA kernel function to integrate a TSDF voxel volume given depth images
__global__
void Integrate(float * cam_K, float * cam2base, float * depth_im,
               int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
               float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
               float * voxel_grid_TSDF, float * voxel_grid_weight) {

  int pt_grid_z = blockIdx.x;
  int pt_grid_y = threadIdx.x;

  for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; ++pt_grid_x) {

    // Convert voxel center from grid coordinates to base frame camera coordinates
    float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
    float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
    float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

    // Convert from base frame camera coordinates to current frame camera coordinates
    float tmp_pt[3] = {0};
    tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
    tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
    tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
    float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
    float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
    float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];
    
    if (pt_cam_z <= 0)
      continue;

    int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
    int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
    if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
      continue;

    float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];

    if (depth_val <= 0.5 || depth_val > 6)
      continue;

    float diff = depth_val - pt_cam_z;

    if (diff <= -trunc_margin)
      continue;

    // Integrate
    int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
    float dist = fmin(1.0f, diff / trunc_margin);
    float weight_optim = calculate_weight(depth_val, pt_pix_x, pt_pix_y);
    float weight_old = voxel_grid_weight[volume_idx];
    float weight_new = weight_old + weight_optim;

    voxel_grid_weight[volume_idx] = weight_new;
    voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
  }
}


TSDFFusion::TSDFFusion(){

}

TSDFFusion::~TSDFFusion(){

}

void TSDFFusion::init(){

  // Read camera intrinsics
  //set to 0 for prototyping

  // Initialize voxel grid
  voxel_grid_TSDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  voxel_grid_weight = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
    voxel_grid_TSDF[i] = 1.0f;
  memset(voxel_grid_weight, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);

  // Load variables to GPU memory
  cudaMalloc(&gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
  cudaMalloc(&gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
  //checkCUDA(__LINE__, cudaGetLastError());
  cudaMemcpy(gpu_voxel_grid_TSDF, voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_voxel_grid_weight, voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
  //checkCUDA(__LINE__, cudaGetLastError());

  cudaMalloc(&gpu_cam_K, 3 * 3 * sizeof(float));
  cudaMemcpy(gpu_cam_K, cam_K, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&gpu_cam2base, 4 * 4 * sizeof(float));
  cudaMalloc(&gpu_depth_im, im_height * im_width * sizeof(float));
  //checkCUDA(__LINE__, cudaGetLastError());
}

void TSDFFusion::integrate_frame(cv::Mat depth_frame,  float pose[16]){
    if(!first_frame_transform){
        //Store First frame transformation
        std::copy(pose, pose+16, base2world);

        // Invert base frame camera pose to get world-to-base frame transform 
        memset(base2world_inv, 0, sizeof(float) * 16);

        invert_matrix(base2world, base2world_inv);

        first_frame_transform = true;
    }
    
    
    // convert frame to array
    //ReadDepth(depth_im_file, im_height, im_width, depth_im);
    float depth_im[im_height * im_width];
    cv_mat_to_array(depth_frame, depth_im);

    // Compute relative camera pose (camera-to-base frame)
    multiply_matrix(base2world_inv, pose, cam2base);

    cudaMemcpy(gpu_cam2base, cam2base, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_depth_im, depth_im, im_height * im_width * sizeof(float), cudaMemcpyHostToDevice);
    //checkCUDA(__LINE__, cudaGetLastError());

    Integrate <<< voxel_grid_dim_z, voxel_grid_dim_y >>>(gpu_cam_K, gpu_cam2base, gpu_depth_im,
                                                         im_height, im_width, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                                                         voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z, voxel_size, trunc_margin,
                                                         gpu_voxel_grid_TSDF, gpu_voxel_grid_weight);
    

    // Load TSDF voxel grid from GPU to CPU memory
    cudaMemcpy(voxel_grid_TSDF, gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(voxel_grid_weight, gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
    
    /*for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++)
      if(std::abs(voxel_grid_TSDF[i]) != 1)
        std::cout << voxel_grid_TSDF[i] << std::endl;*/

    std::cout << "Processing frame: " <<  counter++ << std::endl;
    if(counter % 200 == 0)
        export_model("tsdf_new.ply");
}


void TSDFFusion::export_model(std::string filename){
    // Load TSDF voxel grid from GPU to CPU memory
    cudaMemcpy(voxel_grid_TSDF, gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(voxel_grid_weight, gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
    //checkCUDA(__LINE__, cudaGetLastError());

    // Compute surface points from TSDF voxel grid and save to point cloud .ply file
    std::cout << "Saving surface point cloud (tsdf.ply)..." << std::endl;
    SaveVoxelGrid2SurfacePointCloud(filename, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z, 
                                    voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
                                    voxel_grid_TSDF, voxel_grid_weight, 0.2f, 0.0f);

}



void TSDFFusion::cv_mat_to_array(cv::Mat depth_mat, float * depth_array){
    for (int r = 0; r < im_height; ++r)
        for (int c = 0; c < im_width; ++c) {
            depth_array[r * im_width + c] = (depth_mat.at<float>(r, c)) / 1000.0f;
            //if((depth_mat.at<float>(r, c)) != 0)
            //    std::cout << (float)(depth_mat.at<float>(r, c))/1000.0f << std::endl;
            if (depth_array[r * im_width + c] > 6.0f) // Only consider depth < 6m
                depth_array[r * im_width + c] = 0;
    }
}

void TSDFFusion::set_intrinsics(float fx, float fy, float cx, float cy){
    cam_K[0] = fx;
    cam_K[1] = 0;
    cam_K[2] = cx;
    cam_K[3] = 0;
    cam_K[4] = fy;
    cam_K[5] = cy;
    cam_K[6] = 0;
    cam_K[7] = 0;
    cam_K[8] = 1; 
    cudaMemcpy(gpu_cam_K, cam_K, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

}
