#include "caffe/util/prediction.hpp"

#include <vector>
#include <Eigen/Core>
// #include <cv.h>
// #include <highgui.h>

namespace caffe {
	
template <typename Dtype>
Grid<Dtype> rotate_voxels_prediction(const Grid<Dtype> &vox,
								   const Eigen::Matrix<Dtype,4,4> &model,
								   const Eigen::Matrix<Dtype,4,4> &view,
								   const Eigen::Matrix<Dtype,4,4> &proj)
{
int size = vox[0].rows();
//std::cout<<size<<std::endl;
	Grid<Dtype> rot_vox(size);
	for(int c = 0; c < size; c++)
	{
		rot_vox[c] = Slice<Dtype>(size, size);
		for(int i = 0; i <size; i++)
		{
			for(int j = 0; j < size; j++)
			{
				rot_vox[c](i,j) = rotated_proba_value<Dtype>(vox, model, view, proj, c, i, j);
			}
		}
	}
	return rot_vox;	
}

	template <typename Dtype>
Dtype rotated_proba_value(const Grid<Dtype> &vox,
						const Eigen::Matrix<Dtype,4,4> &model,
						const Eigen::Matrix<Dtype,4,4> &view,
						const Eigen::Matrix<Dtype,4,4> &proj,
						Dtype z,Dtype x, Dtype y)
{
	int size = vox[0].rows();
	//pass coord in the unit cube and rotate it
	// std::cout<<x<<" "<<y<<" "<<z<<" -> ";
	Dtype depth = z_to_depth(z, size, proj);;
	// std::cout<<depth<<std::endl;
	Eigen::Matrix<Dtype,3,1> rotated =
		rotate_coords<Dtype>((y+0.5)/(size), 1-(x+0.5)/(size),
					  depth, model, view, proj);
	// std::cout<<rotated<<std::endl;

	int new_i = (int)std::floor((1-rotated.y())*(size));
	int new_j = (int)std::floor(rotated.x()*(size));
	int new_z = depth_to_z(rotated.z(), size, proj);
	new_z = std::min<int>(size - 1,std::max<int>(0,new_z));
	// std::cout<<new_i<<" "<<new_j<<" "<<new_z<<std::endl;
	Dtype v0 = vox[new_z](new_i, new_j);
	return v0;
}

	template <typename Dtype>
	Dtype z_to_depth(int z, int size, const Eigen::Matrix<Dtype,4,4> &proj)
{
	z=z*0.8+8;
	Dtype depth = -(z+0.5)/size*5.5-2.5;
	depth = -proj(2,3)/depth-proj(2,2);
	depth=depth/2+0.5;
	return depth;
}

	template <typename Dtype>
int depth_to_z(Dtype depth, int size, const Eigen::Matrix<Dtype,4,4> &proj)
{
	depth = depth * 2 - 1;
	depth = -proj(2,3)/(depth+proj(2,2));
	Dtype z = (-(depth+2.5)/5.5)*size-0.5;
	return (int)std::floor((z-8)/0.8);
}

	template <typename Dtype>
Eigen::Matrix<Dtype,3,1> rotate_coords(Dtype x, Dtype y, Dtype z,
							  const Eigen::Matrix<Dtype,4,4> &model, 
							  const Eigen::Matrix<Dtype,4,4> &view,
							  const Eigen::Matrix<Dtype,4,4> &proj)
{
	Eigen::Matrix<Dtype,4,4> PV = proj*view;
	Eigen::Matrix<Dtype,4,1> NDC(x*2-1, y*2-1, z*2-1, 1.0);
	Eigen::Matrix<Dtype,4,1> position = PV.inverse()*NDC;
	//rotate
	Eigen::Matrix<Dtype,4,1> coords = proj * view * model * position;
	coords /= coords.w();
	//goes back into (0,1)
	coords = coords.array()/2+0.5;
	//make sure
	coords.x() = std::min<Dtype>(0.99,std::max<Dtype>(0.01,coords.x()));
	coords.y() = std::min<Dtype>(0.99,std::max<Dtype>(0.01,coords.y()));
	coords.z() = std::min<Dtype>(0.99,std::max<Dtype>(0.01,coords.z()));
	return coords.head(3);
}



// float interpolate( float val, float y0, float x0, float y1, float x1 ) {
//     return (val-x0)*(y1-y0)/(x1-x0) + y0;
// }

// float base( float val ) {
//     if ( val <= -0.75 ) return 0;
//     else if ( val <= -0.25 ) return interpolate( val, 0.0, -0.75, 1.0, -0.25 );
//     else if ( val <= 0.25 ) return 1.0;
//     else if ( val <= 0.75 ) return interpolate( val, 1.0, 0.25, 0.0, 0.75 );
//     else return 0.0;
// }

// cv::Vec3b jetColor(float gray)
// {
// 	return cv::Vec3b(base( gray + 0.5 )*255,base( gray )*255, base( gray - 0.5 )*255);
// }

// 	template <typename Dtype>
// cv::Mat iso_surface(const  Grid<Dtype> &vox,
// 					float threshold)
// {
// 	int size = vox[0].rows();
// 	cv::Mat img(size, size,CV_8UC3);
// 	for(int i = 0; i <size; i++)
// 	{
// 		for(int j = 0; j < size; j++)
// 		{
// 			int c_depth = -1;
// 			for(int c = 0; c < size; c++)
// 			{
// float val_depth= vox[c](i,j);
// 				if (val_depth > threshold)
// 				{
// 					c_depth = c;
// 					break;
// 				}
// 			}
// 			float depth = (c_depth+0.5)/(size);
// 			cv::Vec3b color = jetColor(depth*2-1);
// 			img.at<cv::Vec3b>(i, j)=color;
// 		}
// 	}

// 	return img;
// }
	

//takes only blobs and do the work (go to eigen and rotate) AVOID COPYING!
	template <typename Dtype>
	void rotate_blobs(const Blob<Dtype> * pred,
					  const Dtype* viewpoint1,
					  const Dtype* viewpoint2,
					  const Dtype* view_mat,
					  const Dtype* proj_mat,
					  Dtype * output)  //WARNING a voir
	{
		//go from pred to a Grid<Dtype>
		int output_channels=pred->channels();
		int output_width = pred->width();
		int output_height = pred->height();
		int size = output_width;

		Grid<Dtype> grid(size);
		const Dtype* output_data=pred->cpu_data();
		//chenger methode selon taille du blob  (unfold ou 3d)
		if (output_height == size*size) //if pred from a net, skip first classif layer
			output_data+=output_width * output_height;

		for(int c = 0; c < size; c++)
		{
			Slice<Dtype> channel(size, size);
			std::memcpy(channel.data(), output_data, size*size* sizeof(Dtype));
			output_data += size*size;
			grid[c] = channel;
		}
		//transform viewpoint, view_mat, proj_mat into eigen matrices
		Eigen::Map<const Eigen::Matrix<Dtype,4,4> > view(view_mat);
		Eigen::Map<const Eigen::Matrix<Dtype,4,4> > proj(proj_mat);
		Eigen::Map<const Eigen::Matrix<Dtype,4,4> > model_old(viewpoint1);
		Eigen::Map<const Eigen::Matrix<Dtype,4,4> > new_view(viewpoint2);
		//compute model
		Eigen::Matrix<Dtype,4,4> model;
		model = new_view*model_old.inverse();
		model=model.inverse();

		//rotate
		Grid<Dtype> gt12 = rotate_voxels_prediction<Dtype>(grid,  model, view, proj);
		//put into output

		for(int c = 0; c < size; c++)
		{
			std::memcpy(output, gt12[c].data() ,size * size * sizeof(Dtype));
			output += size*size;
		}
	}

	template void rotate_blobs(const Blob<double> * pred, const double* viewpoint1, const double* viewpoint2, const double* view_mat, const double* proj_mat, double * output);
	template void rotate_blobs(const Blob<float> * pred, const float* viewpoint1, const float* viewpoint2, const float* view_mat, const float* proj_mat, float * output);
}
