#include "caffe/util/prediction.hpp"

#include <vector>
#include <Eigen/Core>

#include <opencv2/core/core.hpp>

// #include <cv.h>
// #include <highgui.h>

namespace caffe {
	
template <typename Dtype>
Grid<Dtype> rotate_voxels_prediction(const Grid<Dtype> &vox,
								   const Eigen::Matrix<Dtype,4,4> &model1,
								   const Eigen::Matrix<Dtype,4,4> &view1,
								   const Eigen::Matrix<Dtype,4,4> &model2,
								   const Eigen::Matrix<Dtype,4,4> &view2,
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
				rot_vox[c](i,j) = rotated_proba_value<Dtype>(vox, model1, view1,model2, view2, proj, c, i, j);
			}
		}
	}
	return rot_vox;	
}

	template <typename Dtype>
Dtype rotated_proba_value(const Grid<Dtype> &vox,
						  const Eigen::Matrix<Dtype,4,4> &model1,
						  const Eigen::Matrix<Dtype,4,4> &view1,
						  const Eigen::Matrix<Dtype,4,4> &model2,
						  const Eigen::Matrix<Dtype,4,4> &view2,
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
					  depth, model1, view1,model2, view2, proj);
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
	Dtype zf=z*0.8+(size/16);
	Dtype depth = -(zf+0.5)/size*5.5-2.5;
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
	return (int)std::round((z-(size/16))/0.8);
}

	template <typename Dtype>
Eigen::Matrix<Dtype,3,1> rotate_coords(Dtype x, Dtype y, Dtype z,
						  const Eigen::Matrix<Dtype,4,4> &model1,
						  const Eigen::Matrix<Dtype,4,4> &view1,
						  const Eigen::Matrix<Dtype,4,4> &model2,
						  const Eigen::Matrix<Dtype,4,4> &view2,
							  const Eigen::Matrix<Dtype,4,4> &proj)
{
	Eigen::Matrix<Dtype,4,1> NDC(x*2-1, y*2-1, z*2-1, 1.0);
	Eigen::Matrix<Dtype,4,1> position = (proj*view2*model2).inverse()*NDC;
	//rotate
	Eigen::Matrix<Dtype,4,1> coords = proj * view1 * model1 * position;
	coords /= coords.w();
	//goes back into (0,1)
	coords = coords.array()/2+0.5;
	//make sure
	coords.x() = std::min<Dtype>(0.99,std::max<Dtype>(0.01,coords.x()));
	coords.y() = std::min<Dtype>(0.99,std::max<Dtype>(0.01,coords.y()));
	coords.z() = std::min<Dtype>(0.99,std::max<Dtype>(0.01,coords.z()));
	return coords.head(3);
}



float interpolate( float val, float y0, float x0, float y1, float x1 ) {
    return (val-x0)*(y1-y0)/(x1-x0) + y0;
}

float base( float val ) {
    if ( val <= -0.75 ) return 0;
    else if ( val <= -0.25 ) return interpolate( val, 0.0, -0.75, 1.0, -0.25 );
    else if ( val <= 0.25 ) return 1.0;
    else if ( val <= 0.75 ) return interpolate( val, 1.0, 0.25, 0.0, 0.75 );
    else return 0.0;
}

cv::Vec3b jetColor(float gray)
{
	return cv::Vec3b(base( gray + 0.5 )*255,base( gray )*255, base( gray - 0.5 )*255);
}

	template <typename Dtype>
cv::Mat iso_surface(const  Grid<Dtype> &vox,
					float threshold)
{
	int size = vox[0].rows();
	cv::Mat img(size, size,CV_8UC3);
	for(int i = 0; i <size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			int c_depth = -1;
			for(int c = 0; c < size; c++)
			{
float val_depth= vox[c](i,j);
				if (val_depth > threshold)
				{
					c_depth = c;
					break;
				}
			}
			float depth = (c_depth+0.5)/(size);
			cv::Vec3b color = jetColor(depth*2-1);
			img.at<cv::Vec3b>(i, j)=color;
		}
	}

	return img;
}
	

//takes only blobs and do the work (go to eigen and rotate) AVOID COPYING!
	template <typename Dtype>
	void rotate_blobs(const Blob<Dtype> * pred,
					  const Dtype* model1,
					  const Dtype* view_mat1,
					  const Dtype* model2,
					  const Dtype* view_mat2,
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
		Eigen::Map<const Eigen::Matrix<Dtype,4,4> > view1(view_mat1);
		Eigen::Map<const Eigen::Matrix<Dtype,4,4> > view2(view_mat2);
		Eigen::Map<const Eigen::Matrix<Dtype,4,4> > proj(proj_mat);
		Eigen::Map<const Eigen::Matrix<Dtype,4,4> > mv1(model1);
		Eigen::Map<const Eigen::Matrix<Dtype,4,4> > mv2(model2);
		// std::cout<<view1<<std::endl<<std::endl;
		// std::cout<<view2<<std::endl<<std::endl;
		// std::cout<<mv1<<std::endl<<std::endl;
		// std::cout<<mv2<<std::endl<<std::endl;
		//compute model
		// Eigen::Matrix<Dtype,4,4> model;
		// model = new_view*model_old.inverse();
		// model=model.inverse();

		//rotate
		Grid<Dtype> gt12 = rotate_voxels_prediction<Dtype>(grid, mv1, view1, mv2, view2, proj);
		// std::cout<<"rotate ok"<<std::endl;
		// cv::Mat pred1 = iso_surface(grid, 0.3);
		// cv::namedWindow( "pred1", CV_WINDOW_NORMAL );
		// cv::imshow("pred1",pred1);
		// cv::Mat pred2 = iso_surface(gt12, 0.3);
		// cv::namedWindow( "pred2", CV_WINDOW_NORMAL );
		// cv::imshow("pred2",pred2);
		// cv::waitKey(0);
		//put into output

		for(int c = 0; c < size; c++)
		{
			std::memcpy(output, gt12[c].data() ,size * size * sizeof(Dtype));
			output += size*size;
		}
	}

	template <typename Dtype>
	Grid<Dtype> unpack_pred_in_image( cv::Mat &image, int grid_rows, int grid_cols)
{
	//assert(grid_rows*grid_cols == grid.size()/4);
	int nrows = image.rows/grid_rows;
	int ncols = image.cols/grid_cols;
	int nchannels = nrows/4;//std::min<int>(grid.size(), grid_rows*grid_cols);
	//cv::Mat img(nrows*grid_rows, ncols*grid_cols,CV_8UC4);
	//std::cout<<"start to unpack\n";
	std::vector<cv::Mat> imgs;
	int from_to[4*2] = {0,2,1,1,2,0,3,3};
	std::vector<cv::Mat> img;
	img.push_back(image);
	// std::cout<<img[0].type()<<std::endl;
	//std::cout<<img[0].channels()<<std::endl;
	for(int i = 0; i<4; i++)
	{
		imgs.push_back(cv::Mat(image.size(),CV_8UC1));
	}
	cv::mixChannels(img,imgs,from_to,4);
	// for(int i = 0; i<4; i++)
	// 	imShow::show(imgs[i],"img"+std::to_string(i));
	// imShow::wait();
	// cv::namedWindow("grid");
	
	Grid<Dtype> grid(nrows);
	for(int c = 0; c<nchannels; c++)
	{
		int i_cell = c/grid_cols*nrows;
		int j_cell = c%grid_cols*ncols;
		//std::cout<<"c = "<<c<<", icell = "<<i_cell<<", jcell = "<<j_cell<<std::endl;
		for(int i = 0; i<4; i++)
		{
			int channel = c*4 + i;
			//std::cout<<"copying channel "<<channel<<std::endl;
			cv::Mat part = imgs[i](cv::Rect(j_cell,i_cell,nrows,ncols)),partF;
			part.convertTo(partF, CV_type<Dtype>(), 1.0/255);

			Eigen::Map<Slice<Dtype> > chan(reinterpret_cast<Dtype*>(partF.data),nrows,ncols);
			grid[channel] =chan;
		
		}
	}
	// cv::imshow("grid",pred_to_grid(grid,8,8));
	// cv::waitKey(0);
	return grid;
}

template <typename Dtype>
int CV_type()
{
	return 0;
}
	template <>
	int CV_type<float>()
	{
		return CV_32F;
	}
	template <>
	int CV_type<double>()
	{
		return CV_64F;
	}

	template 	Grid<float> unpack_pred_in_image( cv::Mat &image, int grid_rows, int grid_cols);
	template 	Grid<double> unpack_pred_in_image( cv::Mat &image, int grid_rows, int grid_cols);

	
	template void rotate_blobs(const Blob<double> * pred, const double* model1, const double* model2, const double* view_mat1, const double* view_mat2,  const double* proj_mat, double * output);
	template void rotate_blobs(const Blob<float> * pred, const float* model1, const float* model2, const float* view_mat1, const float* view_mat2,  const float* proj_mat, float * output);
}
