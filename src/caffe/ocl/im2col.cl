template <class T>
__kernel void im2col(const int n, __global T* data_im, const int img_offset, const int height, const int width, const int ksize, const int pad, const int stride, const int height_col, const int width_col, __global T* data_col, const int col_offset){
    int index=get_global_id(0);
    data_im = data_im + img_offset;
    data_col =  data_col + col_offset;
    if(index < n){
        int w_out=index %width_col;
        index /= width_col;
        int h_out=index%height_col;
        int channel_in = index/height_col;
        int channel_out=channel_in *ksize *ksize;
        int h_in = h_out *stride-pad;
        int w_in = w_out *stride-pad;
        data_col +=(channel_out *height_col + h_out) *width_col + w_out;
        data_im +=(channel_in * height + h_in) *width + w_in;
        int i=0,j=0;
        for(i=0;i<ksize;++i){
            for(j=0;j<ksize;++j){
                int h = h_in+i;
                int w = w_in+j;
                if(h >= 0 && w >= 0 && h < height && w < width)
                    *data_col=data_im[i * width + j];
                else *data_col=0;
                data_col +=height_col *width_col;
            }
        }
    }
}

template __attribute__((mangled_name(im2colfloat))) __kernel void im2col(const int n, __global float* data_im, const int img_offset, const int height, const int width, const int ksize, const int pad, const int stride, const int height_col, const int width_col, __global float* data_col, const int col_offset); 
template __attribute__((mangled_name(im2coldouble))) __kernel void im2col(const int n, __global double* data_im, const int img_offset, const int height, const int width, const int ksize, const int pad, const int stride, const int height_col, const int width_col, __global double* data_col, const int col_offset); 

template <class T>
__kernel void im2col_opt(const int n, __global T* data_im, const int channels, const int img_offset, const int height, const int width, const int ksize, const int pad, const int stride, const int height_col, const int width_col, __global T* data_col, const int col_offset, const int optnum){

    int index = get_global_id(0);

    data_im = data_im + img_offset;
    data_col = data_col + col_offset;

    int x_out = index % width_col;
    int y_out = (index / width_col) % height_col;
    int channel_in = (index / width_col / height_col) % channels;
    int channel_out = channel_in * ksize * ksize;
    int im_id = index / width_col / height_col / channels;

    int y_in = y_out * stride - pad;
    int x_in = x_out * stride - pad;
    int offset_col = channel_out * optnum * height_col * width_col + im_id * height_col * width_col;
    int offset_im = im_id * channels * height * width + channel_in * height * width;

    for(int k_h = 0; k_h < ksize; k_h++){
        for(int k_w = 0; k_w < ksize; k_w++){
            int x_im = x_in + k_w;
            int y_im = y_in + k_h;
            int index_im = y_im * width + x_im;
            int index_col = (k_h * ksize + k_w) * optnum * height_col * width_col + y_out * width_col + x_out;
            if(y_im >= 0 && y_im < height && x_im >= 0 && x_im < width)
                data_col[offset_col + index_col] = data_im[offset_im + index_im];
            else
                data_col[offset_col + index_col] = 0;
        }
    }
}

template __attribute__((mangled_name(im2col_opt_float))) __kernel void im2col_opt(const int n, __global float* data_im, const int channels, const int lmg_offset, const int height, const int width, const int ksize, const int pad, const int stride, const int height_col, const int width_col, __global float* data_col, const int col_offset, const int optnum); 
template __attribute__((mangled_name(im2col_opt_double))) __kernel void im2col_opt(const int n, __global double* data_im, const int channels, const int img_offset, const int height, const int width, const int ksize, const int pad, const int stride, const int height_col, const int width_col, __global double* data_col, const int col_offset, const int optnum); 


template <class T>
__kernel void im2col_gpu_kernel(const int n, __global const T* data_im, const int img_offset,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    __global T* data_col, const int col_offset) {
    data_im = data_im + img_offset;
    data_col = data_col + col_offset;     

    int index = get_global_id(0);
    if(index < n) {
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * kernel_h * kernel_w;
        int h_in = h_out * stride_h - pad_h;
        int w_in = w_out * stride_w - pad_w;
        __global T* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        __global const T* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                            data_im_ptr[i * width + j] : 0;
                data_col_ptr += height_col * width_col;
        }
    }
  }
}

template __attribute__((mangled_name(im2col_gpu_kernel_float))) void im2col_gpu_kernel<float>(const int n, __global const float* data_im,
           const int img_offset, const int height, const int width, const int kernel_h, const int kernel_w,
           const int pad_h, const int pad_w, const int stride_h, const int stride_w,
           const int height_col, const int width_col, __global float* data_col, const int col_offset);
template __attribute__((mangled_name(im2col_gpu_kernel_double)))  void im2col_gpu_kernel<double>(const int n, __global const double* data_im,
           const int img_offset, const int height, const int width, const int kernel_h, const int kernel_w,
           const int pad_h, const int pad_w, const int stride_h, const int stride_w,
           const int height_col, const int width_col, __global double* data_col, const int col_offset);

template <class T>
__kernel void col2im_gpu_kernel(const int n, __global const T* data_col, const int col_offset,
    const int height, const int width, const int channels,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    __global T* data_im, const int img_offset) {
    data_col = data_col + col_offset;
    data_im = data_im + img_offset;
   int index = get_global_id(0);
    if(index < n) {
        T val = 0;
        int w = index % width + pad_w;
        int h = (index / width) % height + pad_h;
        int c = index / (width * height);
        // compute the start and end of the output
        int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
        int w_col_end = min(w / stride_w + 1, width_col);
        int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
        int h_col_end = min(h / stride_h + 1, height_col);
        // equivalent implementation
        int offset =
            (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
        int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
        int coeff_w_col = (1 - stride_w * height_col * width_col);
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
            }
        }
        data_im[index] = val;
  }
}

template __attribute__((mangled_name(col2im_gpu_kernel_float))) __kernel void col2im_gpu_kernel(const int n, __global const float* data_col, const int col_offset,
    									const int height, const int width, const int channels,
    									const int patch_h, const int patch_w,const int pad_h, const int pad_w,
    									const int stride_h, const int stride_w,const int height_col, const int width_col,
    									__global float* data_im, const int img_offset);
template __attribute__((mangled_name(col2im_gpu_kernel_double))) __kernel void col2im_gpu_kernel(const int n, __global const double* data_col,
                                         const int col_offset, const int height, const int width, const int channels,
                                         const int patch_h, const int patch_w, const int pad_h, const int pad_w,
                                         const int stride_h, const int stride_w, const int height_col, const int width_col, __global double* data_im, const int img_offset);

template <class T>
__kernel void col2im(const int n, __global T* data_col, const int col_offset, const int height, const int width, const int channels, const int ksize, const int pad, const int stride, const int height_col, const int width_col, __global T* data_im, const int img_offset){
    int index = get_global_id(0);
    data_col = data_col + col_offset;
    data_im = data_im + img_offset;
    if(index < n){
      T val = 0;
      int w = index % width + pad;
      int h = (index / width) % height + pad;
      int c = index / (width * height);
      // compute the start and end of the output
      int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
      int w_col_end = min(w / stride + 1, width_col);
      int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
      int h_col_end = min(h / stride + 1, height_col);
      // equivalent implementation
      int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
      int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
      int coeff_w_col = (1 - stride * height_col * width_col);
      for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
          val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
        }
      }
      data_im[index] = val;
  }
}
template __attribute__((mangled_name(col2imfloat))) __kernel void col2im(const int n, __global float* data_col, const int col_offset, const int height, const int width, const int channels, const int ksize, const int pad, const int stride, const int height_col, const int width_col, __global float* data_im, const int img_offset); 
template __attribute__((mangled_name(col2imdouble))) __kernel void col2im(const int n, __global double* data_col, const int col_offset, const int height, const int width, const int channels, const int ksize, const int pad, const int stride, const int height_col, const int width_col, __global double* data_im, const int img_offset); 

template <class T>
__kernel void col2im_opt(const int n, __global T* data_col, const int col_offset, const int height, const int width, const int channels, const int ksize, const int pad, const int stride, const int height_col, const int width_col, __global T* data_im, const int img_offset, const int optnum){
    int index = get_global_id(0);
    data_col = data_col + col_offset;
    data_im = data_im + img_offset;
    if(index < n){
      T val = 0;
      int w = index % width + pad;
      int h = (index / width) % height + pad;
      int c = index / (width * height) % channels;
      int im = index / width / height / channels;
      // compute the start and end of the output
      int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
      int w_col_end = min(w / stride + 1, width_col);
      int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
      int h_col_end = min(h / stride + 1, height_col);
      // equivalent implementation
      int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col * optnum + im * height_col * width_col;
      int coeff_h_col = (1 - stride * ksize * height_col * optnum) * width_col;
      int coeff_w_col = (1 - stride * height_col * width_col * optnum);
      for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
          val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
        }
      }
      data_im[index] = val;
  }
}
template __attribute__((mangled_name(col2im_opt_float))) __kernel void col2im_opt(const int n, __global float* data_col, const int col_offset, const int height, const int width, const int channels, const int ksize, const int pad, const int stride, const int height_col, const int width_col, __global float* data_im, const int img_offset, const int optnum); 
template __attribute__((mangled_name(col2im_opt_double))) __kernel void col2im_opt(const int n, __global double* data_col, const int col_offset, const int height, const int width, const int channels, const int ksize, const int pad, const int stride, const int height_col, const int width_col, __global double* data_im, const int img_offset, const int optnum); 

template <class T>
__kernel void opttrans(const int n, __global T* data_im, const int im_offset, const int height, const int width, const int channels, __global T* data_opt, const int opt_offset, const int optnum){

    int index = get_global_id(0);
    data_opt = data_opt + opt_offset;
    data_im = data_im + im_offset;
    if(index < n){
      int w = index % width;
      int h = (index / width) % height;
      int c = index / (width * height) % channels;
      int im = index / width / height / channels;

      int opt_index = c * height * optnum * width + h * optnum * width + im * width + w;
      data_opt[opt_index] = data_im[index];
    }
}
template __attribute__((mangled_name(opttrans_float))) __kernel void opttrans(const int n, __global float* data_im, const int im_offset, const int height, const int width, const int channels, __global float* data_opt, const int opt_offset, const int optnum); 
template __attribute__((mangled_name(opttrans_double))) __kernel void opttrans(const int n, __global double* data_im, const int im_offset, const int height, const int width, const int channels, __global double* data_opt, const int opt_offset, const int optnum); 

template <class T>
__kernel void transpose(__global const T *src, __global T* dst, int width, int height, int optnum){
     int gidx = get_global_id(0);
     int gidy = get_global_id(1);
     int gidyy = gidy;
     int index = gidy / height;
     int offset = index * width * height;
     gidy = gidy % height;
     if( gidx < width && gidyy < height * optnum )
         dst[offset + height * gidx + gidy] = src[offset + width * gidy + gidx];
}
template __attribute__((mangled_name(transpose_float))) __kernel void transpose(__global const float* src, __global float* dst, const int width, const int height, int optnum); 
template __attribute__((mangled_name(transpose_double))) __kernel void transpose(__global const double* src, __global double* dst, const int width, const int heighti, int optnum);

template <class T>
__kernel void transform(__global const T *src, __global T* dst, int top_offset, int width, int height, int optnum){
     int gidx = get_global_id(0);
     int index;
     index = (optnum==1) ? 0: gidx % optnum;
     dst = dst + top_offset; // now we point at (*top)[n]
     int offset = gidx / optnum;
     int i = 0;
     for(i = 0 ; i < width; i++)
         dst[(index * height + offset)* width + i] = src[gidx * width + i];
}
template __attribute__((mangled_name(transform_float))) __kernel void transform(__global const float* src, __global float* dst, int top_offset, const int width, const int height, const int optnum); 
template __attribute__((mangled_name(transform_double))) __kernel void transform(__global const double* src, __global double* dst, int top_offset, const int width, const int height, const int optnum); 