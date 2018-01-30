### Install caffe

* 原作者的[Caffe](https://github.com/rbgirshick/caffe-fast-rcnn/tree/0dcd397b29507b8314e252e850518c5695efbb83)

* [Install step](https://github.com/alisure-ml/Installation/blob/master/Caffe.md)

* 需要说明的问题  
   * 若直接使用原作者的[Caffe](https://github.com/rbgirshick/caffe-fast-rcnn/tree/0dcd397b29507b8314e252e850518c5695efbb83),会在运行Faster-RCNN的Demo时出现有关于cuDNN的问题：
   ```
   In file included from ./include/caffe/util/cudnn.hpp:5:0,
   from ./include/caffe/util/device_alternate.hpp:40,
   from ./include/caffe/common.hpp:19,
   from src/caffe/data_reader.cpp:6:
   /usr/local/cuda/include/cudnn.h:799:27: note: declared here
   cudnnStatus_t CUDNNWINAPI cudnnSetPooling2dDescriptor(
   ```
  
   * 原因：cuDNN的版本太高
  
   * 解决办法：修改caffe源码或者降低cuDNN的版本
  
   * 修改caffe源码: 直接下载caffe到最新版本，然后采用[解决办法](https://www.cnblogs.com/zjutzz/p/6099720.html)替换文件后，再正常编译caffe。
   ```
   1. 用最新caffe源码的以下文件替换掉faster rcnn 的对应文件
      include/caffe/layers/cudnn_relu_layer.hpp, 
      src/caffe/layers/cudnn_relu_layer.cpp, 
      src/caffe/layers/cudnn_relu_layer.cu

      include/caffe/layers/cudnn_sigmoid_layer.hpp, 
      src/caffe/layers/cudnn_sigmoid_layer.cpp,
      src/caffe/layers/cudnn_sigmoid_layer.cu

      include/caffe/layers/cudnn_tanh_layer.hpp, 
      src/caffe/layers/cudnn_tanh_layer.cpp, 
      src/caffe/layers/cudnn_tanh_layer.cu

      include/caffe/util/cudnn.hpp

   2. 将 faster rcnn 中的 src/caffe/layers/cudnn_conv_layer.cu 文件中的所有
      cudnnConvolutionBackwardData_v3 函数名替换为 cudnnConvolutionBackwardData
      cudnnConvolutionBackwardFilter_v3函数名替换为 cudnnConvolutionBackwardFilter
   ```
 
