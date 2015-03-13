# Caffe Debug Tools
The matlab program is used for analyzing the weights and gradients for each layer of neural network in caffe.

# Usage:
1. Add the following code in the cpp file
  DebugTool<Dtype> dbg;
  dbg.open("filename.bin");
  dbg.write_blob("bottom", *((*bottom)[0]), flag);
  dbg.write_blob("top", *top[0], flag);
where 
  flag = 0: print weights of the blob
  flag = 1: print graidents of the blob
2. open the saved file in matlab use load_bin.m
  dic = load_bin('filename.bin');
  blob_top = dic('top');
  blob_bottom = dic('bottom');
  