## Build WebNN-native to enable native implementation for DNN_BACKEND_WEBNN  in modules/dnn


### Build WebNN and set the environment variable

 Refer to [WebNN's build instructions](https://github.com/otcshare/webnn-native/blob/main/README.md) to complete the build of Dawn (build the Release version). Set environment variable `WEBGPU_ROOT_DIR` to enable native DNN_BACKEND_WEBGPU build: `export WEBGPU_ROOT_DIR=${PATH_TO_Dawn}`.

### Test native DNN_BACKEND_WEBGPU backend
Add -DWITH_WEBGPU=ON to the cmake command to build the webgpu module such as:
`cmake -D CMAKE_BUILD_TYPE=Release -DWITH_WEBGPU=ON -D CMAKE_INSTALL_PREFIX=/usr/local ..`

Temporarily following these instructions to do the test:
```
git clone https://github.com/opencv/opencv_extra.git
export OPENCV_TEST_DATA_PATH=${PATH_TO}/opencv_extra/testdata
sudo apt-get install libvulkan1 mesa-vulkan-drivers vulkan-utils
git clone https://github.com/NALLEIN/opencv.git
cd opencv/
git checkout -b DawnTest origin/gsoc_2020_webgpu