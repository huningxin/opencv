## Build Instructions
### Build WebNN-native and set the environment variable

 Refer to [WebNN's build instructions](https://github.com/webmachinelearning/webnn-native) to complete the build of WebNN-native (build the Release version). Set environment variable `WEBNN_NATIVE_DIR` to enable native DNN_BACKEND_WEBNN build: `export WEBNN_NATIVE_DIR=${PATH_TO_WebNN}`.

### Test native DNN_BACKEND_WEBNN backend
Add -DWITH_WEBNN=ON to the cmake command to build the WebNN module such as:
`cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules -DWITH_WEBNN=ON ../opencv` (according to the official docs https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)