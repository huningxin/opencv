## Build WebNN-native to enable native implementation for DNN_BACKEND_WEBNN  in modules/dnn


### Build WebNN-native and set the environment variable

 Refer to [WebNN's build instructions](https://github.com/otcshare/webnn-native/blob/main/README.md) to complete the build of WebNN-native (build the Release version). Set environment variable `WEBNN_NATIVE_DIR` to enable native DNN_BACKEND_WEBNN build: `export WEBNN_NATIVE_DIR=${PATH_TO_WebNN}`.

### Test native DNN_BACKEND_WEBNN backend
Add -DWITH_WEBNN=ON to the cmake command to build the WebNN module such as:
`cmake -D CMAKE_BUILD_TYPE=Release -DWITH_WEBNN=ON -D CMAKE_INSTALL_PREFIX=/usr/local ..`
or
`cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules -DWITH_WEBNN=ON ../opencv` (according to the official guide.)