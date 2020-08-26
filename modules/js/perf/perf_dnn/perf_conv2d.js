const isNodeJs = (typeof window) === 'undefined'? true : false;

if　(isNodeJs)　{
  var Benchmark = require('benchmark');
  var cv = require('../../opencv');
  var HelpFunc = require('../perf_helpfunc');
  var Base = require('../base');
} else {
  var paramsElement = document.getElementById('params');
  var runButton = document.getElementById('runButton');
  var logElement = document.getElementById('log');
}

createFileFromUrl = function(path, url, callback) {
  let request = new XMLHttpRequest();
  request.open('GET', url, true);
  request.responseType = 'arraybuffer';
  request.onload = function(ev) {
      if (request.readyState === 4) {
          if (request.status === 200) {
              let data = new Uint8Array(request.response);
              cv.FS_createDataFile('/', path, data, true, false, false);
              callback();
          } else {
              console.error('Failed to load ' + url + ' status: ' + request.status);
          }
      }
  };
  request.send();
};

// get name of model and config file from url
function getNameFromUrl(url) {
  const modelParts = url.modelUrl.split('/');
  const modelPath = modelParts[modelParts.length-1];
  const configParts = url.configUrl.split('/');
  const configPath = configParts[configParts.length-1];
  return {
      modelPath: modelPath,
      configPath: configPath
  }
}

let modelLoaded = [];
loadModel = async function(url) {
  path = getNameFromUrl(url);
  return new Promise((resolve) => {
      // check if the model has been loaded before
      if(modelLoaded.indexOf(path.modelPath) == -1){
          createFileFromUrl(path.modelPath, url.modelUrl, () => {
              modelLoaded.push(path.modelPath);
              // check if need to load config file
              if(url.configUrl !== "") {
                  createFileFromUrl(path.configPath, url.configUrl, () => {
                      resolve(path);
                  });
              } else {
                  resolve(path);
              }
          });
      } else {
          resolve(path);
      }
  });
}

function asyncForwardWrapper(net) {
  let outputs = new cv.MatVector();
  net.forward1(outputs);
  return new Promise(function(resolve) {
      Module.Asyncify.asyncFinalizers.push(function() {
        resolve(outputs.get(0));
        outputs.delete();
      });
  });
}

async function testCaffeLayer()
{
    const prototxt = "layer_softmax.prototxt";
    const caffemodel = "layer_convolution.caffemodel";
    const url = {
      modelUrl: caffemodel,
      configUrl: prototxt
    }; 
    const path = await loadModel(url);
    let net, net1;
    const input = new cv.Mat([2, 6, 75, 113], cv.CV_32F);
    net = cv.readNetFromCaffe(path.configPath, '');
    net.setInput(input);
    net.setPreferableBackend(cv.DNN_BACKEND_WGPU);
    net.setPreferableTarget(cv.DNN_TARGET_WGPU);
    const start = performance.now();
    const out = await asyncForwardWrapper(net);
    const time = (performance.now() - start);
    console.log("Time cost(ms) :", time);
    console.log(out.data);
    net1 = cv.readNetFromCaffe(path.configPath, '');
    net1.setInput(input);
    net1.setPreferableBackend(cv.DNN_BACKEND_DEFAULT);
    net1.setPreferableTarget(cv.DNN_TARGET_CPU);
    let start1 = performance.now();
    for(let i = 0; i < 100; ++i)
    {
      net1.forward();
    }
    const out1 = net1.forward();
    let time1 = (performance.now() - start1) / 101;
    console.log("Time1 cost(ms) :", time1);
    console.log(out1.data);
    console.log("Net compute succeed");
    input.delete();
    net.delete();
    net1.delete();
    if(out) out.delete();
    out1.delete();
}

async function perf() {
    console.log('opencv.js loaded');
    if (isNodeJs) {
      global.cv = cv;
      global.combine = HelpFunc.combine;
      global.cvtStr2cvSize = HelpFunc.cvtStr2cvSize;
      global.cvSize = Base.getCvSize();
    } else {
      enableButton();
    }
    if (isNodeJs) {
        await testCaffeLayer();
      } else {
        runButton.onclick = async function()　{
            await testCaffeLayer();
        }
      }
}

async function main() {
    if (cv instanceof Promise) {
      cv = await cv;
      await perf();
    } else {
      cv.onRuntimeInitialized = perf;
    }
  }
  
  main();