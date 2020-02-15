# resonance_classifier

Tested with
tensorflow                1.14.0          gpu_py37h63f5f00_0 
tensorflow-base           1.14.0          gpu_py37h611c6d2_0 
tensorflow-estimator      1.14.0                     py_0 
tensorflow-gpu            1.14.0               h0d30ee6_0
keras                     2.2.4                         0 
keras-applications        1.0.8                      py_0 
keras-base                2.2.4                    py37_0 
keras-preprocessing       1.1.0                      py_1 

Keras settings:
{
    "image_data_format": "channels_first", 
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "backend": "tensorflow"
}


To fix problems with Batchnorm dimensions see
https://github.com/keras-team/keras/issues/10648#issuecomment-502937188
