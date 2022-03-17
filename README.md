# NeRF
---
NeRF related project re-implementation of mine. **<u>This project is currently not finished.</u>**

To build pytorch extension for python. Run:

```shell
cd cuda/
python ./setup.py install --user
```

There is probably a compilation error, which might be caused by include directory specified in `setup.py`.  Note that:

- If you are using arch_70+, (for example, I am using RTX 3060 whose architect is sm_86), eigen 3.3.7 or higher is required. Otherwise you will get an error `No such file or directory <math_functions.h>`. I'm using the manually installed Eigen 3.4.0, which is located in `/usr/local/include/eigen3`
- If the GPU architecture is lower than arch_70, normal `libeigen3_dev` (version 3.3.4 on Ubuntu 18.04) will suffice. 
