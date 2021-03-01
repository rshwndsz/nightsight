# On-Device Nightsight

Download LibTorch from [pytorch.org](https://pytorch.org) and place the unzipped folder in `nightsight/app/include`.

Serialize the model into `nightsight/app/` using
```bash
python serialize.py
```

Build the binary 
From `nightsight/app` do
```bash
mkdir build
cd build

cmake -DCMAKE_PREFIX_PATH=<full-path-to-libtorch> ..
cmake --build . --config Release

make
```

To execute the model from `nightsight/app` use 
```bash
./build/nightsight <path-to-serialized-model-file>
```

