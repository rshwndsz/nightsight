# On-Device Nightsight

Download LibTorch from [pytorch.org](https://pytorch.org) and place the unzipped folder in `nightsight/app/include`.

Serialize the model by running the following in `nightsight/app/`
```bash
python serialize.py
```

Build the binary by running the following in `nightsight/app`
```bash
mkdir build
cd build

cmake -DCMAKE_PREFIX_PATH=<full-path-to-libtorch> ..
cmake --build . --config Release

make
```

To execute the model run the following from `nightsight/app`
```bash
./build/nightsight <path-to-serialized-model-file>
```

