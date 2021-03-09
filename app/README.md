# On-Device Nightsight

Download LibTorch from [pytorch.org](https://pytorch.org) and place the unzipped folder in `nightsight/app/include/external/`.

Download the files `stb_image.h`, `stb_image_resize.h` and `stb_image_write.h` from [github.com/nothings/stb](https://github.com/nothings/stb) into `app/include/external/stb`.

Serialize the model by running 
```bash
python serialize.py
```
Place the serialized file in `app/`

Build the binary by running the following in `nightsight/app`
```bash
mkdir build
cd build

./../scripts/configure.sh
./../scripts/build.sh
cd ..
```

To execute the model run the following from `nightsight/app`
```bash
./build/nightsight <path-to-serialized-model-file> <path-to-input-image> <desired-path-to-output-image>
```
