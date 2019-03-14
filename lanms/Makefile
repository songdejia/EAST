CXXFLAGS = -I include  -std=c++11 -O3 -std=c++11 -O3 -I/data/jiangmiaotong/anaconda3/envs/tf/include/python3.6m -I/data/jiangmiaotong/anaconda3/envs/tf/include/python3.6m  -Wno-unused-result -Wsign-compare -march=nocona -mtune=generic -ftree-vectorize -fPIC -fstack-protector-strong -O3 -pipe  -fdebug-prefix-map==/usr/local/src/conda/- -fdebug-prefix-map==/usr/local/src/conda-prefix -fuse-linker-plugin -ffat-lto-objects -flto-partition=none -flto -DNDEBUG -fwrapv -O3 -Wall -L/data/jiangmiaotong/anaconda3/envs/tf/lib/python3.6/config-3.6m-x86_64-linux-gnu -L/data/jiangmiaotong/anaconda3/envs/tf/lib -lpython3.6m -lpthread -ldl  -lutil -lrt -lm  -Xlinker -export-dynamic adaptor.cpp include/clipper/clipper.cpp --shared -fPIC
LDFLAGS = $(shell python3-config --ldflags)

DEPS = lanms.h $(shell find include -xtype f)
CXX_SOURCES = adaptor.cpp include/clipper/clipper.cpp

LIB_SO = adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC

clean:
	rm -rf $(LIB_SO)
