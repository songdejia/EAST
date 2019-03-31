CXXFLAGS = -I include  -std=c++11 -O3 -std=c++11 -O3 -I/usr/include/python3.7m -I/usr/include/python3.7m  -Wno-unused-result -Wsign-compare -march=x86-64 -mtune=generic -O3 -pipe -fstack-protector-strong -fno-plt -flto=4 -fuse-linker-plugin -ffat-lto-objects -flto-partition=none -DNDEBUG -g -fwrapv -O3 -Wall

LDFLAGS = $(shell python3-config --ldflags)

DEPS = lanms.h $(shell find include -xtype f)
CXX_SOURCES = adaptor.cpp include/clipper/clipper.cpp

LIB_SO = adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC

clean:
	rm -rf $(LIB_SO)
