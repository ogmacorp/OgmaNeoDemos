#include "helpers.h"

void File_Reader::read(void* data, long len) {
    ins.read(static_cast<char*>(data), len);
}

void File_Writer::write(const void* data, long len) {
    outs.write(static_cast<const char*>(data), len);
}

