#include <cstdio>

int outputHelpXml(const unsigned char* text, unsigned int size) {
  return std::fwrite(text, 1, size, stdout) == size ? 0 : 1;
}
