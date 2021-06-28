#include <webnn/webnn_cpp.h>

int main(int /*argc*/, char** /*argv*/)
{
    webnn::MLContext context = webnn::CreateContext();
    return 0;
}