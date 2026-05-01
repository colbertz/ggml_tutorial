#include <ggml-cpu.h>
#include <ggml.h>
#include <iostream>

int main()
{
    std::cout << "ggml version: " << ggml_version() << std::endl;

    // 创建一个简单的上下文
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };

    struct ggml_context *ctx = ggml_init(params);
    if (!ctx)
    {
        std::cerr << "Failed to initialize ggml context" << std::endl;
        return 1;
    }

    // 创建两个输入张量
    struct ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    struct ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);

    // 初始化数据
    float *a_data = (float *)a->data;
    float *b_data = (float *)b->data;
    a_data[0] = 1.0f;
    a_data[1] = 2.0f;
    a_data[2] = 3.0f;
    a_data[3] = 4.0f;
    b_data[0] = 1.0f;
    b_data[1] = 0.0f;
    b_data[2] = 0.0f;
    b_data[3] = 1.0f;

    // 计算 a + b
    struct ggml_tensor *result = ggml_add(ctx, a, b);

    // 执行计算
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // 打印结果
    float *result_data = (float *)result->data;
    std::cout << "Result (a + b):" << std::endl;
    std::cout << result_data[0] << " " << result_data[1] << std::endl;
    std::cout << result_data[2] << " " << result_data[3] << std::endl;

    ggml_free(ctx);

    std::cout << "ggml sample run successfully!" << std::endl;
    return 0;
}
