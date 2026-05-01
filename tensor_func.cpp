#include <ggml-cpu.h>
#include <ggml.h>
#include <iostream>

void tensor_func_1d()
{
    // get a input from user
    float input_value;
    std::cout << "Enter a value for x: ";
    std::cin >> input_value;

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
        return;
    }

    // 定一个函数 f(x) =3 * x^2 + 4 * x + 5
    // x是一个scaler，输出是一个scalar
    struct ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    float *x_data = (float *)x->data;
    x_data[0] = input_value;                             // 输入值
    struct ggml_tensor *x_squared = ggml_mul(ctx, x, x); // x^2
    struct ggml_tensor *three_x_squared =
        ggml_mul(ctx, ggml_new_f32(ctx, 3.0f), x_squared);                  // 3 * x^2
    struct ggml_tensor *four_x = ggml_mul(ctx, ggml_new_f32(ctx, 4.0f), x); // 4 * x
    struct ggml_tensor *result = ggml_add(ctx,
                                          ggml_add(ctx, three_x_squared, four_x),
                                          ggml_new_f32(ctx, 5.0f)); // 3 * x^2 + 4 * x + 5

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // 打印结果
    float *result_data = (float *)result->data;
    std::cout << "Result (f(x)) using ggml:" << std::endl;
    std::cout << result_data[0] << std::endl;
    // 手动计算结果
    float manual_result = 3.0f * input_value * input_value + 4.0f * input_value + 5.0f;
    std::cout << "Result (f(x)) using manual calculation:" << std::endl;
    std::cout << manual_result << std::endl;

    ggml_free(ctx);

    std::cout << "ggml sample run successfully!" << std::endl;
}

void tensor_func_2d()
{
    // get three input from user
    std::cout << "Enter 3 values for x (3x1 vector): ";
    float x_values[3];
    for (int i = 0; i < 3; i++)
    {
        std::cin >> x_values[i];
    }

    // 定一个函数 f(x) = a*x+b
    // 其中a是3*3的矩阵
    // x是3*1的向量
    // b是3*1的向量
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
        return;
    }
    struct ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 3);
    struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 1);
    struct ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 1);

    float *a_data = (float *)a->data;
    float *x_data = (float *)x->data;
    float *b_data = (float *)b->data;

    // 初始化a [5,9,1; 2,3,4; 7,8,6]
    a_data[0] = 5.0f;
    a_data[1] = 9.0f;
    a_data[2] = 1.0f;
    a_data[3] = 2.0f;
    a_data[4] = 3.0f;
    a_data[5] = 4.0f;
    a_data[6] = 7.0f;
    a_data[7] = 8.0f;
    a_data[8] = 6.0f;

    // 初始化b [1; 2; 3]
    b_data[0] = 1.0f;
    b_data[1] = 2.0f;
    b_data[2] = 3.0f;

    // 初始化x
    for (int i = 0; i < 3; i++)
    {
        x_data[i] = x_values[i];
    }
    struct ggml_tensor *ax = ggml_mul_mat(ctx, a, x);  // a*x
    struct ggml_tensor *result = ggml_add(ctx, ax, b); // a*x + b
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // 打印结果
    float *result_data = (float *)result->data;
    std::cout << "Result (f(x)) using ggml:" << std::endl;
    for (int i = 0; i < 3; i++)
    {
        std::cout << result_data[i] << std::endl;
    }
    std::cout << "Result (f(x)) using manual calculation:" << std::endl;
    for (int i = 0; i < 3; i++)
    {
        float manual_result = 0.0f;
        for (int j = 0; j < 3; j++)
        {
            manual_result += a_data[i * 3 + j] * x_data[j];
        }
        manual_result += b_data[i];
        std::cout << manual_result << std::endl;
    }
    ggml_free(ctx);
    std::cout << "ggml sample run successfully!" << std::endl;
}

// 鸡兔同笼
void tensor_func_jttl()
{
    // Ask user for two input one for head, one for feet
    int heads, feet;
    std::cout << "Enter the number of heads: ";
    std::cin >> heads;
    std::cout << "Enter the number of feet: ";
    std::cin >> feet;

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
        return;
    }

    // a = [1,1; 2,4], x = [heads; feet]
    // 手动计算逆矩阵: inv(a) = (1/det) * [4,-1; -2,1], det = 1*4 - 1*2 = 2
    // inv(a) = [2, -0.5; -1, 0.5]
    struct ggml_tensor *a_inv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    float *a_inv_data = (float *)a_inv->data;
    float *x_data = (float *)x->data;

    a_inv_data[0] = 2.0f;
    a_inv_data[1] = -0.5f;
    a_inv_data[2] = -1.0f;
    a_inv_data[3] = 0.5f;

    x_data[0] = (float)heads;
    x_data[1] = (float)feet;

    struct ggml_tensor *result = ggml_mul_mat(ctx, a_inv, x);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    float *result_data = (float *)result->data;
    std::cout << "Result using ggml (chicken, rabbit):" << std::endl;
    std::cout << "Chickens: " << result_data[0] << std::endl;
    std::cout << "Rabbits: " << result_data[1] << std::endl;

    ggml_free(ctx);
    std::cout << "ggml sample run successfully!" << std::endl;
}

int main()
{
    // Ask user to choose which function to run
    int choice;
    std::cout << "Choose a function to run:" << std::endl;
    std::cout << "1. f(x) = 3 * x^2 + 4 * x + 5" << std::endl;
    std::cout << "2. f(x) = a*x + b" << std::endl;
    std::cout << "3. Chicken-Rabbit problem (鸡兔同笼)" << std::endl;
    std::cin >> choice;

    switch (choice)
    {
    case 1:
        tensor_func_1d();
        break;
    case 2:
        tensor_func_2d();
        break;
    case 3:
        tensor_func_jttl();
        break;
    default:
        std::cerr << "Invalid choice" << std::endl;
    }

    return 0;
}
