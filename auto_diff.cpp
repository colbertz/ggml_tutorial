#include <ggml-cpu.h>
#include <ggml.h>
#include <iostream>
#include <stdio.h>

// example 1: 计算 f(x) = 3*x^2 + 4 的值和梯度
void example1()
{
    float input_value;
    std::cout << "Enter a value for x: ";
    std::cin >> input_value;

    // 1. 初始化上下文
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024, // 16 MB
        .mem_buffer = NULL,
    };
    struct ggml_context *ctx = ggml_init(params);

    // 2. 定义变量和参数
    struct ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    // 标记需要计算梯度的参数
    ggml_set_param(x); // x 是输入变量

    // 3. 构建计算图：f(x) = a*x^2 + b
    struct ggml_tensor *x2 = ggml_mul(ctx, x, x);
    struct ggml_tensor *ax2 = ggml_mul(ctx, a, x2);
    struct ggml_tensor *f = ggml_add(ctx, ax2, b);

    // 4. 标记损失函数（关键变化！）
    ggml_set_loss(f);

    // 5. 构建前向+反向计算图
    struct ggml_cgraph *gf =
        ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true); // grads=true
    ggml_build_forward_expand(gf, f);
    ggml_build_backward_expand(ctx, gf, NULL); // NULL 表示自动创建梯度累加器

    // 6. 设置数值
    ggml_set_f32(x, input_value); // x = 2.0
    ggml_set_f32(a, 3.0f);
    ggml_set_f32(b, 4.0f);

    // 7. 执行计算（前向+反向统一在同一个图中）
    ggml_graph_reset(gf);                    // 重置梯度
    ggml_graph_compute_with_ctx(ctx, gf, 1); // 单线程

    // 8. 获取梯度
    struct ggml_tensor *grad_x = ggml_graph_get_grad(gf, x);

    // 10. 输出结果
    float f_value = ggml_get_f32_1d(f, 0);
    printf("f(%.2f) = %.2f\n", input_value, f_value);        // 3*4 + 4 = 16
    printf("∂ f/∂ x  = %.2f\n", ggml_get_f32_1d(grad_x, 0)); // 期望: 12.0

    // 清理
    ggml_free(ctx);
}

// exmaple 2 计算函数 f(a, b, x) = (a * ln(x)) / (b - a) 的值和梯度
void example2()
{
    float a = 3.0f;
    float b = 5.0f;
    float x = 7.0f;

    // 1. 初始化上下文
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024, // 16 MB
        .mem_buffer = NULL,
    };
    struct ggml_context *ctx = ggml_init(params);
    // 2. 定义变量和参数
    struct ggml_tensor *a_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor *b_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor *x_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    // 标记需要计算梯度的参数
    ggml_set_param(a_t); // a 是输入变量
    ggml_set_param(b_t); // b 是输入变量
    ggml_set_param(x_t); // x 是输入变量

    // 3. 构建计算图：f(a, b, x) = (a * ln(x)) / (b - a)
    struct ggml_tensor *ln_x = ggml_log(ctx, x_t);
    struct ggml_tensor *a_ln_x = ggml_mul(ctx, a_t, ln_x);
    struct ggml_tensor *b_minus_a = ggml_sub(ctx, b_t, a_t);
    struct ggml_tensor *f = ggml_div(ctx, a_ln_x, b_minus_a);

    // 4. 标记损失函数
    ggml_set_loss(f);

    // 5. 构建前向+反向计算图
    struct ggml_cgraph *gf =
        ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true); // grads=true
    ggml_build_forward_expand(gf, f);
    ggml_build_backward_expand(ctx, gf, NULL); // NULL 表示自动创建梯度累加器

    // 6. 设置数值
    ggml_set_f32(a_t, a);
    ggml_set_f32(b_t, b);
    ggml_set_f32(x_t, x);

    // 7. 执行计算（前向+反向统一在同一个图中）
    ggml_graph_reset(gf);                    // 重置梯度
    ggml_graph_compute_with_ctx(ctx, gf, 1); // 单线程

    // 8. 获取梯度
    struct ggml_tensor *grad_a = ggml_graph_get_grad(gf, a_t);
    struct ggml_tensor *grad_b = ggml_graph_get_grad(gf, b_t);
    struct ggml_tensor *grad_x = ggml_graph_get_grad(gf, x_t);

    // 10. 输出结果
    float f_value = ggml_get_f32_1d(f, 0);
    printf("f(%.2f, %.2f, %.2f) = %.5f\n", a, b, x, f_value);
    printf("∂ f/∂ a  = %.5f\n", ggml_get_f32_1d(grad_a, 0));
    printf("∂ f/∂ b  = %.5f\n", ggml_get_f32_1d(grad_b, 0));
    printf("∂ f/∂ x  = %.5f\n", ggml_get_f32_1d(grad_x, 0));

    ggml_free(ctx);
}

int main()
{

    // ask user which exmaple to be executed
    int choice;
    std::cout << "Choose an example to run (1 or 2): ";
    std::cin >> choice;
    switch (choice)
    {
    case 1:
        example1();
        break;
    case 2:
        example2();
        break;
    default:
        std::cerr << "Invalid choice" << std::endl;
    }

    return 0;
}