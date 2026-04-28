#include <cstddef>
#include <cstring>
#include <format>
#include <fstream>
#include <ggml-cpu.h>
#include <ggml.h>
#include <iostream>
#include <sstream>
#include <vector>

// load data from data.txt
// the first column to std::vector<float> x_trains
// the third column to std::vector<float> y_trains
int load_training_data(std::vector<float> &x_trains,
                       std::vector<float> &y_trains) {
  std::ifstream file("data.txt");
  if (!file.is_open()) {
    std::cerr << "Error: cannot open data.txt" << std::endl;
    return -1;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    float col1, col2, col3;

    std::getline(ss, cell, ',');
    col1 = std::stof(cell);
    std::getline(ss, cell, ',');
    col2 = std::stof(cell);
    std::getline(ss, cell, ',');
    col3 = std::stof(cell);

    x_trains.push_back(col1 / 1000.0f);
    y_trains.push_back(col3 / 1000.0f);
  }

  file.close();
  return 0;
}

ggml_cgraph *buildgraph(ggml_context *ctx, ggml_tensor *x, ggml_tensor *y,
                        ggml_tensor *m2, ggml_tensor *w, ggml_tensor *b,
                        ggml_tensor **loss) {
  struct ggml_tensor *wx = ggml_mul(ctx, x, w);
  struct ggml_tensor *y_predict = ggml_add(ctx, wx, b);
  struct ggml_tensor *e = ggml_sub(ctx, y_predict, y);
  struct ggml_tensor *e_sqr = ggml_mul(ctx, e, e);
  struct ggml_tensor *e_sqr_sum = ggml_sum(ctx, e_sqr);
  *loss = ggml_div(ctx, e_sqr_sum, m2);
  ggml_set_loss(*loss);

  struct ggml_cgraph *gf =
      ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true); // grads=true
  ggml_build_forward_expand(gf, *loss);
  ggml_build_backward_expand(ctx, gf, NULL);

  return gf;
}

int main() {
  std::vector<float> x_trains, y_trains;
  if (load_training_data(x_trains, y_trains) != 0) {
    return 1;
  }

  // 1. 初始化上下文
  struct ggml_init_params ggml_params = {.mem_size =
                                             64 * 1024 * 1024, // 64 MB
                                         .mem_buffer = NULL,
                                         .no_alloc = false};
  struct ggml_context *ctx = ggml_init(ggml_params);

  size_t m_samples = x_trains.size();

  struct ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, m_samples);
  struct ggml_tensor *y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, m_samples);
  struct ggml_tensor *m2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  struct ggml_tensor *w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

  float *x_data = (float *)x->data;
  float *y_data = (float *)y->data;
  memcpy(x_data, x_trains.data(), m_samples * sizeof(float));
  memcpy(y_data, y_trains.data(), m_samples * sizeof(float));
  ggml_set_f32(m2, 2.0f * (float)m_samples);

  ggml_set_param(w);
  ggml_set_param(b);

  struct ggml_tensor *loss;
  ggml_cgraph *gf = buildgraph(ctx, x, y, m2, w, b, &loss);

  // 从极小值开始尝试
  float lr = 1e-2f; // 或者 1e-9f, 1e-8f
  float w_value = 0.0f;
  float b_value = 0.0f;

  size_t iter;
  std::cin >> iter;
  for (int i = 0; i < iter; i++) {
    ggml_set_f32(w, w_value);
    ggml_set_f32(b, b_value);

    ggml_graph_reset(gf);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    float loss_value = ggml_get_f32_1d(loss, 0);
    std::cout << std::format("Iteration {}, Loss: {}, w: {}, b: {}\n", i,
                             loss_value, w_value, b_value);

    struct ggml_tensor *dl_dw = ggml_graph_get_grad(gf, w);
    struct ggml_tensor *dl_db = ggml_graph_get_grad(gf, b);
    float dl_dw_value = ggml_get_f32_1d(dl_dw, 0);
    float dl_db_value = ggml_get_f32_1d(dl_db, 0);
    w_value = w_value - lr * dl_dw_value;
    b_value = b_value - lr * dl_db_value;
  }

  std::cout << std::format("Final w {}, b: {}\n", w_value, b_value);

  ggml_free(ctx);
  return 0;
}