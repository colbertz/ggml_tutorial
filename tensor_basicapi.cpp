#include <ggml.h>

#include <cstring>
#include <format>
#include <iostream>

// Helper to print tensor info
static void print_tensor(const char *name, struct ggml_tensor *t) {
  std::string shape;
  for (int i = 0; i < GGML_MAX_DIMS; i++) {
    if (i > 0)
      shape += " x ";
    shape += std::to_string((int)t->ne[i]);
  }
  std::cout << std::format("{}: shape({})\n", name, shape);
}

static void print_tensor_metadata(struct ggml_tensor *t) {
  std::cout << std::format("Tensor metadata:\n");
  std::cout << std::format("  n_dims  = {}\n", ggml_n_dims(t));
  for (int i = 0; i < ggml_n_dims(t); i++) {
    std::cout << std::format("  ne[{}]   = {}\n", i, t->ne[i]);
    std::cout << std::format("  nb[{}]   = {}\n", i, t->nb[i]);
  }
  std::cout << std::format("  type    = {} ({})\n", static_cast<int>(t->type), ggml_type_name(t->type));
  std::cout << std::format("  data    = {}\n", t->data);
  std::cout << std::format("  name    = '{}'\n", t->name);
  std::cout << std::format("ggml_nelements(t)  = {}\n", ggml_nelements(t));
  std::cout << std::format("ggml_nbytes(t)      = {}\n", ggml_nbytes(t));
  std::cout << std::format("ggml_is_contiguous(t) = {}\n",
                           ggml_is_contiguous(t) ? "true" : "false");
}

static void print_2d_tensor_pretty(struct ggml_tensor *t) {
  float *data = (float *)t->data;
  std::cout << std::format("Data:\n");
  for (size_t i = 0; i < t->ne[1]; i++) {
    std::string row = "   [";
    for (size_t j = 0; j < t->ne[0]; j++) {
      size_t idx = i * t->nb[1] / ggml_type_size(t->type) +
                   j * t->nb[0] / ggml_type_size(t->type);
      row += std::format(" {:.1f}", data[idx]);
    }
    row += " ]\n";
    std::cout << row;
  }
}

int main(void) {
  std::cout << std::format("=== ggml_tensor API Demo ===\n\n");

  // ============================================
  // 1. Create context with memory buffer
  // ============================================
  size_t ctx_size = 0;
  ctx_size += 4 * 3 * ggml_type_size(GGML_TYPE_F32);
  ctx_size += 4 * ggml_tensor_overhead();
  ctx_size += 8096;

  struct ggml_init_params params;
  params.mem_size = ctx_size;
  params.mem_buffer = nullptr;
  params.no_alloc = false;
  struct ggml_context *ctx = ggml_init(params);

  std::cout << std::format("1. Basic tensor creation and initialization\n");
  std::cout << std::format("------------------------\n");

  // ggml_new_tensor_2d: create 2D tensor
  struct ggml_tensor *t2d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
  float *t2d_data = (float *)t2d->data;
  for (int i = 0; i < 4 * 3; i++) {
    t2d_data[i] = (float)i;
  }
  print_tensor("t2d", t2d);
  print_2d_tensor_pretty(t2d);
  print_tensor_metadata(t2d);

  std::cout << std::format("\n2. Create a transposed tensor\n");
  std::cout << std::format("-----------------------------------\n");
  // create a transposed view of t2d
  struct ggml_tensor *t2d_transposed = ggml_transpose(ctx, t2d);
  std::cout << std::format("t2d_transposed shares data with t2d: {}\n",
                           t2d_transposed->data == t2d->data ? "YES" : "NO");
  print_tensor("t2d_transposed", t2d_transposed);
  print_2d_tensor_pretty(t2d_transposed);
  print_tensor_metadata(t2d_transposed);

  std::cout << std::format("\n3. Create a view of a tensor\n");
  std::cout << std::format("-----------------------------------\n");

  // the origin tensor is t2d (4 column x3 rows), let's create a view of middle
  // 2 columns and last 2 rows
  struct ggml_tensor *view_2d =
      ggml_view_2d(ctx, t2d, 2, 2, t2d->nb[1], 1 * t2d->nb[1] + 1 * t2d->nb[0]);

  std::cout << std::format("view_2d shares data with t2d: {}\n",
                           view_2d->data == t2d->data ? "YES" : "NO");
  print_tensor("view_2d", view_2d);
  print_2d_tensor_pretty(view_2d);
  print_tensor_metadata(view_2d);

  std::cout << std::format("\n4. Reshape tensor to another with different dimensions\n");
  std::cout << std::format("------------------------------------\n");

  // ggml_reshape_3d: reshape t2d to another 2D tensor with 2*6
  struct ggml_tensor *reshape_2d = ggml_reshape_2d(ctx, t2d, 2, 6);
  std::cout << std::format("reshape_2d shares data with t2d: {}\n",
                           reshape_2d->data == t2d->data ? "YES" : "NO");
  print_tensor("reshape_2d", reshape_2d);
  print_2d_tensor_pretty(reshape_2d);
  print_tensor_metadata(reshape_2d);

  // Cleanup
  ggml_free(ctx);

  std::cout << std::format("\n=== Demo completed ===\n");
  return 0;
}
