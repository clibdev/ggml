#include "gguf.h"
#include "ggml-backend.h"

struct Perceptron {
    ggml_backend_t backend;
    ggml_backend_buffer_t buffer;
    ggml_context *ctx;

    ggml_tensor *linear_weight;
    ggml_tensor *linear_bias;
};

static ggml_backend_t create_backend() {
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

static void load_model(const char *model_file, Perceptron &model) {
    ggml_context *tmp_ctx = nullptr;
    const gguf_init_params gguf_params = {false, &tmp_ctx};

    gguf_context *gguf_ctx = gguf_init_from_file(model_file, gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return;
    }

    const int64_t num_tensors = gguf_get_n_tensors(gguf_ctx);
    const ggml_init_params params{
        ggml_tensor_overhead() * num_tensors,
        nullptr,
        true,
    };
    model.ctx = ggml_init(params);
    for (int i = 0; i < num_tensors; i++) {
        const char *name = gguf_get_tensor_name(gguf_ctx, i);
        const ggml_tensor *src = ggml_get_tensor(tmp_ctx, name);
        ggml_tensor *dst = ggml_dup_tensor(model.ctx, src);
        ggml_set_name(dst, name);
    }

    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

    for (ggml_tensor *cur = ggml_get_first_tensor(model.ctx); cur != nullptr;
         cur = ggml_get_next_tensor(model.ctx, cur)) {
        const ggml_tensor *src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        ggml_backend_tensor_set(cur, ggml_get_data(src), 0, ggml_nbytes(src));
    }
    gguf_free(gguf_ctx);
    ggml_free(tmp_ctx);

    model.linear_weight = ggml_get_tensor(model.ctx, "linear.weight");
    model.linear_bias = ggml_get_tensor(model.ctx, "linear.bias");
}

static ggml_cgraph *build_graph(ggml_context *ctx_cgraph, const Perceptron &model) {
    ggml_cgraph *gf = ggml_new_graph(ctx_cgraph);

    ggml_tensor *input = ggml_new_tensor_2d(ctx_cgraph, GGML_TYPE_F32, model.linear_weight->ne[0], 1);
    ggml_set_name(input, "input");

    ggml_tensor *result = ggml_mul_mat(ctx_cgraph, model.linear_weight, input);
    result = ggml_add(ctx_cgraph, result, model.linear_bias);
    result = ggml_sigmoid(ctx_cgraph, result);
    ggml_set_name(result, "output");
    ggml_set_output(result);

    ggml_build_forward_expand(gf, result);

    return gf;
}

void inference(ggml_cgraph *gf, const Perceptron &model, const float *input_data, float *output_data) {
    ggml_tensor *input = ggml_graph_get_tensor(gf, "input");
    ggml_backend_tensor_set(input, input_data, 0, ggml_nbytes(input));

    if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        return;
    }

    const ggml_tensor *output = ggml_graph_get_tensor(gf, "output");
    ggml_backend_tensor_get(output, output_data, 0, ggml_nbytes(output));
}

int main(int _, char **argv) {
    Perceptron model{};
    model.backend = create_backend();

    load_model(argv[1], model);

    const ggml_init_params params = {
        ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        nullptr,
        true,
    };
    ggml_context *ctx_cgraph = ggml_init(params);
    ggml_cgraph *gf = build_graph(ctx_cgraph, model);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    constexpr float input_data[2] = {0.0f, 0.0f};
    float output_data[1];
    inference(gf, model, input_data, output_data);

    printf("%f\n", output_data[0]);

    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);
    ggml_free(model.ctx);
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);

    return 0;
}
