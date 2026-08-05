extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (n_rot > head_dim || !cuda_tensor_has_elems2(x, n_tok, head_dim, sizeof(float))) return 0;
    if (n_tok == 0u || head_dim == 0u) return 1;
    const uint32_t n_nope = head_dim - n_rot;
    if (n_nope == 0) return 1;
    const uint32_t groups = (n_nope + 63u) / 64u;
    fp8_kv_quantize_kernel<<<dim3(n_tok, groups), 64>>>((float *)x->ptr, n_tok, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize launch");
}

extern "C" int ds4_gpu_dsv4_turbo3_comp_pack_tensor(
        const ds4_gpu_tensor *src, ds4_gpu_tensor *dst,
        uint32_t n_rows, uint64_t dst_first_row,
        uint32_t head_dim, uint64_t dst_row_bytes) {
    (void)src; (void)dst; (void)n_rows; (void)dst_first_row;
    (void)head_dim; (void)dst_row_bytes;
    return 0;
}

extern "C" int ds4_gpu_dsv4_turbo3_comp_dequant_to_scratch_tensor(
        const ds4_gpu_tensor *src, ds4_gpu_tensor *dst,
        uint64_t src_first_row, uint32_t n_rows,
        uint32_t head_dim, uint64_t src_row_bytes) {
    (void)src; (void)dst; (void)src_first_row; (void)n_rows;
    (void)head_dim; (void)src_row_bytes;
    return 0;
}
