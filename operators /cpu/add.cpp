void add_kernel_cpu(const tensor::Tensor& in1, const tensor::Tensor& in2,
    const tensor::Tensor& out, void* stream) {
UNUSED(stream);
CHECK_EQ(in1.is_empty(), false);
CHECK_EQ(in2.is_empty(), false);
CHECK_EQ(out.is_empty(), false);

CHECK_EQ(in1.size(), in2.size());
CHECK_EQ(in1.size(), out.size());

arma::fvec input_vec1(const_cast<float*>(in1.ptr<float>()), in1.size(), false, true);
arma::fvec input_vec2(const_cast<float*>(in2.ptr<float>()), in2.size(), false, true);
arma::fvec output_vec(const_cast<float*>(out.ptr<float>()), out.size(), false, true);
output_vec = input_vec1 + input_vec2;
}