import MLLibrary.Contracts;

int main()
{
    using namespace mllibrary::contracts;
    const TensorDescriptor descriptor{
        .shape = { 2, 3 },
        .strides = {},
        .dataType = DataType::Float32,
        .layout = TensorLayout::RowMajor,
        .device = DeviceType::CPUScalar,
        .ownership = StorageOwnership::Owned
    };
    descriptor.validate();
    return descriptor.element_count() == 6 && descriptor.byte_size() == 24 ? 0 : 1;
}
