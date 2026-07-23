export module MLLibrary.Training;

export import MLLibrary.Matrix;
export import MLLibrary.Model;

export struct ModelTrainingDesc {
    Matrix* train_images;
    Matrix* train_labels;
    Matrix* test_images;
    Matrix* test_labels;

    u32 epochs;
    u32 batch_size;
    f32 learning_rate;
};

export void model_train(ModelContext* model, const ModelTrainingDesc* desc);
