#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/core/macros.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/core/subgraph.h"
#include <iostream>

using namespace tflite;

bool IsAllIntegerTypeTensor(const tflite::Model* model){
    TFLITE_LOG(TFLITE_LOG_INFO, "model has %d subgraphs", model->subgraphs()->size());
    for(const SubGraph* subgraph: *(model->subgraphs())) {
        TFLITE_LOG(TFLITE_LOG_INFO, "subgraph has %d tensors", subgraph->tensors()->size());
        for(const Tensor* tensor: *(subgraph->tensors())) {
            auto type = tensor->type();
            TFLITE_LOG(TFLITE_LOG_INFO, "tensor type: %s", EnumNameTensorType(type));
            if(!(type == TensorType_UINT8 || type ==  TensorType_UINT16 ||
               type == TensorType_UINT32 || type ==  TensorType_UINT64 ||
               type == TensorType_INT8  || type ==  TensorType_INT16 ||
               type == TensorType_INT32 || type ==  TensorType_INT64)) {
                return false;
            }
        }
    }
    return true;
}

void ShowBuiltinOperator(const tflite::Model* model) {
    TFLITE_LOG(TFLITE_LOG_INFO, "model has %d operator codes", model->operator_codes()->size());
    for(const OperatorCode* op_code: *(model->operator_codes())) {
        BuiltinOperator builtin_op = GetBuiltinCode(op_code);
        TFLITE_LOG(TFLITE_LOG_INFO, "BuiltinOperator: %s", EnumNameBuiltinOperator(builtin_op));
        TFLITE_EXPECT_TRUE(builtin_op != BuiltinOperator_CUSTOM);
    }
}


int main(int argc, const char* argv[]) {
    StderrReporter error_reporter;
    if(argc != 2) {
        std::cerr << "USAGE: tmp_main <tflite-file>\n";
    }
    auto model = FlatBufferModel::BuildFromFile(argv[1],
                                                &error_reporter);
    if(!model->initialized()) {
        std::cerr << "Failed to load TFLite file: " << argv[1];
    }
    std::cerr << "loaded\n";

    const Model* tflite_model = model->GetModel();

    ShowBuiltinOperator(tflite_model);
    assert(IsAllIntegerTypeTensor(tflite_model));


    std::vector<std::unique_ptr<Subgraph>> subgraphs;


    return 0;
}