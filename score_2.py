import json
import numpy as np
import os
import onnxruntime


def init():
    
    model_name = 'AutoML.model'
    global sess
    sess = onnxruntime.InferenceSession(
        os.path.join(os.getenv("AZUREML_MODEL_DIR"), model_name)
    )


def run(request):
    print(request)
    data = json.loads(request)
    
    # Run inference
    test = sess.run(
        None,
        {"query_word": qw, "query_char": qc, "context_word": cw, "context_char": cc},
    )
    
    print(ans)
    return ans
