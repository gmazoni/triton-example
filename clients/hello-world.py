from tritonclient.utils import *
import tritonclient.http as httpclient
import sys
import numpy as np

model_name = "hello-world"

with httpclient.InferenceServerClient("localhost:8000") as client:
    my_name = "Joselito"
    input_data = np.array([my_name]).astype(bytes)
    
    inputs = [
        httpclient.InferInput("INPUT0", input_data.shape, np_to_triton_dtype(input_data.dtype))
    ]

    inputs[0].set_data_from_numpy(input_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0")
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()
    print("Response:", result)

    data = response.as_numpy("OUTPUT0")
    print("Data:", data[0].decode('utf8'))

    print('PASS: hello-world')
    sys.exit(0)
