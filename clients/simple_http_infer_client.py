
import numpy as np
import sys
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


def test_infer(model_name,
               input0_data,
               input1_data,
               headers=None,
               request_compression_algorithm=None,
               response_compression_algorithm=None):
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput('INPUT0', [1, 16], "INT32"))
    inputs.append(httpclient.InferInput('INPUT1', [1, 16], "INT32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data, binary_data=False)
    inputs[1].set_data_from_numpy(input1_data, binary_data=True)

    outputs.append(httpclient.InferRequestedOutput('OUTPUT0', binary_data=True))
    outputs.append(httpclient.InferRequestedOutput('OUTPUT1',
                                                   binary_data=False))
    query_params = {'test_1': 1, 'test_2': 2}
    results = triton_client.infer(
        model_name,
        inputs,
        outputs=outputs,
        query_params=query_params,
        headers=headers,
        request_compression_algorithm=request_compression_algorithm,
        response_compression_algorithm=response_compression_algorithm)

    return results


def test_infer_no_outputs(model_name,
                          input0_data,
                          input1_data,
                          headers=None,
                          request_compression_algorithm=None,
                          response_compression_algorithm=None):
    inputs = []
    inputs.append(httpclient.InferInput('INPUT0', [1, 16], "INT32"))
    inputs.append(httpclient.InferInput('INPUT1', [1, 16], "INT32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data, binary_data=False)
    inputs[1].set_data_from_numpy(input1_data, binary_data=True)

    query_params = {'test_1': 1, 'test_2': 2}
    results = triton_client.infer(
        model_name,
        inputs,
        outputs=None,
        query_params=query_params,
        headers=headers,
        request_compression_algorithm=request_compression_algorithm,
        response_compression_algorithm=response_compression_algorithm)

    return results


if __name__ == '__main__':
    try:
        triton_client = httpclient.InferenceServerClient(url='localhost:8000',verbose=False)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    model_name = "simple"

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    input0_data = np.expand_dims(input0_data, axis=0)
    input1_data = np.full(shape=(1, 16), fill_value=-1, dtype=np.int32)


    # Infer with requested Outputs
    results = test_infer(model_name, input0_data, input1_data)

    print(results.get_response())

    statistics = triton_client.get_inference_statistics(model_name=model_name)
    print(statistics)
    if len(statistics['model_stats']) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)

    # Validate the results by comparing with precomputed values.
    output0_data = results.as_numpy('OUTPUT0')
    output1_data = results.as_numpy('OUTPUT1')
    for i in range(16):
        print(
            str(input0_data[0][i]) + " + " + str(input1_data[0][i]) + " = " +
            str(output0_data[0][i]))
        print(
            str(input0_data[0][i]) + " - " + str(input1_data[0][i]) + " = " +
            str(output1_data[0][i]))
        if (input0_data[0][i] + input1_data[0][i]) != output0_data[0][i]:
            print("sync infer error: incorrect sum")
            sys.exit(1)
        if (input0_data[0][i] - input1_data[0][i]) != output1_data[0][i]:
            print("sync infer error: incorrect difference")
            sys.exit(1)

    # Infer without requested Outputs
    results = test_infer(model_name, input0_data, input1_data)
    print(results.get_response())

    # Validate the results by comparing with precomputed values.
    output0_data = results.as_numpy('OUTPUT0')
    output1_data = results.as_numpy('OUTPUT1')
    for i in range(16):
        print(
            str(input0_data[0][i]) + " + " + str(input1_data[0][i]) + " = " +
            str(output0_data[0][i]))
        print(
            str(input0_data[0][i]) + " - " + str(input1_data[0][i]) + " = " +
            str(output1_data[0][i]))
        if (input0_data[0][i] + input1_data[0][i]) != output0_data[0][i]:
            print("sync infer error: incorrect sum")
            sys.exit(1)
        if (input0_data[0][i] - input1_data[0][i]) != output1_data[0][i]:
            print("sync infer error: incorrect difference")
            sys.exit(1)

    # Infer with incorrect model name
    try:
        response = test_infer("wrong_model_name", input0_data,
                              input1_data).get_response()
        print("expected error message for wrong model name")
        sys.exit(1)
    except InferenceServerException as ex:
        print(ex)
        if not (ex.message().startswith("Request for unknown model")):
            print("improper error message for wrong model name")
            sys.exit(1)