import numpy as np
import triton_python_backend_utils as pb_utils
import logging

import os

from gensim import  models
import nltk
import ssl

import json

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

from nltk.tokenize import word_tokenize

# Set up the logging configuration
logging.basicConfig(level=logging.DEBUG)


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        logging.debug("Initializing the model...")
        self.lda = models.ldamodel.LdaModel.load("/opt/tritonserver/recommendation.model")
        logging.debug("Model loaded...")
        


    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        logging.debug("Executing the model...")

        # list filenames in current dir

        logging.debug(f"Current dir: {os.getcwd()}")
        logging.debug(f"Files in current dir: {os.listdir()}")

        logging.debug(word_tokenize("Hello world"))

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            user_data = in_0.as_numpy()[0].decode("utf-8")

            user_tokens = word_tokenize(user_data)
            user_vector = self.lda.infer_vector(user_tokens)
            similar_documents = self.lda.docvecs.most_similar([user_vector])

            out_tensor_0 = pb_utils.Tensor(
                "OUTPUT0", np.array([json.dumps(similar_documents)], dtype=np.bytes_)
            )

            responses.append(pb_utils.InferenceResponse([out_tensor_0]))

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        logging.debug("Finalizing the model...")
