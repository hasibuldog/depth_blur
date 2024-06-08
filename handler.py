import os
from typing import Any, List

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from transformers import DPTImageProcessor, DPTForDepthEstimation
from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler


class DepthHandler(BaseHandler):
    def __init__(self):
        super(DepthHandler, self).__init__()
        self.initialized = False

    def load_pipeline(self, context: Context) -> TextClassificationPipeline:
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        config_path = os.path.join(model_dir, "config.json")

        model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config_path)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, config=config_path)
        return TextClassificationPipeline(model=model, tokenizer=tokenizer, device="cpu", return_all_scores=True)

    def initialize(self, context: Context):
        self.initialized = True
        self.model_pipeline = self.load_pipeline(context=context)

    def preprocess(self, data: List[dict]) -> List[dict]:
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")
        return preprocessed_data            

    def inference(self, input: List[dict]) -> List[List[Any]]:
        classifications = []
        for data in input:
            query = data.get("query")
            if query:
                classification = self.model_pipeline(query)
                classifications.append(classification)
            else:
                classifications.append([])
        return classifications

    def postprocess(self, output: List[List[Any]]) -> List[List[List[Any]]]:
        return [output]

    def handle(self, data: List[dict], context: Context) -> List[List[List[Any]]]:
        model_input = self.preprocess(data=data)
        model_output = self.inference(input=model_input)
        return self.postprocess(output=model_output)
    
    