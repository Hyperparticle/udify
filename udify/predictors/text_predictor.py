"""
The main UDify predictor to output conllu files
"""

from typing import List
from overrides import overrides
import json

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

from udify.dataset_readers.universal_dependencies import UniversalDependenciesRawDatasetReader
from udify.predictors.predictor import UdifyPredictor


@Predictor.register("udify_text_predictor")
class UdifyTextPredictor(Predictor):
    """
    Predictor for a UDify model that takes in a raw sentence and outputs json.
    """
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 output_conllu: bool = False) -> None:
        super().__init__(model, dataset_reader)
        self._dataset_reader = UniversalDependenciesRawDatasetReader(self._dataset_reader)
        self.predictor = UdifyPredictor(model, dataset_reader)
        self.output_conllu = output_conllu

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        return self.predictor.predict_batch_instance(instances)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        return self.predictor.predict_instance(instance)

    def _predict_unknown(self, instance: Instance):
        """
        Maps each unknown label in each namespace to a default token
        :param instance: the instance containing a list of labels for each namespace
        """
        return self.predictor._predict_unknown(instance)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = [word.text for word in self._dataset_reader.tokenizer.split_words(sentence)]
        return self._dataset_reader.text_to_instance(tokens)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        if self.output_conllu:
            return self.predictor.dump_line(outputs)
        else:
            return json.dumps(outputs) + "\n"
