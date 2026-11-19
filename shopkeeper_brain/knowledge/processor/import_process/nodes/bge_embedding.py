from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.state import ImportGraphState


class BgeEmbeddingChunksNode(BaseNode):
    name = "bge_embedding_node"
    pass
    def process(self, state: ImportGraphState) -> ImportGraphState:
        return state