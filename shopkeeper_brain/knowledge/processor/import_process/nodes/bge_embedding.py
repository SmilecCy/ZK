import json
from pathlib import Path
from typing import Dict

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.config import get_config, ImportConfig
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients


class BgeEmbeddingChunksNode(BaseNode):
    name = "bge_embedding_node"
    pass
    def process(self, state: ImportGraphState) -> ImportGraphState:
        validated_chunks,config = self._validate_get_inputs(state)
        embedding_batch_chunk_size = getattr(config, 'embedding_batch_size', 16)
        total_length = len(validated_chunks)
        final_chunks = []
        for i in range(0,total_length,embedding_batch_chunk_size):
            batch = validated_chunks[i:i+embedding_batch_chunk_size]
            batch_chunks = self._process_batch_chunks(batch, i, total_length)
            final_chunks.extend(batch_chunks)
        state['chunks'] = final_chunks
        return state

    def _validate_get_inputs(self, state: ImportGraphState) -> tuple[list[Dict[str, str]], ImportConfig]:
        config = get_config()
        chunks = state.get('chunks')
        if not chunks or not isinstance(chunks, list):
            raise ValidationError(f"chunks为空或者无效", self.name)
        self.logger.info(f"嵌入的块数：{len(chunks)}")
        return chunks,config

    def _process_batch_chunks(self, batch: list[Dict[str, str]], star_index: int, total_length: int) -> list[Dict[str, str]]:
        """处理一个批次的切片"""
        self.log_step("step2", f"开始批量处理chunk嵌入:批次{star_index + 1}-{star_index + len(batch)}")
        embedding_contents = []
        for _, chunk in enumerate(batch):
            content = chunk.get('content')
            item_name = chunk.get('item_name')
            embedding_content = f"{item_name}\n{content}"
            embedding_contents.append(embedding_content)
        try:
            bge_m3_model = AIClients.get_bge_m3_client()
            embedding_result = bge_m3_model.encode_documents(documents=embedding_contents)
            if not embedding_result:
                self.logger.warning(f"嵌入后的结果不存在...")
                return batch
        except Exception as e:
            self.logger.warning(f"嵌入向量嵌入失败...{str(e)}")
            return batch
        for index, chunk in enumerate(batch):
            dense_vector = embedding_result['dense'][index].tolist()
            csr_array = embedding_result['sparse']
            ind_ptr = csr_array.indptr
            start_ind_ptr = ind_ptr[index]
            end_ind_ptr = ind_ptr[index + 1]
            token_id = csr_array.indices[start_ind_ptr:end_ind_ptr].tolist()
            weight = csr_array.data[start_ind_ptr:end_ind_ptr].tolist()
            sparse_vector = dict(zip(token_id, weight))
            chunk['dense_vector'] = dense_vector
            chunk['sparse_vector'] = sparse_vector
            self.logger.info(f"开始批量处理chunk嵌入:批次{star_index + 1}-{star_index + len(batch)}/{total_length}")
        return batch
if __name__ == '__main__':
    setup_logging()

    base_temp_dir = Path(r"G:\PY_SD\SD\ZK\文档\md")

    input_path = base_temp_dir / "chunks.json"
    output_path = base_temp_dir / "chunks_vector.json"

    # 1. 读取上游状态
    if not input_path.exists():
        print(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    # 2. 构建模拟的图状态
    state = {
        "file_title": content.get("file_title"),
        "chunks": content.get("chunks")
    }

    # 3. 触发节点执行
    node_bge_embedding = BgeEmbeddingChunksNode()
    proceed_result = node_bge_embedding.process(state)

    # 4. 结果落盘
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(proceed_result, f, ensure_ascii=False, indent=4)

    print(f"向量生成测试完成！结果已成功备份至:\n{output_path}")