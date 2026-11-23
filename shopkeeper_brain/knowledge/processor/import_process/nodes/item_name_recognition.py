import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType



from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompt.import_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE, ITEM_NAME_SYSTEM_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class ItemNameRecognitionNode(BaseNode):
    name = "item_name_rec_node"
    pass

    def process(self, state: ImportGraphState) -> ImportGraphState:
        file_title,chunks,item_mane_chunks_k,item_name_size = self._validate_state(state)

        item_name_recogbition_context = self._get_item_name_recogbition_context(chunks,item_mane_chunks_k,item_name_size)
        item_name =self._recognition_name(item_name_recogbition_context,file_title)
        dense_vector, sparse_vector = self._embedding_item_name(item_name)
        self._insert_milvus(file_title,item_name,dense_vector,sparse_vector)
        self._fill_item_name(item_name, state, chunks)
        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[str,List[Dict[str,Any]],int,int]:
        file_title = state.get("file_title")
        chunks = state.get("chunks")
        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title", expected_type=str)
        if not chunks:
            if not chunks or not isinstance(chunks, list):
                raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)
        item_mane_chunks_k = self.config.item_name_chunk_k
        item_name_size = self.config.item_name_chunk_size

        return file_title,chunks,item_mane_chunks_k,item_name_size

    def _get_item_name_recogbition_context(self, chunks: List[Dict[str,Any]], item_mane_chunks_k: int, item_name_size: int)->str:
        final_conttext = []
        chunks = chunks[:item_mane_chunks_k]
        total = 0
        for idx,chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            chunk_content = chunk.get("content")
            context = f"【切片】-【{idx+1}】--{chunk_content}"
            total += len(chunk_content)
            if total >= item_name_size:
                break
            final_conttext.append(context)
        return "\n\n".join(final_conttext)

    def _recognition_name(self, item_name_recogbition_context:str,file_title:str)->str:
        llm_client = AIClients.get_llm_openai(response_format=False)
        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(context=item_name_recogbition_context,file_title=file_title)

        try:
            llm_response = llm_client.invoke([SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
                                          HumanMessage(content=user_prompt)])
            item_name_data = llm_response.content.strip()
            data = json.loads(item_name_data)
            item_name = data.get("item_name")

            if not item_name or item_name == "UNKNOWN":
                return file_title
            self.logger.info(f"LLM提取到商品名: {item_name}")
            return item_name
        except Exception as e:
            self.logger.error(f"LLM调用失败，降级使用标题: {file_title}，异常: {e}")
            return file_title

    def _embedding_item_name(self, item_name:str)->Tuple[List[float],Dict[int, float]]:
        try:
            beg_m3_client = AIClients.get_bge_m3_client()
            vector_result = beg_m3_client.encode_documents([item_name])
            dense_vector = vector_result['dense'][0].tolist()
            start_index = vector_result['sparse'].indptr[0]
            end_index = vector_result['sparse'].indptr[1]
            token_id = vector_result['sparse'].indices[start_index:end_index].tolist()
            weight = vector_result['sparse'].data[start_index:end_index].tolist()
            sparse_vector = dict(zip(token_id, weight))
            return dense_vector, sparse_vector

        except Exception as e:
            self.logger.error(f"BGE M3模型调用失败，异常: {e}")
            return None, None

    def _insert_milvus(self, file_title:str, item_name:str, dense_vector:List[float], sparse_vector:Dict[int, float]):
        if not dense_vector or not sparse_vector:
            self.logger.error(f"文档{file_title} 对应的商品名{item_name} 向量生成不完整")
            return
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"Milvus 客户端创建失败: {e}")
            return
        item_name_collection = self.config.item_name_collection
        try:
            if not milvus_client.has_collection(item_name_collection):
                self._create_item_name_collection(item_name_collection, milvus_client)
            data = {
                    "file_title": file_title,
                    "item_name": item_name,
                    "dense_vector": dense_vector,
                    "sparse_vector": sparse_vector
            }
            result = milvus_client.insert(collection_name=item_name_collection, data=data)
            self.logger.info(f"已成功保存到 Milvus，ID: {result['ids'][0]}")
        except Exception as e:
            self.logger.error(f"Milvus 数据操作失败: {e}")

    def _create_item_name_collection(self, item_name_collection:str, milvus_client):
        schema = milvus_client.create_schema()
        schema.add_field(field_name="pk",datatype=DataType.INT64, is_primary=True,auto_id=True,max_length = 100)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_param = milvus_client.prepare_index_params()
        index_param.add_index(field_name="dense_vector", index_name="dense_vector_index",
                              index_type="AUTOINDEX", metric_type="COSINE")
        index_param.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                              index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        milvus_client.create_collection(collection_name=item_name_collection, schema=schema,index_params=index_param)
        self.logger.info(f"集合 {item_name_collection} 创建成功并构建了索引")

    def _fill_item_name(self, item_name:str, state:ImportGraphState, chunks:List[Dict[str,Any]])->ImportGraphState:
        for chunk in chunks:
            chunk["item_name"] = item_name
        state["item_name"] = item_name



if __name__ == '__main__':
    setup_logging()
    chunk_json_path = r"G:\PY_SD\SD\ZK\文档\md\chunks.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)
    state = {"file_title": "万用表的使用", "chunks": chunk_content.get("chunks")}
    node = ItemNameRecognitionNode()
    result = node.process(state)
    print(f"商品名: {result.get('item_name')}")
    print(f"chunks数量: {len(result.get('chunks', []))}")
    print(f"首个chunk是否含item_name: {'item_name' in result['chunks'][0]}")








