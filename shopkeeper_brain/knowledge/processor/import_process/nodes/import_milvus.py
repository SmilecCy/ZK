import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, List

from pymilvus import MilvusClient, DataType

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import ImportConfig, get_config
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.storage_clients import StorageClients


@dataclass(frozen=True)
class ScalarFieldSpec:
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None
_SCALAR_FIELDS: Sequence[ScalarFieldSpec] = (
    ScalarFieldSpec(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),

)


def build_schema(client: MilvusClient, collection_name: str, dim: int):
    print(f"开始构建集合 {collection_name} 索引...")
    schema = client.create_schema(enable_dynamic_field=True)
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64,is_primary=True,auto_id=True)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR, )
    for scalar_field in _SCALAR_FIELDS:
        kwargs: Dict[str, Any] = {
            "field_name": scalar_field.field_name,
            "datatype": scalar_field.datatype,
        }
        if scalar_field.max_length is not None:
            kwargs["max_length"] = scalar_field.max_length
        schema.add_field(**kwargs)
    return schema


def IndexBuilder(milvus_client:MilvusClient, collection_name:str):
    index = milvus_client.prepare_index_params(collection_name=collection_name)
    index.add_index(field_name="dense_vector", index_name="dense_vector_index", index_type="AUTOINDEX",
        metric_type="COSINE")
    index.add_index(field_name="sparse_vector", index_name="sparse_vector_index", index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",)
    return index



class ImportMilvusNode(BaseNode):
    name = "import_milvus_node"
    pass

    def process(self, state: ImportGraphState) -> ImportGraphState:
        validated_chunks,dim,config = self._validate_get_inputs(state)
        milvus_client =StorageClients.get_milvus_client()
        collection = getattr(config, 'chunks_collection')
        self._ensure_has_collection(milvus_client, collection, dim)
        final_chunks = self.inserter(milvus_client,collection,validated_chunks)
        state['chunks'] = final_chunks

        return state

    def _validate_get_inputs(self, state: ImportGraphState)->tuple[list,int,ImportConfig]:
        config = get_config()
        chunks = state.get("chunks", [])
        if not chunks:
            raise ValidationError("待入库的切块chunk不存在", self.name)
        validated_chunks = []
        for chunk in chunks:
            if chunk.get('dense_vector') and chunk.get('sparse_vector'):
                validated_chunks.append(chunk)
            else:
                self.logger.error("待入库的切块chunk的混合向量不存在")
        if not validated_chunks:
            raise ValidationError("入库的chunk都无效", self.name)
        dim = len(validated_chunks[0].get('dense_vector'))

        return validated_chunks, dim, config

    def _ensure_has_collection(self, milvus_client:MilvusClient, collection_name:str, dim:int,delete_flag: bool = True):
        # if delete_flag and milvus_client.has_collection(collection_name=collection_name):
        #     self.logger.info(f"Milvus中的集合 {collection_name}已被删除")
        #     milvus_client.drop_collection(collection_name=collection_name)

        if milvus_client.has_collection(collection_name=collection_name):
            return

        schema = build_schema(milvus_client, collection_name,dim)
        index = IndexBuilder(milvus_client, collection_name)
        milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index)

    def inserter(self, milvus_client:MilvusClient, collection_name:str,validated_chunks:List[Dict[str, Any]])->List[dict[str, Any]]:
        inserted_result = milvus_client.insert(collection_name=collection_name,data=validated_chunks)
        inserted_count = inserted_result.get('insert_count')
        ids = inserted_result.get('ids')
        self._fill_chunk_ids(validated_chunks, ids)
        return validated_chunks

    def _fill_chunk_ids(self, validated_chunks:List[Dict[str, Any]], ids:List[Any]):
        for chunk, id in zip(validated_chunks, ids):
            chunk['chunk_id'] = id

def _cli_main() -> None:
    setup_logging()

    temp_dir = Path(r"G:\PY_SD\SD\ZK\文档\md")

    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        print(f"找不到输入文件: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as fh:
        content = json.load(fh)

    state: ImportGraphState = {
        "chunks": content.get("chunks", [])
    }

    import_milvus = ImportMilvusNode()
    result_state = import_milvus.process(state)

    with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(result_state, fh, ensure_ascii=False, indent=4)

    print(f"备份临时文件{output_path}成功")
if __name__ == "__main__":
    _cli_main()