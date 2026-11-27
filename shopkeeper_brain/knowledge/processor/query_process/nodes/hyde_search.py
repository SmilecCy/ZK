import logging
from typing import Union, Dict, Any, Tuple, List

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import execute_hybrid_search_query, create_hybrid_search_requests, item_names_filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class HyDeSearchNode(BaseNode):
    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        validated_query, validate_item_names = self._validate_query_inputs(state)
        hy_document = self._generate_hy_document(validated_query, validate_item_names)
        try:
            embedding_molde = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型获取失败 原因:{str(e)}")
            return state
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端获取失败 原因:{str(e)}")
            return state

        embedding_document = f"{validated_query}\n{hy_document}"
        try:
            embedding_result = generate_bge_m3_hybrid_vectors(model=embedding_molde, embedding_documents=[embedding_document])
        except Exception as e:
            self.logger.error(f"问题{validated_query}嵌入失败")
            return state
        filter_expr, filter_expr_param = item_names_filter(validate_item_names)

        try:
            hybrid_search_request = create_hybrid_search_requests(dense_vector=embedding_result['dense'][0],
                                                              sparse_vector=embedding_result['sparse'][0],
                                                              expr=filter_expr,
                                                              expr_params=filter_expr_param)
            hybrid_search_reps = execute_hybrid_search_query(milvus_client=milvus_client,
                                                         collection_name=self.config.chunks_collection,
                                                         search_requests=hybrid_search_request,
                                                         output_fields=['chunk_id', 'content', 'item_name', 'title'])
            if not hybrid_search_reps or not hybrid_search_reps[0]:
                return state
            state['hyde_embedding_chunks'] = hybrid_search_reps[0]
        except Exception as e:
            self.logger.error(f"混合检索失败 原因:{str(e)}")
        return state

    def _generate_hy_document(self, validated_query: str, validate_item_names: List[str]) -> str:
        llm_client = AIClients.get_llm_openai(False)
        if llm_client is None:
            return ""
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(item_names=validate_item_names, rewritten_query=validated_query)
        system_prompt = f"您是一位{validate_item_names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"
        try:
            # 4. 获取 AIMessage
            llm_response = llm_client.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            llm_response_content = getattr(llm_response, 'content', "").strip()
            if not llm_response_content:
                return ""
            return llm_response_content
        except Exception as e:
            self.logger.error(f"LLM调用失败:{str(e)}")
            return ""


    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        rewritten_query = state.get('rewritten_query', "")
        item_names = state.get('item_names', "")
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)
        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)
        return rewritten_query, item_names
if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging
    setup_logging()

    print("=" * 60)
    print("开始测试: HyDE 检索节点 (HydeSearchNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS PRO RS-12 数字万用表"],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  商品: {mock_state['item_names']}")
    print("-" * 60)

    node = HyDeSearchNode()
    result = node.process(mock_state)

    chunks = result.get("hyde_embedding_chunks", [])
    print(f"\n【HyDE 检索结果】: {len(chunks)} 条")
    for i, chunk in enumerate(chunks, 1):
        entity = chunk.get("entity", {})
        print(f"  [{i}] chunk_id={entity.get('chunk_id')} "
              f"item_name={entity.get('item_name')} "
              f"distance={chunk.get('distance', 'N/A')}")
        content = entity.get("content", "")
        print(f"      内容: {content[:80]}...")

    print("-" * 60)
    print("测试完成")