import json
from typing import Union, Dict, Any

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import item_names_filter, create_hybrid_search_requests, execute_hybrid_search_query


class VectorSearchNode(BaseNode):
    name = "vector_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        validated_query, validate_item_names = self._validate_state(state)

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
        try:
            embed_query = generate_bge_m3_hybrid_vectors(model=embedding_molde, embedding_documents=[validated_query])
        except Exception as e:
            self.logger.error(f"问题{validated_query}嵌入失败")
            return state
        filter_expr, filter_expr_param = item_names_filter(validate_item_names)
        try:
            hybrid_search_request = create_hybrid_search_requests(dense_vector=embed_query['dense'][0],
                                                                    sparse_vector=embed_query['sparse'][0],
                                                                    expr=filter_expr,
                                                                    expr_params=filter_expr_param)
            hybrid_search_reps = execute_hybrid_search_query(milvus_client=milvus_client,
                                                             collection_name=self.config.chunks_collection,
                                                             search_requests=hybrid_search_request,
                                                             output_fields=['chunk_id', 'content', 'item_name', 'title'])
            if not hybrid_search_reps or not hybrid_search_reps[0]:
                return state
            state['embedding_chunks'] = hybrid_search_reps[0]
            return state
        except Exception as e:
            self.logger.error(f"混合检索失败 原因:{str(e)}")
            return state

    def _validate_state(self, state: QueryGraphState):
        rewritten_query = state.get('rewritten_query')
        item_names = state.get('item_names')
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)
        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)

        return rewritten_query, item_names


if __name__ == '__main__':
    state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS PRO RS-12 数字万用表"]
    }

    vector_search = VectorSearchNode()
    result = vector_search.process(state)

    for r in result.get('embedding_chunks',[]):
        print(json.dumps(r, ensure_ascii=False, indent=2))

    """G:\PY_SD\SD\ZK\.venv\Scripts\python.exe G:\PY_SD\SD\ZK\shopkeeper_brain\knowledge\processor\query_process\nodes\vector_search.py 
Loading weights: 100%|██████████| 391/391 [00:09<00:00, 41.31it/s]
INFO:FlagEmbedding.finetune.embedder.encoder_only.m3.runner:loading existing colbert_linear and sparse_linear---------
INFO:knowledge.utils.client.base:bge_m3客户端初始化成功
{
  "chunk_id": 470681092194572844,
  "distance": 0.7403039932250977,
  "entity": {
    "chunk_id": 470681092194572844,
    "content": "## 电阻测量\n\n\n**警告:** 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。\n\n1. 将功能转盘置于最高电阻Ω位置.\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口\n\n3. 把表笔接触被测电路或元件。测试时最好断开电路的一端，以使剩余的电路不会干扰被测电阻数值。\n\n4. 读取显示屏上读数，然后将功能转盘调至最低电阻Ω档位，通常大于实际电阻或预测电阻.读数由精确的小数点和数值表示。\n\n![万用表电阻测量示意图](http://192.168.88.160:9000/knowledge-base-files/万用表RS-12的使用/239.jpg)\n",
    "item_name": "RS PRO RS-12 数字万用表",
    "title": "## 电阻测量"
  }
}
{
  "chunk_id": 470681092194572834,
  "distance": 0.6994400024414062,
  "entity": {
    "chunk_id": 470681092194572834,
    "content": "## 规格-3\n\n- 【电阻】(对应功能)：量程为200kΩ，分辨率为0.1kΩ，精确度为± (0.8% reading + 2 digits)。\n- 【电阻】(对应功能)：量程为2000kΩ，分辨率为1kΩ，精确度为± (1.0% reading + 2 digits)。\n- 【电池】(对应功能)：量程为9V，分辨率为10mV，精确度为± (1.0% reading + 2 digits)。\n- 【电池】(对应功能)：量程为1.5V，分辨率为1mV，精确度为± (1.0% reading + 2 digits)。",
    "item_name": "RS PRO RS-12 数字万用表",
    "title": "## 规格-3"
  }
}
{
  "chunk_id": 470681092194572841,
  "distance": 0.4197039008140564,
  "entity": {
    "chunk_id": 470681092194572841,
    "content": "万用表RS-12的使用\n\n\n**警告:**谨防触电。\n\n若表笔长度不够不能接触到某些240V用具插座的带电部位，则可能出现插座有电而读到的数值却为0的情况。因此若无电压显示，应检查表笔是否接触到了插座内的金属接口。\n\n**注意:**正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n![交流电压测量示意图](http://192.168.88.160:9000/knowledge-base-files/万用表RS-12的使用/193.jpg)\n\n1. 将功能转盘置于V AC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值和(AC,V等)符号。\n\n在显示屏上读取电压数据。不断重调功能转盘至低交流电压档位获得高分辨率读数。读数由精确的小数点和数值表示。\n\n\n## 直流电流测量-1\n\n内容：**注意:**在10A情况下测量时间不能超过30秒，否则将可能损坏仪表或表笔。\n\n![直流电流测量示意图",
    "item_name": "RS PRO RS-12 数字万用表",
    "title": "万用表RS-12的使用"
  }
}
{
  "chunk_id": 470681092194572840,
  "distance": 0.4161059260368347,
  "entity": {
    "chunk_id": 470681092194572840,
    "content": "## 直流电压测量\n\n\n**注意:**正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n1. 将功能转盘置于V DC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物,确保极性正确(红色连正极,黑色连负极)。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值。若极性颠倒，数值前将显示负号。\n",
    "item_name": "RS PRO RS-12 数字万用表",
    "title": "## 直流电压测量"
  }
}
{
  "chunk_id": 470681092194572843,
  "distance": 0.41108477115631104,
  "entity": {
    "chunk_id": 470681092194572843,
    "content": "## 直流电流测量-4\n\n**标题：** 直流电流测量示意图（或：万用表直流电流测量连接图）\n\n**最终输出：** 直流电流测量示意图](http://192.168.88.160:9000/knowledge-base-files/万用表RS-12的使用/209.jpg)\n\n1. 将黑色表笔插入负极COM端口。\n\n2. 测量直流200mA 以下的电流,将功能转盘置于最高DC mA档位，并将红色表笔插入mA端口。\n\n3. 测量直流10A时,将功能转盘置于10A档位，并将红色表笔(10A)端口。\n\n4. 断开被测电路的电源。在你想测量电流的位置打开电路绝缘层。\n\n5. 将黑色表笔接触被测电路的负极，红色表笔接触被测电路正极。\n\n6. 接通电源。\n\n7. 在显示屏上读取读数。进行mA DC测量时,不断重调功能转盘至低mA DC档位获得高分辨率读数.读数由精确的小数点和数值表示。\n\n![直流电流测量接线示意图](http://192.168.88.160:9000/knowledge-base-files/万用表RS-12的使用/225.jpg)",
    "item_name": "RS PRO RS-12 数字万用表",
    "title": "## 直流电流测量-4"
  }
}
INFO:knowledge.utils.milvus_util:Milvus 混合搜索完成，共处理 1 个查询，总计找到 5 个结果

进程已结束，退出代码为 0
"""
