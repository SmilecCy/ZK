import json
import re
from typing import Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    name = "document_split_node"
    pass

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.logger.info("开始文档切分流程")
        md_content,file_title,max_content_length,min_content_length = self._get_inputs(state)
        sections = self._split_by_headings(md_content,file_title)
        final_sections = self._split_long_merge_short(sections,max_content_length,min_content_length)
        final_chunks = self._assemble_chunk(final_sections)
        state["chunks"] = final_chunks
        self.logger.info(f"切分流程完成, 最终切片数: {len(final_chunks)}")
        self._log_summary(md_content, final_chunks, max_content_length)
        self._backup_chunks(state, final_chunks)

        return state

    def _get_inputs(self, state: ImportGraphState) -> tuple[str, str, int, int]:
        self.log_step("step1", "切分文档的参数校验以及获取...")
        config = get_config()
        md_content:str = state.get("md_content")
        file_title:str = state.get('file_title')
        min_content_length = config.min_content_length
        max_content_length = config.max_content_length
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        self.logger.debug(f"参数: file_title={file_title}, md_content 长度={len(md_content) if md_content else 0}, "
                          f"min_content_length={min_content_length}, max_content_length={max_content_length}")

        if max_content_length <= 0 or min_content_length <= 0 or max_content_length > min_content_length >= max_content_length:
            raise ValueError(f"切片长度参数校验失败")
        return md_content,file_title,max_content_length,min_content_length

    def _split_by_headings(self, md_content:str, file_title:str) -> list[Dict[str,Any]]:
        self.log_step("step2", "根据标题进行切分...")
        body_lines = []
        sections = []
        hierarchy = [""] * 7
        current_title = ""
        current_level = 0
        in_fence = False




        content_lines = md_content.split("\n")
        self.logger.debug(f"文档总行数: {len(content_lines)}")
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        def _flush():
            body = "\n".join(body_lines)
            if current_title or body:

                parent_title = ""
                for i in range(current_level-1,0,-1):
                    parent_title = hierarchy[i] if hierarchy[i] else ""
                    if parent_title :
                        break
                if not parent_title :
                    parent_title = parent_title if parent_title else file_title



                section = {"title":current_title if current_title else file_title,
                           "parent_title":parent_title,
                           "file_title":file_title,
                           "body":body

                }
                self.logger.debug(f"flush 章节: title={section['title']}, parent_title={parent_title}, body 长度={len(body)}")
            sections.append(section)


        for content_line in content_lines:
            match = heading_re.match(content_line)
            if content_line.strip().startswith("```") or content_line.strip().startswith("~~~"):
                in_fence = not in_fence

            if match:
                _flush()

                current_title = content_line
                current_level = len(match.group(1))
                self.logger.debug(f"识别到标题(level={current_level}): {current_title.strip()}")
                hierarchy[current_level] = current_title
                for i in range(current_level+1,0,-1):
                    hierarchy[i] = ""
                


                body_lines = []
            else:
                body_lines.append(content_line)
        _flush()
        self.logger.info(f"标题切分完成, 共 {len(sections)} 个章节")
        return sections

    def _split_long_merge_short(self, sections:list[Dict[str,str]], max_content_length:int=1000, min_content_length:int=50) -> list[Dict[str,str]]:
        self.log_step("step3", f"长章节拆分与短章节合并, 输入章节数: {len(sections)}")
        current_session:list[Dict[str,str]] = []
        for section in sections:
            sub_section = self._split_long(section,max_content_length)

            current_session.extend(sub_section)
        self.logger.debug(f"长章节拆分后章节数: {len(current_session)}")
        final_sections = self.merge_short_section(current_session,min_content_length)
        self.logger.info(f"短章节合并后章节数: {len(final_sections) if final_sections else 0}")
        return final_sections



    def _split_long(self, section:Dict[str,str], max_content_length:int):
        title:str|None = section.get("title")
        parent_title = section.get("parent_title")
        file_title = section.get("file_title")
        body:str|None = section.get("body")
        self.logger.debug(f"处理章节: title={title}, body 长度={len(body) if body else 0}, max_content_length={max_content_length}")
        if "<table>" in body:
            body = MarkdownTableLinearizer.process(body)
            section["body"] = body
        if len(title) > 50:
            title = title[:50]
        title_prefix = f"{title}\n\n"
        content_length = len(title_prefix) + len(body)
        if content_length < max_content_length:
            self.logger.debug("章节内容未超过 max_content_length, 无需拆分")
            return [section]
        if content_length-len(title_prefix ) < 0:
            return [section]
        splitter_false = RecursiveCharacterTextSplitter(chunk_size=max_content_length,
                                                        chunk_overlap=0,
                                                        keep_separator=False,
                                                        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]

        )
        texts = splitter_false.split_text(body)
        self.logger.debug(f"章节被拆分为 {len(texts)} 个子块")
        if len(texts) < 2:
            return [section]
        sub_section = []
        part_counter = {}
        for idx,text  in enumerate(texts):
            part_counter[parent_title] = part_counter.get(parent_title, 0) + 1
            n = part_counter[parent_title]
            sub_section.append({"title":f"{title}-{n}",
                                "parent_title":parent_title,
                                "file_title":file_title,
                                "body":text
            })
        return sub_section


        pass

    def merge_short_section(self, current_sessions:list[Dict[str,Any]], min_content_length:int)->list[Dict[str,Any]]:
        self.log_step("step4", f"合并短章节, 输入数量: {len(current_sessions)}, min_content_length={min_content_length}")
        current_section = current_sessions[0]
        final_sections = []
        for next_section in current_sessions[1:]:
            same_parent = (current_section.get("parent_title") == next_section.get("parent_title"))
            if same_parent and len(next_section.get("body")) < min_content_length:
                self.logger.debug(f"合并短章节: {next_section.get('title')} -> {current_section.get('title')}")
                current_section["body"] += (f"\n\n{next_section.get('title')}\n\n内容：{next_section['body']}")
                current_section['title'] = current_section['parent_title']


            else:
                final_sections.append(current_section)
                current_section = next_section
        final_sections.append(current_section)
        self.logger.info(f"合并完成, 最终章节数: {len(final_sections)}")
        return final_sections






    def _assemble_chunk(self, final_sections:list[Dict[str,str]])->list[Dict[str,Any]]:
        self.log_step("step5", f"组装最终切片, 输入章节数: {len(final_sections) if final_sections else 0}")
        final_chunks = []
        for idx,section in enumerate(final_sections):
            chunk = ({"title":section.get("title"),
                                 "parent_title":section.get("parent_title"),
                                 "file_title":section.get("file_title"),
                                 "content":section.get("title")+"\n\n"+section.get("body")

            })
            chunk["part"] = idx+1
            final_chunks.append(chunk)
        self.logger.info(f"组装完成, 最终切片数: {len(final_chunks)}")
        return final_chunks

    def _log_summary(self, md_content:str, final_chunks:list[Dict[str,Any]], max_content_length:int):
        lines_count = md_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(final_chunks)}")
        self.logger.info(f"最大切片长度: {max_content_length}")
        if final_chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(final_chunks[:5]):
                title = sec.get("title", "")[:30]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(final_chunks) > 5:
                self.logger.info(f"  ... 还有 {len(final_chunks) - 5} 个章节")

        pass

    def _backup_chunks(self, state:ImportGraphState, final_chunks:list[Dict[str,Any]]) :
        """将切分结果备份到 JSON 文件"""
        self.log_step("step6", "备份切片")

        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置 file_dir，跳过备份")
            return

        try:
            import os
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_chunks, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")

if __name__ == '__main__':
    setup_logging()

    document_node = DocumentSplitNode()
    # 构造状态字典
    file_path = r"G:\PY_SD\SD\ZK\文档\md\万用表RS-12的使用_new.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    state = {
        "file_title": "万用表的使用",
        "md_content": content,
        "file_dir": r"G:\PY_SD\SD\ZK\文档\md"
    }
    document_node.process(state)