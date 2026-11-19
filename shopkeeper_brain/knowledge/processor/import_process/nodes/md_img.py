import base64
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Deque, Optional, Dict
from collections import deque

from openai import OpenAI

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import StateFieldError, FileProcessingError, ImageProcessingError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


@dataclass
class ImageContext:
    """图片在 MD 中的上下文信息。"""
    heading: str =""     # 最近的章节标题
    pre_text: str =""    # 图片上方的正文内容
    post_text: str =""   # 图片下方的正文内容


@dataclass
class ImageInfo:
    """一张图片的完整信息。"""
    name: str =""                # 图片文件名，如 "abc123.jpg"
    path: str =""
    Number: int = 0 # 图片序号

    context: Optional[ImageContext] = None    # 在 MD 中的上下文



class MdFileHandler:
    def __init__(self,logger:logging.Logger):
        self.logger = logger
    def read_md(self,state:ImportGraphState)->tuple[str,Path]:
        md_path = state.get("md_path")
        if not md_path:
            self.logger.error("state 中缺少 md_path 字段")
            raise StateFieldError(node_name="md_path",
                                  field_name="md_path",
                                  expected_type=str)
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError("md_path文件不存在")
        self.logger.info(f"开始读取 Markdown 文件: {md_path_obj}")
        with open(md_path_obj,"r",encoding="utf-8") as f:
            md_content = f.read()
        self.logger.info(f"读取完成，共 {len(md_content)} 字符")
        return md_content,md_path_obj

    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        self.logger.info("备份新文件")

        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )
        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(
                f"文件写入失败: {e}", node_name="md_img_node"
            )
        return str(new_file_path)


class ImageScanner:
    def __init__(self,logger:logging.Logger):
        self.logger = logger
    def scanner(self,md_content:str,img_content_length:int)->list[ImageInfo]:

        image_info_list:list[ImageInfo] = []
        md_lines = md_content.split("\n")
        pattern = re.compile(r'!\[\]\([^)]+\)')
        self.logger.info(f"开始扫描图片，文档共 {len(md_lines)} 行")

        for line_idx, line in enumerate(md_lines):

            if pattern.search(line):

                image_info = ImageInfo()
                image_info.path = pattern.search(line).group(0)
                image_info.Number = line_idx



                prev_title_idx, prev_title = self._find_heading_above(line_idx=line_idx, md_lines=md_lines)
                prev_ctx = md_lines[prev_title_idx + 1:line_idx]
                img_pre_ctx = self._exrtact_limited_ctx(prev_ctx=prev_ctx, img_content_length=img_content_length,
                                                        direction="front")
                next_title_idx = self._find_heading_below(md_lines, line_idx=line_idx)
                next_title = md_lines[line_idx + 1:next_title_idx]
                img_next_ctx = self._exrtact_limited_ctx(next_title=next_title, img_content_length=img_content_length,
                                                         direction="back")
                image_info.context = ImageContext(heading=prev_title, pre_text=img_pre_ctx, post_text=img_next_ctx)




                image_info_list.append(image_info)
        self.logger.info(f"扫描完成，共发现 {len(image_info_list)} 张图片")
        return image_info_list




    @staticmethod
    def _find_heading_above(line_idx:int, md_lines:list[str])->tuple[int,str]:
        for i in range(line_idx-1,1,-1):
            if re.match(r"^#{1,6}\s+",md_lines[i]):
                return i,md_lines[i]

        return -1,""
    @staticmethod
    def _find_heading_below( md_lines:list[str], line_idx:int)->int:
        for i in range(line_idx+1,len(md_lines)):
            if re.match(r"^#{1,6}\s+",md_lines[i]):
                return i

        return len(md_lines)

    @staticmethod
    def _exrtact_limited_ctx(prev_ctx:list[str]=[], next_title:list[str]=[], img_content_length:int = None, direction:str = None)->str:
        current_paragraph:list[str] = []
        paragraph:list[str] = []

        if direction == "front":
            for line in prev_ctx:
                is_blank_line = not line.strip()
                is_other_img = re.match(r"^!\[.*?\]\[.*?]$",line.strip())

                if is_other_img or is_blank_line:
                    if current_paragraph:
                        paragraph.append("\n".join(current_paragraph))
                        current_paragraph = []
                    continue
                current_paragraph.append(line)
            if current_paragraph:
                paragraph.append("\n".join(current_paragraph))

            paragraph.reverse()
            total = 0
            select_ctx : list[str]=[]
            for line in paragraph:
                if (total+len(line))>img_content_length and select_ctx:
                    break
                select_ctx.append(line)
                total += len(line)

            select_ctx.reverse()


        elif direction == "back":
            for line in next_title:
                is_blank_line = not line.strip()
                is_other_img = re.match(r"^!\[.*？\]\[.*?]$", line.strip())
                if is_other_img or is_blank_line:
                    if current_paragraph:
                        paragraph.append("\n".join(current_paragraph))
                        current_paragraph = []
                    continue
                current_paragraph.append(line)
            select_ctx : list[str] = []
            total = 0
            for line in paragraph:
                if (total + len(line)) > img_content_length and select_ctx:
                    break
                select_ctx.append(line)
                total += len(line)
        return "\n\n".join(select_ctx)

class VLMSummarizer:
    def __init__(self,logger:logging.Logger):
        self.logger = logger
    def summarizer(self,image_info_list:list[ImageInfo],document_title:str,vl_model:str,max_requests:int)->list[str]:
        summaries:list[str]=[]

        self.logger.info(f"开始生成图片摘要，共 {len(image_info_list)} 张，模型: {vl_model}")

        request_timestamps: Deque[float] = deque()
        try:
            client = AIClients.get_openai()
        except Exception as e:
            self.logger.error(f"获取OpenAI客户端失败:{e}")
            for img in image_info_list:
                img.name = "默认图片描述"
                summaries.append(img.name)
            return summaries
        for idx, img in enumerate(image_info_list, start=1):
            self._enforce_rate_limit(request_timestamps,max_requests)
            self.logger.info(f"[{idx}/{len(image_info_list)}] 正在为第 {img.Number} 行的图片生成摘要...")
            summarie = self._summarize_one(client,img,document_title,vl_model)
            summaries.append(summarie)
            self.logger.info(f"[{idx}/{len(image_info_list)}] 摘要结果: {summarie}")

        self.logger.info(f"摘要生成完成，共 {len(summaries)} 条")
        return summaries

    def _enforce_rate_limit(self, request_timestamps:Deque[float],max_requests:int,window: int = 20):
        now = time.time()
        while request_timestamps and now - request_timestamps[0] > window:
            request_timestamps.popleft()
        if len (request_timestamps) >= max_requests:
            sleep_time = window - (now - request_timestamps[0])
            if sleep_time > 0:
                self.logger.info(f"达到速率限制，暂停 {sleep_time:.2f} 秒...")
                time.sleep(sleep_time)
            now = time.time()
            while now - request_timestamps[0] >= window:
                request_timestamps.popleft()
        request_timestamps.append(now)

    def _summarize_one(self, client:OpenAI, img: ImageInfo, document_title:str, vl_model:str)->str:

        img_ctx=[p for p in (img.context.heading,img.context.pre_text,img.context.post_text) if p]
        final_context = "\n".join(img_ctx)
        img_url = img.path
        m = re.search(r'!\[.*?\]\((.*?)\)', img_url)
        img_url = m.group(1)

        try:
            resp = client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_title}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_url
                            },
                        },
                    ],
                }]
            )

            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.warning(f"图片摘要生成失败 {img.Number}: {e}")
            return "图片描述"




class ImageUploader:
    def __init__(self,logger:logging.Logger):
        self.logger = logger

    def upload_and_replace(self, md_content:str, md_path_obj:Path, image_info_list:list[ImageInfo], summaries:list[str], minio_bucket:str, minio_base_url:str)->str:

        self.logger.info("开始上传图片到 MinIO 并替换 Markdown 链接")
        remote_urls: Dict[str, str]=self._upload_all(image_info_list=image_info_list, minio_bucket=minio_bucket,document_name=md_path_obj.name,summaries=summaries,minio_base_url=minio_base_url)
        self.logger.info(f"图片上传完成，共 {len(remote_urls)} 张")
        new_md_content = self.replace_md(remote_urls=remote_urls,image_info_list=image_info_list,summaries=summaries,md_content=md_content)
        self.logger.info("Markdown 链接替换完成")
        return new_md_content

    def _upload_all(self, image_info_list:list[ImageInfo], minio_bucket:str, document_name:str,minio_base_url:str, summaries:list[str])->Dict[str, str]:
        remote_urls: Dict[str, str]={}
        self.logger.info(f"开始上传图片，共 {len(image_info_list)} 张，目标桶: {minio_bucket}")
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            self.logger.error(f"获取MinIO客户端失败:{e}")
            for img in image_info_list:
                remote_urls[img.path] = img.path
            return remote_urls

        images_dir = Path(r"G:\PY_SD\SD\ZK\文档\images")
        images_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"本地图片保存目录: {images_dir}")

        success_count = 0
        for img in image_info_list:
            try:
                b64_data = self._extract_base64(img.path)
                local_path = images_dir / f"image_{img.Number}.jpg"
                local_path.write_bytes(base64.b64decode(b64_data))

                object_name = f"{Path(document_name).stem}/{img.Number}.jpg"
                minio_client.fput_object(
                    bucket_name=minio_bucket,
                    object_name=object_name,
                    file_path=str(local_path),
                )
                remote_url = f"{minio_base_url}/{minio_bucket}/{object_name}"
                remote_urls[img.path] = remote_url
                success_count += 1
                self.logger.info(f"[{success_count}/{len(image_info_list)}] 成功上传图片到 MinIO: {object_name}")
            except Exception as e:
                self.logger.warning(f"图片 {img.Number} 上传失败: {e}")
                remote_urls[img.path] = img.path
        self.logger.info(f"上传完成，成功 {success_count}/{len(image_info_list)} 张")
        return remote_urls

    @staticmethod
    def _extract_base64(markdown_img: str) -> str:
        m = re.search(r'!\[.*?\]\((.*?)\)', markdown_img)
        if not m:
            raise ValueError(f"无法从 markdown 提取图片地址: {markdown_img[:50]}")
        data_uri = m.group(1)
        if data_uri.startswith("data:"):
            return data_uri.split(",", 1)[1]
        return data_uri

    def replace_md(self, remote_urls: Dict[str, str], image_info_list:list[ImageInfo], summaries:list[str], md_content:str)->str:
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
        img_to_summary = {img.path: summary for img, summary in zip(image_info_list, summaries)}
        replaced_count = 0

        def replacer(match: re.Match) -> str:
            nonlocal replaced_count
            full_md = match.group(0)
            remote_url = remote_urls.get(full_md)
            summary = img_to_summary.get(full_md)
            if remote_url and summary:
                replaced_count += 1
                return f"![{summary}]({remote_url})"
            return full_md

        result = pattern.sub(replacer, md_content)
        self.logger.info(f"共替换 {replaced_count} 处图片链接")
        return result




class MarkDownImageNode(BaseNode):
    name="md_img_node"
    def __init__(self):
        super().__init__()
        self.file_handler = MdFileHandler(self.logger)
        self.image_scanner = ImageScanner(self.logger)
        self.summarizer = VLMSummarizer(self.logger)
        self.uploader = ImageUploader(self.logger)
    pass

    def process(self, state: ImportGraphState) -> ImportGraphState:
         self.logger.info("=" * 50)
         self.logger.info("开始处理 Markdown 图片流水线")

         start_time = time.time()
         md_content,md_path_obj= self.file_handler.read_md(state)
         if md_content:
             state["md_content"] = md_content
         self.logger.info(f"[步骤1] 读取文档完成，耗时 {time.time() - start_time:.2f}s")

         image_info_list:list[ImageInfo] = self.image_scanner.scanner(md_content,img_content_length=self.config.img_content_length)
         self.logger.info(f"[步骤2] 扫描图片完成，共 {len(image_info_list)} 张，耗时 {time.time() - start_time:.2f}s")

         summaries:list[str] = self.summarizer.summarizer(image_info_list=image_info_list,document_title=md_path_obj.name,vl_model=self.config.vl_model,max_requests=self.config.max_requests)
         self.logger.info(f"[步骤3] 生成摘要完成，共 {len(summaries)} 条，耗时 {time.time() - start_time:.2f}s")
         for summary in summaries:
             print(summary)

         new_md_content = self.uploader.upload_and_replace(md_content=md_content,md_path_obj=md_path_obj,image_info_list=image_info_list,summaries=summaries,minio_bucket=self.config.minio_bucket,minio_base_url=self.config.get_minio_base_url())
         self.logger.info(f"[步骤4] 上传并替换完成，耗时 {time.time() - start_time:.2f}s")

         self.file_handler.backup(md_path_obj, new_md_content)
         state["md_content"] = new_md_content
         self.logger.info(f"[完成] 全部处理完成，总耗时 {time.time() - start_time:.2f}s")
         self.logger.info("=" * 50)
         return state
if __name__ == '__main__':
    setup_logging()
    node = MarkDownImageNode()
    state = {"md_path": r"G:\PY_SD\SD\ZK\文档\md\万用表RS-12的使用.md"}
    node(state)