import json
import subprocess
import time
from pathlib import Path
from typing import Tuple


from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    name = "pdf_to_md_node"
    pass

    def process(self, state: ImportGraphState) -> ImportGraphState:
        import_file_path,file_dir = self._validate_state_inputs_path(state)

        processed_code = self._execute_mineru(import_file_path,file_dir)
        if processed_code != 0:
            raise ValidationError(f"MinerU解析PDF文件失败，返回码：{self.name}")

        md_path = self._get_md_path(import_file_path,file_dir)

        state["md_path"] = md_path


        return state

    def _validate_state_inputs_path(self, state)->Tuple[Path, Path]:
        import_file_path = state.get("import_file_path")
        file_dir = state.get("file_dir")

        if import_file_path:
            import_file_path_obj =Path(import_file_path)
            if not import_file_path_obj.exists():
                raise ValidationError("文件不存在")
        else:
            raise ValidationError("文件不存在")

        file_dir_obj = Path(file_dir)
        if not file_dir_obj.exists():
            raise ValidationError("文件路径不存在")
        return import_file_path_obj, file_dir_obj

    def _execute_mineru(self, import_file_path, file_dir)->int:

        cmd = ["mineru-kit", "parse", import_file_path, "-o", file_dir, "--tier", "standard"]
        process_start_time = time.time()


        sub_process = subprocess.Popen(args=cmd,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       errors="replace",
                                       encoding="utf-8",
                                       bufsize=1)
        assert sub_process.stdout is not None
        for line in sub_process.stdout:
            self.logger.info(f"执行MinerU产生的日志：{line}")
        sub_process_code =sub_process.wait()
        process_end_time = time.time()
        if sub_process_code ==0:
            self.logger.info(f"MinerU成功解析PDF文件：{import_file_path.name}"
                             f"耗时:{process_end_time - process_start_time:.2f}s")
        else:
            self.logger.error(f"MinerU解析PDF文件：{import_file_path.name}失败")




        return sub_process_code

    def _get_md_path(self, import_file_path, file_dir)->str:
        file_name = import_file_path.stem
        md_path = file_dir/f"{file_name}.md"


        return str(md_path)

if __name__ == '__main__':
    setup_logging()
    pdf_to_md_node = PdfToMdNode()
    pdf_to_md_node_init_state: ImportGraphState = {"import_file_path":r"G:\PY_SD\SD\ZK\文档\pdf\万用表RS-12的使用.pdf",
                                 "file_dir":r"G:\PY_SD\SD\ZK\文档\md"}

    processed_result = pdf_to_md_node.process(pdf_to_md_node_init_state)
    print(json.dumps(processed_result, indent=4,ensure_ascii=False))