import os
import subprocess
from pathlib import Path
import json



from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState

class EntryNode(BaseNode):
    name = "entry_node"
    pass

    def process(self, state: ImportGraphState) -> ImportGraphState:
        file_dir:str = state.get("file_dir")
        import_file_path:str = state.get("import_file_path")

        if not file_dir or not import_file_path:
            raise ValidationError("文件不存在")

        path = Path(import_file_path)

        suffix = path.suffix.lower()

        if suffix == ".md":
            state['is_md_read_enabled'] = True
            state['md_path'] = import_file_path
        elif suffix == ".pdf":
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = import_file_path
        else:
            self.logger.debug(f"文件类型{suffix}不支持")
            raise ValidationError(f"文件类型{suffix}不支持")

        file_title = path.stem
        state['file_title'] = file_title


        return state



def text(inpout:Path,output:Path):
    # 基础用法：run() - 同步执行，等待完成
    os.environ['MINERU_MODEL_SOURCE'] = 'modelscope'  # 设置MinerU的模型来源 ModelScope（阿里魔搭）
    os.environ['MODELSCOPE_OFFLINE'] = '1'  # 1 表示开启离线模式（MODELSCOPE_CACHE 指定的目录）
    os.environ['HF_HUB_OFFLINE'] = '1'  # 启用 HuggingFace Hub 离线模式
    os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 启用 Transformers 库离线模式

    print("✅ 环境变量已配置")
    print(f"   MINERU_MODEL_SOURCE = {os.environ.get('MINERU_MODEL_SOURCE')}")
    print(f"   MODELSCOPE_OFFLINE = {os.environ.get('MODELSCOPE_OFFLINE')}")
    print(f"   HF_HOME = {os.environ.get('HF_HOME')}")
    print()
    subprocess.run( [
    "mineru",
    "-p",
    inpout,
    "-o",
    output,
    "--backend",
    "pipeline"
],
capture_output=True,  # 捕获输出
encoding='utf-8',  # 指定 UTF-8 编码（Windows 下避免 GBK 问题）
errors='replace',  # 遇到无法解码的字符用  替换
check=False  # 不自动抛异常，手动检查
)

def text2(inpout:Path,output:Path):
    os.environ['MINERU_MODEL_SOURCE'] = 'modelscope'
    os.environ['MODELSCOPE_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    print("✅ 环境变量已配置")
    print(f"   MINERU_MODEL_SOURCE = {os.environ.get('MINERU_MODEL_SOURCE')}")
    print(f"   MODELSCOPE_OFFLINE = {os.environ.get('MODELSCOPE_OFFLINE')}")
    print(f"   HF_HOME = {os.environ.get('HF_HOME')}")
    print()
    proc = subprocess.Popen(
        args=["mineru-kit", "parse", inpout, "-o", output, "--tier", "standard"],
        stdout=subprocess.PIPE,  # 捕获标准输出
        stderr=subprocess.STDOUT,  # 合并错误到标准输出
        text=True, encoding="utf-8", errors="replace",  # 遇到乱码时替换
        bufsize=1  # 行缓冲，实时输出
    )

    for line in proc.stdout:
        print(line.rstrip())

    return_code = proc.wait()
    print(return_code)
    pass
def text3(import_file_path:Path,file_dir_path:Path):
    file_name = import_file_path.stem
    md_path = file_dir_path.parents[0]/"md"/f"{file_name}.md"
    print(md_path)




if __name__ == '__main__':


    setup_logging()
    test_entry_state = {"file_dir": r"G:\PY_SD\SD\ZK\文档\pdf",
                        "import_file_path": r"G:\PY_SD\SD\ZK\文档\pdf\万用表RS-12的使用.pdf"}
    entry_node = EntryNode()

    processed_state = entry_node(test_entry_state)
    print(json.dumps(processed_state, ensure_ascii=False, indent=4))



    # inpout = Path(r"G:\PY_SD\SD\ZK\文档\pdf\万用表RS-12的使用.pdf")
    # output = Path(r"G:\PY_SD\SD\ZK\文档\md")
    # # text(inpout, output)
    # # text2(inpout, output)
    # text3(inpout, output)
