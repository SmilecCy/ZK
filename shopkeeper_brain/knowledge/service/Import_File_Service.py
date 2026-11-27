import datetime
import json
import logging
import os
import shutil
import uuid
from typing import Tuple

from fastapi import UploadFile


from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.main_graph import kb_import_graph_app
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import add_done_task, add_running_task, update_task_status

logger = logging.getLogger(__name__)

class ImportFileService:


    def process_upload_file(self, file:UploadFile) -> Tuple[str,str,str]:
        date_dir = self._get_date_dir()
        task_id = str(uuid.uuid4().hex[:8])
        file_dir = os.path.join(date_dir, task_id)
        add_running_task(task_id, "upload_file")
        import_file_path = self._save_upload_file_to_local(file,file_dir)
        self._save_upload_file_to_minio(import_file_path, file.filename)
        add_done_task(task_id, "upload_file")
        return task_id, file_dir, import_file_path

    def run_import_graph(self,task_id:str,file_dir:str,import_file_path:str):
        try:
            update_task_status(task_id, "processing")
            global_graph_init_status: ImportGraphState={"task_id": task_id,
                                                    "file_dir": file_dir,
                                                    "import_file_path": import_file_path}
            for event in kb_import_graph_app.stream(global_graph_init_status):
                for node_name, state in event.items():
                    print(f"[{task_id}] Completed Node: {node_name}")

                update_task_status(task_id, "completed")
        except Exception as e:
            update_task_status(task_id, "failed")
            print(f"[{task_id}] Error: {e}")

    def _get_date_dir(self)->str:
        return os.path.join(get_local_base_dir(),datetime.datetime.now().strftime("%Y%m%d"))

    def _save_upload_file_to_local(self, file:UploadFile, file_dir:str)->str:
        os.makedirs(file_dir, exist_ok=True)
        import_file_path = os.path.join(file_dir, file.filename)
        try:
            with open (import_file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        except Exception as e:
            logger.info(f"{file.filename}写入临时目录失败 原因:{str(e)}")
            raise e

        return import_file_path

    def _save_upload_file_to_minio(self, import_file_path:str, filename:str):
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.error(f"MinIO客户端获取失败 原因:{str(e)}")
            return
        bucket_name = os.getenv('MINIO_BUCKET_NAME')
        object_name = f"origin_files/{datetime.datetime.now().strftime('%Y%m%d')}/{filename}"
        try:
            minio_client.fput_object(bucket_name, object_name, import_file_path)
        except Exception as e:
            logger.error(f"{filename}上传到MinIO失败 原因：{str(e)}")
