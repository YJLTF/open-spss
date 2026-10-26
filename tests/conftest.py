"""测试全局配置:在任何 open_spss 导入之前,把数据落盘目录指到临时目录,
避免测试读写真实的用户数据/仓库 data 目录。"""
import os
import tempfile

os.environ.setdefault(
    "OPENSPSS_DATA_DIR",
    os.path.join(tempfile.mkdtemp(prefix="ospss-test-data-"), "datasets"),
)
