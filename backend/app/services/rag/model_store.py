"""模型权重的落地位置：仓库内 ``models/`` 优先，否则回落到 ModelScope 缓存。

嵌入与 rerank 都是几个 GB 的权重，放在仓库根 ``models/<模型名>`` 下便于离线部署、
备份和挂载进容器；没有本地副本时仍按原来的方式从 ModelScope 下载（缓存在
``~/.cache/modelscope``）。设 ``RAG_MODELS_DIR`` 可把这个目录移到仓库外。

下载一个模型到本地目录（以 BGE-m3 为例，跳过 onnx/ 与 imgs/）::

    uv run python -m app.services.rag.model_store BAAI/bge-m3
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# 与 crawler.common.paths 同样的算法：定位到工作区根，而不是 backend/。
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_ROOT = _REPO_ROOT.parent if _REPO_ROOT.name == "backend" else _REPO_ROOT

MODELS_DIR = Path(os.getenv("RAG_MODELS_DIR") or _PROJECT_ROOT / "models")

# 只取 PyTorch 那条链路：onnx/ 里的 model.onnx_data 与 pytorch_model.bin 内容重复
# （各 2.3GB），imgs/ 是 README 用的截图。
_ALLOW_PATTERNS = [
    "pytorch_model.bin",
    "model.safetensors",
    "colbert_linear.pt",  # BGE-m3 的 colbert 表示
    "sparse_linear.pt",  # BGE-m3 的 sparse(lexical) 表示，hybrid 检索需要
    "*.json",
    "sentencepiece.bpe.model",
    "1_Pooling/config.json",
]
_IGNORE_PATTERNS = ["onnx/*", "imgs/*"]


def local_dir_for(model_id: str) -> Path:
    """``BAAI/bge-m3`` → ``<repo>/models/bge-m3``（不保证存在）。"""
    return MODELS_DIR / model_id.split("/")[-1]


def has_local_copy(model_id: str) -> bool:
    """本地目录里是否有一份看起来完整的权重。

    以 config.json + 权重文件同时存在为准：只有 config.json 的目录通常是一次被
    中断的下载，交给 FlagEmbedding 会在加载时才炸。
    """
    path = local_dir_for(model_id)
    if not (path / "config.json").is_file():
        return False
    return any((path / name).is_file() for name in ("pytorch_model.bin", "model.safetensors"))


def resolve_model_dir(model_id: str) -> str:
    """模型权重所在目录：本地副本优先，否则从 ModelScope 下载后返回缓存路径。"""
    if has_local_copy(model_id):
        path = local_dir_for(model_id)
        logger.info("使用本地模型目录: %s", path)
        return str(path)

    from modelscope import snapshot_download

    logger.info("本地无 %s，改从 ModelScope 下载（缓存目录）...", model_id)
    return snapshot_download(model_id, ignore_file_pattern=["onnx"])


def download_to_local(model_id: str) -> Path:
    """把模型下载到仓库内 ``models/<模型名>``，返回该目录。

    ``snapshot_download`` 在 ``local_dir`` 模式下会忽略 ``ignore_file_pattern``，
    所以这里用 ``allow_patterns`` / ``ignore_patterns`` 白名单来避开 onnx 权重。
    """
    from modelscope import snapshot_download

    target = local_dir_for(model_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("下载 %s → %s", model_id, target)
    snapshot_download(
        model_id,
        local_dir=str(target),
        allow_patterns=_ALLOW_PATTERNS,
        ignore_patterns=_IGNORE_PATTERNS,
        max_workers=4,
    )
    logger.info("下载完成: %s", target)
    return target


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if len(sys.argv) != 2:
        logger.error("用法: python -m app.services.rag.model_store <model_id>  (例: BAAI/bge-m3)")
        raise SystemExit(2)
    logger.info("模型目录: %s", download_to_local(sys.argv[1]))
