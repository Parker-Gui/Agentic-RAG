"""RAG 评估：检索指标（recall@k/MRR/nDCG）+ 端到端 faithfulness（复用 agents.audit）。

标注数据集的口径与规模待讨论；本包先把模块结构与指标函数落好。
运行：uv run python -m eval.run_eval
"""
