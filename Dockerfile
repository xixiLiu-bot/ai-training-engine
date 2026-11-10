# 智训引擎 · 自进化闭环镜像
# 目标环境：4 核 8G、无 GPU、无外网（评测时只允许访问 PyTRIO 模型服务）
# 一次运行：造漂移流 → 完整闭环 → 产物写到 /app/out

FROM python:3.11-slim

ENV PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MPLBACKEND=Agg \
    SWAN_MODE=offline \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 依赖单独一层，利用构建缓存；必须用国内源，否则构建会卡在 pypi
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

# 代码 + 技能（data/ 不在仓库里，由 make_data.py 运行时生成）
COPY evolver ./evolver
COPY scripts ./scripts
COPY skills ./skills

# data/ 在干净 clone 里不存在，直接 COPY 会让构建失败；
# 这里只建空目录，数据由下面的 make_data.py 生成
RUN mkdir -p data

# 一次运行：造数据 → 跑完整闭环；日志同时写 out/run.log
# SwanLab 默认 offline，只写本地 swanlog/，评测时无网络也能跑
CMD ["sh", "-c", "mkdir -p out && python scripts/make_data.py && python scripts/run_pipeline.py 2>&1 | tee out/run.log"]
