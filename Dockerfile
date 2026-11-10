# 智训引擎 · 评测镜像
#
# 目标：一条命令跑完闭环，全程无网络、无 GPU、无交互。
#
# 构建（构建阶段可以联网）：
#     docker build -t zhixun-engine:1.0 .
#
# 运行（运行阶段完全离线）：
#     docker run --rm --network none \
#         -v "$PWD/data:/app/data:ro" \
#         -v "$PWD/out:/app/out" \
#         zhixun-engine:1.0
#
# 说明：
#   - 基础镜像选 slim，最终体积约 400~600MB，远低于 2GB 上限
#   - 不打包任何模型权重；驱动模型走远端推理服务（QWEN_BASE_URL）
#   - 依赖版本在 requirements.lock 里固定，保证可复现
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    MPLBACKEND=Agg \
    TZ=Asia/Shanghai

WORKDIR /app

# 只装锁定的运行时依赖；不装开发依赖（pytest / swanlab 等）
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir --disable-pip-version-check -r /app/requirements.lock

# 代码
COPY evolver /app/evolver
COPY scripts /app/scripts
COPY skills /app/skills
COPY run_offline.sh /app/run_offline.sh
RUN chmod +x /app/run_offline.sh

# 数据与产物用挂载卷，镜像里不放数据（保持体积小、也避免泄漏测试集）
RUN mkdir -p /app/data /app/out

# 非 root 运行：评测机不一定给 root
RUN useradd -m -u 1000 runner && chown -R runner:runner /app
USER runner

ENTRYPOINT ["/app/run_offline.sh"]
