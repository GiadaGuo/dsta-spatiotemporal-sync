
###
 # @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 # @Date: 2026-10-23 10:35:33
 # @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 # @LastEditTime: 2026-10-23 10:35:44
 # @FilePath: /dsta-spatiotemporal-sync/scripts/sync_to_server.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
#!/usr/bin/env bash
set -euo pipefail

REMOTE="zju"
REMOTE_DIR="/data/slx/project/dsta-spatiotemporal-sync"

# 确保远程目录存在
ssh "$REMOTE" "mkdir -p '$REMOTE_DIR/data/processed' '$REMOTE_DIR/outputs'"

echo "=== 同步项目代码 ==="

rsync -avz \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.DS_Store' \
  --exclude='data/' \
  --exclude='outputs/' \
  ./ "$REMOTE:$REMOTE_DIR/"

echo "=== 同步训练所需数据 ==="

rsync -avz \
  data/processed/city_state.parquet \
  data/processed/anomaly_states.parquet \
  data/processed/city_state_embeddings.parquet \
  "$REMOTE:$REMOTE_DIR/data/processed/"

echo "=== 同步完成 ==="
