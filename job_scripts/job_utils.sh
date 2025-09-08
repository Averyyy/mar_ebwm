if [ -z "${MASTER_PORT}" ]; then
  if command -v shuf >/dev/null 2>&1; then
    MASTER_PORT=$(shuf -i 20000-65000 -n 1)
  else
    MASTER_PORT=$(( (RANDOM % 45000) + 20000 ))
  fi
fi
echo "MASTER_PORT: ${MASTER_PORT}"

MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_NODELIST:-${SLURM_JOB_NODELIST:-}}" 2>/dev/null | head -n1)}
MASTER_ADDR=${MASTER_ADDR:-$(hostname -f 2>/dev/null || echo localhost)}
echo "MASTER_ADDR: ${MASTER_ADDR}"