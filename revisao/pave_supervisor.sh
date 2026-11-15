#!/bin/bash
# Supervisiona pave_extract.py: reinicia se o JSONL parar de crescer.
# Motivo: uma chamada da API pode ficar pendurada sem timeout (aconteceu quando
# a máquina dormiu), deixando o processo vivo em 0% de CPU e sem progresso.
cd /Users/MezzowTecnologia/projects/invoice-data-normalizer
JSONL=revisao/pave/tail_pred.jsonl
LOG=revisao/pave/extract.log
ALVO=8274
PARADO=300   # segundos sem crescer -> considera travado

while :; do
  n=$(wc -l < "$JSONL" 2>/dev/null || echo 0)
  [ "$n" -ge "$ALVO" ] && { echo "SUPERVISOR: completo ($n/$ALVO)"; break; }

  if ! pgrep -f pave_extract.py >/dev/null; then
    echo "SUPERVISOR: processo ausente, iniciando ($n/$ALVO)"
    nohup python3 revisao/pave_extract.py >> "$LOG" 2>&1 &
    sleep 60; continue
  fi

  idade=$(( $(date +%s) - $(stat -f %m "$JSONL") ))
  if [ "$idade" -gt "$PARADO" ]; then
    echo "SUPERVISOR: travado ha ${idade}s em $n/$ALVO — reiniciando"
    pkill -9 -f pave_extract.py; sleep 3
    nohup python3 revisao/pave_extract.py >> "$LOG" 2>&1 &
  fi
  sleep 60
done
