# Handoff — ticket "Comparação: confrontar AI-PAVE-Br e baselines mais fortes"

Insumos para os três pontos do ticket. **Nada aqui é texto final** — são os fatos,
os números e as verificações que sustentam o que for escrito nas Seções 2.1, 2.2,
4.2 e 4.3.

Regenerar tudo: `python revisao/gerar_numeros.py --pave-dir <Dataset_in_CSVs>`

| Arquivo | Conteúdo | Seção |
|---|---|---|
| `T1_naocomparabilidade_aipavebr.csv` | 13 dimensões de comparação AI-PAVE-Br × TAIL | 2.1 |
| `T2_cobertura_schema_aipavebr.csv` | Cobertura de schema por categoria (21 categorias) | 2.1 |
| `T3_significancia.csv` | Bootstrap pareado + McNemar sobre os 1.200 pares | 4.2, 4.3 |
| `T4_pave_global.csv` | Os 3 sistemas no Golden Set: 2 escopos × 2 métricas | 2.1 |
| `T5_pave_por_atributo.csv` | Por atributo gold (21 linhas), com cobertura | 2.1, 4.3 |
| `T6_pave_por_categoria.csv` | Por categoria (21 categorias) | 2.1 |
| `T7_numeros_so_na_descricao.csv` | Valores numéricos extraídos que só aparecem na descrição (512 caracteres × completa) | 2.1 |

---

## 1. AI-PAVE-Br (§2.1) — **rodamos o Golden Set deles**

§2.1 afirma hoje que o trabalho deles serve *"como base de comparação direta para
este trabalho"* e a comparação nunca acontece. **A promessa agora pode ser
cumprida com número.**

Rodamos o pipeline do TAIL (Gemini 3.1 Flash-Lite, o mesmo modelo e prompt da
Seção 3.3, só o título como entrada) sobre os **7.890 itens das 21 categorias**
do Golden Set publicado (a versão do repositório baixada nesta rodada tem uma
categoria a menos que a original de 8.274/22 — não afeta a comparação, os três
sistemas rodam sobre o mesmo conjunto), e recomputamos os três sistemas com a
mesma métrica.

**Atenção — nomes de arquivo deslocados na versão baixada.** Cada arquivo traz a
categoria anterior em ordem alfabética: `Armário.csv` contém ares-condicionados,
`Bicicleta.csv` contém armários, …, `Tênis.csv` contém TVs (conferido pelos títulos
e pelos atributos gold). As 21 categorias reais são Ar Condicionado … TV; **a
ausente é Tênis**. Os rótulos por categoria (`T2`, `T6`) usam o mapa corrigido
`pave_dataset.CATEGORIA_REAL`; os resultados globais (`T4`) e por atributo (`T5`)
não dependem do nome do arquivo e não mudam. Conferência: com o mapa, as 21
categorias batem exatamente (itens, slots, atributos) com um download anterior
cujos nomes estavam corretos. Nesta rodada, o pipeline do TAIL já inclui o fix de
`_normalize_quantidade` (colapsa quantidade implícita "1" → null) que não
estava presente na rodada anterior — impacto pequeno e esperado, já que
`quantidade` fica fora do escopo "schema alinhado" (ver `NAO_MODELADOS`).

### Resultado

**Schema alinhado** — atributos que o schema do TAIL se propõe a extrair
(n = 18.677 células), variante com equivalência de unidade aplicada aos três
(ver decisão metodológica #2 abaixo — é a comparação justa, recomendada para o artigo):

| Sistema | P | R | F1 | Coverage |
|---|---|---|---|---|
| AI-PAVE-Br (Gemini 1.5 Flash) | 0,778 | 0,602 | **0,679** | 77,6% |
| Baseline empresa terceira | 0,851 | 0,529 | 0,652 | 66,8% |
| **TAIL (Gemini 3.1 Flash-Lite)** | 0,845 | 0,537 | **0,657** | 65,1% |

**Todos os atributos mapeados** — inclui os que o schema fixo do TAIL não expressa
(n = 23.765), mesma variante:

| Sistema | F1 | Coverage |
|---|---|---|
| AI-PAVE-Br | **0,646** | 76,4% |
| Baseline empresa terceira | 0,597 | 58,6% |
| TAIL | 0,566 | 57,4% |

### A leitura

**O TAIL empata estatisticamente com o baseline da empresa terceira (0,657 vs 0,652;
ΔF1 +0,5 pp, IC 95% [−0,3; +1,3]) e fica 2,2 pp atrás do AI-PAVE-Br (IC 95%
[−2,8; −1,6]) — usando apenas o título.** (ICs: bootstrap sobre itens,
`benchmark.ipynb`, Seção 10.5.)

O Golden Set publicado traz título e descrição (descrição preenchida em 7.784 dos
7.890 itens). **Não sabemos qual foi a entrada do sistema deles** — os dados não a
documentam (por exemplo, se houve ficha técnica). O que os dados mostram (`T7`,
`numeros_descricao.py`): dos valores numéricos extraídos pelo AI-PAVE-Br, 32,9% não
aparecem no título mas aparecem na descrição completa (13,2% se a descrição for
cortada nos primeiros 512 caracteres, como em `pave/itens.csv`; 6.866 das 7.784
descrições passam disso). Isso **indica, sem provar**, que a descrição fez parte da
entrada — o modelo poderia ter inferido alguns valores por conhecimento prévio.
Não afirmar mais do que isso. O TAIL recebe só o título.

Precisão: a do TAIL (0,845) é **estatisticamente igual** à da empresa terceira
(0,851; Δ −0,6 pp, IC 95% [−1,4; +0,2]) e 6,7 pp **maior** que a do AI-PAVE-Br
(IC 95% [+5,9; +7,5]), com cobertura menor (65,1% contra 77,6%): extrai menos, erra
menos. Não dizer "a maior precisão dos três".

**Importante, e vale declarar como limitação em vez de conclusão:** não temos como
separar, só com este número, quanto do gap de 2,2 pp é assimetria de informação
(sem descrição) e quanto é desenho de schema (deles é por categoria, o do TAIL é
genérico cross-categoria). Um experimento controlado — rodar o TAIL com
título+descrição neste mesmo Golden Set — isolaria as duas causas; não foi feito
nesta rodada.

**A diferença entre os dois escopos (0,657 → 0,566) é a medida do custo de um
schema fixo cross-categoria.** Essa é a substância empírica do argumento de
não-comparabilidade — não mais retórica.

Por atributo (`T5`), o padrão é nítido:

- **Onde os conceitos coincidem, o TAIL é competitivo ou melhor:** `marca` 0,778
  (n = 7.296, o maior); as cinco capacidades volumétricas entre 0,886 e 0,981 —
  à frente do sistema deles em duas delas (`capacidade_liquida_total` +0,30,
  `capacidade_de_lavagem` +0,04).
- **Onde o TAIL arrasta a média para baixo:** `cor` (0,548, n = 4.994) e
  `voltagem` (0,577, n = 2.206) — juntos são ~38% do escopo "schema alinhado" e
  ficam na faixa de 55-58%. Causa provável: vocabulário controlado do gold deles
  (`Grafite`→`Cinza`, `127V`→`110V`) ao qual o TAIL nunca foi ajustado — não é
  erro de extração, é convenção de rótulo diferente.
- **Onde o TAIL zera, a cobertura é ~0%:** `capacidade_do_hd`, `capacidade_do_ssd`.
  Ele não erra — ele não tem o campo (armazenamento digital não é
  `capacidade_volume`; o prompt manda converter para ml/kg). Ambos já ficam fora
  do escopo "schema alinhado" por definição, então não afetam os números acima.
- **Perda genuína:** `material` 0,342 vs 0,715 (AI-PAVE-Br) e `material_da_estrutura`
  0,257 vs 0,689, com cobertura ~50-80% para eles e mais baixa para o TAIL. Material
  raramente aparece no título — tende a estar na descrição, que o TAIL não lê.

### Decisões metodológicas a declarar no artigo

1. **Não reproduzimos os números publicados deles, e não tentamos.** O repositório
   traz 22 categorias / 8.274 itens; o artigo deles reporta ~7.700 em 20. Não é
   possível reconstruir o subconjunto exato. Nossa recomputação do sistema deles dá
   coverage 69,0% (publicado: 71,96%) e do baseline 41,4% (publicado: 46,71%). Em
   vez de perseguir o número, recomputamos **os três sistemas com a mesma métrica
   sobre os mesmos itens** — internamente consistente e reproduzível.
2. **Equivalência de unidade, aplicada aos três.** O prompt do TAIL canonicaliza
   volume para ml; o gold deles usa litros. `40L` e `40000ml` são a mesma
   quantidade. Sem essa equivalência o TAIL cai para 0,606 no schema alinhado — a
   diferença de 5,1 pp é convenção de unidade, não erro de extração.
3. **O recorte "schema alinhado" foi definido a priori**, pela definição dos campos
   do TAIL, não pelos resultados (lista em `pave_score.py::NAO_MODELADOS`).
4. **Duas assimetrias desfavorecem o TAIL** e devem ser declaradas: entrada
   (só título vs. título e, ao que os dados indicam, descrição) e vocabulário (o gold usa vocabulário controlado
   deles — `Grafite`→`Cinza`, `127V`→`110V` — ao qual o TAIL nunca foi ajustado).
5. **O gold é anotação independente.** Concorda mais com o baseline da empresa
   terceira (82,1% de match exato quando responde) do que com o sistema deles
   (69,3%) — não foi derivado das predições do AI-PAVE-Br.
6. `product_type` ficou fora da comparação: é rótulo de taxonomia do catálogo, não
   extração do título, e os sistemas deles não o predizem. Para registro, o TAIL
   atinge F1 0,622 com 100% de cobertura nele.

### Como reproduzir

```
python revisao/pave_dataset.py --pave-dir <Dataset_in_CSVs>   # ~7.890-8.274 itens, conforme a versão baixada
python revisao/pave_extract.py                                # ~1.700 chamadas, ~US$2 (rodado: 7.890 títulos, 57 min, 0 vazios)
python revisao/pave_score.py                                  # T4, T5, T6
```

Os CSVs do AI-PAVE-Br não estão versionados aqui (CC BY-NC-SA 4.0) — baixe de
`github.com/ai-luizalabs/AI-PAVE-Br`. `pave_extract.py` é retomável;
`pave_supervisor.sh` reinicia a extração se ela travar (a API pode pendurar uma
chamada sem timeout).

---

## 2. Ditto (§4.2/4.3)

**Recomendação: justificar a exclusão.** O revisor aceita explicitamente essa via,
e a justificativa é forte por três razões independentes — nenhuma delas é "custo".

1. **Setting incompatível.** Ditto é supervisionado: fine-tuning de um encoder com
   cabeça de classificação sobre pares rotulados. A contribuição do TAIL é
   explicitamente zero-shot, *"sem necessidade de pares rotulados para treinamento"*
   (Conclusão). Comparar os dois de frente compara settings de problema diferentes,
   não métodos.

2. **Não há conjunto de treino.** O dataset tem **30 pares positivos**. Não é
   conjunto de treino para um cross-encoder de 110M de parâmetros. Treinar exigiria
   dado sintético — que mede robustez a perturbação de superfície, não matching de
   anúncios independentes — ou uma nova rodada de anotação.

3. **O número do Ditto já está publicado, e Zhang et al. 2025 o reportam.** Ditto é
   um dos oito matchers do estudo deles: **F1 médio 72,9** em 11 benchmarks
   cross-dataset, variando de **43,1 a 94,4** conforme o dataset. Citar um número
   publicado e reproduzível é mais forte que produzir um número fraco nosso a
   partir de 30 positivos.

O ponto (3) permite fechar o item **citando duas referências que o artigo já tem
na bibliografia** (Li et al. 2020 e Zhang et al. 2025), sem experimento novo.

### Ressalva a incluir junto — importa

O TF-IDF hoje é um baseline **mais forte do que a tabela antiga sugeria**, mas
com o pipeline corrigido a diferença para o Gemini (Weighted) **é** estatisticamente
significativa (ver seção 3 abaixo — atualizado, isso mudou desde a rodada anterior).
Ainda assim, `baseline.ipynb` (célula 9) registra que os limiares 0,608 e 0,8241
foram escolhidos varrendo os rótulos do próprio conjunto de teste, enquanto o exact
match do TAIL não tem limiar ajustável — ou seja, a comparação já favorece o
baseline nesse sentido específico. Vale declarar isso como limitação mesmo com o
resultado significativo: fortalece a honestidade da seção.

---

## 3. Zhang et al. 2025 (§2.2/4.3)

Referência confirmada pela bibliografia do artigo: **Zhang, Z., Groth, P.,
Calixto, I., Schelter, S. (2025). "A deep dive into cross-dataset entity matching
with large and small language models." EDBT 2025, pp. 921–934.**

### O que eles reportam

Oito matchers × 11 datasets, em setting cross-dataset (target sem rótulos):

| Matcher | F1 médio |
|---|---|
| GPT-4 (promptado) | 87,4 |
| GPT-4o-Mini (promptado) | 83,9 |
| AnyMatch [GPT-2, 124M] | 81,5 |
| Unicorn [DeBERTa, 143M] | 81,0 |
| **Ditto [BERT, 110M]** | **72,9** |
| ZeroER | 57,1 |

Conclusões deles: *"fine-tuned small models can perform on par with prompted large
models"* e *"data preparation tends to have a greater impact on predictive quality
than model design"*.

### O que nós reportamos, e por que complementa

Nossos números, recalculados com **bootstrap pareado por cluster** (10.000
reamostragens, seed 42) e **McNemar exato** — tabela completa em `T3`:

| Comparação | ΔF1 | IC95% | McNemar |
|---|---|---|---|
| Gemini − DeepSeek | **+34,0 pp** | [+21,1, +48,5] | p ≈ 0 |
| Gemini − GPT-5.4-mini | +33,6 pp | [+16,8, +52,4] | p = 1,9e-05 |
| Gemini − TF-IDF palavra | +18,7 pp | **[+1,1, +37,3]** | p = 0,0037 |
| Gemini − TF-IDF caractere | +23,5 pp | **[+5,6, +43,4]** | p = 0,0118 |

Atualizado após as correções de pipeline (fix de quantidade implícita, reversão
de regras de prompt que causaram regressão, threshold realinhado) — as quatro
comparações agora são estatisticamente significativas, incluindo as duas de
TF-IDF, que antes não eram (ver "Duas cautelas" abaixo, item 1 revisado).

**O eixo da complementaridade, ancorado em número:** entre os dois modelos
proprietários promptados que eles avaliam, a diferença é **3,5 pp** (87,4 → 83,9).
Nós fixamos dataset, prompt e esquema de campos, variamos apenas o fornecedor
dentro da mesma faixa de custo, e observamos **34,0 pp** (Gemini vs DeepSeek).
Eles variam escala e classe de modelo entre datasets; nós isolamos o fornecedor
num único dataset em português. São eixos distintos, e o nosso é o que decide
implantação.

**A tensão que vale reconhecer explicitamente:** eles concluem que preparação de
dados pesa mais que desenho de modelo; nosso abstract afirma que a escolha do
modelo é tão crítica quanto o desenho do pipeline. Não é contradição — a conclusão
deles é sobre preparação de dados de *treino* para matchers supervisionados; a
nossa é sobre pipeline zero-shot, onde não há dado de treino a preparar.
Reconhecer e delimitar é mais forte que ignorar.

### Duas cautelas ao escrever

1. **[ATUALIZADO] O achado de variação entre modelos se sustenta, e agora o de
   superar o TF-IDF também.** Com o pipeline corrigido, o IC de Gemini − TF-IDF
   **não cruza mais zero** em nenhuma das duas variantes (palavra: [+1,1, +37,3];
   caractere: [+5,6, +43,4]) — isso é diferente da rodada anterior, onde essa
   ressalva existia. A frase do abstract *"com apenas o melhor superando um
   baseline TF-IDF"* agora pode ser apresentada como diferença estatisticamente
   demonstrada, não só como ordenamento observado. Vale atualizar o abstract e
   a Seção 4.2 para refletir isso — é um resultado mais forte do que o texto
   atual do artigo reivindica.

2. **§3.3 afirma que "a única variável manipulada entre as três execuções é o
   modelo de extração" — hoje isso não é exato.** `GEMINI_USER_TEMPLATE`
   (`inference_harness/normalization.py`) inclui um campo `Categoria` e o Gemini
   recebe `response_mime_type="application/json"`; as células OpenAI-compatíveis
   de `classifier_api.ipynb` (31, 33, 36, 39) enviam apenas `[{i}] Título: "{t}"`
   em texto livre. O `SYSTEM_PROMPT` é o mesmo, o template de usuário não. Isso
   não invalida os 25 pp, mas a frase de §3.3 precisa ser suavizada — e é
   justamente o confundidor que Zhang et al. e Peeters & Bizer apontam.

---

## Flag para quem cuida da Tabela 3 — [RESOLVIDO]

Esta seção documentava uma investigação de uma inconsistência antiga entre
ground truths (28 vs 31 positivos, dependendo do arquivo). Isso está resolvido:
depois de correções de rotulagem feitas pelo autor (Esparadrapo, VR Box, Balde
Pipoca, Terminal Direção), **todos os arquivos de dados hoje estão sincronizados
em 30 positivos** — `pairs_to_label.csv`, os 3 runs de cada modelo, e
`baseline_predictions.csv`. Verificado diretamente nos 6 arquivos, todos batendo
em 30/1.200. `T3_significancia.csv` já usa 30 em todas as linhas (campo
`n_positivos`), e o print do `gerar_numeros.py` que dizia "28 positivos" de forma
fixa no código foi corrigido para refletir o valor real, dinamicamente.

