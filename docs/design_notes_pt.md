# Setup de IA Local para Previsão de Jogos do Mundial

Este documento serve como resumo executivo e ficha de contexto técnico para alimentar extensões de IA (como o Gemini no VS Code) ou para guiar o desenvolvimento do projeto local.

---

## 1. Visão Geral do Projeto & Abordagem Tática

* **Objetivo:** Criar um sistema local que utilize um Modelo de Linguagem (LLM) para prever e analisar os jogos do Mundial de Futebol.
* **Filosofia Técnica:** **As LLMs não são calculadoras nem bolas de cristal.** O modelo não deve ser instado a "adivinhar" o vencedor diretamente a partir do seu conhecimento estático (o que causaria alucinações). Em vez disso, a LLM atuará como um **Analista de Dados Avançado**, recebendo dados estatísticos frios e estruturados em tempo real e transformando-os em análises táticas estruturadas e percentagens de probabilidade.
* **Arquitetura:** Pipeline em Python que consome dados de APIs/Datasets desportivos, limpa os dados em Pandas, injeta-os num prompt contextualizado (RAG/Agentic approach) e faz a inferência através de um motor local (Ollama).

---

## 2. Perfil de Hardware & Restrições

O sistema será executado localmente na seguinte máquina:
* **GPU:** NVIDIA GeForce RTX 3060 Ti (8GB VRAM GDDR6).
* **Memória RAM:** 32GB DDR4 @ 3200 MHz.
* **Impacto no Setup:**
    * **VRAM (8GB):** Limita o uso a modelos de tamanho **7B ou 8B** (7 a 8 mil milhões de parâmetros) obrigatoriamente **quantizados em 4-bit**. Estes modelos ocupam cerca de 4.5GB a 5GB em memória, deixando ~3GB livres para a janela de contexto (dados das equipas e histórico).
    * **RAM (32GB):** Garante margem absoluta para executar scripts Python pesados, manipulação de tabelas extensas em Pandas (históricos de mundiais) e processamento em paralelo sem gargalos no sistema operacional.

---

## 3. Stack Tecnológica Recomendada

### Motor de Inferência Local
* **Ollama:** Escolha padrão para gerir e servir os modelos locais de forma leve na porta `11434`.

### Modelos de Linguagem Selecionados (Uso de versões 4-bit via Ollama)
1.  **Llama 3.1 (8B):** Excelente em raciocínio geral, interpretação de tabelas de texto e seguimento de instruções estritas (`ollama run llama3.1`).
2.  **Qwen 2.5 (7B):** Modelo altamente refinado para lógica, matemática e análise de dados estruturados; frequentemente supera o Llama nesta categoria (`ollama run qwen2.5`).
3.  **Mistral v0.3 (7B):** Alternativa europeia leve, focada em sumarização e extração de tópicos concretos (`ollama run mistral`).

### Fontes de Dados & Métricas Avançadas (Foco em xG - Expected Goals)
* **API-Football (via RapidAPI):** Ideal para o pipeline de produção. O plano gratuito oferece 100 requisições diárias. Dá acesso a dados limpos, escalações, lesões e estatísticas dos jogos do Mundial.
* **Biblioteca `soccerdata` (Python):** Ferramenta de scraping ético que extrai tabelas completas do portal **FBref** (dados oficiais Opta) diretamente para DataFrames do Pandas. Excelente para criar a base de dados histórica sem custos.
* **StatsBomb Open Data (GitHub):** Datasets gratuitos e ultra-detalhados de Mundiais anteriores. Perfeito para calibrar e testar o prompt do modelo com dados reais do passado.
* *Nota de exclusão:* Evitar o portal *Understat*, pois embora tenha ótimos dados de xG, não cobre jogos de seleções (Mundiais).

---

## 4. Arquitetura do Fluxo de Trabalho (Pipeline)

1.  **Extração de Dados:** O script Python faz um pedido à API (ou lê o DataFrame do `soccerdata`) para recolher as métricas recentes das duas seleções (ex: xG dos últimos 5 jogos, posse de bola, golos sofridos, golos esperados contra, ausências por lesão).
2.  **Formatação do Contexto:** O Python transforma os dados estruturados em texto legível ou estruturas JSON simples que a LLM compreende facilmente.
3.  **Engenharia de Prompt (Prompt Engineering):** Os dados são injetados num prompt que define o comportamento da IA. Exemplo de estrutura:
    > *"Atua como um analista tático de elite. Analisa os seguintes dados do Confronto X vs Y: [Dados Injetados]. Considera o xG ofensivo da Equipa A e as fragilidades defensivas da Equipa B. Fornece uma análise tática de 2 parágrafos e atribui probabilidades em % para Vitória A, Empate e Vitória B."*
4.  **Inferência & Resposta:** O script envia o prompt via API local para o Ollama e devolve o resultado estruturado.

---

## 5. Próximos Passos para Desenvolvimento

* [ ] Configurar e testar o Ollama localmente com o comando `ollama run qwen3.5:9b`.
* [ ] Instalar as dependências de Python (`pip install pandas requests langchain_community`).
* [ ] Obter a chave API gratuita na API-Football ou testar a extração do FBref via `soccerdata`.
* [ ] Escrever o script base em Python para unificar a extração de dados e a chamada à API do Ollama.
