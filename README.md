# ADL 2025 Final Project — Team 10

本專案為 [ADL 2025 Final Challenge: Jailbreak Olympics](https://github.com/yenshan0530/2025-ADL-Final-Challenge-Release) 的 Team 10 實作：設計一個 agent，將 toxic jailbreak prompt 改寫成能繞過 guard model、同時保留原始意圖的 prompt。

> **Disclaimer**：本專案僅供課程作業與 AI 安全研究使用，資料與結果中含有有害 prompt 的改寫內容，請勿用於任何實際攻擊用途。

## 方法

以 QLoRA 微調 [`huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2`](https://huggingface.co/huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2)，讓模型直接學會「toxic prompt → 看起來安全但保留意圖的 prompt」的改寫（`algorithms.py` 中的 `finetuned_attack`）。

- 4-bit NF4 量化 + double quant
- LoRA r=64, alpha=16, dropout=0.05，套用於所有 attention / MLP projection
- 500 steps, lr 2e-4, cosine scheduler

本 repo 無提供訓練好的模型權重，請依照下方「模型訓練方法」自行訓練 LoRA adapter。

## 結果

`ADL_Final_25W_part2_with_cost`（389 筆）：

| Metric                     | Score  |
| -------------------------- | ------ |
| Average safety score       | 0.6748 |
| Average relevance score    | 0.6478 |
| Final acc                  | 0.4165 |
| Weighted final acc         | 0.4103 |

`results/` 中的檔名後綴：`_1` 為 Dolphin 版本、`_2` 為 Qwen2.5 版本的改寫結果。

# 使用方法

## 1. 環境建置

```bash
# 進入程式碼資料夾
cd code

# 建立 conda 虛擬環境
conda create -n adl-final python=3.12
conda activate adl-final
pip install -r requirements.txt
```

## 2. 準備模型

推論需要 LoRA adapter 放在 `code/adapter_model/`。請先依照「模型訓練方法」訓練，再將 checkpoint 中的 adapter 複製過來：

```bash
cp -r src/finetune/finetuned_models/Qwen2.5-14B-finetuned/checkpoint-500/adapter_model adapter_model
```

Base model `huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2` 會在推論時自動從 Hugging Face 下載。

## 3. 執行

```bash
# 執行隱藏集推論
python run_inference.py --dataset theblackcat102/ADL_Final_25W_part2_with_cost --algorithm finetuned_attack

# 執行隱藏集評估
python run_eval.py --dataset theblackcat102/ADL_Final_25W_part2_with_cost --algorithm finetuned_attack
```

# 資料夾結構

```
./
├── code/
│   ├── src/
│   │   ├── finetune/       // 訓練 model 之子資料夾
│   │   │   ├── data/
│   │   │   │   └── example_data.jsonl
│   │   │   ├── preprocess.py
│   │   │   ├── qlora.py
│   │   │   ├── README.md
│   │   │   ├── requirements.txt
│   │   │   └── train.sh
│   │   ├── agent.py        // 原有 sample code
│   │   ├── algorithms.py   // model finetune attack 實做
│   │   ├── eval.py         // 原有 sample code
│   │   ├── prompts.py      // 原有 sample code
│   ├── adapter_model/      // LoRA adapter（需自行訓練，不在 repo 中）
│   ├── results/            // finetuned_attack 在 part2 的推論與評估結果
│   ├── requirements.txt    // 新增 peft, bitsandbytes packages
│   ├── run_eval.py         // 原有 sample code
│   └── run_inference.py    // 原有 sample code
├── results/
└── README.md
```

# 模型訓練方法

## 1. 環境設定

```bash
# 進入 finetune 資料夾
cd code/src/finetune

# 建立 conda 虛擬環境
conda create -n finetune python=3.10 -y
conda activate finetune

# 安裝相依套件
pip install -r requirements.txt
```

## 2. 訓練模型

使用 `train.sh` 進行 QLoRA 微調訓練：

```bash
bash train.sh
```

### 需要調整的參數

在 `train.sh` 中：

| 參數         | 說明               | 範例                                          |
| ------------ | ------------------ | --------------------------------------------- |
| `MODEL`      | 基底模型名稱或路徑 | `cognitivecomputations/dolphin-2.9-llama3-8b` |
| `INPUT_FILE` | 訓練資料路徑       | `data/example_data.jsonl`                     |
| `OUTPUT_DIR` | 模型輸出目錄       | `finetuned_models/llama3-8b-finetuned`        |

`python qlora.py` 的相關參數：
batch_size 請根據 GPU vram 大小調整，預設這樣是跑 18~19G
其他參數就做調參吧（佛系）

在 `preprocess.py` 中：

| 參數                   | 說明                          |
| ---------------------- | ----------------------------- |
| `INSTRUCTION_TEMPLATE` | Prompt 模板，定義模型輸入格式 |

>  **重要**：`INSTRUCTION_TEMPLATE` 會影響後續推論與評估，請確保訓練與推論時使用相同的模板。
