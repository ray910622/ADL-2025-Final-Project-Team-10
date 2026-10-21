# Fine-tune Model Attack

此文件用於說明微調 LLM 模型，使其學會將 toxic jailbreak prompts 轉換為「看起來安全但保留原始意圖」的 prompts。

# Training Stage

## 1. 環境設定

```bash
# 進入 finetune 資料夾
cd finetune

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

> ⚠️ **重要**：`INSTRUCTION_TEMPLATE` 會影響後續推論與評估，請確保訓練與推論時使用相同的模板。
