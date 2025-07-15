import json
import os
import random
from pathlib import Path
from prompt import prompt


def create_and_split_chat_dataset(
    input_path: str, 
    output_dir: str, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15
):
    """
    讀取原始 JSON 檔案，將 QA pair 轉換為 NeMo Chat SFT 格式，
    並將字典類型的 output 序列化為 JSON 字串，最後分割成 .jsonl 檔案。

    Args:
        input_path (str): 來源 .json 檔案的路徑。
        output_dir (str): 分割後檔案的輸出目錄。
        train_ratio (float): 訓練集的比例。
        val_ratio (float): 驗證集的比例。
    """
    print(f"--- 開始處理檔案並轉換為 Chat 格式：'{input_path}' ---")
    
    try:
        # 1. 讀取來源 JSON 檔案
        with open(input_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        if not isinstance(original_data, list):
            raise ValueError("輸入檔案必須包含一個 JSON 列表。")
            
        print(f"成功載入 {len(original_data)} 筆記錄。")

        # 2. 將每一筆記錄轉換為 NeMo Chat SFT 格式
        processed_data = []
        for item in original_data:
            if not isinstance(item, dict) or 'question' not in item or 'answer' not in item:
                print(f"警告：跳過格式不正確的記錄：{item}")
                continue

            question_text = item['question']
            original_answer = item['answer']
            
            # 如果 answer 是字典，使用 json.dumps 將其轉換為字串
            if isinstance(original_answer, dict):
                answer_text = "[" + json.dumps(original_answer, ensure_ascii=False) + "]"
            else:
                answer_text = str(original_answer)
            
            # 建立新的記錄，符合 system/mask/conversations 格式
            new_record = {
                "system": prompt,
                "mask": "User",
                "conversations": [
                    {"from": "User", "value": question_text},
                    {"from": "Assistant", "value": answer_text}
                ]
            }
            processed_data.append(new_record)
        
        print(f"成功將 {len(processed_data)} 筆記錄轉換為 NeMo Chat SFT 格式。")

        # 3. 隨機打亂資料
        random.shuffle(processed_data)
        print("資料集已隨機打亂。")

        # 4. 分割資料
        total_size = len(processed_data)
        train_end = int(total_size * train_ratio)
        val_end = train_end + int(total_size * val_ratio)

        train_data = processed_data[:train_end]
        val_data = processed_data[train_end:val_end]
        test_data = processed_data[val_end:]

        print("\n資料分割摘要:")
        print(f"  訓練集大小:     {len(train_data)}")
        print(f"  驗證集大小:     {len(val_data)}")
        print(f"  測試集大小:       {len(test_data)}")

        # 5. 建立輸出目錄並寫入 .jsonl 檔案
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        def write_to_jsonl(data: list, file_path: str):
            with open(file_path, 'w', encoding='utf-8') as f:
                for record in data:
                    # 將每個字典物件寫為單獨的一行
                    json_line = json.dumps(record, ensure_ascii=False)
                    f.write(json_line + '\n')
            print(f"成功將 {len(data)} 筆記錄寫入 '{file_path}'")

        print("\n正在寫入分割後的檔案...")
        write_to_jsonl(train_data, os.path.join(output_dir, 'training.jsonl'))
        write_to_jsonl(val_data, os.path.join(output_dir, 'validation.jsonl'))
        write_to_jsonl(test_data, os.path.join(output_dir, 'test.jsonl'))
        
        print(f"\n處理完成！所有檔案都已儲存在 '{output_dir}' 資料夾中。")

    except FileNotFoundError:
        print(f"錯誤：找不到輸入檔案 '{input_path}'。")
    except Exception as e:
        print(f"處理過程中發生未預期的錯誤：{e}")

# --- 主程式執行區塊 ---
if __name__ == "__main__":
    # 設定來源檔案和輸出目錄的完整路徑
    SOURCE_JSON_FILE = '/datasets/soc-20250703225140/dataset/dataset.json'
    OUTPUT_DIRECTORY = '/datasets/soc-20250703225140/dataset_split'
    # 執行資料處理與分割
    create_and_split_chat_dataset(SOURCE_JSON_FILE, OUTPUT_DIRECTORY)
