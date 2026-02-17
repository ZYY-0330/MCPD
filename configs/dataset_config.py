# configs/dataset_config.py
from pathlib import Path

# 基础路径 (模块级变量)
BASE_DATA_DIR = Path('/root/autodl-tmp/A_data/Eedi')

# 数据路径配置
RAW_DATA = {
    'train': BASE_DATA_DIR / 'q_train_set.csv',
    'val': BASE_DATA_DIR / 'q_val_set.csv',
    'test': BASE_DATA_DIR / 'q_test_set.csv',
    'knowledge': BASE_DATA_DIR / 'q.csv'
}

# ==================== 列名映射配置 ====================
USER_ID_COL = 'UserId'
PROBLEM_ID_COL = 'QuestionId'
SKILL_ID_COL = 'SubjectId'
CORRECT_COL = 'IsCorrect'
TEXT_CONTENT_COL = 'content'

TOTAL_SKILLS = 86

IMAGE_DIR = Path('/root/autodl-tmp/A_data/Eedi/images')
IMAGE_OUTPUT_FILE = Path('/root/autodl-tmp/A_data/Eedi/images/precomputed_images.pt')
TEXT_DIR = Path('/root/autodl-tmp/A_data/Eedi/text/questions.json')
KNOWLEDGE_JSON = Path('/root/autodl-tmp/A_data/Eedi/text/know.json')
KNOW_OUTPUT_FILE = Path('/root/autodl-tmp/A_data/Eedi/knowledge_bert_features.pt')
OUTPUT_FILE = Path('/root/autodl-tmp/A_data/Eedi/offline_features_cache.pt')

# 输出路径（目录）
OUTPUT_DIR = Path('/root/autodl-tmp/data_2/Eedi')

RESULTS_DIR = OUTPUT_DIR / 'results'
LOG_DIR = OUTPUT_DIR / 'logs'

# 模型配置
MODEL_PATH = "../MathBERT"

KNOWLEDGE_FREQ_CSV = Path('/root/autodl-tmp/data_2/Eedi/student_concept_relative_frequency.csv')
STUDENT_WEIGHT = Path('/root/autodl-tmp/data_2/Eedi/student_weights.csv')
KNOW_PT = Path('/root/autodl-tmp/data_2/Eedi/text_img/text_features.pt')

# 预处理参数
IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
