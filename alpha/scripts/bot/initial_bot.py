from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)
import torch


# # downlaod model
# snapshot_download(
#     repo_id="facebook/blenderbot-400M-distill", cache_dir="./huggingface_models"
# )

model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/blenderbot-400M-distill",
    cache_dir="huggingface_models",
    local_files_only=True,
)
