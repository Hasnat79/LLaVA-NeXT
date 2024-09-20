from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle


from PIL import Image
import requests
import copy
import torch

import sys
import warnings

from PIL import Image
import requests
import copy
import torch

import sys
import warnings
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from functools import lru_cache

warnings.filterwarnings("ignore")

def prepare_inputs(_question, image):
  image_tensor = process_images([image], image_processor, model.config)
  image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
  conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
  question = DEFAULT_IMAGE_TOKEN + _question  #"\nWhat is shown in this image?"
  conv = copy.deepcopy(conv_templates[conv_template])
  conv.append_message(conv.roles[0], question)
  conv.append_message(conv.roles[1], None)
  prompt_question = conv.get_prompt()
  input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
  image_sizes = [image.size]

  return input_ids,image_tensor , image_sizes
@lru_cache(maxsize=1)
def setup_model(pretrained, model_name, device, device_map, attn_implementation=None):
  tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation=attn_implementation)  # Add any other thing you want to pass in llava_model_args
  model.eval()
  return tokenizer, model, image_processor, max_length

# tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map,attn_implementation=None)  # Add any other thing you want to pass in llava_model_args
def prettify(text):
  print("\n\n ===========Generated Output ============================\n\n")
  highlighted_text = highlight(text, PythonLexer(), TerminalFormatter())
  print(highlighted_text)
  print("\n\n ===========End of Generated Output =====================")

if __name__ == "__main__":
  pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-si"
  model_name = "llava_qwen"
  device = "cuda"
  device_map = "auto"
  attn_implementation = None
  tokenizer ,model , image_processor, max_length = setup_model(pretrained, model_name, device, device_map, attn_implementation)
  model.eval()

# ============================================================
 
  chart = "data/image/chart.png"
  sofa_under_water = "data/image/sofa_under_water.jpeg"


  # image = Image.open(chart)
  image= Image.open(sofa_under_water)

  # question = "\nWhat is shown in this image?"
  question= "\nPlease describe this image in detail. Do you think the image is unusual or not?"

  input_ids,image_tensor, image_sizes = prepare_inputs(question,image)
  cont = model.generate(
      input_ids,
      images=image_tensor,
      image_sizes=image_sizes,
      do_sample=False,
      temperature=0,
      max_new_tokens=4096,
  )
  text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
  prettify(text_outputs[0])
# print(text_outputs)
