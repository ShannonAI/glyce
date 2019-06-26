__version__ = "0.1.0"


import os 
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path) 




from .layers.char_glyph_embedding import CharGlyphEmbedding as CharGlyceEmbedding
from .layers.word_glyph_embedding import WordGlyphEmbedding as WordGlyceEmbedding
from .utils.default_config import GlyphEmbeddingConfig as GlyceConfig
from .glyph_cnn_models.glyph_group_cnn import GlyphGroupCNN










