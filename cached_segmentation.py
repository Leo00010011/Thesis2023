from utils.SAM_utils import create_cached_sam_inter, review_images_with_saved_emb, TYPE_VIT_B, WPATH_VIT_B , save_embeddings
from pathlib import Path
'''
Usar las flechas para navegar por las imágenes
Usar espacio para crear el embedding y poder hacer el prompt
    - click izquierdo para poner un promtp de foreground
    - click derecho para poner un prompt de background
    - 'C' para borrar el prompt
    - 'ENTER' para hacer la segmentación
    - 'Escape' para volver a poder navegar en las fotos
'''

data_path = Path('output\\color')
path_list = [str(path) for path in data_path.rglob('*.jpg')]
path_list = path_list[0:15]
data_path = save_embeddings(path_list,'output\\embeddings',TYPE_VIT_B, WPATH_VIT_B)

# data_path = 'output\\embeddings\\data.json'
review_images_with_saved_emb(data_path,create_cached_sam_inter(TYPE_VIT_B, WPATH_VIT_B))